from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from braid.contract import (
    ContractValidationError,
    DerivationRecord,
    EventMarginal,
    ForecastDistribution,
    ForecastRequest,
    GraphBundleV2,
    GraphEventV2,
    Operation,
    PatchWindow,
    PayloadSnapshot,
    ProvenanceRecord,
    RelationDecl,
    RoleBinding,
    SchemaDecl,
    TaskDeclaration,
    validate_bundle,
    validate_forecast_distribution,
    validate_forecast_request,
    validate_proposed_patch,
)


def _request(
    bundle: GraphBundleV2, cutoff: datetime, *, support: tuple[GraphEventV2, ...] = ()
) -> ForecastRequest:
    return ForecastRequest(
        request_id="request:causal-test",
        schema=bundle.schema,
        prefix=bundle,
        cutoff=cutoff,
        horizon=timedelta(days=5),
        task=TaskDeclaration("patch"),
        support_events=support,
    )


def _create_proposal(request: ForecastRequest, observed_at: datetime) -> GraphEventV2:
    return GraphEventV2(
        event_id="proposal:create-person",
        observed_at=observed_at,
        valid_from=observed_at,
        valid_to=None,
        operation=Operation.CREATE_NODE,
        schema_version=request.schema.version,
        relation=None,
        arguments=(RoleBinding("node", "proposal:person"),),
        payload={"node_id": "proposal:person", "node_type": "Person"},
        basis_refs=(),
        derivation=DerivationRecord("model"),
        provenance=ProvenanceRecord("model-proposal", None, None, observed_at),
    )


def test_forecast_request_rejects_future_payload_snapshots(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    node = valid_bundle.nodes[0]
    future = PayloadSnapshot(
        observed_at=times[5],
        valid_from=times[5],
        payload={"secret_future": True},
    )
    prefix = replace(
        valid_bundle,
        nodes=(
            replace(node, payload_history=(*node.payload_history, future)),
            *valid_bundle.nodes[1:],
        ),
    )

    with pytest.raises(ContractValidationError, match="not visible at cutoff"):
        validate_forecast_request(_request(prefix, times[2]))


def test_support_events_receive_full_contract_validation(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    original = valid_bundle.events[0]
    invalid = replace(
        original,
        event_id="support:invalid",
        observed_at=times[2],
        valid_from=times[2],
        schema_version="missing-schema",
        relation="UNKNOWN",
        arguments=(RoleBinding("bogus", "missing-node"),),
        basis_refs=("missing-evidence",),
        provenance=replace(original.provenance, acquired_at=times[2]),
    )

    with pytest.raises(ContractValidationError) as caught:
        validate_forecast_request(_request(valid_bundle, times[3], support=(invalid,)))
    message = str(caught.value)
    assert "unknown schema 'missing-schema'" in message
    assert "references unknown node 'missing-node'" in message
    assert "dangling evidence reference 'missing-evidence'" in message


def test_temporal_request_rejects_explicitly_imputed_clocks(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    imputed = replace(valid_bundle.events[0], observed_at_imputed=True)
    prefix = replace(valid_bundle, events=(imputed,))

    with pytest.raises(ContractValidationError, match="imputed observation time"):
        validate_forecast_request(_request(prefix, times[2]))


def _schema_v2(valid_bundle: GraphBundleV2, observed_at: datetime) -> SchemaDecl:
    original = valid_bundle.schema.relations[0]
    added = RelationDecl(
        "REVIEWS",
        roles=original.roles,
        description="A person reviews a repository.",
    )
    return replace(
        valid_bundle.schema,
        version="2.1.0",
        observed_at=observed_at,
        relations=(*valid_bundle.schema.relations, added),
    )


def test_novel_schema_use_requires_an_ordered_schema_change(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    schema_v2 = _schema_v2(valid_bundle, times[4])
    original = valid_bundle.events[0]
    use = replace(
        original,
        event_id="event:review",
        observed_at=times[4],
        valid_from=times[4],
        schema_version=schema_v2.version,
        relation="REVIEWS",
        provenance=replace(original.provenance, acquired_at=times[4]),
    )
    bundle = replace(
        valid_bundle,
        schemas=(*valid_bundle.schemas, schema_v2),
        events=(*valid_bundle.events, use),
    )

    with pytest.raises(ContractValidationError, match="requires exactly one SCHEMA_CHANGE"):
        validate_bundle(bundle)


def test_ordered_schema_change_authorizes_new_schema_use(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    schema_v2 = _schema_v2(valid_bundle, times[4])
    original = valid_bundle.events[0]
    change = GraphEventV2(
        event_id="event:schema-change",
        observed_at=times[4],
        valid_from=times[4],
        valid_to=None,
        operation=Operation.SCHEMA_CHANGE,
        schema_version=valid_bundle.schema.version,
        relation=None,
        arguments=(),
        payload={
            "previous_schema_version": valid_bundle.schema.version,
            "new_schema_version": schema_v2.version,
            "schema_hash": schema_v2.canonical_hash(),
        },
        basis_refs=(),
        derivation=DerivationRecord("schema-test"),
        provenance=ProvenanceRecord("test", None, None, times[4]),
    )
    use = replace(
        original,
        event_id="event:review",
        observed_at=times[4],
        valid_from=times[4],
        schema_version=schema_v2.version,
        relation="REVIEWS",
        provenance=replace(original.provenance, acquired_at=times[4]),
    )
    bundle = replace(
        valid_bundle,
        schemas=(*valid_bundle.schemas, schema_v2),
        events=(*valid_bundle.events, change, use),
    )

    validate_bundle(bundle)


def test_contract_native_patch_validation_uses_private_proposal_state(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    request = _request(valid_bundle, times[2])
    evidence_id = valid_bundle.evidence[0].evidence_id
    create = GraphEventV2(
        event_id="proposal:create-person",
        observed_at=times[3],
        valid_from=times[3],
        valid_to=None,
        operation=Operation.CREATE_NODE,
        schema_version=request.schema.version,
        relation=None,
        arguments=(RoleBinding("node", "proposal:person"),),
        payload={"node_id": "proposal:person", "node_type": "Person"},
        basis_refs=(evidence_id,),
        derivation=DerivationRecord("model", (evidence_id,)),
        provenance=ProvenanceRecord("model-proposal", None, None, times[3]),
    )
    assertion = GraphEventV2(
        event_id="proposal:assert",
        observed_at=times[4],
        valid_from=times[4],
        valid_to=None,
        operation=Operation.ASSERT,
        schema_version=request.schema.version,
        relation="CONTRIBUTES_TO",
        arguments=(
            RoleBinding("actor", "proposal:person"),
            RoleBinding("target", "repo:braid"),
        ),
        payload={},
        basis_refs=(evidence_id,),
        derivation=DerivationRecord("model", (create.event_id, evidence_id)),
        provenance=ProvenanceRecord("model-proposal", None, None, times[4]),
    )

    validate_proposed_patch(request, (create, assertion))
    distribution = ForecastDistribution(
        sampled_patch_windows=(PatchWindow((create, assertion), 0.5),),
        event_marginals=(),
        calibrated_uncertainty=0.2,
        abstention_reason=None,
        retrieval_coverage=1.0,
        model_manifest_id="manifest:test",
    )
    validate_forecast_distribution(distribution, request=request)
    assert "proposal:person" not in {node.node_id for node in request.prefix.nodes}


def test_patch_rejects_non_cutoff_evidence_and_duplicate_assertion(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    request = _request(valid_bundle, times[2])
    duplicate = replace(
        valid_bundle.events[0],
        event_id="proposal:duplicate",
        observed_at=times[3],
        valid_from=times[3],
        basis_refs=("future-or-missing",),
        provenance=replace(valid_bundle.events[0].provenance, acquired_at=times[3]),
    )

    with pytest.raises(ContractValidationError) as caught:
        validate_proposed_patch(request, (duplicate,))
    assert "duplicate assertion" in str(caught.value)
    assert "was not visible at cutoff" in str(caught.value)


def test_distribution_validates_each_marginal_event_against_request(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    request = _request(valid_bundle, times[2])
    invalid = replace(
        _create_proposal(request, times[3]),
        basis_refs=("evidence:not-visible",),
    )
    distribution = ForecastDistribution(
        sampled_patch_windows=(),
        event_marginals=(EventMarginal(invalid, 0.5),),
        calibrated_uncertainty=0.2,
        abstention_reason=None,
        retrieval_coverage=1.0,
        model_manifest_id="manifest:test",
    )

    with pytest.raises(ContractValidationError, match="was not visible at cutoff"):
        validate_forecast_distribution(distribution, request=request)


def test_distribution_allows_repeated_draws_but_rejects_marginal_duplicates_and_excess_mass(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    request = _request(valid_bundle, times[2])
    proposal = _create_proposal(request, times[3])
    distribution = ForecastDistribution(
        sampled_patch_windows=(
            PatchWindow((proposal,), 0.6),
            PatchWindow((proposal,), 0.5),
        ),
        event_marginals=(
            EventMarginal(proposal, 0.5),
            EventMarginal(proposal, 0.5),
        ),
        calibrated_uncertainty=0.2,
        abstention_reason=None,
        retrieval_coverage=1.0,
        model_manifest_id="manifest:test",
    )

    with pytest.raises(ContractValidationError) as caught:
        validate_forecast_distribution(distribution, request=request)
    message = str(caught.value)
    assert "window probability mass must not exceed 1" in message
    assert "duplicate marginal event ID" in message
    assert "duplicate marginal event identity" in message
