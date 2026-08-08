from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from braid.contract import (
    ContractValidationError,
    DerivationRecord,
    ForecastRequest,
    GraphBundleV2,
    GraphEventV2,
    NodeRecord,
    NodeTypeDecl,
    Operation,
    PayloadSnapshot,
    ProvenanceRecord,
    RelationDecl,
    RoleBinding,
    RoleDecl,
    TaskDeclaration,
    validate_bundle,
    validate_forecast_request,
    validate_proposed_patch,
)


def _request(bundle: GraphBundleV2, cutoff: datetime) -> ForecastRequest:
    return ForecastRequest(
        request_id="request:operations",
        schema=bundle.schema,
        prefix=bundle,
        cutoff=cutoff,
        horizon=timedelta(days=5),
        task=TaskDeclaration("patch"),
    )


def _at(
    template: GraphEventV2,
    event_id: str,
    observed_at: datetime,
    operation: Operation,
    *,
    relation: str | None = None,
    arguments: tuple[RoleBinding, ...] | None = None,
    payload: dict | None = None,
    schema_version: str | None = None,
) -> GraphEventV2:
    return replace(
        template,
        event_id=event_id,
        observed_at=observed_at,
        valid_from=observed_at,
        valid_to=None,
        operation=operation,
        schema_version=schema_version or template.schema_version,
        relation=relation,
        arguments=template.arguments if arguments is None else arguments,
        payload={} if payload is None else payload,
        basis_refs=(),
        derivation=DerivationRecord("operation-test"),
        provenance=ProvenanceRecord("operation-test", None, None, observed_at),
        tie_group=None,
    )


def test_bundle_replays_retract_reassert_supersede_and_retract(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    original = valid_bundle.events[0]
    retract = _at(
        original,
        "event:retract-original",
        times[2],
        Operation.RETRACT,
        relation=original.relation,
        payload={"retracts_event_id": original.event_id},
    )
    reassert = _at(
        original,
        "event:reassert",
        times[3],
        Operation.ASSERT,
        relation=original.relation,
        payload={"revision": 2},
    )
    supersede = _at(
        original,
        "event:supersede",
        times[4],
        Operation.SUPERSEDE,
        relation=original.relation,
        payload={"supersedes_event_id": reassert.event_id, "revision": 3},
    )
    retract_supersession = _at(
        original,
        "event:retract-supersession",
        times[5],
        Operation.RETRACT,
        relation=original.relation,
        payload={"target_event_id": supersede.event_id},
    )

    validate_bundle(
        replace(
            valid_bundle,
            events=(original, retract, reassert, supersede, retract_supersession),
        )
    )


def test_bundle_rejects_inactive_future_and_mismatched_lifecycle_targets(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    original = valid_bundle.events[0]
    first = _at(
        original,
        "event:first-retract",
        times[2],
        Operation.RETRACT,
        relation=original.relation,
        payload={"retracts_event_id": original.event_id},
    )
    again = _at(
        original,
        "event:second-retract",
        times[3],
        Operation.RETRACT,
        relation=original.relation,
        payload={"target_event_id": original.event_id},
    )
    with pytest.raises(ContractValidationError, match="must be an active ASSERT or SUPERSEDE"):
        validate_bundle(replace(valid_bundle, events=(original, first, again)))

    future_target = _at(
        original,
        "event:future-assertion",
        times[4],
        Operation.ASSERT,
        relation=original.relation,
        payload={"revision": 2},
    )
    early_retract = _at(
        original,
        "event:early-retract",
        times[3],
        Operation.RETRACT,
        relation=original.relation,
        payload={"target_event_id": future_target.event_id},
    )
    with pytest.raises(ContractValidationError, match="observed strictly before"):
        validate_bundle(replace(valid_bundle, events=(future_target, early_retract)))

    mismatched = _at(
        original,
        "event:mismatched-retract",
        times[2],
        Operation.RETRACT,
        relation="HAS_CONTRIBUTOR",
        arguments=(
            RoleBinding("target", "repo:braid"),
            RoleBinding("actor", "person:ada"),
        ),
        payload={"target_event_id": original.event_id},
    )
    with pytest.raises(ContractValidationError, match="must match its target"):
        validate_bundle(replace(valid_bundle, events=(original, mismatched)))


def test_create_and_update_bind_materialized_node_identity_and_type(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    template = valid_bundle.events[0]
    node = NodeRecord(
        node_id="person:grace",
        node_type="Person",
        schema_version=valid_bundle.schema.version,
        observed_at=times[2],
        payload_history=(PayloadSnapshot(times[2], {"name": "Grace"}, valid_from=times[2]),),
    )
    create = _at(
        template,
        "event:create-grace",
        times[2],
        Operation.CREATE_NODE,
        relation=None,
        arguments=(RoleBinding("node", node.node_id),),
        payload={"node_id": node.node_id, "node_type": node.node_type},
    )
    update = _at(
        template,
        "event:update-grace",
        times[3],
        Operation.UPDATE_NODE,
        relation=None,
        arguments=(RoleBinding("node", node.node_id),),
        payload={
            "node_id": node.node_id,
            "node_type": node.node_type,
            "name": "Grace Hopper",
        },
    )
    bundle = replace(
        valid_bundle,
        nodes=(*valid_bundle.nodes, node),
        events=(*valid_bundle.events, create, update),
    )
    validate_bundle(bundle)

    wrong = replace(update, payload={"node_id": node.node_id, "node_type": "Repository"})
    with pytest.raises(ContractValidationError) as caught:
        validate_bundle(replace(bundle, events=(*valid_bundle.events, create, wrong)))
    message = str(caught.value)
    assert "cannot change the node type" in message
    assert "requires a payload field beyond node identity and type" in message


def test_proposal_virtual_state_supports_every_non_schema_operation(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    request = _request(valid_bundle, times[2])
    template = valid_bundle.events[0]
    base = times[2]

    def clock(hour: int) -> datetime:
        return base + timedelta(hours=hour)

    create = _at(
        template,
        "proposal:create",
        clock(1),
        Operation.CREATE_NODE,
        relation=None,
        arguments=(RoleBinding("node", "person:new"),),
        payload={"node_id": "person:new", "node_type": "Person"},
    )
    update = _at(
        template,
        "proposal:update",
        clock(2),
        Operation.UPDATE_NODE,
        relation=None,
        arguments=(RoleBinding("node", "person:new"),),
        payload={"node_id": "person:new", "node_type": "Person", "name": "New"},
    )
    expose = _at(
        template,
        "proposal:expose",
        clock(3),
        Operation.EXPOSE,
        relation=None,
        arguments=(RoleBinding("candidate", "person:new"),),
        payload={"surface": "shadow"},
    )
    judge = _at(
        template,
        "proposal:judge",
        clock(4),
        Operation.JUDGE,
        relation=None,
        arguments=(RoleBinding("subject", "person:new"),),
        payload={"outcome": "complete-slate-label"},
    )
    retract = _at(
        template,
        "proposal:retract",
        clock(5),
        Operation.RETRACT,
        relation=template.relation,
        payload={"target_event_id": template.event_id},
    )
    reassert = _at(
        template,
        "proposal:reassert",
        clock(6),
        Operation.ASSERT,
        relation=template.relation,
        payload={"revision": 2},
    )
    supersede = _at(
        template,
        "proposal:supersede",
        clock(7),
        Operation.SUPERSEDE,
        relation=template.relation,
        payload={"supersedes_event_id": reassert.event_id, "revision": 3},
    )

    validate_proposed_patch(
        request,
        (create, update, expose, judge, retract, reassert, supersede),
    )
    assert "person:new" not in {node.node_id for node in request.prefix.nodes}


def test_schema_proposal_is_hash_bound_and_advances_only_private_schema_state(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    request = _request(valid_bundle, times[2])
    template = valid_bundle.events[0]
    changed_at = times[2] + timedelta(hours=1)
    new_schema = replace(
        request.schema,
        version="2.1.0",
        observed_at=changed_at,
        node_types=(*request.schema.node_types, NodeTypeDecl("Issue")),
        relations=(
            *request.schema.relations,
            RelationDecl("TRACKS", (RoleDecl("issue", ("Issue",)),)),
        ),
    )
    change = _at(
        template,
        "proposal:schema-change",
        changed_at,
        Operation.SCHEMA_CHANGE,
        relation=None,
        arguments=(),
        payload={
            "previous_schema_version": request.schema.version,
            "new_schema_version": new_schema.version,
            "schema_hash": new_schema.canonical_hash(),
            "schema_decl": new_schema.to_dict(),
        },
    )
    create = _at(
        template,
        "proposal:create-issue",
        changed_at + timedelta(hours=1),
        Operation.CREATE_NODE,
        relation=None,
        arguments=(RoleBinding("node", "issue:new"),),
        payload={"node_id": "issue:new", "node_type": "Issue"},
        schema_version=new_schema.version,
    )

    validate_proposed_patch(request, (change, create))
    assert request.schema.version == "2.0.0"

    corrupt = replace(change, payload={**change.payload, "schema_hash": "0" * 64})
    with pytest.raises(ContractValidationError, match="canonical hash"):
        validate_proposed_patch(request, (corrupt, create))


def test_forecast_request_requires_latest_prefix_schema(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    old = valid_bundle.schema
    new = replace(old, version="2.1.0", observed_at=times[2])
    change = _at(
        valid_bundle.events[0],
        "event:schema-change",
        times[2],
        Operation.SCHEMA_CHANGE,
        relation=None,
        arguments=(),
        payload={
            "previous_schema_version": old.version,
            "new_schema_version": new.version,
            "schema_hash": new.canonical_hash(),
        },
    )
    prefix = replace(
        valid_bundle,
        schemas=(old, new),
        events=(*valid_bundle.events, change),
    )
    request = replace(_request(prefix, times[3]), schema=old)

    with pytest.raises(ContractValidationError, match="latest active prefix schema"):
        validate_forecast_request(request)


def test_schema_evolution_rejects_same_timestamp_lexical_version_trap(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    root = valid_bundle.schema
    shared_clock = times[2]
    schema_v9 = replace(root, version="v9", observed_at=shared_clock)
    schema_v10 = replace(root, version="v10", observed_at=shared_clock)
    change_v9 = _at(
        valid_bundle.events[0],
        "event:schema-v9",
        shared_clock,
        Operation.SCHEMA_CHANGE,
        relation=None,
        arguments=(),
        payload={
            "previous_schema_version": root.version,
            "new_schema_version": schema_v9.version,
            "schema_hash": schema_v9.canonical_hash(),
        },
    )
    change_v10 = _at(
        valid_bundle.events[0],
        "event:schema-v10",
        shared_clock,
        Operation.SCHEMA_CHANGE,
        relation=None,
        arguments=(),
        payload={
            "previous_schema_version": schema_v9.version,
            "new_schema_version": schema_v10.version,
            "schema_hash": schema_v10.canonical_hash(),
        },
        schema_version=schema_v9.version,
    )
    bundle = replace(
        valid_bundle,
        schemas=(root, schema_v9, schema_v10),
        events=(*valid_bundle.events, change_v9, change_v10),
    )

    # Lexicographic max would select v9 over v10.  Concurrent activations are
    # rejected instead of allowing a version string or tuple order to signal cause.
    assert bundle.schema.version == "v9"
    with pytest.raises(ContractValidationError, match="same observed_at"):
        validate_bundle(bundle)


def test_model_confidence_literals_are_never_graph_payload(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    event = replace(
        valid_bundle.events[0],
        payload={"prediction": {"model_confidence": 0.9}},
    )
    with pytest.raises(ContractValidationError, match="forecast distribution"):
        validate_bundle(replace(valid_bundle, events=(event,)))

    request = _request(valid_bundle, times[2])
    proposal = _at(
        valid_bundle.events[0],
        "proposal:confidence",
        times[3],
        Operation.EXPOSE,
        relation=None,
        arguments=(RoleBinding("candidate", "person:ada"),),
        payload={"nested": {"confidence": 0.9}},
    )
    with pytest.raises(ContractValidationError, match="forecast distribution"):
        validate_proposed_patch(request, (proposal,))
