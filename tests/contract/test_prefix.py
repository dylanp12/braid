from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from braid.contract import (
    ContractValidationError,
    EvidenceRecord,
    GraphBundleV2,
    NodeRecord,
    PayloadSnapshot,
    SourceLineage,
    build_as_of_prefix,
    prefixes_are_invariant,
)


def test_as_of_prefix_excludes_future_nodes_payloads_events_evidence_and_schema(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    cutoff = times[2]
    future_evidence = EvidenceRecord("evidence:future", times[4], "future")
    existing = valid_bundle.nodes[0]
    future_payload = PayloadSnapshot(
        observed_at=times[4],
        valid_from=times[4],
        payload={"secret_future_value": True},
        basis_refs=(future_evidence.evidence_id,),
    )
    updated_existing = replace(
        existing, payload_history=(*existing.payload_history, future_payload)
    )
    future_node = NodeRecord(
        node_id="repo:future",
        node_type="Repository",
        schema_version=valid_bundle.schema.version,
        observed_at=times[4],
        payload_history=(
            PayloadSnapshot(times[4], {"secret_future_node": True}, valid_from=times[4]),
        ),
    )
    future_event = replace(
        valid_bundle.events[0],
        event_id="event:future",
        observed_at=times[4],
        valid_from=times[4],
        basis_refs=(future_evidence.evidence_id,),
        provenance=replace(valid_bundle.events[0].provenance, acquired_at=times[4]),
    )
    future_schema = replace(valid_bundle.schema, version="3.0.0", observed_at=times[4])
    extended = replace(
        valid_bundle,
        schemas=(*valid_bundle.schemas, future_schema),
        nodes=(updated_existing, valid_bundle.nodes[1], future_node),
        evidence=(*valid_bundle.evidence, future_evidence),
        events=(*valid_bundle.events, future_event),
    )

    prefix = build_as_of_prefix(extended, cutoff)

    assert [schema.version for schema in prefix.schemas] == ["2.0.0"]
    assert {node.node_id for node in prefix.nodes} == {"person:ada", "repo:braid"}
    assert len(prefix.nodes[0].payload_history) == 1
    assert {record.evidence_id for record in prefix.evidence} == {"evidence:commit-1"}
    assert {event.event_id for event in prefix.events} == {"event:contribution-1"}
    assert "secret_future" not in prefix.to_json()


def test_prefix_is_invariant_to_even_malformed_future_appends(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    # These future rows deliberately reuse IDs.  The full extension is invalid,
    # but causal slicing happens first and therefore cannot perturb the past.
    future_duplicate_node = replace(
        valid_bundle.nodes[0],
        observed_at=times[5],
        payload_history=(PayloadSnapshot(times[5], {"leak": 1}, valid_from=times[5]),),
    )
    future_duplicate_evidence = replace(
        valid_bundle.evidence[0], observed_at=times[5], payload={"leak": 2}
    )
    future_duplicate_event = replace(
        valid_bundle.events[0],
        observed_at=times[5],
        valid_from=times[5],
        payload={"leak": 3},
        provenance=replace(valid_bundle.events[0].provenance, acquired_at=times[5]),
    )
    extended = replace(
        valid_bundle,
        nodes=(*valid_bundle.nodes, future_duplicate_node),
        evidence=(*valid_bundle.evidence, future_duplicate_evidence),
        events=(*valid_bundle.events, future_duplicate_event),
    )

    assert prefixes_are_invariant(valid_bundle, extended, times[2])


def test_known_future_validity_is_visible_when_observed_before_cutoff(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    scheduled = replace(
        valid_bundle.events[0],
        event_id="event:scheduled",
        observed_at=times[2],
        valid_from=times[5],
        provenance=replace(valid_bundle.events[0].provenance, acquired_at=times[2]),
    )
    prefix = build_as_of_prefix(replace(valid_bundle, events=(scheduled,)), times[3])

    assert prefix.events == (scheduled,)


def test_naive_cutoff_is_rejected(valid_bundle: GraphBundleV2, times: tuple[datetime, ...]) -> None:
    with pytest.raises(ContractValidationError, match="UTC offset"):
        build_as_of_prefix(valid_bundle, times[2].replace(tzinfo=None))


def test_future_lineage_targeting_a_historical_record_is_excluded(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    future_lineage = SourceLineage(
        source_contract="1.x",
        source_kind="event",
        source_id="future:event",
        source_hash="f" * 64,
        target_kind="node",
        target_id=valid_bundle.nodes[0].node_id,
        converter="test",
        observed_at=times[5],
    )
    extended = replace(valid_bundle, lineage=(future_lineage,))

    assert prefixes_are_invariant(valid_bundle, extended, times[2])


def test_exact_cutoff_is_inclusive_and_reconstructs_mutable_payload(
    valid_bundle: GraphBundleV2, times: tuple[datetime, ...]
) -> None:
    node = valid_bundle.nodes[0]
    at_cutoff = PayloadSnapshot(
        observed_at=times[2],
        valid_from=times[1],
        payload={"name": "Ada at cutoff"},
    )
    after_cutoff = PayloadSnapshot(
        observed_at=times[3],
        valid_from=times[1],
        payload={"name": "future rewrite"},
    )
    changed = replace(
        node,
        payload_history=(*node.payload_history, at_cutoff, after_cutoff),
    )
    bundle = replace(valid_bundle, nodes=(changed, *valid_bundle.nodes[1:]))

    prefix = build_as_of_prefix(bundle, times[2])
    reconstructed = prefix.nodes[0]

    assert reconstructed.payload_history[-1] == at_cutoff
    assert reconstructed.payload_as_of(times[2]) == {"name": "Ada at cutoff"}
    assert "future rewrite" not in prefix.to_json()
