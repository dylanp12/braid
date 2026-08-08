from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from braid.contract import (
    ContractValidationError,
    EvidenceRecord,
    GraphBundleV2,
    Operation,
    RoleBinding,
    collect_bundle_issues,
    validate_bundle,
)


def _messages(bundle: GraphBundleV2) -> str:
    return "\n".join(str(issue) for issue in collect_bundle_issues(bundle))


def test_valid_bundle_passes(valid_bundle: GraphBundleV2) -> None:
    assert collect_bundle_issues(valid_bundle) == ()
    validate_bundle(valid_bundle)


@pytest.mark.parametrize("kind", ["node", "event", "evidence"])
def test_duplicate_ids_are_rejected(kind: str, valid_bundle: GraphBundleV2) -> None:
    if kind == "node":
        broken = replace(valid_bundle, nodes=(*valid_bundle.nodes, valid_bundle.nodes[0]))
    elif kind == "event":
        broken = replace(valid_bundle, events=(*valid_bundle.events, valid_bundle.events[0]))
    else:
        broken = replace(valid_bundle, evidence=(*valid_bundle.evidence, valid_bundle.evidence[0]))

    with pytest.raises(ContractValidationError, match=rf"duplicate {kind} ID"):
        validate_bundle(broken)


def test_semantic_duplicate_assertion_is_rejected_despite_new_id_and_time(
    valid_bundle: GraphBundleV2,
) -> None:
    original = valid_bundle.events[0]
    assert original.observed_at is not None and original.valid_from is not None
    duplicate = replace(
        original,
        event_id="event:contribution-again",
        observed_at=original.observed_at + timedelta(days=1),
        valid_from=original.valid_from + timedelta(days=1),
        provenance=replace(
            original.provenance,
            acquired_at=original.provenance.acquired_at + timedelta(days=1),
        ),
        arguments=tuple(reversed(original.arguments)),
    )
    broken = replace(valid_bundle, events=(original, duplicate))

    with pytest.raises(ContractValidationError, match="duplicate assertion"):
        validate_bundle(broken)


@pytest.mark.parametrize("clock", ["missing", "naive"])
def test_missing_and_naive_observation_clocks_are_rejected(
    clock: str, valid_bundle: GraphBundleV2
) -> None:
    observed = (
        None if clock == "missing" else valid_bundle.events[0].observed_at.replace(tzinfo=None)
    )
    broken = replace(
        valid_bundle,
        events=(replace(valid_bundle.events[0], observed_at=observed),),
    )

    with pytest.raises(ContractValidationError, match="observed_at"):
        validate_bundle(broken)


def test_missing_valid_clock_and_naive_provenance_clock_are_rejected(
    valid_bundle: GraphBundleV2,
) -> None:
    event = valid_bundle.events[0]
    broken_event = replace(
        event,
        valid_from=None,
        provenance=replace(
            event.provenance,
            acquired_at=event.provenance.acquired_at.replace(tzinfo=None),
        ),
    )
    messages = _messages(replace(valid_bundle, events=(broken_event,)))

    assert "valid_from: clock is required" in messages
    assert "provenance.acquired_at: clock must include a UTC offset" in messages


def test_dangling_and_future_evidence_are_rejected(valid_bundle: GraphBundleV2) -> None:
    event = valid_bundle.events[0]
    assert event.observed_at is not None
    future_evidence = EvidenceRecord(
        evidence_id="evidence:future",
        observed_at=event.observed_at + timedelta(days=1),
        kind="commit",
        parent_refs=("evidence:missing",),
    )
    broken_event = replace(event, basis_refs=(future_evidence.evidence_id, "evidence:missing"))
    broken = replace(
        valid_bundle,
        evidence=(*valid_bundle.evidence, future_evidence),
        events=(broken_event,),
    )
    messages = _messages(broken)

    assert "dangling evidence reference 'evidence:missing'" in messages
    assert "references future evidence" in messages


def test_role_name_cardinality_and_node_type_are_enforced(valid_bundle: GraphBundleV2) -> None:
    original = valid_bundle.events[0]
    wrong_type = replace(valid_bundle.nodes[0], node_type="Repository")
    bad_event = replace(
        original,
        arguments=(
            RoleBinding("who", "person:ada"),
            RoleBinding("target", "repo:braid"),
            RoleBinding("target", "repo:braid"),
        ),
    )
    messages = _messages(
        replace(
            valid_bundle,
            nodes=(wrong_type, valid_bundle.nodes[1]),
            events=(bad_event,),
        )
    )

    assert "role 'who' is not declared" in messages
    assert "role 'actor' has 0 bindings" in messages
    assert "role 'target' has 2 bindings" in messages

    wrong_type_event = replace(
        original,
        arguments=(RoleBinding("actor", "repo:braid"), RoleBinding("target", "person:ada")),
    )
    type_messages = _messages(replace(valid_bundle, events=(wrong_type_event,)))
    assert "does not allow node type 'Repository'" in type_messages
    assert "does not allow node type 'Person'" in type_messages


def test_bad_chronology_is_rejected(valid_bundle: GraphBundleV2) -> None:
    event = valid_bundle.events[0]
    assert event.valid_from is not None and event.observed_at is not None
    bad_event = replace(event, valid_to=event.valid_from - timedelta(seconds=1))
    future_node = replace(valid_bundle.nodes[0], observed_at=event.observed_at + timedelta(days=2))
    original_snapshot = future_node.payload_history[0]
    out_of_order_node = replace(
        future_node,
        payload_history=(
            replace(original_snapshot, observed_at=event.observed_at + timedelta(days=3)),
            replace(original_snapshot, observed_at=event.observed_at + timedelta(days=2)),
        ),
    )
    messages = _messages(
        replace(valid_bundle, nodes=(out_of_order_node, valid_bundle.nodes[1]), events=(bad_event,))
    )

    assert "valid_to precedes valid_from" in messages
    assert "payload history is not chronological" in messages
    assert "references a node not yet observed" in messages


def test_invalid_relationless_operation_roles_are_rejected(valid_bundle: GraphBundleV2) -> None:
    original = valid_bundle.events[0]
    broken_event = replace(
        original,
        operation=Operation.CREATE_NODE,
        relation=None,
        arguments=(RoleBinding("subject", "person:ada"),),
    )
    with pytest.raises(ContractValidationError, match="requires exactly one 'node'"):
        validate_bundle(replace(valid_bundle, events=(broken_event,)))


def test_evidence_cycles_are_rejected(valid_bundle: GraphBundleV2) -> None:
    clock = valid_bundle.evidence[0].observed_at
    first = EvidenceRecord("evidence:a", clock, "derived", parent_refs=("evidence:b",))
    second = EvidenceRecord("evidence:b", clock, "derived", parent_refs=("evidence:a",))

    with pytest.raises(ContractValidationError, match="evidence derivation cycle"):
        validate_bundle(replace(valid_bundle, evidence=(first, second), events=()))
