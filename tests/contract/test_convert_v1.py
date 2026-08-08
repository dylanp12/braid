from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest

from braid.contract import (
    Operation,
    V1ConversionError,
    build_as_of_prefix,
    convert_v1_bundle,
    prefixes_are_invariant,
    validate_bundle,
)


def _legacy_source() -> dict:
    event = {
        "event_id": "ev:duplicate",
        "event_type": "inference",
        "relation": "CONTRIBUTES_TO",
        "participants": [["actor", "person:ada"], ["target", "repo:braid"]],
        "event_time": ["2025-01-02T00:00:00Z", None],
        "derived_at": None,
        "evidence": ["https://example.test/commit/1"],
        "derivation": [],
        "attrs": {"source": "fixture"},
        "extraction": {"pipeline_version": 1, "confidence": 0.8},
        "human_confirmed": False,
    }
    return {
        "manifest": {"contract_version": "0.1.0", "generator": "fixture"},
        "nodes": [
            {"id": "person:ada", "type": "Person", "label": "Ada", "attrs": {}},
            {"id": "repo:braid", "type": "Repository", "label": "Braid", "attrs": {}},
        ],
        "events": [event, copy.deepcopy(event)],
        "judgments": [{"id": "proposal:1", "kind": "proposal", "outcome": "kept"}],
    }


def test_converter_is_deterministic_valid_and_does_not_mutate_source() -> None:
    source = _legacy_source()
    untouched = copy.deepcopy(source)
    fallback = datetime(2025, 1, 1, tzinfo=UTC)

    first = convert_v1_bundle(source, default_observed_at=fallback)
    second = convert_v1_bundle(source, default_observed_at=fallback)

    assert source == untouched
    assert first == second
    assert first.canonical_hash() == second.canonical_hash()
    validate_bundle(first)


def test_converter_collapses_duplicate_assertions_but_keeps_full_lineage() -> None:
    converted = convert_v1_bundle(
        _legacy_source(), default_observed_at=datetime(2025, 1, 1, tzinfo=UTC)
    )

    assertion_events = [event for event in converted.events if event.operation.value == "ASSERT"]
    assert len(assertion_events) == 1
    event_lineage = [item for item in converted.lineage if item.source_kind == "event"]
    assert len(event_lineage) == 2
    assert {item.target_id for item in event_lineage} == {assertion_events[0].event_id}
    assert all(len(item.source_hash) == 64 for item in event_lineage)
    assert any(record.kind == "legacy-judgment" for record in converted.evidence)


def test_relation_named_supersedes_remains_a_relation_assertion() -> None:
    source = _legacy_source()
    for event in source["events"]:
        event["relation"] = "SUPERSEDES"

    converted = convert_v1_bundle(source, default_observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    relation_events = [event for event in converted.events if event.relation == "SUPERSEDES"]

    assert len(relation_events) == 1
    assert relation_events[0].operation is Operation.ASSERT
    validate_bundle(converted)


def test_explicit_supersession_with_a_resolvable_target_becomes_an_operation() -> None:
    source = _legacy_source()
    source["events"] = source["events"][:1]
    supersession = copy.deepcopy(source["events"][0])
    supersession.update(
        {
            "event_id": "ev:replacement",
            "operation": "SUPERSEDE",
            "target_event_id": "ev:duplicate",
            "derived_at": "2025-01-03T00:00:00Z",
            "event_time": ["2025-01-03T00:00:00Z", None],
            "attrs": {"source": "explicit-lifecycle-fixture", "revision": 2},
        }
    )
    source["events"].append(supersession)

    converted = convert_v1_bundle(source, default_observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    assertion = next(event for event in converted.events if event.operation is Operation.ASSERT)
    replacement = next(
        event for event in converted.events if event.operation is Operation.SUPERSEDE
    )

    assert replacement.payload["supersedes_event_id"] == assertion.event_id
    validate_bundle(converted)


def test_converter_infers_open_schema_without_global_entity_ids() -> None:
    converted = convert_v1_bundle(
        _legacy_source(), default_observed_at=datetime(2025, 1, 1, tzinfo=UTC)
    )
    schema = converted.schema
    relation = next(item for item in schema.relations if item.name == "CONTRIBUTES_TO")

    assert {item.name for item in schema.node_types} == {"Person", "Repository"}
    assert {role.name for role in relation.roles} == {"actor", "target"}
    assert all("person:ada" not in role.allowed_node_types for role in relation.roles)


def test_converter_requires_explicit_aware_fallback_clock() -> None:
    with pytest.raises(V1ConversionError, match="UTC offset"):
        convert_v1_bundle(_legacy_source(), default_observed_at=datetime(2025, 1, 1))


def test_converter_rejects_conflicting_duplicate_nodes() -> None:
    source = _legacy_source()
    source["nodes"].append(
        {"id": "person:ada", "type": "Repository", "label": "Conflict", "attrs": {}}
    )
    with pytest.raises(V1ConversionError, match="conflicting duplicate node"):
        convert_v1_bundle(source, default_observed_at=datetime(2025, 1, 1, tzinfo=UTC))


def test_future_events_do_not_change_converted_historical_prefix() -> None:
    fallback = datetime(2025, 1, 1, tzinfo=UTC)
    cutoff = datetime(2025, 1, 3, tzinfo=UTC)
    source = _legacy_source()
    source["events"] = source["events"][:1]
    extended = copy.deepcopy(source)
    future = copy.deepcopy(source["events"][0])
    future.update(
        {
            "event_id": "ev:future",
            "relation": "REVIEWS",
            "derived_at": "2025-02-01T00:00:00Z",
            "event_time": ["2025-02-01T00:00:00Z", None],
        }
    )
    extended["events"].append(future)

    original = convert_v1_bundle(source, default_observed_at=fallback)
    with_future = convert_v1_bundle(extended, default_observed_at=fallback)

    assert prefixes_are_invariant(original, with_future, cutoff)
    assert (
        build_as_of_prefix(original, cutoff).to_json()
        == build_as_of_prefix(with_future, cutoff).to_json()
    )


def test_future_duplicate_lineage_and_first_reference_do_not_change_prefix() -> None:
    fallback = datetime(2025, 1, 1, tzinfo=UTC)
    cutoff = datetime(2025, 1, 3, tzinfo=UTC)
    source = _legacy_source()
    source["events"] = source["events"][:1]
    source["nodes"].append(
        {"id": "person:unreferenced", "type": "Person", "label": "Future", "attrs": {}}
    )
    extended = copy.deepcopy(source)
    duplicate = copy.deepcopy(source["events"][0])
    duplicate.update(
        {
            "event_id": "ev:future-duplicate",
            "derived_at": "2025-02-01T00:00:00Z",
            "event_time": ["2025-02-01T00:00:00Z", None],
        }
    )
    first_reference = copy.deepcopy(duplicate)
    first_reference.update(
        {
            "event_id": "ev:first-reference",
            "relation": "MENTORS",
            "participants": [
                ["actor", "person:ada"],
                ["target", "person:unreferenced"],
            ],
        }
    )
    extended["events"].extend((duplicate, first_reference))

    original = convert_v1_bundle(source, default_observed_at=fallback)
    with_future = convert_v1_bundle(extended, default_observed_at=fallback)

    assert prefixes_are_invariant(original, with_future, cutoff)
    assert (
        build_as_of_prefix(original, cutoff).to_json()
        == build_as_of_prefix(with_future, cutoff).to_json()
    )


def test_padded_legacy_node_ids_are_content_addressed_without_trimming() -> None:
    source = _legacy_source()
    source["nodes"][0]["id"] = " person:ada "
    for event in source["events"]:
        event["participants"][0][1] = " person:ada "

    converted = convert_v1_bundle(source, default_observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    validate_bundle(converted)

    remapped = next(node for node in converted.nodes if node.node_type == "Person")
    assert remapped.node_id.startswith("v2:legacy-node:")
    assert remapped.payload_history[0].payload["legacy_node_id"] == " person:ada "


def test_converter_marks_every_synthesized_observation_clock() -> None:
    converted = convert_v1_bundle(
        _legacy_source(), default_observed_at=datetime(2025, 1, 1, tzinfo=UTC)
    )

    assert all(schema.observed_at_imputed for schema in converted.schemas)
    assert all(node.observed_at_imputed for node in converted.nodes)
    assert all(
        snapshot.observed_at_imputed
        for node in converted.nodes
        for snapshot in node.payload_history
    )
    assert all(record.observed_at_imputed for record in converted.evidence)
    assert all(event.observed_at_imputed for event in converted.events)
