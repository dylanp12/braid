from __future__ import annotations

from datetime import timedelta, timezone

import pytest

from braid.contract import (
    EventMarginal,
    ForecastDistribution,
    ForecastRequest,
    GraphBundleV2,
    ModelManifest,
    Operation,
    PatchWindow,
    TaskDeclaration,
    canonical_hash,
    canonical_json,
)


def test_operation_vocabulary_is_exact_and_closed() -> None:
    assert [(operation.name, operation.value) for operation in Operation] == [
        ("CREATE_NODE", "CREATE_NODE"),
        ("UPDATE_NODE", "UPDATE_NODE"),
        ("ASSERT", "ASSERT"),
        ("RETRACT", "RETRACT"),
        ("SUPERSEDE", "SUPERSEDE"),
        ("SCHEMA_CHANGE", "SCHEMA_CHANGE"),
        ("EXPOSE", "EXPOSE"),
        ("JUDGE", "JUDGE"),
    ]
    with pytest.raises(ValueError):
        Operation("OBSERVE")


def test_bundle_json_round_trip_is_lossless_and_canonical(valid_bundle: GraphBundleV2) -> None:
    encoded = valid_bundle.to_json()
    decoded = GraphBundleV2.from_json(encoded)

    assert decoded == valid_bundle
    assert decoded.to_json() == encoded
    assert decoded.canonical_hash() == valid_bundle.canonical_hash()
    assert ": " not in encoded
    assert ", " not in encoded
    assert "\n" not in encoded


def test_hash_normalizes_mapping_order_and_equivalent_timezones(
    valid_bundle: GraphBundleV2,
) -> None:
    first = {"b": 2, "a": {"z": 3, "x": 1}}
    second = {"a": {"x": 1, "z": 3}, "b": 2}
    assert canonical_json(first) == canonical_json(second)
    assert canonical_hash(first) == canonical_hash(second)

    instant = valid_bundle.events[0].observed_at
    assert instant is not None
    alternate_zone = instant.astimezone(timezone(timedelta(hours=-5)))
    assert canonical_hash(instant) == canonical_hash(alternate_zone)


def test_caller_owned_payloads_are_deeply_frozen(valid_bundle: GraphBundleV2) -> None:
    payload = {"outer": {"answer": 42}, "items": [1, 2]}
    manifest = ModelManifest(*("a" * 64 for _ in range(8)), metadata=payload)
    payload["outer"]["answer"] = 0
    payload["items"].append(3)

    assert manifest.metadata["outer"]["answer"] == 42
    assert manifest.metadata["items"] == (1, 2)
    with pytest.raises(TypeError):
        manifest.metadata["new"] = True  # type: ignore[index]


def test_all_forecast_interfaces_round_trip(valid_bundle: GraphBundleV2) -> None:
    cutoff = valid_bundle.events[0].observed_at
    assert cutoff is not None
    request = ForecastRequest(
        request_id="request:1",
        schema=valid_bundle.schema,
        prefix=valid_bundle,
        cutoff=cutoff,
        horizon=timedelta(days=7),
        task=TaskDeclaration(
            name="future-patch",
            query_node_ids=("repo:braid",),
            target_relations=("CONTRIBUTES_TO",),
            options={"samples": 8},
        ),
        support_events=valid_bundle.events,
    )
    distribution = ForecastDistribution(
        sampled_patch_windows=(PatchWindow(valid_bundle.events, 0.7),),
        event_marginals=(EventMarginal(valid_bundle.events[0], 0.8),),
        calibrated_uncertainty=0.2,
        abstention_reason=None,
        retrieval_coverage=1.0,
        model_manifest_id="manifest:1",
    )

    assert ForecastRequest.from_json(request.to_json()) == request
    assert ForecastDistribution.from_json(distribution.to_json()) == distribution


def test_model_manifest_id_binds_every_required_hash() -> None:
    hashes = [f"{index:x}" * 64 for index in range(8)]
    manifest = ModelManifest(*hashes)
    changed = ModelManifest(*hashes[:-1], "f" * 64)

    assert len(manifest.manifest_id) == 64
    assert manifest.manifest_id != changed.manifest_id


def test_naive_datetime_cannot_cross_json_boundary(valid_bundle: GraphBundleV2) -> None:
    from dataclasses import replace

    naive_clock = valid_bundle.schemas[0].observed_at.replace(tzinfo=None)
    naive = replace(valid_bundle.schemas[0], observed_at=naive_clock)
    with pytest.raises(ValueError, match="naive"):
        naive.to_json()
