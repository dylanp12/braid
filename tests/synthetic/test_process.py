from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from braid.contract import Operation, build_as_of_prefix, validate_bundle
from braid.synthetic import ProcessGeneratorConfig, generate_world


def test_generated_world_is_deterministic_valid_and_unique() -> None:
    config = ProcessGeneratorConfig(seed=11, relation_events=120)
    first = generate_world(config)
    second = generate_world(config)

    assert first.bundle.canonical_hash() == second.bundle.canonical_hash()
    validate_bundle(first.bundle)
    ids = [event.event_id for event in first.bundle.events]
    assert len(ids) == len(set(ids))

    assertions = [
        (
            event.schema_version,
            event.relation,
            tuple(sorted((binding.role, binding.node_id) for binding in event.arguments)),
        )
        for event in first.bundle.events
        if event.operation is Operation.ASSERT
    ]
    assert len(assertions) == len(set(assertions))


def test_generator_exercises_dual_clocks_retractions_and_schema_evolution() -> None:
    world = generate_world(
        ProcessGeneratorConfig(
            seed=7,
            relation_events=180,
            retraction_probability=0.3,
            supersede_probability=0.2,
        )
    )
    assert len(world.bundle.schemas) == 2
    assert any(event.operation is Operation.SCHEMA_CHANGE for event in world.bundle.events)
    assert any(event.operation is Operation.RETRACT for event in world.bundle.events)
    assert any(event.operation is Operation.SUPERSEDE for event in world.bundle.events)
    assert any(event.valid_from < event.observed_at for event in world.bundle.events)


def test_censor_prefix_excludes_every_future_record() -> None:
    world = generate_world(ProcessGeneratorConfig(seed=3, relation_events=80))
    prefix = build_as_of_prefix(world.bundle, world.censor_at)
    assert all(schema.observed_at <= world.censor_at for schema in prefix.schemas)
    assert all(node.observed_at <= world.censor_at for node in prefix.nodes)
    assert all(record.observed_at <= world.censor_at for record in prefix.evidence)
    assert all(event.observed_at <= world.censor_at for event in prefix.events)


def test_invalid_generator_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        ProcessGeneratorConfig(seed=1, censor_fraction=1.1)
    with pytest.raises(ValueError):
        generate_world(ProcessGeneratorConfig(seed=1), start=datetime(2025, 1, 1))


def test_seed_changes_world_identity() -> None:
    config = ProcessGeneratorConfig(seed=1, relation_events=20)
    first = generate_world(config).bundle.canonical_hash()
    second = generate_world(replace(config, seed=2)).bundle.canonical_hash()
    assert first != second
