from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest
import torch

from braid.contract import build_as_of_prefix
from braid.contract.types import (
    DerivationRecord,
    ForecastRequest,
    GraphBundleV2,
    GraphEventV2,
    NodeRecord,
    NodeTypeDecl,
    Operation,
    ProvenanceRecord,
    RelationDecl,
    RoleBinding,
    RoleDecl,
    SchemaDecl,
    TaskDeclaration,
)
from braid.model.config import TINY_CONFIG
from braid.model.model import EventGraphModel
from braid.model.tensorize import tensorize_forecast_request
from braid.model.tokenizer import BraidTokenizer
from braid.synthetic import ProcessGeneratorConfig, generate_world


def tied_request(order: tuple[int, int]) -> ForecastRequest:
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    observed_at = cutoff - timedelta(hours=1)
    schema = SchemaDecl(
        version="v2",
        observed_at=cutoff - timedelta(days=3),
        node_types=(NodeTypeDecl("person"),),
        relations=(
            RelationDecl(
                "knows",
                (RoleDecl("subject", ("person",)), RoleDecl("object", ("person",))),
            ),
        ),
    )
    nodes = (
        NodeRecord("n1", "person", "v2", cutoff - timedelta(days=2)),
        NodeRecord("n2", "person", "v2", cutoff - timedelta(days=2)),
    )
    derivation = DerivationRecord("observed-source")
    provenance = ProvenanceRecord("observed-source", None, None, observed_at)
    events = (
        GraphEventV2(
            event_id="event-assert",
            observed_at=observed_at,
            valid_from=observed_at,
            valid_to=None,
            operation=Operation.ASSERT,
            schema_version="v2",
            relation="knows",
            arguments=(RoleBinding("subject", "n1"), RoleBinding("object", "n2")),
            payload={"weight": 1},
            basis_refs=(),
            derivation=derivation,
            provenance=provenance,
            tie_group="same-observation",
        ),
        GraphEventV2(
            event_id="event-expose",
            observed_at=observed_at,
            valid_from=observed_at - timedelta(hours=2),
            valid_to=None,
            operation=Operation.EXPOSE,
            schema_version="v2",
            relation=None,
            arguments=(RoleBinding("candidate", "n1"),),
            payload={"channel": "test"},
            basis_refs=(),
            derivation=derivation,
            provenance=provenance,
            tie_group="same-observation",
        ),
    )
    bundle = GraphBundleV2(
        bundle_id="tied-prefix",
        schemas=(schema,),
        nodes=nodes,
        evidence=(),
        events=tuple(events[index] for index in order),
    )
    return ForecastRequest(
        request_id="tied-request",
        schema=schema,
        prefix=bundle,
        cutoff=cutoff,
        horizon=timedelta(days=1),
        task=TaskDeclaration("patch", query_node_ids=("n1",), target_relations=("knows",)),
    )


def test_multi_event_contract_prefix_has_a_full_sized_non_start_mask() -> None:
    episode = tensorize_forecast_request(tied_request((0, 1)), BraidTokenizer())

    assert episode.event_batch.start_mask is not None
    assert episode.event_batch.start_mask.tolist() == [[False, False]]
    episode.event_batch.validate()


def test_contract_tensorization_is_invariant_to_same_timestamp_order() -> None:
    tokenizer = BraidTokenizer()
    first = tensorize_forecast_request(tied_request((0, 1)), tokenizer)
    second = tensorize_forecast_request(tied_request((1, 0)), tokenizer)
    torch.testing.assert_close(
        first.event_batch.observed_deltas,
        second.event_batch.observed_deltas,
    )

    torch.manual_seed(17)
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size).eval()
    with torch.no_grad():
        first_output = model(first.schema_batch, first.event_batch)
        second_output = model(second.schema_batch, second.event_batch)

    torch.testing.assert_close(
        first_output.distribution.hidden_states,
        second_output.distribution.hidden_states,
    )
    torch.testing.assert_close(first_output.final_node_memory, second_output.final_node_memory)


def test_contract_clocks_use_the_declared_fixed_time_unit() -> None:
    episode = tensorize_forecast_request(
        tied_request((0, 1)),
        BraidTokenizer(),
        time_unit_seconds=3_600.0,
    )

    assert episode.event_batch.observed_deltas.tolist() == [[71.0, 71.0]]
    assert episode.event_batch.valid_lags.tolist() == [[0.0, -2.0]]


def test_contract_time_normalization_guard_rejects_absurd_values() -> None:
    with pytest.raises(ValueError, match="normalized-time guard"):
        tensorize_forecast_request(
            tied_request((0, 1)),
            BraidTokenizer(),
            time_unit_seconds=1e-9,
        )


def test_future_bundle_append_cannot_change_historical_representation_or_scores() -> None:
    world = generate_world(ProcessGeneratorConfig(seed=41, relation_events=32))
    cutoff = world.censor_at
    from_full = build_as_of_prefix(world.bundle, cutoff, bundle_id="causal-prefix")
    from_historical = build_as_of_prefix(from_full, cutoff, bundle_id="causal-prefix")
    assert from_full.to_json() == from_historical.to_json()

    def request(prefix: GraphBundleV2) -> ForecastRequest:
        return ForecastRequest(
            request_id="future-invariance",
            schema=prefix.schema,
            prefix=prefix,
            cutoff=cutoff,
            horizon=timedelta(days=1),
            task=TaskDeclaration("patch"),
        )

    tokenizer = BraidTokenizer()
    full_episode = tensorize_forecast_request(request(from_full), tokenizer)
    historical_episode = tensorize_forecast_request(request(from_historical), tokenizer)
    for record_field in fields(full_episode.event_batch):
        left = getattr(full_episode.event_batch, record_field.name)
        right = getattr(historical_episode.event_batch, record_field.name)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)

    torch.manual_seed(29)
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size).eval()
    with torch.no_grad():
        full_output = model(full_episode.schema_batch, full_episode.event_batch)
        historical_output = model(
            historical_episode.schema_batch,
            historical_episode.event_batch,
        )
    for record_field in fields(full_output.distribution):
        left = getattr(full_output.distribution, record_field.name)
        right = getattr(historical_output.distribution, record_field.name)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(full_output.final_node_memory, historical_output.final_node_memory)
