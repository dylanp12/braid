from dataclasses import replace

import pytest
import torch

from braid.model.config import TINY_CONFIG
from braid.model.model import EventGraphModel, EventTensorBatch, SchemaTensorBatch
from braid.model.tokenizer import BraidTokenizer


def schema_batch() -> SchemaTensorBatch:
    # type-0, type-1, relation-0, role-0, role-1
    tokens = torch.tensor(
        [
            [1, 3, 20, 2],
            [1, 3, 21, 2],
            [1, 4, 22, 2],
            [1, 5, 23, 2],
            [1, 5, 24, 2],
        ]
    )
    return SchemaTensorBatch(
        token_ids=tokens,
        padding_mask=torch.zeros_like(tokens, dtype=torch.bool),
        type_item_indices=torch.tensor([0, 1]),
        relation_item_indices=torch.tensor([2]),
        role_item_indices=torch.tensor([3, 4]),
        role_to_relation=torch.tensor([0, 0]),
        role_to_type=torch.tensor([0, 1]),
    )


def event_batch(
    order: tuple[int, int] = (0, 1),
    tie_groups: tuple[int, int] = (7, 7),
) -> EventTensorBatch:
    operation_ids = torch.tensor([[2, 6]])[:, order]
    relation_indices = torch.tensor([[0, -1]])[:, order]
    observed = torch.tensor([[3.0, 0.0]])[:, order]
    valid = torch.tensor([[0.0, -2.0]])[:, order]
    payload = torch.tensor([[[1, 20, 2], [1, 21, 2]]])[:, order]
    nodes = torch.tensor([[[0], [1]]])[:, order]
    roles = torch.tensor([[[0], [1]]])[:, order]
    argument_mask = torch.ones_like(nodes, dtype=torch.bool)
    return EventTensorBatch(
        operation_ids=operation_ids,
        relation_indices=relation_indices,
        observed_deltas=observed,
        valid_lags=valid,
        payload_token_ids=payload,
        argument_node_indices=nodes,
        argument_role_indices=roles,
        argument_mask=argument_mask,
        tie_groups=torch.tensor([tie_groups])[:, order],
        event_mask=torch.ones((1, 2), dtype=torch.bool),
        node_type_indices=torch.tensor([[0, 1]]),
        evidence_token_ids=torch.tensor([[[1, 25, 2]]]),
        evidence_mask=torch.ones((1, 1), dtype=torch.bool),
    )


def test_eventgraph_forward_exposes_every_factor() -> None:
    torch.manual_seed(3)
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size).eval()
    output = model(schema_batch(), event_batch())
    result = output.distribution
    assert result.delta_log_rate.shape == (1, 1)
    assert result.valid_lag_mean.shape == (1, 1)
    assert result.operation_logits.shape == (1, 1, 8)
    assert result.relation_logits.shape == (1, 1, 1)
    assert result.argument_logits.shape == (1, 1, 2)
    assert result.payload_logits.shape == (1, 1, tokenizer.vocab_size)
    assert result.evidence_logits.shape == (1, 1, 1)
    assert output.final_node_memory.shape == (1, 2, TINY_CONFIG.d_model)


def test_model_is_invariant_to_order_inside_marked_tie() -> None:
    torch.manual_seed(5)
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size).eval()
    first = model(schema_batch(), event_batch((0, 1)))
    second = model(schema_batch(), event_batch((1, 0)))
    torch.testing.assert_close(first.distribution.hidden_states, second.distribution.hidden_states)
    torch.testing.assert_close(first.final_node_memory, second.final_node_memory)


def test_model_enforces_declared_context_and_memory_limits() -> None:
    tokenizer = BraidTokenizer()
    context_limited = EventGraphModel(
        replace(TINY_CONFIG, context_events=1),
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    with pytest.raises(ValueError, match="configured context limit"):
        context_limited(schema_batch(), event_batch())

    memory_limited = EventGraphModel(
        replace(TINY_CONFIG, memory_slots=1),
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    with pytest.raises(ValueError, match="configured memory limit"):
        memory_limited(schema_batch(), event_batch())


def test_node_memory_can_be_carried_across_stream_chunks() -> None:
    torch.manual_seed(23)
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size).eval()
    complete = event_batch(tie_groups=(7, 8))
    first_chunk = _event_slice(complete, 0)
    second_chunk = _event_slice(complete, 1)

    with torch.no_grad():
        complete_output = model(schema_batch(), complete)
        first_output = model(schema_batch(), first_chunk)
        second_output = model(
            schema_batch(),
            second_chunk,
            initial_node_memory=first_output.final_node_memory,
        )

    torch.testing.assert_close(second_output.final_node_memory, complete_output.final_node_memory)


def test_initial_node_memory_shape_is_validated() -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    with pytest.raises(ValueError, match="initial node memory must have shape"):
        model(
            schema_batch(),
            event_batch(),
            initial_node_memory=torch.zeros(1, 1, TINY_CONFIG.d_model),
        )


def _event_slice(batch: EventTensorBatch, index: int) -> EventTensorBatch:
    selection = slice(index, index + 1)
    return EventTensorBatch(
        operation_ids=batch.operation_ids[:, selection],
        relation_indices=batch.relation_indices[:, selection],
        observed_deltas=batch.observed_deltas[:, selection],
        valid_lags=batch.valid_lags[:, selection],
        payload_token_ids=batch.payload_token_ids[:, selection],
        argument_node_indices=batch.argument_node_indices[:, selection],
        argument_role_indices=batch.argument_role_indices[:, selection],
        argument_mask=batch.argument_mask[:, selection],
        tie_groups=batch.tie_groups[:, selection],
        event_mask=batch.event_mask[:, selection],
        node_type_indices=batch.node_type_indices,
        evidence_token_ids=batch.evidence_token_ids,
        evidence_mask=batch.evidence_mask,
        start_mask=None if batch.start_mask is None else batch.start_mask[:, selection],
    )
