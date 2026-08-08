"""Causal conversion from the public contract to episode-local model tensors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import torch

from braid.contract.serde import canonical_json
from braid.contract.types import ForecastRequest, GraphEventV2
from braid.contract.validate import validate_forecast_request
from braid.model.config import DEFAULT_TIME_UNIT_SECONDS, MAX_NORMALIZED_TIME
from braid.model.grammar import (
    DecodeContext,
    GrammarSchema,
    RelationRule,
    RoleRule,
    VisibleEvidence,
    VisibleNode,
)
from braid.model.grammar import Operation as ModelOperation
from braid.model.model import EventTensorBatch, SchemaTensorBatch
from braid.model.retrieval import NodeCandidate
from braid.model.tokenizer import (
    BraidTokenizer,
    DynamicHandleTable,
    EncodedSchemaItem,
    HandleKind,
)


@dataclass(slots=True)
class TensorizedForecast:
    schema_batch: SchemaTensorBatch
    event_batch: EventTensorBatch
    handles: DynamicHandleTable
    node_ids: tuple[str, ...]
    node_type_handles: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    relation_names: tuple[str, ...]
    role_incidence: Mapping[tuple[str, str, str], int]
    grammar_context: DecodeContext
    retrieval_nodes: tuple[NodeCandidate, ...]


def tensorize_forecast_request(
    request: ForecastRequest,
    tokenizer: BraidTokenizer,
    *,
    max_payload_tokens: int = 256,
    max_evidence_tokens: int = 256,
    time_unit_seconds: float = DEFAULT_TIME_UNIT_SECONDS,
    device: str | torch.device = "cpu",
) -> TensorizedForecast:
    """Validate and tensorize only the already-sliced causal request prefix."""

    validate_forecast_request(request)
    if not math.isfinite(time_unit_seconds) or time_unit_seconds <= 0:
        raise ValueError("time_unit_seconds must be positive and finite")
    assert request.cutoff is not None
    handles, schema_items = tokenizer.encode_schema_decl(request.schema)
    schema_batch, role_incidence = _schema_tensors(
        request, tokenizer, handles, schema_items, device=device
    )

    node_ids: list[str] = []
    node_type_handles: list[int] = []
    node_observed: list[datetime] = []
    for node in request.prefix.nodes:
        handle = handles.register(HandleKind.NODE, node.node_id)
        if handle.index != len(node_ids):  # validated contract IDs should be unique
            raise ValueError(f"duplicate node handle during tensorization: {node.node_id}")
        node_ids.append(node.node_id)
        node_type_handles.append(handles.resolve(HandleKind.TYPE, node.node_type).index)
        if node.observed_at is None:
            raise ValueError("temporal tensorization rejects unknown node observation time")
        node_observed.append(node.observed_at)

    evidence_ids: list[str] = []
    evidence_sequences: list[tuple[int, ...]] = []
    evidence_observed: list[datetime] = []
    for evidence in request.prefix.evidence:
        handle = handles.register(HandleKind.EVIDENCE, evidence.evidence_id)
        if handle.index != len(evidence_ids):
            raise ValueError(
                f"duplicate evidence handle during tensorization: {evidence.evidence_id}"
            )
        evidence_ids.append(evidence.evidence_id)
        if evidence.observed_at is None:
            raise ValueError("temporal tensorization rejects unknown evidence observation time")
        evidence_observed.append(evidence.observed_at)
        semantic = canonical_json({"kind": evidence.kind, "payload": evidence.payload})
        evidence_sequences.append(tokenizer.encode_text(semantic)[:max_evidence_tokens])

    all_events = (*request.prefix.events, *request.support_events)
    events = tuple(
        event
        for _, event in sorted(
            enumerate(all_events),
            key=lambda item: (
                item[1].observed_at is not None,
                item[1].observed_at or request.cutoff,
                item[0],
            ),
        )
    )
    for event in events:
        handles.register(HandleKind.EVENT, event.event_id)
    event_batch = _event_tensors(
        events,
        request,
        tokenizer,
        handles,
        role_incidence,
        node_type_handles,
        evidence_sequences,
        max_payload_tokens=max_payload_tokens,
        time_unit_seconds=time_unit_seconds,
        device=device,
    )
    grammar_context = _grammar_context(
        request,
        handles,
        role_incidence,
        node_ids,
        node_type_handles,
        node_observed,
        evidence_ids,
        evidence_observed,
    )
    neighbors: dict[int, set[int]] = {index: set() for index in range(len(node_ids))}
    for event in request.prefix.events:
        bound = [handles.resolve(HandleKind.NODE, item.node_id).index for item in event.arguments]
        for source in bound:
            neighbors[source].update(target for target in bound if target != source)
    retrieval_nodes = tuple(
        NodeCandidate(
            handle=index,
            type_handle=node_type_handles[index],
            observed_at=node_observed[index],
            neighbors=frozenset(neighbors[index]),
        )
        for index in range(len(node_ids))
    )
    return TensorizedForecast(
        schema_batch=schema_batch,
        event_batch=event_batch,
        handles=handles,
        node_ids=tuple(node_ids),
        node_type_handles=tuple(node_type_handles),
        evidence_ids=tuple(evidence_ids),
        relation_names=tuple(relation.name for relation in request.schema.relations),
        role_incidence=role_incidence,
        grammar_context=grammar_context,
        retrieval_nodes=retrieval_nodes,
    )


def _schema_tensors(
    request: ForecastRequest,
    tokenizer: BraidTokenizer,
    handles: DynamicHandleTable,
    items: tuple[EncodedSchemaItem, ...],
    *,
    device: str | torch.device,
) -> tuple[SchemaTensorBatch, dict[tuple[str, str, str], int]]:
    token_ids, padding_mask = _pad_sequences(
        [item.token_ids for item in items], tokenizer.PAD, device=device
    )
    type_items = {
        item.handle: index for index, item in enumerate(items) if item.kind is HandleKind.TYPE
    }
    relation_items = {
        item.handle: index for index, item in enumerate(items) if item.kind is HandleKind.RELATION
    }
    role_items = {
        item.handle: index for index, item in enumerate(items) if item.kind is HandleKind.ROLE
    }
    role_item_indices: list[int] = []
    role_to_relation: list[int] = []
    role_to_type: list[int] = []
    role_incidence: dict[tuple[str, str, str], int] = {}
    for relation in request.schema.relations:
        relation_handle = handles.resolve(HandleKind.RELATION, relation.name).index
        for role in relation.roles:
            role_handle = handles.resolve(HandleKind.ROLE, f"{relation.name}:{role.name}").index
            for node_type in role.allowed_node_types:
                incidence = len(role_item_indices)
                role_item_indices.append(role_items[role_handle])
                role_to_relation.append(relation_handle)
                type_handle = handles.resolve(HandleKind.TYPE, node_type).index
                role_to_type.append(type_handle)
                role_incidence[(relation.name, role.name, node_type)] = incidence
    return (
        SchemaTensorBatch(
            token_ids=token_ids,
            padding_mask=padding_mask,
            type_item_indices=torch.tensor(
                [type_items[index] for index in range(len(type_items))],
                dtype=torch.long,
                device=device,
            ),
            relation_item_indices=torch.tensor(
                [relation_items[index] for index in range(len(relation_items))],
                dtype=torch.long,
                device=device,
            ),
            role_item_indices=torch.tensor(role_item_indices, dtype=torch.long, device=device),
            role_to_relation=torch.tensor(role_to_relation, dtype=torch.long, device=device),
            role_to_type=torch.tensor(role_to_type, dtype=torch.long, device=device),
        ),
        role_incidence,
    )


def _event_tensors(
    events: tuple[GraphEventV2, ...],
    request: ForecastRequest,
    tokenizer: BraidTokenizer,
    handles: DynamicHandleTable,
    role_incidence: Mapping[tuple[str, str, str], int],
    node_type_handles: list[int],
    evidence_sequences: list[tuple[int, ...]],
    *,
    max_payload_tokens: int,
    time_unit_seconds: float,
    device: str | torch.device,
) -> EventTensorBatch:
    # An explicit learned start state allows schema-only and right-censored prefixes.
    is_start = not events
    source_events: tuple[GraphEventV2 | None, ...] = events if events else (None,)
    operation_ids: list[int] = []
    relation_indices: list[int] = []
    observed_deltas: list[float] = []
    valid_lags: list[float] = []
    payload_sequences: list[tuple[int, ...]] = []
    arguments: list[list[tuple[int, int]]] = []
    tie_groups: list[int] = []
    previous_observed = request.schema.observed_at or request.cutoff
    current_group = -1
    previous_group_clock: datetime | None = None
    operation_order = tuple(item.value for item in ModelOperation)

    for event in source_events:
        if event is None:
            operation_ids.append(0)
            relation_indices.append(-1)
            observed_deltas.append(0.0)
            valid_lags.append(0.0)
            payload_sequences.append((tokenizer.BOS, tokenizer.EOS))
            arguments.append([])
            tie_groups.append(0)
            continue
        if event.observed_at is None or event.valid_from is None:
            raise ValueError("temporal tensorization rejects unknown event clocks")
        operation_ids.append(operation_order.index(event.operation.value))
        relation_indices.append(
            -1
            if event.relation is None
            else handles.resolve(HandleKind.RELATION, event.relation).index
        )
        if previous_group_clock != event.observed_at:
            group_delta = max(
                0.0,
                (event.observed_at - previous_observed).total_seconds() / time_unit_seconds,
            )
            _validate_normalized_time(group_delta, "observed-time delta")
            previous_observed = event.observed_at
        else:
            # Every member of a tie group receives the same inter-group delta.
            # Assigning it only to the first serialized member would make arbitrary
            # same-timestamp order visible to the model before group pooling.
            group_delta = observed_deltas[-1]
        observed_deltas.append(group_delta)
        valid_lag = (event.valid_from - event.observed_at).total_seconds() / time_unit_seconds
        _validate_normalized_time(valid_lag, "valid-time lag")
        valid_lags.append(valid_lag)
        if previous_group_clock != event.observed_at:
            current_group += 1
            previous_group_clock = event.observed_at
        tie_groups.append(current_group)
        payload_sequences.append(
            tokenizer.encode_text(canonical_json(event.payload))[:max_payload_tokens]
        )
        event_arguments: list[tuple[int, int]] = []
        for binding in event.arguments:
            node_handle = handles.resolve(HandleKind.NODE, binding.node_id).index
            node_type_name = request.prefix.nodes[node_handle].node_type
            role_index = role_incidence.get(
                (event.relation or "", binding.role, node_type_name), -1
            )
            event_arguments.append((node_handle, role_index))
        arguments.append(event_arguments)

    max_arguments = max(1, *(len(items) for items in arguments))
    argument_nodes = torch.full(
        (1, len(source_events), max_arguments), -1, dtype=torch.long, device=device
    )
    argument_roles = torch.full_like(argument_nodes, -1)
    argument_mask = torch.zeros_like(argument_nodes, dtype=torch.bool)
    for index, bindings in enumerate(arguments):
        for argument_index, (node, role) in enumerate(bindings):
            argument_nodes[0, index, argument_index] = node
            argument_roles[0, index, argument_index] = role
            argument_mask[0, index, argument_index] = True
    payload_ids, _ = _pad_sequences(payload_sequences, tokenizer.PAD, device=device)
    if evidence_sequences:
        evidence_ids, _ = _pad_sequences(evidence_sequences, tokenizer.PAD, device=device)
        evidence_ids = evidence_ids.unsqueeze(0)
        evidence_mask = torch.ones((1, len(evidence_sequences)), dtype=torch.bool, device=device)
    else:
        evidence_ids = torch.full((1, 1, 1), tokenizer.PAD, dtype=torch.long, device=device)
        evidence_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
    return EventTensorBatch(
        operation_ids=torch.tensor([operation_ids], dtype=torch.long, device=device),
        relation_indices=torch.tensor([relation_indices], dtype=torch.long, device=device),
        observed_deltas=torch.tensor([observed_deltas], dtype=torch.float32, device=device),
        valid_lags=torch.tensor([valid_lags], dtype=torch.float32, device=device),
        payload_token_ids=payload_ids.unsqueeze(0),
        argument_node_indices=argument_nodes,
        argument_role_indices=argument_roles,
        argument_mask=argument_mask,
        tie_groups=torch.tensor([tie_groups], dtype=torch.long, device=device),
        event_mask=torch.ones((1, len(source_events)), dtype=torch.bool, device=device),
        node_type_indices=torch.tensor([node_type_handles], dtype=torch.long, device=device),
        evidence_token_ids=evidence_ids,
        evidence_mask=evidence_mask,
        start_mask=torch.full(
            (1, len(source_events)),
            is_start,
            dtype=torch.bool,
            device=device,
        ),
    )


def _validate_normalized_time(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if abs(value) > MAX_NORMALIZED_TIME:
        raise ValueError(f"{name} exceeds the normalized-time guard; use a larger fixed time unit")


def _grammar_context(
    request: ForecastRequest,
    handles: DynamicHandleTable,
    role_incidence: Mapping[tuple[str, str, str], int],
    node_ids: list[str],
    node_type_handles: list[int],
    node_observed: list[datetime],
    evidence_ids: list[str],
    evidence_observed: list[datetime],
) -> DecodeContext:
    relation_rules: dict[int, RelationRule] = {}
    for relation in request.schema.relations:
        relation_handle = handles.resolve(HandleKind.RELATION, relation.name).index
        roles: list[RoleRule] = []
        for role in relation.roles:
            # A role rule uses a stable local role handle, while motif incidences may
            # expand it once per allowed target type.
            role_handle = handles.resolve(HandleKind.ROLE, f"{relation.name}:{role.name}").index
            targets = frozenset(
                handles.resolve(HandleKind.TYPE, name).index for name in role.allowed_node_types
            )
            roles.append(RoleRule(role_handle, targets, role.min_count, role.max_count))
        relation_rules[relation_handle] = RelationRule(relation_handle, tuple(roles))
    assertions = set()
    for event in request.prefix.events:
        if event.operation.value != "ASSERT" or event.relation is None:
            continue
        relation = handles.resolve(HandleKind.RELATION, event.relation).index
        arguments: dict[int, list[int]] = {}
        for binding in event.arguments:
            role = handles.resolve(HandleKind.ROLE, f"{event.relation}:{binding.role}").index
            arguments.setdefault(role, []).append(
                handles.resolve(HandleKind.NODE, binding.node_id).index
            )
        assertions.add(
            (
                relation,
                tuple(sorted((role, tuple(sorted(values))) for role, values in arguments.items())),
            )
        )
    assert request.cutoff is not None
    return DecodeContext(
        cutoff=request.cutoff,
        horizon_end=request.cutoff + request.horizon,
        schema=GrammarSchema(
            request.schema.version,
            frozenset(range(handles.count(HandleKind.TYPE))),
            relation_rules,
        ),
        nodes={
            index: VisibleNode(index, node_type_handles[index], node_observed[index])
            for index, _ in enumerate(node_ids)
        },
        evidence={
            index: VisibleEvidence(index, evidence_observed[index])
            for index, _ in enumerate(evidence_ids)
        },
        event_handles=frozenset(
            handles.resolve(HandleKind.EVENT, event.event_id).index
            for event in request.prefix.events
        ),
        assertion_keys=frozenset(assertions),
    )


def _pad_sequences(
    sequences: list[tuple[int, ...]],
    padding: int,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty sequence collection")
    width = max(map(len, sequences))
    result = torch.full((len(sequences), width), padding, dtype=torch.long, device=device)
    mask = torch.ones_like(result, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        result[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = False
    return result, mask
