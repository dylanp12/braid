"""The schema-conditioned Braid EventGraph Transformer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from braid.model.config import EventGraphConfig
from braid.model.decoder import CausalEventDecoder, FactorizedEventDistribution
from braid.model.encoder import MotifEncoding, RelationMotifEncoder, SchemaEncoder
from braid.model.grammar import Operation
from braid.model.memory import RoleAwareNodeMemory


@dataclass(slots=True)
class SchemaTensorBatch:
    """One episode schema shared by a batch of event prefixes."""

    token_ids: Tensor
    padding_mask: Tensor
    type_item_indices: Tensor
    relation_item_indices: Tensor
    role_item_indices: Tensor
    role_to_relation: Tensor
    role_to_type: Tensor

    def validate(self) -> None:
        if self.token_ids.ndim != 2 or self.padding_mask.shape != self.token_ids.shape:
            raise ValueError("schema token IDs and padding mask must align")
        for name in (
            "type_item_indices",
            "relation_item_indices",
            "role_item_indices",
            "role_to_relation",
            "role_to_type",
        ):
            if getattr(self, name).ndim != 1:
                raise ValueError(f"{name} must be a vector")
        if self.role_item_indices.shape != self.role_to_relation.shape:
            raise ValueError("every role item needs a relation incidence")
        if self.role_item_indices.shape != self.role_to_type.shape:
            raise ValueError("every role item needs a type incidence")


@dataclass(slots=True)
class EventTensorBatch:
    """Causally visible prefix tensors using only episode-local handles."""

    operation_ids: Tensor
    relation_indices: Tensor
    observed_deltas: Tensor
    valid_lags: Tensor
    payload_token_ids: Tensor
    argument_node_indices: Tensor
    argument_role_indices: Tensor
    argument_mask: Tensor
    tie_groups: Tensor
    event_mask: Tensor
    node_type_indices: Tensor
    evidence_token_ids: Tensor
    evidence_mask: Tensor
    start_mask: Tensor | None = None

    def validate(self) -> None:
        shape = self.operation_ids.shape
        if self.operation_ids.ndim != 2:
            raise ValueError("operations must have shape [batch,events]")
        for name in (
            "relation_indices",
            "observed_deltas",
            "valid_lags",
            "tie_groups",
            "event_mask",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must match operation shape")
        if self.payload_token_ids.shape[:2] != shape:
            raise ValueError("payload fields must align with events")
        if self.argument_node_indices.shape != self.argument_role_indices.shape:
            raise ValueError("argument node and role tensors must align")
        if self.argument_mask.shape != self.argument_node_indices.shape:
            raise ValueError("argument mask must match arguments")
        if self.argument_node_indices.shape[:2] != shape:
            raise ValueError("arguments must align with events")
        if self.node_type_indices.ndim != 2:
            raise ValueError("node type handles must have shape [batch,nodes]")
        if self.evidence_token_ids.ndim != 3:
            raise ValueError("evidence token IDs must have shape [batch,evidence,tokens]")
        if self.evidence_mask.shape != self.evidence_token_ids.shape[:2]:
            raise ValueError("evidence mask must match evidence items")
        if self.start_mask is not None and self.start_mask.shape != shape:
            raise ValueError("start mask must match events")
        if not torch.all(self.event_mask.any(dim=1)):
            raise ValueError("every batch row needs at least one event")
        active_groups = self.tie_groups.masked_fill(~self.event_mask, -1)
        for row in active_groups:
            values = row[row >= 0]
            if values.numel() > 1 and torch.any(values[1:] < values[:-1]):
                raise ValueError("tie groups must be chronological and nondecreasing")


@dataclass(slots=True)
class EventGraphOutput:
    distribution: FactorizedEventDistribution
    final_node_memory: Tensor
    schema: MotifEncoding
    group_ids: Tensor


class EventGraphModel(nn.Module):
    """Schema-conditioned temporal event transducer.

    This class initializes weights but makes no claim that they are trained.  The
    forecasting wrapper refuses generation unless a checkpoint records training.
    """

    def __init__(self, config: EventGraphConfig, *, tokenizer_vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.tokenizer_vocab_size = tokenizer_vocab_size
        width = config.d_model
        self.schema_encoder = SchemaEncoder(
            vocab_size=tokenizer_vocab_size,
            d_model=width,
            n_heads=config.n_heads,
            n_layers=config.schema_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.motif_encoder = RelationMotifEncoder(width)
        self.node_memory = RoleAwareNodeMemory(width)
        self.operation_embedding = nn.Embedding(len(Operation), width)
        self.start_event = nn.Parameter(torch.zeros(width))
        self.time_embedding = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.event_norm = nn.LayerNorm(width)
        self.decoder = CausalEventDecoder(
            d_model=width,
            n_heads=config.n_heads,
            n_layers=config.decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            payload_vocab_size=tokenizer_vocab_size,
        )

    @property
    def architecture_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "architecture": type(self).__name__,
                "config": self.config.to_dict(),
                "tokenizer_vocab_size": self.tokenizer_vocab_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_schema(self, batch: SchemaTensorBatch) -> MotifEncoding:
        batch.validate()
        encoded = self.schema_encoder(batch.token_ids, batch.padding_mask)
        return self.motif_encoder(
            encoded.item_states[batch.relation_item_indices],
            encoded.item_states[batch.type_item_indices],
            encoded.item_states[batch.role_item_indices],
            batch.role_to_relation,
            batch.role_to_type,
        )

    def forward(
        self,
        schema_batch: SchemaTensorBatch,
        event_batch: EventTensorBatch,
        *,
        initial_node_memory: Tensor | None = None,
    ) -> EventGraphOutput:
        event_batch.validate()
        active_events = event_batch.event_mask.sum(dim=1)
        if torch.any(active_events > self.config.context_events):
            largest = int(active_events.max().item())
            raise ValueError(
                f"event prefix has {largest} active events; configured context limit is "
                f"{self.config.context_events}"
            )
        node_count = event_batch.node_type_indices.shape[1]
        if node_count > self.config.memory_slots:
            raise ValueError(
                f"event prefix has {node_count} nodes; configured memory limit is "
                f"{self.config.memory_slots}"
            )
        schema = self.encode_schema(schema_batch)
        if torch.any(event_batch.node_type_indices >= schema.types.shape[0]):
            raise ValueError("node references an unknown local type handle")
        node_types = schema.types[event_batch.node_type_indices]
        if initial_node_memory is None:
            memory = self.node_memory.initial_state(node_types)
        else:
            expected_shape = (*event_batch.node_type_indices.shape, self.config.d_model)
            if initial_node_memory.shape != expected_shape:
                raise ValueError(
                    "initial node memory must have shape "
                    f"{expected_shape}; received {tuple(initial_node_memory.shape)}"
                )
            if initial_node_memory.device != node_types.device:
                raise ValueError("initial node memory must be on the event-batch device")
            if initial_node_memory.dtype != node_types.dtype:
                raise ValueError("initial node memory dtype must match model activations")
            if not torch.isfinite(initial_node_memory).all():
                raise ValueError("initial node memory must be finite")
            memory = initial_node_memory
        evidence_shape = event_batch.evidence_token_ids.shape
        evidence_states = self.schema_encoder.mean_embed(
            event_batch.evidence_token_ids.reshape(-1, evidence_shape[-1])
        ).reshape(evidence_shape[0], evidence_shape[1], -1)
        base_events = self._base_event_states(schema, event_batch)
        grouped, group_mask, group_ids, memories = self._run_memory(
            base_events, schema, memory, event_batch
        )
        schema_context = torch.cat([schema.types, schema.relations, schema.roles], dim=0).mean(0)
        distribution = self.decoder(
            grouped,
            event_mask=group_mask,
            schema_context=schema_context,
            relation_states=schema.relations,
            node_memory=memories,
            evidence_states=evidence_states,
            evidence_mask=event_batch.evidence_mask,
        )
        final_indices = group_mask.sum(dim=1) - 1
        final_memory = memories[
            torch.arange(memories.shape[0], device=memories.device), final_indices
        ]
        return EventGraphOutput(distribution, final_memory, schema, group_ids)

    def _base_event_states(self, schema: MotifEncoding, batch: EventTensorBatch) -> Tensor:
        safe_operations = batch.operation_ids.clamp(0, len(Operation) - 1)
        if schema.relations.shape[0]:
            safe_relations = batch.relation_indices.clamp(0, schema.relations.shape[0] - 1)
            relation_states = schema.relations[safe_relations]
            relation_states = relation_states * batch.relation_indices.ge(0).unsqueeze(-1)
        else:
            relation_states = self.operation_embedding.weight.new_zeros(
                (*batch.relation_indices.shape, self.config.d_model)
            )
        payload_shape = batch.payload_token_ids.shape
        payload_states = self.schema_encoder.mean_embed(
            batch.payload_token_ids.reshape(-1, payload_shape[-1])
        ).reshape(*payload_shape[:2], -1)
        time_features = torch.stack(
            [
                torch.log1p(batch.observed_deltas.clamp_min(0)),
                batch.valid_lags.sign() * torch.log1p(batch.valid_lags.abs()),
            ],
            dim=-1,
        ).to(payload_states.dtype)
        result = (
            self.operation_embedding(safe_operations)
            + relation_states
            + payload_states
            + self.time_embedding(time_features)
        )
        result = self.event_norm(result)
        if batch.start_mask is not None:
            result = torch.where(
                batch.start_mask.unsqueeze(-1), self.start_event.view(1, 1, -1), result
            )
        return result * batch.event_mask.unsqueeze(-1)

    def _run_memory(
        self,
        base_events: Tensor,
        schema: MotifEncoding,
        memory: Tensor,
        batch: EventTensorBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        grouped_rows: list[list[Tensor]] = []
        memory_rows: list[list[Tensor]] = []
        group_id_rows: list[list[int]] = []
        for row in range(base_events.shape[0]):
            groups = torch.unique_consecutive(batch.tie_groups[row][batch.event_mask[row]])
            row_groups: list[Tensor] = []
            row_memories: list[Tensor] = []
            row_ids: list[int] = []
            current = memory[row : row + 1]
            for group in groups:
                select = batch.event_mask[row] & batch.tie_groups[row].eq(group)
                event_states = base_events[row : row + 1, select]
                node_indices = batch.argument_node_indices[row : row + 1, select]
                role_indices = batch.argument_role_indices[row : row + 1, select]
                argument_mask = batch.argument_mask[row : row + 1, select]
                if schema.roles.shape[0]:
                    safe_roles = role_indices.clamp(0, schema.roles.shape[0] - 1)
                    role_states = schema.roles[safe_roles]
                    role_states = role_states * role_indices.ge(0).unsqueeze(-1)
                else:
                    role_states = event_states.new_zeros(
                        (*node_indices.shape, event_states.shape[-1])
                    )
                if current.shape[1]:
                    safe_nodes = node_indices.clamp(0, current.shape[1] - 1)
                    gathered_memory = current[0, safe_nodes[0]].unsqueeze(0)
                    argument_context = (
                        (gathered_memory + role_states) * argument_mask.unsqueeze(-1)
                    ).sum(dim=2) / argument_mask.sum(dim=2, keepdim=True).clamp_min(1)
                else:
                    argument_context = event_states.new_zeros(event_states.shape)
                event_states = self.event_norm(event_states + argument_context)
                row_groups.append(event_states.mean(dim=1).squeeze(0))
                current = self.node_memory.update_tie_group(
                    current,
                    event_states,
                    node_indices,
                    role_states,
                    argument_mask,
                )
                row_memories.append(current.squeeze(0))
                row_ids.append(int(group.item()))
            grouped_rows.append(row_groups)
            memory_rows.append(row_memories)
            group_id_rows.append(row_ids)

        max_groups = max(map(len, grouped_rows))
        grouped = base_events.new_zeros((base_events.shape[0], max_groups, base_events.shape[-1]))
        memories = memory.new_zeros((memory.shape[0], max_groups, memory.shape[1], memory.shape[2]))
        group_mask = torch.zeros(
            (base_events.shape[0], max_groups), dtype=torch.bool, device=base_events.device
        )
        group_ids = torch.full(
            (base_events.shape[0], max_groups),
            -1,
            dtype=batch.tie_groups.dtype,
            device=base_events.device,
        )
        for row, row_groups in enumerate(grouped_rows):
            count = len(row_groups)
            grouped[row, :count] = torch.stack(row_groups)
            memories[row, :count] = torch.stack(memory_rows[row])
            group_mask[row, :count] = True
            group_ids[row, :count] = torch.tensor(
                group_id_rows[row], dtype=group_ids.dtype, device=group_ids.device
            )
        return grouped, group_mask, group_ids, memories
