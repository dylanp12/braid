"""Schema and relation-motif encoders with no global schema identifiers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def sinusoidal_positions(length: int, width: int, *, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies[: result[:, 1::2].shape[1]])
    return result


@dataclass(slots=True)
class SchemaEncoding:
    token_states: Tensor
    item_states: Tensor


class SchemaEncoder(nn.Module):
    """Bidirectional byte-level encoder for schema declaration semantics."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )

    def forward(self, token_ids: Tensor, padding_mask: Tensor | None = None) -> SchemaEncoding:
        if token_ids.ndim != 2:
            raise ValueError("schema token_ids must have shape [items, tokens]")
        if padding_mask is None:
            padding_mask = token_ids.eq(self.padding_idx)
        if padding_mask.shape != token_ids.shape:
            raise ValueError("schema padding mask must match token_ids")
        positions = sinusoidal_positions(
            token_ids.shape[1], self.token_embedding.embedding_dim, device=token_ids.device
        )
        embedded = self.token_embedding(token_ids) + positions.unsqueeze(0).to(
            self.token_embedding.weight.dtype
        )
        states = self.encoder(embedded, src_key_padding_mask=padding_mask)
        keep = (~padding_mask).unsqueeze(-1)
        pooled = (states * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)
        return SchemaEncoding(states, pooled)

    def mean_embed(self, token_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Cheap shared encoder for payload/evidence fields."""

        if padding_mask is None:
            padding_mask = token_ids.eq(self.padding_idx)
        embedded = self.token_embedding(token_ids)
        keep = (~padding_mask).unsqueeze(-1)
        return (embedded * keep).sum(dim=-2) / keep.sum(dim=-2).clamp_min(1)


@dataclass(slots=True)
class MotifEncoding:
    relations: Tensor
    types: Tensor
    roles: Tensor


class RelationMotifEncoder(nn.Module):
    """Exchange structural messages over relation-role-type hyperedges.

    The only inputs are semantic schema states and episode-local incidence tensors;
    there are no learned relation, role, or type ID tables.
    """

    def __init__(self, d_model: int, *, rounds: int = 2) -> None:
        super().__init__()
        if rounds <= 0:
            raise ValueError("motif rounds must be positive")
        self.rounds = rounds
        self.role_message = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.relation_update = nn.GRUCell(d_model, d_model)
        self.type_update = nn.GRUCell(d_model, d_model)
        self.role_norm = nn.LayerNorm(d_model)
        self.relation_norm = nn.LayerNorm(d_model)
        self.type_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        relation_states: Tensor,
        type_states: Tensor,
        role_states: Tensor,
        role_to_relation: Tensor,
        role_to_type: Tensor,
    ) -> MotifEncoding:
        if role_to_relation.shape != role_to_type.shape or role_to_relation.ndim != 1:
            raise ValueError("role incidence tensors must be matching vectors")
        if role_states.shape[0] != role_to_relation.shape[0]:
            raise ValueError("every role needs one incidence")
        relations = relation_states
        types = type_states
        roles = role_states
        for _ in range(self.rounds):
            messages = self.role_message(
                torch.cat([roles, relations[role_to_relation], types[role_to_type]], dim=-1)
            )
            relation_messages = _scatter_mean(messages, role_to_relation, relations.shape[0])
            type_messages = _scatter_mean(messages, role_to_type, types.shape[0])
            relations = self.relation_norm(self.relation_update(relation_messages, relations))
            types = self.type_norm(self.type_update(type_messages, types))
            roles = self.role_norm(
                roles
                + self.role_message(
                    torch.cat([roles, relations[role_to_relation], types[role_to_type]], dim=-1)
                )
            )
        return MotifEncoding(relations, types, roles)


def _scatter_mean(values: Tensor, indices: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, values.shape[-1]))
    counts = values.new_zeros((size, 1))
    output.index_add_(0, indices, values)
    counts.index_add_(0, indices, values.new_ones((indices.shape[0], 1)))
    return output / counts.clamp_min(1)
