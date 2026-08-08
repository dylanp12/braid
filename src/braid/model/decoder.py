"""Causal EventGraph decoder with factorized open-schema prediction heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from braid.model.encoder import sinusoidal_positions
from braid.model.grammar import Operation


@dataclass(slots=True)
class FactorizedEventDistribution:
    """Parameters of the next-event factors at every causal timestep."""

    hidden_states: Tensor
    delta_log_rate: Tensor
    valid_lag_mean: Tensor
    valid_lag_log_scale: Tensor
    operation_logits: Tensor
    relation_logits: Tensor
    argument_logits: Tensor
    payload_logits: Tensor
    evidence_logits: Tensor
    event_mask: Tensor


class CausalEventDecoder(nn.Module):
    """Decode time, operation, schema handle, arguments, payload, and evidence."""

    operation_order = tuple(Operation)

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        payload_vocab_size: int,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.schema_context = nn.Linear(d_model, d_model, bias=False)
        self.delta_head = nn.Linear(d_model, 1)
        self.valid_lag_head = nn.Linear(d_model, 2)
        self.operation_head = nn.Linear(d_model, len(self.operation_order))
        self.relation_query = nn.Linear(d_model, d_model, bias=False)
        self.argument_query = nn.Linear(d_model, d_model, bias=False)
        self.payload_head = nn.Linear(d_model, payload_vocab_size)
        self.evidence_query = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(
        self,
        event_states: Tensor,
        *,
        event_mask: Tensor,
        schema_context: Tensor,
        relation_states: Tensor,
        node_memory: Tensor,
        evidence_states: Tensor,
        evidence_mask: Tensor | None = None,
    ) -> FactorizedEventDistribution:
        if event_states.ndim != 3 or event_mask.shape != event_states.shape[:2]:
            raise ValueError("event states must be [batch,time,width] with a matching mask")
        if not torch.all(event_mask.any(dim=1)):
            raise ValueError("every batch item needs at least one causal event")
        if schema_context.ndim == 1:
            schema_context = schema_context.unsqueeze(0).expand(event_states.shape[0], -1)
        positions = sinusoidal_positions(
            event_states.shape[1], event_states.shape[2], device=event_states.device
        ).to(event_states.dtype)
        encoded_schema = self.schema_context(schema_context).unsqueeze(1)
        states = event_states + positions.unsqueeze(0) + encoded_schema
        causal_mask = torch.triu(
            torch.ones(
                event_states.shape[1],
                event_states.shape[1],
                dtype=torch.bool,
                device=event_states.device,
            ),
            diagonal=1,
        )
        hidden = self.decoder(
            states,
            mask=causal_mask,
            src_key_padding_mask=~event_mask,
        )
        hidden = hidden.masked_fill(~event_mask.unsqueeze(-1), 0.0)
        lag = self.valid_lag_head(hidden)
        lag_mean = lag[..., 0]
        lag_log_scale = lag[..., 1].clamp(-8.0, 8.0)
        relation_logits = _dynamic_scores(
            self.relation_query(hidden), relation_states, self.d_model
        )
        argument_logits = _dynamic_scores(self.argument_query(hidden), node_memory, self.d_model)
        evidence_logits = _dynamic_scores(
            self.evidence_query(hidden), evidence_states, self.d_model
        )
        if evidence_mask is not None:
            if evidence_mask.shape != evidence_states.shape[:2]:
                raise ValueError("evidence mask must match evidence states")
            evidence_logits = evidence_logits.masked_fill(
                ~evidence_mask.unsqueeze(1), torch.finfo(evidence_logits.dtype).min
            )
        return FactorizedEventDistribution(
            hidden_states=hidden,
            delta_log_rate=self.delta_head(hidden).squeeze(-1).clamp(-12.0, 12.0),
            valid_lag_mean=lag_mean,
            valid_lag_log_scale=lag_log_scale,
            operation_logits=self.operation_head(hidden),
            relation_logits=relation_logits,
            argument_logits=argument_logits,
            payload_logits=self.payload_head(hidden),
            evidence_logits=evidence_logits,
            event_mask=event_mask,
        )


def _dynamic_scores(query: Tensor, candidates: Tensor, width: int) -> Tensor:
    """Dot-product scores for shared or timestep-specific dynamic candidates."""

    scale = math.sqrt(width)
    if candidates.ndim == 2:
        return torch.einsum("btd,cd->btc", query, candidates) / scale
    if candidates.ndim == 3:
        return torch.einsum("btd,bcd->btc", query, candidates) / scale
    if candidates.ndim == 4:
        return torch.einsum("btd,btcd->btc", query, candidates) / scale
    raise ValueError("candidate states must have rank 2, 3, or 4")
