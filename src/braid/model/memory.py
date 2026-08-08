"""Persistent role-aware node memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RoleAwareNodeMemory(nn.Module):
    """Update node state from simultaneous role-labelled event messages."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.type_projection = nn.Linear(d_model, d_model)
        self.message = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.update_cell = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def initial_state(self, node_type_states: Tensor) -> Tensor:
        """Initialize ``[batch, nodes, width]`` memory from local type states."""

        if node_type_states.ndim != 3:
            raise ValueError("node_type_states must have shape [batch, nodes, width]")
        return self.norm(self.type_projection(node_type_states))

    def update_tie_group(
        self,
        memory: Tensor,
        event_states: Tensor,
        node_indices: Tensor,
        role_states: Tensor,
        argument_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply one simultaneous update, invariant to event order in the tie.

        Shapes are memory ``[B,N,D]``, events ``[B,E,D]``, arguments
        ``[B,E,A]``, and role states ``[B,E,A,D]``.
        """

        if memory.ndim != 3 or event_states.ndim != 3:
            raise ValueError("memory and event states must be rank three")
        if node_indices.shape != role_states.shape[:-1]:
            raise ValueError("node indices and role states must align")
        if event_states.shape[:2] != node_indices.shape[:2]:
            raise ValueError("events and arguments must align")
        if argument_mask is None:
            argument_mask = node_indices.ge(0)
        safe_nodes = node_indices.clamp_min(0)
        expanded_events = event_states.unsqueeze(2).expand_as(role_states)
        messages = self.message(torch.cat([expanded_events, role_states], dim=-1))
        messages = messages * argument_mask.unsqueeze(-1)
        aggregated = memory.new_zeros(memory.shape)
        counts = memory.new_zeros((*memory.shape[:2], 1))
        for batch in range(memory.shape[0]):
            flat_mask = argument_mask[batch].reshape(-1)
            if not flat_mask.any():
                continue
            indices = safe_nodes[batch].reshape(-1)[flat_mask]
            if torch.any(indices >= memory.shape[1]):
                raise ValueError("argument node index exceeds memory size")
            batch_messages = messages[batch].reshape(-1, memory.shape[-1])[flat_mask]
            aggregated[batch].index_add_(0, indices, batch_messages)
            counts[batch].index_add_(0, indices, memory.new_ones((indices.shape[0], 1)))
        active = counts.squeeze(-1).gt(0)
        averaged = aggregated / counts.clamp_min(1)
        proposed = self.update_cell(
            averaged.reshape(-1, memory.shape[-1]),
            memory.reshape(-1, memory.shape[-1]),
        ).reshape_as(memory)
        return self.norm(torch.where(active.unsqueeze(-1), proposed, memory))
