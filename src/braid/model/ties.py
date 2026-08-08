"""Utilities for marked same-timestamp event groups."""

from __future__ import annotations

import torch
from torch import Tensor


def randomize_within_ties(
    values: Tensor,
    tie_groups: Tensor,
    *,
    mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomize serialization order only within each marked tie group."""

    if values.ndim < 2 or tie_groups.shape != values.shape[:2]:
        raise ValueError("values must be [batch, events, ...] with matching tie groups")
    if mask is None:
        mask = torch.ones_like(tie_groups, dtype=torch.bool)
    randomized = values.clone()
    permutation = torch.arange(values.shape[1], device=values.device).repeat(values.shape[0], 1)
    for batch in range(values.shape[0]):
        active_groups = torch.unique(tie_groups[batch][mask[batch]], sorted=True)
        for group in active_groups.tolist():
            positions = torch.nonzero(
                (tie_groups[batch] == group) & mask[batch], as_tuple=False
            ).flatten()
            if positions.numel() <= 1:
                continue
            order = positions[
                torch.randperm(positions.numel(), generator=generator, device=values.device)
            ]
            randomized[batch, positions] = values[batch, order]
            permutation[batch, positions] = order
    return randomized, permutation


def pool_tie_groups(
    values: Tensor, tie_groups: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean-pool ties into a causal sequence, invariant to within-tie order.

    Returns ``(pooled, group_mask, group_ids)``.  Group order follows first
    appearance, which must already be chronological in a validated batch.
    """

    if values.ndim != 3 or tie_groups.shape != values.shape[:2]:
        raise ValueError("values must be [batch, events, width] with matching groups")
    if mask is None:
        mask = torch.ones_like(tie_groups, dtype=torch.bool)
    groups_per_batch: list[list[int]] = []
    for batch in range(values.shape[0]):
        seen: set[int] = set()
        ordered: list[int] = []
        for group in tie_groups[batch][mask[batch]].tolist():
            if group not in seen:
                seen.add(group)
                ordered.append(group)
        groups_per_batch.append(ordered)
    max_groups = max((len(groups) for groups in groups_per_batch), default=0)
    pooled = values.new_zeros((values.shape[0], max_groups, values.shape[-1]))
    group_mask = torch.zeros((values.shape[0], max_groups), dtype=torch.bool, device=values.device)
    group_ids = torch.full(
        (values.shape[0], max_groups), -1, dtype=tie_groups.dtype, device=values.device
    )
    for batch, groups in enumerate(groups_per_batch):
        for index, group in enumerate(groups):
            select = (tie_groups[batch] == group) & mask[batch]
            pooled[batch, index] = values[batch, select].mean(dim=0)
            group_mask[batch, index] = True
            group_ids[batch, index] = group
    return pooled, group_mask, group_ids
