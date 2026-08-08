"""Evidence-gated EventGraph training losses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from braid.model.config import DEFAULT_TIME_UNIT_SECONDS, MAX_NORMALIZED_TIME
from braid.model.decoder import FactorizedEventDistribution


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    patch: float = 0.65
    schema_episode: float = 0.15
    reconstruction: float = 0.10
    contrastive: float = 0.10

    def __post_init__(self) -> None:
        values = (self.patch, self.schema_episode, self.reconstruction, self.contrastive)
        if any(value < 0 for value in values):
            raise ValueError("objective weights cannot be negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("objective weights must sum to one")

    def combine(
        self,
        *,
        patch: Tensor,
        schema_episode: Tensor,
        reconstruction: Tensor,
        contrastive: Tensor,
    ) -> Tensor:
        return (
            self.patch * patch
            + self.schema_episode * schema_episode
            + self.reconstruction * reconstruction
            + self.contrastive * contrastive
        )


@dataclass(slots=True)
class PatchTargets:
    """Next-event targets whose temporal fields are expressed in raw seconds."""

    delta_durations: Tensor
    delta_observed: Tensor
    valid_lags: Tensor
    operations: Tensor
    relations: Tensor
    arguments: Tensor
    payload_tokens: Tensor
    evidence: Tensor
    relation_mask: Tensor | None = None
    argument_mask: Tensor | None = None
    evidence_mask: Tensor | None = None


@dataclass(slots=True)
class PatchLoss:
    total: Tensor
    time: Tensor
    valid_lag: Tensor
    operation: Tensor
    relation: Tensor
    argument: Tensor
    payload: Tensor
    evidence: Tensor


def censored_exponential_nll(
    log_rate: Tensor,
    durations: Tensor,
    observed: Tensor,
    *,
    reduction: str = "mean",
    duration_scale: float = 1.0,
    maximum_normalized_duration: float = MAX_NORMALIZED_TIME,
) -> Tensor:
    """Negative log likelihood with right-censored quiet windows.

    Observed intervals contribute ``rate*t - log(rate)``; censored intervals
    contribute the survival term ``rate*t`` only.
    """

    if log_rate.shape != durations.shape or observed.shape != durations.shape:
        raise ValueError("log_rate, durations, and observed must have matching shapes")
    if not torch.isfinite(log_rate).all() or not torch.isfinite(durations).all():
        raise ValueError("log rates and durations must be finite")
    if torch.any(durations < 0):
        raise ValueError("durations cannot be negative")
    if torch.any((observed != 0) & (observed != 1)):
        raise ValueError("observed indicators must be binary")
    normalized_durations, log_scale = _normalize_time(
        durations,
        scale=duration_scale,
        maximum_absolute=maximum_normalized_duration,
        name="durations",
    )
    bounded_log_rate = log_rate.clamp(-20.0, 20.0)
    rate = torch.exp(bounded_log_rate)
    observed_values = observed.to(log_rate.dtype)
    losses = (
        rate * normalized_durations
        - observed_values * bounded_log_rate
        + observed_values * log_scale
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be none, sum, or mean")


def factorized_patch_loss(
    distribution: FactorizedEventDistribution,
    targets: PatchTargets,
    *,
    time_scale: float = DEFAULT_TIME_UNIT_SECONDS,
    maximum_normalized_time: float = MAX_NORMALIZED_TIME,
) -> PatchLoss:
    """Balanced loss across every advertised next-event factor."""

    active = distribution.event_mask
    time_loss = _masked_mean(
        censored_exponential_nll(
            distribution.delta_log_rate,
            targets.delta_durations,
            targets.delta_observed,
            reduction="none",
            duration_scale=time_scale,
            maximum_normalized_duration=maximum_normalized_time,
        ),
        active,
    )
    normalized_lags, log_time_scale = _normalize_time(
        targets.valid_lags,
        scale=time_scale,
        maximum_absolute=maximum_normalized_time,
        name="valid_lags",
    )
    scale = distribution.valid_lag_log_scale.exp()
    lag_loss = (
        _masked_mean(
            0.5 * ((normalized_lags - distribution.valid_lag_mean) / scale).square()
            + distribution.valid_lag_log_scale,
            # Converting a normalized-time density back to caller units adds a
            # constant log-Jacobian. It does not affect gradients but keeps NLLs
            # comparable across declared time units.
            active,
        )
        + log_time_scale
    )
    operation = _masked_cross_entropy(distribution.operation_logits, targets.operations, active)
    relation = _masked_cross_entropy(
        distribution.relation_logits,
        targets.relations,
        active if targets.relation_mask is None else active & targets.relation_mask,
    )
    argument = _masked_cross_entropy(
        distribution.argument_logits,
        targets.arguments,
        active if targets.argument_mask is None else active & targets.argument_mask,
    )
    payload = _masked_cross_entropy(distribution.payload_logits, targets.payload_tokens, active)
    evidence = _masked_cross_entropy(
        distribution.evidence_logits,
        targets.evidence,
        active if targets.evidence_mask is None else active & targets.evidence_mask,
    )
    factors = torch.stack([time_loss, lag_loss, operation, relation, argument, payload, evidence])
    return PatchLoss(
        total=factors.mean(),
        time=time_loss,
        valid_lag=lag_loss,
        operation=operation,
        relation=relation,
        argument=argument,
        payload=payload,
        evidence=evidence,
    )


def _masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return values.sum() * 0.0
    return values[mask].mean()


def _normalize_time(
    values: Tensor,
    *,
    scale: float,
    maximum_absolute: float,
    name: str,
) -> tuple[Tensor, float]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("time scale must be positive and finite")
    if not math.isfinite(maximum_absolute) or maximum_absolute <= 0:
        raise ValueError("maximum normalized time must be positive and finite")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    normalized = values / scale
    if torch.any(normalized.abs() > maximum_absolute):
        raise ValueError(
            f"{name} exceed the normalized-time guard; provide an explicit larger time scale"
        )
    return normalized, math.log(scale)
