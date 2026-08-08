"""Cluster-aware uncertainty estimates and family-wise error correction."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HierarchicalObservation:
    """A paired Braid-minus-baseline improvement with an outer-to-inner path."""

    hierarchy: tuple[str, ...]
    improvement: float

    def __post_init__(self) -> None:
        if not self.hierarchy or any(not value for value in self.hierarchy):
            raise ValueError("bootstrap observations require a non-empty hierarchy")
        if not math.isfinite(self.improvement):
            raise ValueError("bootstrap improvements must be finite")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    estimate: float
    lower_bound: float
    upper_bound: float
    one_sided_p_value: float
    draws: tuple[float, ...]
    confidence: float
    seed: int


def _tree(observations: tuple[HierarchicalObservation, ...]) -> dict[str, object]:
    root: dict[str, object] = {}
    for observation in observations:
        node = root
        for label in observation.hierarchy:
            child = node.setdefault(label, {})
            if not isinstance(child, dict):  # pragma: no cover - guarded by construction
                raise TypeError("invalid bootstrap hierarchy")
            node = child
        node.setdefault("__values__", []).append(observation.improvement)
    return root


def _resample_node(node: dict[str, object], generator: random.Random) -> float:
    values = node.get("__values__", [])
    if not isinstance(values, list):  # pragma: no cover - guarded by construction
        raise TypeError("invalid bootstrap leaf")
    child_names = sorted(name for name in node if name != "__values__")
    if child_names:
        sampled_children: list[float] = []
        for _ in child_names:
            child = node[generator.choice(child_names)]
            if not isinstance(child, dict):  # pragma: no cover - guarded by construction
                raise TypeError("invalid bootstrap child")
            sampled_children.append(_resample_node(child, generator))
        return sum(sampled_children) / len(sampled_children)
    if not values:  # pragma: no cover - guarded by construction
        raise TypeError("invalid bootstrap leaf")
    sampled_values = [generator.choice(values) for _ in range(len(values))]
    return sum(sampled_values) / len(sampled_values)


def _hierarchical_mean(node: dict[str, object]) -> float:
    child_names = sorted(name for name in node if name != "__values__")
    if child_names:
        children = [node[name] for name in child_names]
        if any(not isinstance(child, dict) for child in children):  # pragma: no cover
            raise TypeError("invalid bootstrap child")
        return sum(_hierarchical_mean(child) for child in children) / len(children)
    values = node.get("__values__", [])
    if not isinstance(values, list) or not values:  # pragma: no cover - construction guarantees it
        raise TypeError("invalid bootstrap leaf")
    return sum(values) / len(values)


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_hierarchical_bootstrap(
    observations: Iterable[HierarchicalObservation],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Resample every declared hierarchy level and return a paired improvement CI."""

    materialized = tuple(observations)
    if not materialized:
        raise ValueError("hierarchical bootstrap needs observations")
    if samples < 100:
        raise ValueError("hierarchical bootstrap requires at least 100 draws")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    depths = {len(observation.hierarchy) for observation in materialized}
    if len(depths) != 1:
        raise ValueError("all bootstrap observations must declare the same hierarchy depth")
    tree = _tree(materialized)
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        draws.append(_resample_node(tree, generator))
    ordered = tuple(sorted(draws))
    tail = (1 - confidence) / 2
    nonpositive = sum(value <= 0 for value in draws)
    return BootstrapResult(
        estimate=_hierarchical_mean(tree),
        lower_bound=_quantile(ordered, tail),
        upper_bound=_quantile(ordered, 1 - tail),
        one_sided_p_value=(nonpositive + 1) / (samples + 1),
        draws=tuple(draws),
        confidence=confidence,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class HolmDecision:
    hypothesis: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> tuple[HolmDecision, ...]:
    """Return deterministic Holm-adjusted p-values and step-down decisions."""

    if not p_values:
        raise ValueError("Holm correction needs at least one hypothesis")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be finite and in [0, 1]")

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running_adjusted = 0.0
    decisions: dict[str, HolmDecision] = {}
    continue_rejecting = True
    for index, (name, value) in enumerate(ordered):
        multiplier = count - index
        running_adjusted = max(running_adjusted, multiplier * value)
        adjusted = min(1.0, running_adjusted)
        threshold = alpha / multiplier
        rejected = continue_rejecting and value <= threshold
        if not rejected:
            continue_rejecting = False
        decisions[name] = HolmDecision(name, value, adjusted, rejected)
    return tuple(decisions[name] for name in sorted(decisions))
