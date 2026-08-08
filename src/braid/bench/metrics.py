"""Dependency-free reference metrics with explicit orientation and tie policy."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from braid.contract import (
    ContractValidationError,
    ForecastDistribution,
    ForecastRequest,
    GraphEventV2,
    validate_forecast_distribution,
)


def _finite(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def filtered_rank(
    scores: Sequence[float],
    target_index: int,
    filtered_indices: Iterable[int] = (),
) -> float:
    """Return the average rank of the target, with higher scores preferred.

    Known true alternatives may be removed with ``filtered_indices``.  A tie
    occupies the mean of its rank positions, so an all-tied scorer over ``n``
    candidates receives rank ``(n + 1) / 2`` rather than the fraudulent rank 1.
    """

    checked = _finite(scores, "scores")
    if not 0 <= target_index < len(checked):
        raise IndexError("target_index is outside scores")
    excluded = set(filtered_indices)
    excluded.discard(target_index)
    if any(index < 0 or index >= len(checked) for index in excluded):
        raise IndexError("filtered index is outside scores")
    target = checked[target_index]
    candidates = [score for index, score in enumerate(checked) if index not in excluded]
    greater = sum(score > target for score in candidates)
    tied_others = sum(score == target for score in candidates) - 1
    return 1.0 + greater + tied_others / 2.0


def filtered_mrr(
    examples: Iterable[tuple[Sequence[float], int, Iterable[int]]],
) -> float:
    ranks = [filtered_rank(scores, target, filtered) for scores, target, filtered in examples]
    if not ranks:
        raise ValueError("filtered MRR requires at least one example")
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def macro_filtered_mrr(
    groups: Mapping[Hashable, Iterable[tuple[Sequence[float], int, Iterable[int]]]],
) -> float:
    """Average group-level filtered MRR so prolific repositories cannot dominate."""

    if not groups:
        raise ValueError("macro filtered MRR requires at least one group")
    values = [filtered_mrr(examples) for examples in groups.values()]
    return sum(values) / len(values)


def average_precision(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Average precision with score ties evaluated as a threshold block.

    This is the right-continuous step integral of precision over recall.  Grouping
    ties makes the result invariant to input order.
    """

    checked_scores = _finite(scores, "scores")
    if len(labels) != len(checked_scores):
        raise ValueError("labels and scores must have equal length")
    checked_labels = tuple(int(label) for label in labels)
    if any(label not in (0, 1) for label in checked_labels):
        raise ValueError("labels must be binary")
    positives = sum(checked_labels)
    if positives == 0:
        raise ValueError("average precision is undefined without a positive label")

    grouped: dict[float, list[int]] = {}
    for label, score in zip(checked_labels, checked_scores, strict=True):
        grouped.setdefault(score, []).append(label)
    seen = true_positives = 0
    previous_recall = area = 0.0
    for score in sorted(grouped, reverse=True):
        block = grouped[score]
        seen += len(block)
        true_positives += sum(block)
        recall = true_positives / positives
        precision = true_positives / seen
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def macro_average_precision(
    groups: Mapping[Hashable, tuple[Sequence[int | bool], Sequence[float]]],
) -> float:
    if not groups:
        raise ValueError("macro AUPRC requires at least one group")
    values = [average_precision(labels, scores) for labels, scores in groups.values()]
    return sum(values) / len(values)


def integrated_brier_score(
    probabilities: Sequence[Sequence[float]],
    outcomes: Sequence[Sequence[int | bool]],
    horizon_weights: Sequence[float] | None = None,
    *,
    at_risk: Sequence[Sequence[int | bool]] | None = None,
    inverse_censoring_weights: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return an optionally IPCW-adjusted integrated discrete-time Brier score.

    Right-censored populations must supply both an ``at_risk`` mask and inverse
    censoring-survival weights fitted using training data. Unknown outcomes are
    excluded, and each horizon is normalized by its observable IPCW mass.
    """

    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes need equal, non-zero rows")
    horizons = len(probabilities[0])
    if horizons == 0 or any(len(row) != horizons for row in probabilities):
        raise ValueError("probability rows must have the same non-zero horizon count")
    if any(len(row) != horizons for row in outcomes):
        raise ValueError("outcome rows must match probability horizons")
    if (at_risk is None) != (inverse_censoring_weights is None):
        raise ValueError("right-censored Brier scoring requires both risk masks and IPCW weights")
    if at_risk is None:
        risk = tuple((1,) * horizons for _ in probabilities)
        censoring_weights = tuple((1.0,) * horizons for _ in probabilities)
    else:
        assert inverse_censoring_weights is not None
        if len(at_risk) != len(probabilities) or len(inverse_censoring_weights) != len(
            probabilities
        ):
            raise ValueError("risk masks and IPCW weights must match probability rows")
        if any(len(row) != horizons for row in at_risk) or any(
            len(row) != horizons for row in inverse_censoring_weights
        ):
            raise ValueError("risk masks and IPCW weights must match probability horizons")
        risk = tuple(tuple(int(value) for value in row) for row in at_risk)
        if any(value not in (0, 1) for row in risk for value in row):
            raise ValueError("at-risk masks must be binary")
        censoring_weights = tuple(
            tuple(float(value) for value in row) for row in inverse_censoring_weights
        )
        if any(
            not math.isfinite(value) or value <= 0
            for mask_row, weight_row in zip(risk, censoring_weights, strict=True)
            for mask, value in zip(mask_row, weight_row, strict=True)
            if mask
        ):
            raise ValueError("observed IPCW weights must be positive and finite")
    if horizon_weights is None:
        weights = (1.0 / horizons,) * horizons
    else:
        raw_weights = _finite(horizon_weights, "horizon_weights")
        if len(raw_weights) != horizons or any(weight < 0 for weight in raw_weights):
            raise ValueError("horizon weights must be non-negative and match horizons")
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("horizon weights must have positive mass")
        weights = tuple(weight / total for weight in raw_weights)

    checked_outcomes: list[tuple[int, ...]] = []
    for prediction, outcome in zip(probabilities, outcomes, strict=True):
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in prediction):
            raise ValueError("probabilities must be finite and in [0, 1]")
        binary = tuple(int(value) for value in outcome)
        if any(value not in (0, 1) for value in binary):
            raise ValueError("outcomes must be binary")
        checked_outcomes.append(binary)

    integrated = 0.0
    for horizon, horizon_weight in enumerate(weights):
        weighted_loss = 0.0
        observed_mass = 0.0
        for prediction, outcome, mask_row, ipcw_row in zip(
            probabilities,
            checked_outcomes,
            risk,
            censoring_weights,
            strict=True,
        ):
            if not mask_row[horizon]:
                continue
            ipcw = ipcw_row[horizon]
            weighted_loss += ipcw * (prediction[horizon] - outcome[horizon]) ** 2
            observed_mass += ipcw
        if observed_mass <= 0:
            raise ValueError(f"Brier horizon {horizon} has no observable IPCW mass")
        integrated += horizon_weight * weighted_loss / observed_mass
    return integrated


@dataclass(frozen=True, slots=True)
class BrierGroup:
    probabilities: tuple[tuple[float, ...], ...]
    outcomes: tuple[tuple[int | bool, ...], ...]
    at_risk: tuple[tuple[int | bool, ...], ...] | None = None
    inverse_censoring_weights: tuple[tuple[float, ...], ...] | None = None


def macro_integrated_brier_score(
    groups: Mapping[Hashable, BrierGroup],
    horizon_weights: Sequence[float] | None = None,
) -> float:
    """Average repository-level integrated Brier scores with equal group weight."""

    if not groups:
        raise ValueError("macro integrated Brier requires at least one group")
    scores = [
        integrated_brier_score(
            group.probabilities,
            group.outcomes,
            horizon_weights,
            at_risk=group.at_risk,
            inverse_censoring_weights=group.inverse_censoring_weights,
        )
        for group in groups.values()
    ]
    return sum(scores) / len(scores)


def time_calibration_error(
    cumulative_probabilities: Sequence[Sequence[float]],
    occurred_by_horizon: Sequence[Sequence[int | bool]],
    horizon_weights: Sequence[float] | None = None,
) -> float:
    """Mean absolute marginal calibration error across event-time horizons."""

    if not cumulative_probabilities or len(cumulative_probabilities) != len(occurred_by_horizon):
        raise ValueError("probabilities and outcomes need equal, non-zero rows")
    horizons = len(cumulative_probabilities[0])
    if horizons == 0 or any(len(row) != horizons for row in cumulative_probabilities):
        raise ValueError("probability rows must have the same non-zero horizon count")
    if any(len(row) != horizons for row in occurred_by_horizon):
        raise ValueError("outcome rows must match probability horizons")
    for row in cumulative_probabilities:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in row):
            raise ValueError("probabilities must be finite and in [0, 1]")
        if any(left > right for left, right in zip(row, row[1:], strict=False)):
            raise ValueError("cumulative event probabilities must be non-decreasing")
    binary = tuple(tuple(int(value) for value in row) for row in occurred_by_horizon)
    if any(value not in (0, 1) for row in binary for value in row):
        raise ValueError("outcomes must be binary")
    if any(left > right for row in binary for left, right in zip(row, row[1:], strict=False)):
        raise ValueError("occurred-by-horizon outcomes must be non-decreasing")

    if horizon_weights is None:
        weights = (1.0 / horizons,) * horizons
    else:
        raw_weights = _finite(horizon_weights, "horizon_weights")
        if len(raw_weights) != horizons or any(weight < 0 for weight in raw_weights):
            raise ValueError("horizon weights must be non-negative and match horizons")
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("horizon weights must have positive mass")
        weights = tuple(weight / total for weight in raw_weights)
    return sum(
        weight
        * abs(
            sum(row[horizon] for row in cumulative_probabilities) / len(cumulative_probabilities)
            - sum(row[horizon] for row in binary) / len(binary)
        )
        for horizon, weight in enumerate(weights)
    )


def macro_f1(expected: Sequence[Hashable], predicted: Sequence[Hashable]) -> float:
    if not expected or len(expected) != len(predicted):
        raise ValueError("expected and predicted need equal, non-zero length")
    labels = sorted(set(expected) | set(predicted), key=repr)
    scores: list[float] = []
    for label in labels:
        pairs = tuple(zip(expected, predicted, strict=True))
        true_positive = sum(e == label and p == label for e, p in pairs)
        false_positive = sum(e != label and p == label for e, p in pairs)
        false_negative = sum(e == label and p != label for e, p in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


@dataclass(frozen=True, slots=True)
class PatchEvent:
    operation: str
    relation: str | None
    arguments: tuple[tuple[str, str], ...]
    observed_delta: float
    valid_time_lag: float
    basis_refs: tuple[str, ...] = ()

    def identity(self) -> tuple[str, str | None, tuple[tuple[str, str], ...]]:
        return self.operation, self.relation, tuple(sorted(self.arguments))


@dataclass(frozen=True, slots=True)
class GraphPatchMetrics:
    """Independent patch endpoints.  Deliberately has no aggregate/composite."""

    contract_validity: float
    matched_event_precision: float
    matched_event_recall: float
    matched_event_f1: float
    evidence_basis_accuracy: float
    mean_absolute_observed_time_error: float
    mean_absolute_valid_time_error: float
    distinct_window_rate: float


@dataclass(frozen=True, slots=True)
class ForecastPatchCase:
    """A prediction bound to the exact causal request it must validate against."""

    request: ForecastRequest
    distribution: ForecastDistribution


def _window_key(window: Sequence[PatchEvent]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            *event.identity(),
            event.observed_delta,
            event.valid_time_lag,
            tuple(sorted(event.basis_refs)),
        )
        for event in window
    )


def _project_event(request: ForecastRequest, event: GraphEventV2) -> PatchEvent:
    """Project a contract event into the metric representation without hiding bad clocks."""

    if request.cutoff is None or event.observed_at is None:
        observed_delta = math.inf
    else:
        observed_delta = (event.observed_at - request.cutoff).total_seconds()
    if event.observed_at is None or event.valid_from is None:
        valid_time_lag = math.inf
    else:
        valid_time_lag = (event.valid_from - event.observed_at).total_seconds()
    return PatchEvent(
        operation=event.operation.value,
        relation=event.relation,
        arguments=tuple((binding.role, binding.node_id) for binding in event.arguments),
        observed_delta=observed_delta,
        valid_time_lag=valid_time_lag,
        basis_refs=event.basis_refs,
    )


def _project_window(
    request: ForecastRequest, events: Sequence[GraphEventV2]
) -> tuple[PatchEvent, ...]:
    return tuple(_project_event(request, event) for event in events)


def _distribution_is_contract_valid(case: ForecastPatchCase) -> bool:
    try:
        validate_forecast_distribution(case.distribution, request=case.request)
    except (ContractValidationError, AssertionError, TypeError, ValueError):
        return False
    return True


def graph_patch_metrics(
    predictions: Sequence[ForecastPatchCase],
    reference_windows: Sequence[Sequence[PatchEvent]],
) -> GraphPatchMetrics:
    """Score native distributions, recomputing validity against causal requests.

    The highest-probability sampled window is the point prediction. Diversity is
    measured among all conditional samples for the same request. Contract validity
    is never accepted as caller-supplied metadata.
    """

    count = len(predictions)
    if count == 0 or len(reference_windows) != count:
        raise ValueError("patch metric inputs need equal, non-zero request counts")

    contract_valid = tuple(_distribution_is_contract_valid(case) for case in predictions)
    predicted_windows: list[tuple[PatchEvent, ...]] = []
    conditional_samples: list[tuple[tuple[PatchEvent, ...], ...]] = []
    visible_evidence: list[frozenset[str]] = []
    for case in predictions:
        windows = case.distribution.sampled_patch_windows
        projected = tuple(_project_window(case.request, window.events) for window in windows)
        conditional_samples.append(projected)
        if windows:
            primary_index = max(
                range(len(windows)),
                key=lambda index: (
                    windows[index].probability
                    if math.isfinite(windows[index].probability)
                    else -math.inf
                ),
            )
            predicted_windows.append(projected[primary_index])
        else:
            predicted_windows.append(())
        cutoff = case.request.cutoff
        visible_evidence.append(
            frozenset(
                record.evidence_id
                for record in case.request.prefix.evidence
                if cutoff is not None
                and record.observed_at is not None
                and record.observed_at <= cutoff
            )
        )

    predicted_count = reference_count = matched_count = 0
    evidence_scores: list[float] = []
    observed_errors: list[float] = []
    valid_errors: list[float] = []
    for predicted, reference, visible in zip(
        predicted_windows, reference_windows, visible_evidence, strict=True
    ):
        predicted_count += len(predicted)
        reference_count += len(reference)
        predicted_groups: dict[tuple[object, ...], list[PatchEvent]] = {}
        reference_groups: dict[tuple[object, ...], list[PatchEvent]] = {}
        for event in predicted:
            predicted_groups.setdefault(event.identity(), []).append(event)
        for event in reference:
            reference_groups.setdefault(event.identity(), []).append(event)
        for identity in sorted(set(predicted_groups) & set(reference_groups), key=repr):
            sort_key = lambda item: (  # noqa: E731 - shared deterministic event key
                item.observed_delta,
                item.valid_time_lag,
                item.basis_refs,
            )
            predicted_group = sorted(predicted_groups[identity], key=sort_key)
            reference_group = sorted(reference_groups[identity], key=sort_key)
            for event, target in zip(predicted_group, reference_group, strict=False):
                matched_count += 1
                observed_errors.append(abs(event.observed_delta - target.observed_delta))
                valid_errors.append(abs(event.valid_time_lag - target.valid_time_lag))
                predicted_basis = set(event.basis_refs)
                target_basis = set(target.basis_refs)
                if not predicted_basis.issubset(visible):
                    evidence_scores.append(0.0)
                elif not predicted_basis and not target_basis:
                    evidence_scores.append(1.0)
                else:
                    union = predicted_basis | target_basis
                    evidence_scores.append(len(predicted_basis & target_basis) / len(union))

    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    conditional_diversity = [
        len({_window_key(window) for window in samples}) / len(samples)
        if len(samples) >= 2
        else 0.0
        for samples in conditional_samples
    ]
    return GraphPatchMetrics(
        contract_validity=sum(bool(value) for value in contract_valid) / count,
        matched_event_precision=precision,
        matched_event_recall=recall,
        matched_event_f1=f1,
        evidence_basis_accuracy=(
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        ),
        mean_absolute_observed_time_error=(
            sum(observed_errors) / len(observed_errors) if observed_errors else math.inf
        ),
        mean_absolute_valid_time_error=(
            sum(valid_errors) / len(valid_errors) if valid_errors else math.inf
        ),
        distinct_window_rate=sum(conditional_diversity) / len(conditional_diversity),
    )


def normalized_window_nll(log_likelihoods: Sequence[float], token_counts: Sequence[int]) -> float:
    if not log_likelihoods or len(log_likelihoods) != len(token_counts):
        raise ValueError("likelihoods and token counts need equal, non-zero length")
    if any(count <= 0 for count in token_counts):
        raise ValueError("token counts must be positive")
    if any(not math.isfinite(value) for value in log_likelihoods):
        raise ValueError("log likelihoods must be finite")
    return -sum(log_likelihoods) / sum(token_counts)


def memorization_rate(
    generated_windows: Sequence[Sequence[PatchEvent]],
    training_windows: Sequence[Sequence[PatchEvent]],
) -> float:
    if not generated_windows:
        raise ValueError("generated windows cannot be empty")
    training = {_window_key(window) for window in training_windows}
    return sum(_window_key(window) in training for window in generated_windows) / len(
        generated_windows
    )
