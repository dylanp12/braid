import math
from datetime import UTC, datetime, timedelta

import pytest

from braid.bench.metrics import (
    BrierGroup,
    ForecastPatchCase,
    PatchEvent,
    average_precision,
    filtered_mrr,
    filtered_rank,
    graph_patch_metrics,
    integrated_brier_score,
    macro_f1,
    macro_filtered_mrr,
    macro_integrated_brier_score,
    memorization_rate,
    normalized_window_nll,
    time_calibration_error,
)
from braid.contract import (
    DerivationRecord,
    EvidenceRecord,
    ForecastDistribution,
    ForecastRequest,
    GraphBundleV2,
    GraphEventV2,
    NodeRecord,
    NodeTypeDecl,
    Operation,
    PatchWindow,
    ProvenanceRecord,
    RelationDecl,
    RoleBinding,
    RoleDecl,
    SchemaDecl,
    TaskDeclaration,
)


def event(
    relation: str = "ASSIGNED_TO",
    observed: float = 2.0,
    valid: float = 1.0,
    evidence: tuple[str, ...] = ("ev-1",),
) -> PatchEvent:
    return PatchEvent(
        "ASSERT",
        relation,
        (("assignee", "person-1"), ("subject", "issue-1")),
        observed,
        valid,
        evidence,
    )


def forecast_case(
    samples: tuple[tuple[str, tuple[str, ...], float], ...],
) -> ForecastPatchCase:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    cutoff = start + timedelta(days=1)
    schema = SchemaDecl(
        version="2.0.0",
        observed_at=start,
        node_types=(NodeTypeDecl("Person"), NodeTypeDecl("Issue")),
        relations=tuple(
            RelationDecl(
                name,
                roles=(
                    RoleDecl("assignee", ("Person",)),
                    RoleDecl("subject", ("Issue",)),
                ),
            )
            for name in ("ASSIGNED_TO", "ALTERNATIVE", "RELATES_TO", "A", "B")
        ),
    )
    evidence = EvidenceRecord("ev-1", start, "test")
    prefix = GraphBundleV2(
        bundle_id="bundle:metric",
        schemas=(schema,),
        nodes=(
            NodeRecord("person-1", "Person", schema.version, start),
            NodeRecord("issue-1", "Issue", schema.version, start),
        ),
        evidence=(evidence,),
        events=(),
    )
    request = ForecastRequest(
        request_id="request:metric",
        schema=schema,
        prefix=prefix,
        cutoff=cutoff,
        horizon=timedelta(days=1),
        task=TaskDeclaration("patch"),
    )
    windows = []
    for index, (relation, basis_refs, probability) in enumerate(samples):
        observed_at = cutoff + timedelta(seconds=2)
        proposal = GraphEventV2(
            event_id=f"proposal:{index}",
            observed_at=observed_at,
            valid_from=observed_at + timedelta(seconds=1),
            valid_to=None,
            operation=Operation.ASSERT,
            schema_version=schema.version,
            relation=relation,
            arguments=(
                RoleBinding("assignee", "person-1"),
                RoleBinding("subject", "issue-1"),
            ),
            payload={},
            basis_refs=basis_refs,
            derivation=DerivationRecord("metric-test", basis_refs),
            provenance=ProvenanceRecord("metric-test", None, None, observed_at),
        )
        windows.append(PatchWindow((proposal,), probability))
    return ForecastPatchCase(
        request,
        ForecastDistribution(
            sampled_patch_windows=tuple(windows),
            event_marginals=(),
            calibrated_uncertainty=0.2,
            abstention_reason=None,
            retrieval_coverage=1.0,
            model_manifest_id="manifest:metric",
        ),
    )


def test_all_tied_rank_uses_average_position() -> None:
    assert filtered_rank([0.0, 0.0, 0.0, 0.0], 0) == 2.5
    assert filtered_mrr([([0.0, 0.0, 0.0, 0.0], 0, ())]) == pytest.approx(0.4)


def test_filtered_truths_are_removed_but_target_is_not() -> None:
    assert filtered_rank([0.9, 0.8, 0.8, 0.1], 1, (0, 1)) == 1.5
    with pytest.raises(IndexError):
        filtered_rank([0.1], 0, (4,))


def test_filtered_mrr_is_macro_averaged_across_top_level_groups() -> None:
    groups = {
        "small": [([1.0, 0.0], 0, ())],
        "large": [([0.0, 1.0], 0, ()) for _ in range(9)],
    }
    assert macro_filtered_mrr(groups) == pytest.approx(0.75)


def test_metric_orientation_canaries() -> None:
    perfect_ap = average_precision([1, 0, 1, 0], [1.0, 0.2, 0.9, 0.1])
    reversed_ap = average_precision([1, 0, 1, 0], [0.1, 0.9, 0.2, 1.0])
    assert perfect_ap == 1.0
    assert reversed_ap < perfect_ap

    perfect_brier = integrated_brier_score([[0.0, 1.0]], [[0, 1]])
    reversed_brier = integrated_brier_score([[1.0, 0.0]], [[0, 1]])
    assert perfect_brier == 0.0
    assert reversed_brier == 1.0

    calibrated_time = time_calibration_error(
        [[0.0, 1.0], [1.0, 1.0]],
        [[0, 1], [1, 1]],
    )
    miscalibrated_time = time_calibration_error(
        [[0.0, 0.0], [0.0, 0.0]],
        [[0, 1], [1, 1]],
    )
    assert calibrated_time == 0.0
    assert miscalibrated_time > calibrated_time


def test_integrated_brier_handles_right_censoring_with_ipcw() -> None:
    score = integrated_brier_score(
        [[0.0, 0.2], [0.0, 0.9]],
        [[0, 0], [0, 1]],
        at_risk=[[1, 0], [1, 1]],
        inverse_censoring_weights=[[1.0, 1.0], [1.0, 2.0]],
    )
    assert score == pytest.approx(0.005)
    with pytest.raises(ValueError, match="both risk masks"):
        integrated_brier_score([[0.5]], [[0]], at_risk=[[1]])


def test_integrated_brier_is_macro_averaged_across_repositories() -> None:
    groups = {
        "small": BrierGroup(((0.0,),), ((0,),)),
        "large": BrierGroup(tuple((1.0,) for _ in range(9)), tuple((0,) for _ in range(9))),
    }
    assert macro_integrated_brier_score(groups) == pytest.approx(0.5)


def test_average_precision_ties_are_order_invariant() -> None:
    first = average_precision([1, 0, 1], [0.5, 0.5, 0.1])
    second = average_precision([0, 1, 1], [0.5, 0.5, 0.1])
    assert first == second


def test_macro_f1_is_unweighted_across_labels() -> None:
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)


def test_graph_patch_metrics_report_dimensions_without_composite() -> None:
    predicted = [
        forecast_case((("ASSIGNED_TO", ("ev-1",), 0.8), ("ALTERNATIVE", ("ev-1",), 0.2))),
        forecast_case((("RELATES_TO", ("future",), 0.5), ("RELATES_TO", ("future",), 0.5))),
    ]
    reference = [[event(observed=3.0, valid=2.0)], []]
    result = graph_patch_metrics(predicted, reference)
    assert result.contract_validity == 0.5
    assert result.matched_event_precision == 0.5
    assert result.matched_event_recall == 1.0
    assert result.matched_event_f1 == pytest.approx(2 / 3)
    assert result.evidence_basis_accuracy == 1.0
    assert result.mean_absolute_observed_time_error == 1.0
    assert result.mean_absolute_valid_time_error == 1.0
    assert result.distinct_window_rate == 0.75
    assert not hasattr(result, "composite")


def test_patch_evidence_must_be_visible() -> None:
    result = graph_patch_metrics(
        [forecast_case((("ASSIGNED_TO", ("future",), 0.5), ("ASSIGNED_TO", ("future",), 0.5)))],
        [[event(evidence=("future",))]],
    )
    assert result.contract_validity == 0.0
    assert result.evidence_basis_accuracy == 0.0


def test_window_likelihood_and_memorization_orientation() -> None:
    assert normalized_window_nll([-2.0, -4.0], [2, 4]) == 1.0
    training = [[event()]]
    generated = [[event()], [event("RELATES_TO")]]
    assert memorization_rate(generated, training) == 0.5
    with pytest.raises(ValueError):
        normalized_window_nll([-1.0], [0])


def test_unmatched_patch_time_error_is_explicitly_undefined() -> None:
    result = graph_patch_metrics(
        [forecast_case((("A", ("ev-1",), 0.5), ("A", ("ev-1",), 0.5)))],
        [[event("B")]],
    )
    assert math.isinf(result.mean_absolute_observed_time_error)
