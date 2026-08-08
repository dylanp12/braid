import pytest

from braid.bench.baselines import (
    BaselineFamily,
    BaselineMetadata,
    BaselineRegistry,
    FrequencyBaseline,
    LexicalBaseline,
    NullBaseline,
    QualificationState,
    RandomBaseline,
    RecencyBaseline,
    default_registry,
)
from braid.bench.data import SplitName
from braid.bench.specs import BenchmarkTrack, MetricDirection
from braid.bench.statistics import (
    HierarchicalObservation,
    holm_bonferroni,
    paired_hierarchical_bootstrap,
)


def test_runnable_reference_baselines() -> None:
    random = RandomBaseline(seed=7)
    assert random.score("query", "candidate") == random.score("query", "candidate")
    assert 0 <= random.score("query", "candidate") <= 1

    null = NullBaseline()
    null.fit([1, 0, 1, 1])
    assert null.score() == 0.75

    frequency = FrequencyBaseline()
    frequency.fit(["a", "a", "b"])
    assert frequency.score("a") == pytest.approx(2 / 3)
    assert frequency.score("missing") == 0.0

    recency = RecencyBaseline(half_life_seconds=86_400)
    assert recency.score("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z") == pytest.approx(0.5)

    lexical = LexicalBaseline()
    assert lexical.score("fix parser crash", "parser crash fixed") == 0.5


def test_fitted_baselines_reject_non_train_fit() -> None:
    with pytest.raises(ValueError, match="training split only"):
        NullBaseline().fit([0, 1], split=SplitName.DEVELOPMENT)


def test_registry_has_all_baseline_families_and_does_not_prequalify() -> None:
    registry = default_registry()
    assert {entry.family for entry in registry.entries} == set(BaselineFamily)
    assert registry.get("ULTRA").state is QualificationState.DECLARED
    assert registry.get("random").state is QualificationState.IMPLEMENTED
    assert not registry.get("null").qualified


def test_registry_selects_strongest_qualified_with_metric_orientation() -> None:
    common = dict(
        family=BaselineFamily.CLASSICAL,
        tracks=frozenset({BenchmarkTrack.TEMPORAL_TRANSFER}),
        implementation="adapter",
        pinned_version="1",
        state=QualificationState.QUALIFIED,
        uses_identical_inputs=True,
        within_inference_envelope=True,
        official_result_reproduced=True,
        reason="qualified",
    )
    registry = BaselineRegistry(
        (
            BaselineMetadata(name="a", **common),
            BaselineMetadata(name="b", **common),
        )
    )
    assert (
        registry.strongest_qualified(
            BenchmarkTrack.TEMPORAL_TRANSFER,
            {"a": 0.4, "b": 0.6},
            MetricDirection.HIGHER_IS_BETTER,
        )[0].name
        == "b"
    )
    assert (
        registry.strongest_qualified(
            BenchmarkTrack.TEMPORAL_TRANSFER,
            {"a": 0.4, "b": 0.6},
            MetricDirection.LOWER_IS_BETTER,
        )[0].name
        == "a"
    )


def test_hierarchical_bootstrap_is_deterministic_and_positive() -> None:
    observations = tuple(
        HierarchicalObservation((f"org-{index % 4}", f"repo-{index}"), 0.03 + index / 10_000)
        for index in range(40)
    )
    first = paired_hierarchical_bootstrap(observations, samples=500, seed=11)
    second = paired_hierarchical_bootstrap(observations, samples=500, seed=11)
    assert first == second
    assert first.lower_bound > 0
    assert first.one_sided_p_value == pytest.approx(1 / 501)


def test_hierarchical_estimate_weights_top_level_clusters_equally() -> None:
    observations = (
        HierarchicalObservation(("small", "repo-a"), 1.0),
        *(HierarchicalObservation(("large", f"repo-{index}"), 0.0) for index in range(9)),
    )
    result = paired_hierarchical_bootstrap(observations, samples=100, seed=3)
    assert result.estimate == 0.5


def test_hierarchical_draws_preserve_the_macro_cluster_estimand() -> None:
    observations = [
        HierarchicalObservation(("large", f"leaf-{index}"), 1.0) for index in range(100)
    ]
    observations.append(HierarchicalObservation(("small", "leaf"), -1.0))
    result = paired_hierarchical_bootstrap(observations, samples=2_000, seed=9)
    assert result.estimate == pytest.approx(0.0)
    assert sum(result.draws) / len(result.draws) == pytest.approx(0.0, abs=0.08)


def test_holm_correction_is_step_down_and_order_stable() -> None:
    result = holm_bonferroni({"weak": 0.06, "strong": 0.001, "middle": 0.02})
    by_name = {item.hypothesis: item for item in result}
    assert by_name["strong"].rejected
    assert by_name["middle"].rejected
    assert not by_name["weak"].rejected
    assert by_name["strong"].adjusted_p_value == pytest.approx(0.003)
