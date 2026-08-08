"""Normative public benchmark track and population specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (booleans are not counts)")
    return value


def _require_exact_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float")
    return value


def _require_exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


class BenchmarkTrack(StrEnum):
    """Independent benchmark tracks.  Tracks must never be averaged together."""

    UNSEEN_ENTITIES = "E"
    UNSEEN_RELATIONS = "R"
    UNSEEN_SCHEMAS = "S"
    TEMPORAL_TRANSFER = "T"
    GRAPH_PATCH = "P"


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


class ImprovementRule(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    direction: MetricDirection
    improvement_rule: ImprovementRule
    minimum_improvement: float
    description: str

    def __post_init__(self) -> None:
        if type(self.direction) is not MetricDirection:
            raise TypeError("metric direction must be a MetricDirection")
        if type(self.improvement_rule) is not ImprovementRule:
            raise TypeError("metric improvement_rule must be an ImprovementRule")
        _require_exact_float(self.minimum_improvement, "metric minimum_improvement")


@dataclass(frozen=True, slots=True)
class TrackSpec:
    track: BenchmarkTrack
    description: str
    metrics: tuple[MetricSpec, ...]
    minimum_queries_or_windows: int = 5_000
    minimum_repository_clusters: int = 0
    minimum_relations: int = 0
    minimum_schema_families: int = 0
    maximum_organization_share: float = 0.15

    def __post_init__(self) -> None:
        if type(self.track) is not BenchmarkTrack:
            raise TypeError("track specification track must be a BenchmarkTrack")
        for name, value in (
            ("minimum_queries_or_windows", self.minimum_queries_or_windows),
            ("minimum_repository_clusters", self.minimum_repository_clusters),
            ("minimum_relations", self.minimum_relations),
            ("minimum_schema_families", self.minimum_schema_families),
        ):
            _require_exact_int(value, name)
        _require_exact_float(self.maximum_organization_share, "maximum_organization_share")


TRACK_SPECS: Mapping[BenchmarkTrack, TrackSpec] = MappingProxyType(
    {
        BenchmarkTrack.UNSEEN_ENTITIES: TrackSpec(
            BenchmarkTrack.UNSEEN_ENTITIES,
            "Filtered link prediction over entirely held-out entity/repository clusters.",
            (
                MetricSpec(
                    "filtered_mrr",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Macro-repository filtered mean reciprocal rank.",
                ),
            ),
            minimum_repository_clusters=30,
        ),
        BenchmarkTrack.UNSEEN_RELATIONS: TrackSpec(
            BenchmarkTrack.UNSEEN_RELATIONS,
            "Relation transfer with no global relation IDs.",
            (
                MetricSpec(
                    "semantic_zero_shot_mrr",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Macro-relation semantic zero-shot filtered MRR.",
                ),
                MetricSpec(
                    "symbol_renamed_eight_shot_mrr",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Macro-relation MRR after symbol renaming with eight support events.",
                ),
            ),
            minimum_relations=12,
        ),
        BenchmarkTrack.UNSEEN_SCHEMAS: TrackSpec(
            BenchmarkTrack.UNSEEN_SCHEMAS,
            "Transfer to genuinely different source schemas.",
            (
                MetricSpec(
                    "schema_mapping_macro_f1",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Macro-F1 of schema mappings.",
                ),
                MetricSpec(
                    "schema_link_mrr",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Filtered link MRR on held-out schema families.",
                ),
            ),
            minimum_schema_families=6,
        ),
        BenchmarkTrack.TEMPORAL_TRANSFER: TrackSpec(
            BenchmarkTrack.TEMPORAL_TRANSFER,
            "Causal future-event prediction on held-out repository clusters.",
            (
                MetricSpec(
                    "macro_auprc",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Macro-repository area under the precision-recall curve.",
                ),
                MetricSpec(
                    "integrated_brier",
                    MetricDirection.LOWER_IS_BETTER,
                    ImprovementRule.RELATIVE,
                    0.05,
                    "Integrated Brier score over forecast horizons.",
                ),
            ),
            minimum_repository_clusters=30,
        ),
        BenchmarkTrack.GRAPH_PATCH: TrackSpec(
            BenchmarkTrack.GRAPH_PATCH,
            "Distributional generation of evidence-grounded future graph patches.",
            (
                MetricSpec(
                    "contract_validity",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Fraction of generated windows satisfying the v2 contract.",
                ),
                MetricSpec(
                    "normalized_window_nll",
                    MetricDirection.LOWER_IS_BETTER,
                    ImprovementRule.RELATIVE,
                    0.05,
                    "Length-normalized negative log likelihood of reference windows.",
                ),
                MetricSpec(
                    "matched_event_f1",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "F1 after typed operation/relation/role matching.",
                ),
                MetricSpec(
                    "time_calibration_error",
                    MetricDirection.LOWER_IS_BETTER,
                    ImprovementRule.RELATIVE,
                    0.05,
                    "Calibration error of forecasted event times.",
                ),
                MetricSpec(
                    "evidence_basis_accuracy",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Matched-event accuracy of visible evidence bases.",
                ),
                MetricSpec(
                    "distinct_window_rate",
                    MetricDirection.HIGHER_IS_BETTER,
                    ImprovementRule.ABSOLUTE,
                    0.02,
                    "Fraction of sampled windows that are distinct.",
                ),
                MetricSpec(
                    "memorization_rate",
                    MetricDirection.LOWER_IS_BETTER,
                    ImprovementRule.RELATIVE,
                    0.05,
                    "Rate of prohibited train-window reproduction.",
                ),
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Population:
    """Population counts for one benchmark track.

    ``organization_counts`` counts confirmatory examples, not raw source rows.
    This prevents one prolific organization from dominating macro results.
    """

    queries_or_nonempty_windows: int
    repository_clusters: int = 0
    held_out_relations: int = 0
    schema_families: int = 0
    organization_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        numeric = tuple(
            _require_exact_int(value, name)
            for name, value in (
                ("queries_or_nonempty_windows", self.queries_or_nonempty_windows),
                ("repository_clusters", self.repository_clusters),
                ("held_out_relations", self.held_out_relations),
                ("schema_families", self.schema_families),
            )
        )
        if any(value < 0 for value in numeric):
            raise ValueError("population counts cannot be negative")
        if type(self.organization_counts) is not tuple:
            raise TypeError("organization_counts must be a tuple")
        for item in self.organization_counts:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("organization_counts entries must be (name, count) tuples")
            if type(item[0]) is not str:
                raise TypeError("organization names must be strings")
            _require_exact_int(item[1], "organization count")
        names = [name for name, _ in self.organization_counts]
        if len(names) != len(set(names)):
            raise ValueError("organization names must be unique")
        if any(not name or count < 0 for name, count in self.organization_counts):
            raise ValueError("organization counts need non-empty names and non-negative counts")


@dataclass(frozen=True, slots=True)
class PopulationQualification:
    track: BenchmarkTrack
    passed: bool
    failures: tuple[str, ...]
    largest_organization_share: float

    def __post_init__(self) -> None:
        if type(self.track) is not BenchmarkTrack:
            raise TypeError("population qualification track must be a BenchmarkTrack")
        _require_exact_bool(self.passed, "population qualification passed")
        _require_exact_float(
            self.largest_organization_share,
            "population largest_organization_share",
        )
        if not 0.0 <= self.largest_organization_share <= 1.0:
            raise ValueError("largest organization share must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackPopulation:
    """Raw signed population counts for one independently gated track.

    A caller cannot supply a precomputed ``passed`` flag.  The claim controller
    always derives :class:`PopulationQualification` with
    :func:`validate_population` from this record.
    """

    track: BenchmarkTrack
    population: Population

    def __post_init__(self) -> None:
        if type(self.track) is not BenchmarkTrack:
            raise TypeError("track population track must be a BenchmarkTrack")
        if type(self.population) is not Population:
            raise TypeError("track population must contain raw Population counts")


def validate_population(
    track: BenchmarkTrack,
    population: Population,
) -> PopulationQualification:
    """Apply the non-negotiable population and concentration gates."""

    if type(track) is not BenchmarkTrack:
        raise TypeError("population track must be a BenchmarkTrack")
    if type(population) is not Population:
        raise TypeError("population must contain exact raw Population counts")
    spec = TRACK_SPECS[track]
    failures: list[str] = []
    if population.queries_or_nonempty_windows < spec.minimum_queries_or_windows:
        failures.append(
            f"requires at least {spec.minimum_queries_or_windows} confirmatory examples; "
            f"found {population.queries_or_nonempty_windows}"
        )
    if population.repository_clusters < spec.minimum_repository_clusters:
        failures.append(
            f"requires at least {spec.minimum_repository_clusters} repository clusters; "
            f"found {population.repository_clusters}"
        )
    if population.held_out_relations < spec.minimum_relations:
        failures.append(
            f"requires at least {spec.minimum_relations} held-out relations; "
            f"found {population.held_out_relations}"
        )
    if population.schema_families < spec.minimum_schema_families:
        failures.append(
            f"requires at least {spec.minimum_schema_families} schema families; "
            f"found {population.schema_families}"
        )

    organization_total = sum(count for _, count in population.organization_counts)
    largest_share = (
        max((count for _, count in population.organization_counts), default=0) / organization_total
        if organization_total
        else 0.0
    )
    if organization_total != population.queries_or_nonempty_windows:
        failures.append(
            "organization counts must cover every confirmatory example "
            f"({organization_total} != {population.queries_or_nonempty_windows})"
        )
    if largest_share > spec.maximum_organization_share + 1e-12:
        failures.append(
            f"largest organization share {largest_share:.3f} exceeds "
            f"{spec.maximum_organization_share:.3f}"
        )
    return PopulationQualification(track, not failures, tuple(failures), largest_share)


def required_metric_specs() -> tuple[tuple[BenchmarkTrack, MetricSpec], ...]:
    """Return every independently gated endpoint in stable order."""

    return tuple(
        (track, metric) for track in BenchmarkTrack for metric in TRACK_SPECS[track].metrics
    )
