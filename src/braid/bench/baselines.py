"""Reference baselines and the qualification registry for model comparisons."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .data import SplitName
from .specs import BenchmarkTrack, MetricDirection


class BaselineFamily(StrEnum):
    SHORTCUT = "shortcut"
    CLASSICAL = "classical"
    EXTERNAL = "external"
    HISTORICAL = "historical"


class QualificationState(StrEnum):
    DECLARED = "declared"
    IMPLEMENTED = "implemented"
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class BaselineMetadata:
    name: str
    family: BaselineFamily
    tracks: frozenset[BenchmarkTrack]
    implementation: str
    citation_url: str | None = None
    pinned_version: str | None = None
    state: QualificationState = QualificationState.DECLARED
    uses_identical_inputs: bool = False
    within_inference_envelope: bool = False
    official_result_reproduced: bool = False
    reason: str = "not yet qualified"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise TypeError("baseline name must be a non-empty string")
        if type(self.family) is not BaselineFamily:
            raise TypeError("baseline family must be a BaselineFamily")
        if type(self.tracks) is not frozenset or any(
            type(track) is not BenchmarkTrack for track in self.tracks
        ):
            raise TypeError("baseline tracks must be a frozenset of BenchmarkTrack values")
        if not self.tracks:
            raise ValueError("baseline must declare at least one benchmark track")
        if type(self.implementation) is not str or not self.implementation.strip():
            raise TypeError("baseline implementation must be a non-empty string")
        for name, value in (
            ("citation_url", self.citation_url),
            ("pinned_version", self.pinned_version),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(f"baseline {name} must be a string or None")
        if type(self.state) is not QualificationState:
            raise TypeError("baseline state must be a QualificationState")
        for name, value in (
            ("uses_identical_inputs", self.uses_identical_inputs),
            ("within_inference_envelope", self.within_inference_envelope),
            ("official_result_reproduced", self.official_result_reproduced),
        ):
            if type(value) is not bool:
                raise TypeError(f"baseline {name} must be a boolean")
        if type(self.reason) is not str or not self.reason.strip():
            raise TypeError("baseline reason must be a non-empty string")

    @property
    def qualified(self) -> bool:
        return (
            self.state is QualificationState.QUALIFIED
            and self.uses_identical_inputs
            and self.within_inference_envelope
            and self.official_result_reproduced
            and bool(self.pinned_version)
        )


@dataclass(frozen=True, slots=True)
class BaselineRegistry:
    entries: tuple[BaselineMetadata, ...]

    def __post_init__(self) -> None:
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("baseline names must be unique")

    def get(self, name: str) -> BaselineMetadata:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def qualified_for(self, track: BenchmarkTrack) -> tuple[BaselineMetadata, ...]:
        return tuple(entry for entry in self.entries if track in entry.tracks and entry.qualified)

    def strongest_qualified(
        self,
        track: BenchmarkTrack,
        results: Mapping[str, float],
        direction: MetricDirection,
    ) -> tuple[BaselineMetadata, float]:
        candidates = [
            (entry, float(results[entry.name]))
            for entry in self.qualified_for(track)
            if entry.name in results
        ]
        if not candidates:
            raise ValueError(f"no qualified baseline result for track {track.value}")
        selector = max if direction is MetricDirection.HIGHER_IS_BETTER else min
        return selector(candidates, key=lambda item: item[1])


ALL_TRACKS = frozenset(BenchmarkTrack)
LINK_TRACKS = frozenset(
    {
        BenchmarkTrack.UNSEEN_ENTITIES,
        BenchmarkTrack.UNSEEN_RELATIONS,
        BenchmarkTrack.UNSEEN_SCHEMAS,
    }
)


def _declared(
    name: str,
    family: BaselineFamily,
    tracks: frozenset[BenchmarkTrack],
    implementation: str,
    citation_url: str | None = None,
) -> BaselineMetadata:
    return BaselineMetadata(name, family, tracks, implementation, citation_url)


def default_registry() -> BaselineRegistry:
    """Return the complete comparison registry without pretending entries are qualified."""

    local = (
        "braid.bench.baselines",
        QualificationState.IMPLEMENTED,
    )
    entries = [
        BaselineMetadata(
            "random",
            BaselineFamily.SHORTCUT,
            ALL_TRACKS,
            local[0],
            pinned_version="reference-v1",
            state=local[1],
            reason="implementation available; result qualification is dataset-specific",
        ),
        BaselineMetadata(
            "null",
            BaselineFamily.SHORTCUT,
            ALL_TRACKS,
            local[0],
            pinned_version="reference-v1",
            state=local[1],
            reason="implementation available; result qualification is dataset-specific",
        ),
        BaselineMetadata(
            "frequency",
            BaselineFamily.SHORTCUT,
            ALL_TRACKS,
            local[0],
            pinned_version="reference-v1",
            state=local[1],
            reason="implementation available; result qualification is dataset-specific",
        ),
        BaselineMetadata(
            "recency",
            BaselineFamily.SHORTCUT,
            frozenset({BenchmarkTrack.TEMPORAL_TRANSFER, BenchmarkTrack.GRAPH_PATCH}),
            local[0],
            pinned_version="reference-v1",
            state=local[1],
            reason="implementation available; result qualification is dataset-specific",
        ),
        BaselineMetadata(
            "lexical",
            BaselineFamily.SHORTCUT,
            ALL_TRACKS,
            local[0],
            pinned_version="reference-v1",
            state=local[1],
            reason="implementation available; result qualification is dataset-specific",
        ),
        _declared("embedding", BaselineFamily.SHORTCUT, ALL_TRACKS, "external adapter"),
        _declared("degree_ppr", BaselineFamily.SHORTCUT, LINK_TRACKS, "external adapter"),
        _declared("copy_repeat", BaselineFamily.SHORTCUT, ALL_TRACKS, "external adapter"),
        _declared("linear_tree", BaselineFamily.CLASSICAL, ALL_TRACKS, "external adapter"),
        _declared(
            "survival",
            BaselineFamily.CLASSICAL,
            frozenset({BenchmarkTrack.TEMPORAL_TRANSFER}),
            "external adapter",
        ),
        _declared("rule", BaselineFamily.CLASSICAL, ALL_TRACKS, "external adapter"),
        _declared("relational_gnn", BaselineFamily.CLASSICAL, LINK_TRACKS, "external adapter"),
        _declared(
            "temporal_gnn",
            BaselineFamily.CLASSICAL,
            frozenset({BenchmarkTrack.TEMPORAL_TRANSFER, BenchmarkTrack.GRAPH_PATCH}),
            "external adapter",
        ),
        _declared(
            "InGram",
            BaselineFamily.EXTERNAL,
            frozenset({BenchmarkTrack.UNSEEN_RELATIONS, BenchmarkTrack.UNSEEN_SCHEMAS}),
            "official adapter",
            "https://github.com/bdi-lab/InGram",
        ),
        _declared(
            "ULTRA",
            BaselineFamily.EXTERNAL,
            LINK_TRACKS,
            "official adapter",
            "https://github.com/DeepGraphLearning/ULTRA",
        ),
        _declared(
            "Gamma",
            BaselineFamily.EXTERNAL,
            LINK_TRACKS,
            "official adapter",
            "https://arxiv.org/abs/2512.22931",
        ),
        _declared(
            "GraphBFF",
            BaselineFamily.EXTERNAL,
            LINK_TRACKS,
            "official adapter",
            "https://arxiv.org/abs/2602.04768",
        ),
        _declared("SEMMA_Flock", BaselineFamily.EXTERNAL, LINK_TRACKS, "official adapter"),
        _declared(
            "POSTRA",
            BaselineFamily.EXTERNAL,
            frozenset({BenchmarkTrack.TEMPORAL_TRANSFER}),
            "official adapter",
            "https://arxiv.org/abs/2506.06367",
        ),
        _declared(
            "GET",
            BaselineFamily.EXTERNAL,
            frozenset({BenchmarkTrack.GRAPH_PATCH}),
            "official adapter",
            "https://openreview.net/forum?id=786oOfRVXO",
        ),
        _declared("TGB_RelBench_leader", BaselineFamily.EXTERNAL, ALL_TRACKS, "official adapter"),
        _declared("frozen_retrieval", BaselineFamily.EXTERNAL, ALL_TRACKS, "external adapter"),
        _declared("cross_encoder", BaselineFamily.EXTERNAL, ALL_TRACKS, "external adapter"),
        _declared("open_weight_llm", BaselineFamily.EXTERNAL, ALL_TRACKS, "external adapter"),
        _declared("Braid_1", BaselineFamily.HISTORICAL, ALL_TRACKS, "legacy adapter"),
    ]
    return BaselineRegistry(tuple(entries))


class RandomBaseline:
    """Stable pseudo-random ranking baseline independent of process hash seeds."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def score(self, query_id: str, candidate_id: str) -> float:
        payload = f"{self.seed}\0{query_id}\0{candidate_id}".encode()
        integer = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        return integer / (2**256 - 1)


class _TrainFitted:
    fitted_split: SplitName | None = None

    def _mark_fit(self, split: SplitName) -> None:
        if split is not SplitName.TRAIN:
            raise ValueError("baseline fitting is permitted on the training split only")
        self.fitted_split = split

    def _require_fit(self) -> None:
        if self.fitted_split is not SplitName.TRAIN:
            raise RuntimeError("baseline has not been fitted on the training split")


class NullBaseline(_TrainFitted):
    """Constant empirical-prevalence probability baseline."""

    def __init__(self) -> None:
        self.prevalence = 0.5
        self.fitted_split = None

    def fit(self, labels: Iterable[int | bool], *, split: SplitName = SplitName.TRAIN) -> None:
        values = tuple(int(label) for label in labels)
        if not values or any(label not in (0, 1) for label in values):
            raise ValueError("null baseline needs non-empty binary labels")
        self._mark_fit(split)
        self.prevalence = sum(values) / len(values)

    def score(self, candidate: object = None) -> float:
        del candidate
        self._require_fit()
        return self.prevalence


class FrequencyBaseline(_TrainFitted):
    """Normalized empirical candidate frequency."""

    def __init__(self) -> None:
        self.counts: Counter[Hashable] = Counter()
        self.total = 0
        self.fitted_split = None

    def fit(self, values: Iterable[Hashable], *, split: SplitName = SplitName.TRAIN) -> None:
        counts = Counter(values)
        if not counts:
            raise ValueError("frequency baseline needs training values")
        self._mark_fit(split)
        self.counts = counts
        self.total = counts.total()

    def score(self, candidate: Hashable) -> float:
        self._require_fit()
        return self.counts[candidate] / self.total


class RecencyBaseline:
    """Exponential decay from the last causally visible timestamp."""

    def __init__(self, half_life_seconds: float = 86_400.0) -> None:
        if not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
            raise ValueError("half life must be positive and finite")
        self.half_life_seconds = half_life_seconds

    @staticmethod
    def _time(value: datetime | str) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def score(self, observed_at: datetime | str, cutoff: datetime | str) -> float:
        age = (self._time(cutoff) - self._time(observed_at)).total_seconds()
        if age < 0:
            raise ValueError("recency baseline cannot inspect a future timestamp")
        return math.exp(-math.log(2) * age / self.half_life_seconds)


class LexicalBaseline:
    """Case-folded token Jaccard similarity."""

    _token = re.compile(r"[a-z0-9]+")

    @classmethod
    def tokens(cls, value: str) -> set[str]:
        return set(cls._token.findall(value.casefold()))

    def score(self, query: str, candidate: str) -> float:
        query_tokens = self.tokens(query)
        candidate_tokens = self.tokens(candidate)
        union = query_tokens | candidate_tokens
        return len(query_tokens & candidate_tokens) / len(union) if union else 0.0
