"""Typed, cutoff-safe candidate retrieval and measurable coverage."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class NodeCandidate:
    handle: int
    type_handle: int
    observed_at: datetime
    neighbors: frozenset[int] = frozenset()
    semantic_score: float = 0.0

    def __post_init__(self) -> None:
        if self.handle < 0 or self.type_handle < 0 or any(handle < 0 for handle in self.neighbors):
            raise ValueError("candidate and neighbor handles must be non-negative")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("candidate observation time must include a UTC offset")
        if not math.isfinite(self.semantic_score):
            raise ValueError("candidate semantic score must be finite")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: tuple[NodeCandidate, ...]
    coverage: float
    type_coverage: float
    true_target_recall: float | None
    missing_query_handles: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        handles = [candidate.handle for candidate in self.candidates]
        if len(handles) != len(set(handles)):
            raise ValueError("retrieval candidates must have unique handles")
        for name, value in (("coverage", self.coverage), ("type_coverage", self.type_coverage)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.true_target_recall is not None and (
            not math.isfinite(self.true_target_recall) or not 0.0 <= self.true_target_recall <= 1.0
        ):
            raise ValueError("true_target_recall must be finite and in [0, 1]")
        if any(handle < 0 for handle in self.missing_query_handles):
            raise ValueError("missing query handles must be non-negative")

    @property
    def claim_grade(self) -> bool:
        return (
            self.safe_to_generate(0.99)
            and self.true_target_recall is not None
            and self.true_target_recall >= 0.99
        )

    def safe_to_generate(
        self,
        minimum_coverage: float = 0.99,
        *,
        require_claim_grade: bool = False,
    ) -> bool:
        if not math.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError("minimum coverage must be finite and in [0, 1]")
        coverage_ok = (
            self.coverage >= minimum_coverage
            and self.type_coverage >= minimum_coverage
            and not self.missing_query_handles
        )
        recall_ok = not require_claim_grade or (
            self.true_target_recall is not None and self.true_target_recall >= 0.99
        )
        return coverage_ok and recall_ok


class TypedRetriever:
    """Deterministic hybrid retriever over causally visible nodes."""

    def __init__(self, *, max_candidates: int = 256) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = max_candidates

    def retrieve(
        self,
        nodes: Iterable[NodeCandidate],
        *,
        cutoff: datetime,
        required_types: Iterable[int] = (),
        query_handles: Iterable[int] = (),
        semantic_scores: Mapping[int, float] | None = None,
        true_target_handles: Iterable[int] | None = None,
    ) -> RetrievalResult:
        required_types = tuple(dict.fromkeys(required_types))
        query_handles = tuple(dict.fromkeys(query_handles))
        if any(handle < 0 for handle in (*required_types, *query_handles)):
            raise ValueError("required type and query handles must be non-negative")
        semantic_scores = semantic_scores or {}
        visible = [node for node in nodes if node.observed_at <= cutoff]
        by_handle = {node.handle: node for node in visible}
        allowed_types = set(required_types)
        query_set = set(query_handles)
        eligible = [
            node
            for node in visible
            if not allowed_types or node.type_handle in allowed_types or node.handle in query_set
        ]
        neighbor_set = (
            set().union(
                *(by_handle[handle].neighbors for handle in query_handles if handle in by_handle)
            )
            if query_handles
            else set()
        )

        def score(node: NodeCandidate) -> tuple[float, float, float, int]:
            explicit = 1.0 if node.handle in query_set else 0.0
            neighbor = 1.0 if node.handle in neighbor_set else 0.0
            semantic = float(semantic_scores.get(node.handle, node.semantic_score))
            # Stable final key makes retrieval invariant to input container order.
            return (explicit, neighbor, semantic, -node.handle)

        ranked = sorted(eligible, key=score, reverse=True)[: self.max_candidates]
        ranked_handles = {node.handle for node in ranked}
        missing_queries = tuple(handle for handle in query_handles if handle not in ranked_handles)
        query_coverage = 1.0 - len(missing_queries) / len(query_handles) if query_handles else 1.0
        present_types = {node.type_handle for node in ranked}
        type_coverage = (
            sum(handle in present_types for handle in required_types) / len(required_types)
            if required_types
            else (1.0 if ranked else 0.0)
        )
        target_recall: float | None = None
        if true_target_handles is not None:
            targets = set(true_target_handles)
            if any(handle < 0 for handle in targets):
                raise ValueError("true target handles must be non-negative")
            target_recall = len(targets & ranked_handles) / len(targets) if targets else 1.0
        return RetrievalResult(
            candidates=tuple(ranked),
            coverage=query_coverage,
            type_coverage=type_coverage,
            true_target_recall=target_recall,
            missing_query_handles=missing_queries,
        )
