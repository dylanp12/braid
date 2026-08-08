"""Cross-split leakage audits and train-only fitting enforcement."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from .data import FitMarker, SplitName


class LeakageKind(StrEnum):
    DUPLICATE = "duplicate"
    FORK_OR_MIRROR = "fork_or_mirror"
    TEXT = "text"
    SUBGRAPH = "subgraph"
    INVERSE_RELATION = "inverse_relation"
    NON_TRAIN_FITTING = "non_train_fitting"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    record_id: str
    split: SplitName
    content_hash: str
    fork_or_copy_cluster: str
    text: str
    subgraph_signature: str
    relation: str | None = None
    source: str | None = None
    target: str | None = None


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    kind: LeakageKind
    record_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class LeakageReport:
    findings: tuple[LeakageFinding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def count(self, kind: LeakageKind) -> int:
        return sum(finding.kind is kind for finding in self.findings)


def normalized_text_fingerprint(text: str) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _cross_split_groups(
    records: tuple[AuditRecord, ...],
    key_name: str,
) -> list[tuple[str, tuple[AuditRecord, ...]]]:
    groups: dict[str, list[AuditRecord]] = {}
    for record in records:
        value = getattr(record, key_name)
        if value:
            groups.setdefault(value, []).append(record)
    return [
        (value, tuple(group))
        for value, group in sorted(groups.items())
        if len({record.split for record in group}) > 1
    ]


def audit_leakage(
    records: Iterable[AuditRecord],
    *,
    inverse_relations: Mapping[str, str] | None = None,
    fit_markers: Iterable[FitMarker] = (),
) -> LeakageReport:
    """Run deterministic duplicate, family, text, structure, and inverse audits."""

    materialized = tuple(records)
    findings: list[LeakageFinding] = []
    ids = [record.record_id for record in materialized]
    if len(ids) != len(set(ids)):
        duplicates = tuple(sorted(record_id for record_id in set(ids) if ids.count(record_id) > 1))
        findings.append(LeakageFinding(LeakageKind.DUPLICATE, duplicates, "duplicate record IDs"))

    for value, group in _cross_split_groups(materialized, "content_hash"):
        findings.append(
            LeakageFinding(
                LeakageKind.DUPLICATE,
                tuple(sorted(record.record_id for record in group)),
                f"content hash {value} crosses splits",
            )
        )
    for value, group in _cross_split_groups(materialized, "fork_or_copy_cluster"):
        findings.append(
            LeakageFinding(
                LeakageKind.FORK_OR_MIRROR,
                tuple(sorted(record.record_id for record in group)),
                f"fork/mirror/code-copy cluster {value} crosses splits",
            )
        )
    text_groups: dict[str, list[AuditRecord]] = {}
    for record in materialized:
        if record.text.strip():
            text_groups.setdefault(normalized_text_fingerprint(record.text), []).append(record)
    for fingerprint, group in sorted(text_groups.items()):
        if len({record.split for record in group}) > 1:
            findings.append(
                LeakageFinding(
                    LeakageKind.TEXT,
                    tuple(sorted(record.record_id for record in group)),
                    f"normalized text fingerprint {fingerprint} crosses splits",
                )
            )
    for value, group in _cross_split_groups(materialized, "subgraph_signature"):
        findings.append(
            LeakageFinding(
                LeakageKind.SUBGRAPH,
                tuple(sorted(record.record_id for record in group)),
                f"subgraph signature {value} crosses splits",
            )
        )

    inverses = dict(inverse_relations or {})
    edges: dict[tuple[str, str, str], list[AuditRecord]] = {}
    for record in materialized:
        if record.relation and record.source and record.target:
            edges.setdefault((record.relation, record.source, record.target), []).append(record)
    seen_pairs: set[tuple[str, str]] = set()
    for record in materialized:
        if not record.relation or not record.source or not record.target:
            continue
        inverse = inverses.get(record.relation)
        if inverse is None:
            continue
        for counterpart in edges.get((inverse, record.target, record.source), []):
            if counterpart.split is record.split:
                continue
            pair = tuple(sorted((record.record_id, counterpart.record_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            findings.append(
                LeakageFinding(
                    LeakageKind.INVERSE_RELATION,
                    pair,
                    f"{record.relation} is exposed through inverse relation {inverse}",
                )
            )

    for marker in sorted(fit_markers, key=lambda item: item.artifact_name):
        if not marker.train_only:
            findings.append(
                LeakageFinding(
                    LeakageKind.NON_TRAIN_FITTING,
                    (),
                    f"{marker.artifact_name} fitted on "
                    + ", ".join(split.value for split in marker.fitted_splits),
                )
            )
    findings.sort(key=lambda item: (item.kind.value, item.record_ids, item.detail))
    return LeakageReport(tuple(findings))
