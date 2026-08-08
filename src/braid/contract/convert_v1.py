"""Deterministic, lineage-preserving conversion from Braid contract v1 records."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from .serde import canonical_hash
from .types import (
    DerivationRecord,
    EvidenceRecord,
    GraphBundleV2,
    GraphEventV2,
    NodeRecord,
    NodeTypeDecl,
    Operation,
    PayloadSnapshot,
    ProvenanceRecord,
    RelationDecl,
    RoleBinding,
    RoleDecl,
    SchemaDecl,
    SourceLineage,
)
from .validate import validate_bundle

CONVERTER_ID = "braid.v1-to-v2/1"


class V1ConversionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _PreparedEvent:
    source_index: int
    source_id: str
    source_hash: str
    old_event_id: str
    observed_at: datetime
    valid_from: datetime
    valid_to: datetime | None
    operation: Operation
    relation: str
    arguments: tuple[RoleBinding, ...]
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    derivation_refs: tuple[str, ...]
    derivation_method: str
    observed_at_imputed: bool
    lifecycle_target_ref: str | None = None

    @property
    def assertion_key(self) -> tuple[Any, ...]:
        return (
            self.relation,
            tuple(sorted((binding.role, binding.node_id) for binding in self.arguments)),
        )


def _parse_clock(value: Any, path: str) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise V1ConversionError(f"{path}: invalid datetime {value!r}") from exc
    else:
        raise V1ConversionError(f"{path}: expected an ISO-8601 string or null")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise V1ConversionError(f"{path}: datetime must include a UTC offset")
    return parsed


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "event").lower()).strip("-")
    return slug or "event"


def _event_times(
    raw: Mapping[str, Any], index: int, default_observed_at: datetime
) -> tuple[datetime, datetime, datetime | None, bool]:
    event_time = raw.get("event_time")
    if event_time is None:
        event_time = (None, None)
    if not isinstance(event_time, (list, tuple)) or len(event_time) != 2:
        raise V1ConversionError(f"events[{index}].event_time: expected [start, end]")
    valid_from = _parse_clock(event_time[0], f"events[{index}].event_time[0]")
    valid_to = _parse_clock(event_time[1], f"events[{index}].event_time[1]")
    observed_at = _parse_clock(raw.get("derived_at"), f"events[{index}].derived_at")
    observed_at_imputed = observed_at is None
    # v1 did not consistently distinguish valid and observation time.  A known
    # event time is the least-fabricated fallback; otherwise the caller's explicit
    # migration instant is used and recorded in provenance.
    observed_at = observed_at or valid_from or default_observed_at
    valid_from = valid_from or observed_at
    if valid_to is not None and valid_to < valid_from:
        raise V1ConversionError(f"events[{index}]: valid_to precedes valid_from")
    return observed_at, valid_from, valid_to, observed_at_imputed


def _valid_contract_id(value: str) -> bool:
    return bool(value) and value.strip() == value


def _lineage_source_id(kind: str, value: Any, index: int | None = None) -> str:
    raw = str(value or "anonymous")
    candidate = f"{raw}#{index}" if index is not None else raw
    if _valid_contract_id(candidate):
        return candidate
    material = {"kind": kind, "raw_id": raw, "index": index}
    return f"legacy-{kind}:{canonical_hash(material)}"


def _node_id_mapping(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map legacy padded IDs without trimming or risking normalization collisions."""

    raw_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise V1ConversionError(f"nodes[{index}]: expected an object")
        raw_id = str(raw.get("id") or "")
        if not raw_id:
            raise V1ConversionError(f"nodes[{index}]: id is required")
        raw_ids.add(raw_id)

    mapping: dict[str, str] = {}
    occupied = {raw_id for raw_id in raw_ids if _valid_contract_id(raw_id)}
    for raw_id in sorted(raw_ids):
        if _valid_contract_id(raw_id):
            mapping[raw_id] = raw_id
            continue
        nonce = 0
        while True:
            material = {"legacy_node_id": raw_id, "nonce": nonce}
            candidate = f"v2:legacy-node:{canonical_hash(material)}"
            if candidate not in occupied:
                mapping[raw_id] = candidate
                occupied.add(candidate)
                break
            nonce += 1
    return mapping


def _prepare_events(
    rows: Sequence[Mapping[str, Any]],
    default_observed_at: datetime,
    node_id_by_legacy_id: Mapping[str, str],
) -> list[_PreparedEvent]:
    prepared = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise V1ConversionError(f"events[{index}]: expected an object")
        observed_at, valid_from, valid_to, observed_at_imputed = _event_times(
            raw, index, default_observed_at
        )
        old_event_id = str(raw.get("event_id") or f"anonymous-{index}")
        source_id = _lineage_source_id("event", old_event_id, index)
        relation = raw.get("relation")
        relation = str(relation) if relation else f"legacy:{_slug(raw.get('event_type'))}"
        operation = Operation.ASSERT
        lifecycle_target_ref: str | None = None
        raw_attrs = raw.get("attrs") if isinstance(raw.get("attrs"), Mapping) else {}
        explicit_operation = str(raw.get("operation") or raw.get("op") or "").upper()
        if explicit_operation == Operation.SUPERSEDE.value:
            target_candidates = {
                str(value)
                for value in (
                    raw.get("target_event_id"),
                    raw.get("supersedes_event_id"),
                    raw_attrs.get("target_event_id"),
                    raw_attrs.get("supersedes_event_id"),
                )
                if value is not None and str(value)
            }
            if len(target_candidates) != 1:
                raise V1ConversionError(
                    f"events[{index}]: explicit SUPERSEDE requires exactly one event target"
                )
            lifecycle_target_ref = next(iter(target_candidates))
            operation = Operation.SUPERSEDE
        raw_participants = raw.get("participants") or ()
        arguments: list[RoleBinding] = []
        for participant_index, participant in enumerate(raw_participants):
            if not isinstance(participant, (list, tuple)) or len(participant) != 2:
                raise V1ConversionError(
                    f"events[{index}].participants[{participant_index}]: expected [role, node_id]"
                )
            legacy_node_id = str(participant[1])
            try:
                node_id = node_id_by_legacy_id[legacy_node_id]
            except KeyError as exc:
                raise V1ConversionError(
                    f"events[{index}].participants[{participant_index}]: "
                    f"unknown node {legacy_node_id!r}"
                ) from exc
            arguments.append(RoleBinding(role=str(participant[0]), node_id=node_id))
        if not arguments:
            raise V1ConversionError(
                f"events[{index}]: cannot convert an event without participants"
            )
        extraction = raw.get("extraction") if isinstance(raw.get("extraction"), Mapping) else {}
        method = str(
            extraction.get("model")
            or (
                f"legacy-pipeline-v{extraction.get('pipeline_version')}"
                if extraction.get("pipeline_version") is not None
                else "legacy-import"
            )
        )
        payload = {
            "attrs": raw.get("attrs") or {},
            "legacy_event_type": raw.get("event_type"),
            "legacy_extraction": extraction,
            "legacy_human_confirmed": bool(raw.get("human_confirmed", False)),
        }
        prepared.append(
            _PreparedEvent(
                source_index=index,
                source_id=source_id,
                source_hash=canonical_hash(raw),
                old_event_id=old_event_id,
                observed_at=observed_at,
                valid_from=valid_from,
                valid_to=valid_to,
                operation=operation,
                relation=relation,
                arguments=tuple(arguments),
                payload=payload,
                evidence_refs=tuple(str(item) for item in raw.get("evidence") or ()),
                derivation_refs=tuple(str(item) for item in raw.get("derivation") or ()),
                derivation_method=method,
                observed_at_imputed=observed_at_imputed,
                lifecycle_target_ref=lifecycle_target_ref,
            )
        )
    return prepared


@dataclass(slots=True)
class _RoleStats:
    allowed_types: set[str]
    min_count: int
    max_count: int


@dataclass(slots=True)
class _RelationStats:
    event_count: int
    roles: dict[str, _RoleStats]


def _advance_relation_stats(
    state: dict[str, _RelationStats],
    event: _PreparedEvent,
    node_type_by_id: Mapping[str, str],
) -> bool:
    """Fold one event into exact prefix statistics and report a schema change."""

    counts = Counter(binding.role for binding in event.arguments)
    types_by_role: dict[str, set[str]] = defaultdict(set)
    for binding in event.arguments:
        try:
            node_type = node_type_by_id[binding.node_id]
        except KeyError as exc:
            raise V1ConversionError(
                f"event {event.source_id!r} references unknown node {binding.node_id!r}"
            ) from exc
        types_by_role[binding.role].add(node_type)

    relation = state.get(event.relation)
    if relation is None:
        state[event.relation] = _RelationStats(
            event_count=1,
            roles={
                role: _RoleStats(set(types_by_role[role]), count, count)
                for role, count in counts.items()
            },
        )
        return True

    changed = False
    prior_event_count = relation.event_count
    for role_name in set(relation.roles) | set(counts):
        count = counts[role_name]
        role = relation.roles.get(role_name)
        if role is None:
            # A role absent from earlier examples contributes zero to the true
            # prefix-wide minimum, just as a full rescan would.
            relation.roles[role_name] = _RoleStats(
                set(types_by_role[role_name]),
                0 if prior_event_count else count,
                count,
            )
            changed = True
            continue
        minimum = min(role.min_count, count)
        maximum = max(role.max_count, count)
        allowed_types = role.allowed_types | types_by_role[role_name]
        if (
            minimum != role.min_count
            or maximum != role.max_count
            or allowed_types != role.allowed_types
        ):
            role.min_count = minimum
            role.max_count = maximum
            role.allowed_types = allowed_types
            changed = True
    relation.event_count += 1
    return changed


def _schema_from_stats(
    node_types: set[str],
    relations: Mapping[str, _RelationStats],
    *,
    version: str,
    observed_at: datetime,
) -> SchemaDecl:
    return SchemaDecl(
        version=version,
        observed_at=observed_at,
        node_types=tuple(NodeTypeDecl(name=name) for name in sorted(node_types)),
        relations=tuple(
            RelationDecl(
                name=relation_name,
                roles=tuple(
                    RoleDecl(
                        name=role_name,
                        allowed_node_types=tuple(sorted(role.allowed_types)),
                        min_count=role.min_count,
                        max_count=role.max_count,
                    )
                    for role_name, role in sorted(relation.roles.items())
                ),
            )
            for relation_name, relation in sorted(relations.items())
        ),
        observed_at_imputed=True,
    )


def _schema_timeline(
    node_type_by_id: Mapping[str, str],
    node_observed_at: Mapping[str, datetime],
    events: Sequence[_PreparedEvent],
    *,
    base_version: str,
    default_observed_at: datetime,
) -> tuple[tuple[SchemaDecl, ...], dict[datetime, str]]:
    """Infer only from each causally visible prefix, never from a future suffix."""

    timepoints = sorted(
        {
            default_observed_at,
            *node_observed_at.values(),
            *(event.observed_at for event in events),
        }
    )
    nodes_by_time: dict[datetime, list[str]] = defaultdict(list)
    for node_id, observed_at in node_observed_at.items():
        nodes_by_time[observed_at].append(node_id)
    events_by_time: dict[datetime, list[_PreparedEvent]] = defaultdict(list)
    for event in events:
        events_by_time[event.observed_at].append(event)

    schemas: list[SchemaDecl] = []
    version_at_time: dict[datetime, str] = {}
    visible_node_types: set[str] = set()
    relation_stats: dict[str, _RelationStats] = {}
    active_version = base_version
    for observed_at in timepoints:
        changed = not schemas
        for node_id in nodes_by_time[observed_at]:
            node_type = node_type_by_id[node_id]
            if node_type not in visible_node_types:
                visible_node_types.add(node_type)
                changed = True
        for event in events_by_time[observed_at]:
            changed = _advance_relation_stats(relation_stats, event, node_type_by_id) or changed
        if changed:
            active_version = base_version if not schemas else f"{base_version}+{len(schemas)}"
            schemas.append(
                _schema_from_stats(
                    visible_node_types,
                    relation_stats,
                    version=active_version,
                    observed_at=observed_at,
                )
            )
        version_at_time[observed_at] = active_version
    return tuple(schemas), version_at_time


def _stable_event_id(event: _PreparedEvent) -> str:
    material = {
        "observed_at": event.observed_at,
        "valid_from": event.valid_from,
        "valid_to": event.valid_to,
        "operation": event.operation,
        "relation": event.relation,
        "arguments": event.arguments,
        "payload": event.payload,
        "evidence_refs": sorted(event.evidence_refs),
    }
    return f"v2:event:{canonical_hash(material)}"


def _node_types(
    rows: Sequence[Mapping[str, Any]], node_id_by_legacy_id: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    node_types: dict[str, str] = {}
    node_hashes: dict[str, str] = {}
    for index, raw in enumerate(rows):
        legacy_id = str(raw.get("id") or "")
        node_id = node_id_by_legacy_id[legacy_id]
        node_type = str(raw.get("type") or "")
        if not _valid_contract_id(node_type):
            raise V1ConversionError(f"nodes[{index}]: type must be a non-padded string")
        digest = canonical_hash(raw)
        if node_id in node_hashes and node_hashes[node_id] != digest:
            raise V1ConversionError(f"nodes[{index}]: conflicting duplicate node {legacy_id!r}")
        if node_id in node_types and node_types[node_id] != node_type:
            raise V1ConversionError(f"nodes[{index}]: conflicting duplicate node {legacy_id!r}")
        node_types[node_id] = node_type
        node_hashes[node_id] = digest
    return node_types, node_hashes


def _schema_change_events(schemas: Sequence[SchemaDecl]) -> list[GraphEventV2]:
    events: list[GraphEventV2] = []
    for previous, current in zip(schemas, schemas[1:], strict=False):
        assert current.observed_at is not None
        digest = canonical_hash(
            {
                "previous": previous.version,
                "current": current.version,
                "observed_at": current.observed_at,
                "schema_hash": current.canonical_hash(),
            }
        )
        event_id = f"v2:event:schema-change:{digest}"
        events.append(
            GraphEventV2(
                event_id=event_id,
                observed_at=current.observed_at,
                valid_from=current.observed_at,
                valid_to=None,
                operation=Operation.SCHEMA_CHANGE,
                schema_version=previous.version,
                relation=None,
                arguments=(),
                payload={
                    "previous_schema_version": previous.version,
                    "new_schema_version": current.version,
                    "schema_hash": current.canonical_hash(),
                },
                basis_refs=(),
                derivation=DerivationRecord(
                    method="braid-v1-schema-induction",
                    parameters={"converter": CONVERTER_ID},
                ),
                provenance=ProvenanceRecord(
                    source="braid-v1-converter",
                    source_record_id=None,
                    license=None,
                    acquired_at=current.observed_at,
                ),
                observed_at_imputed=True,
            )
        )
    return events


def convert_v1_bundle(
    source: Mapping[str, Any],
    *,
    default_observed_at: datetime,
    schema_version: str = "2.0.0",
    bundle_id: str | None = None,
) -> GraphBundleV2:
    """Convert an in-memory v1 bundle without mutating any source container.

    ``source`` has the shape ``{manifest, nodes, events, judgments?}``.  Since v1
    nodes and many events have no observation clock, callers must choose an aware
    fallback instant rather than letting the converter invent one silently.
    """

    if default_observed_at.tzinfo is None or default_observed_at.utcoffset() is None:
        raise V1ConversionError("default_observed_at must include a UTC offset")
    source_hash_before = canonical_hash(source)
    manifest = source.get("manifest") or {}
    raw_nodes = source.get("nodes") or ()
    raw_events = source.get("events") or ()
    raw_judgments = source.get("judgments") or ()
    if not isinstance(manifest, Mapping):
        raise V1ConversionError("manifest must be an object")
    if not isinstance(raw_nodes, (list, tuple)) or not isinstance(raw_events, (list, tuple)):
        raise V1ConversionError("nodes and events must be arrays")
    if not isinstance(raw_judgments, (list, tuple)):
        raise V1ConversionError("judgments must be an array or null")

    node_id_by_legacy_id = _node_id_mapping(raw_nodes)
    node_type_by_id, source_node_hashes = _node_types(raw_nodes, node_id_by_legacy_id)
    prepared = _prepare_events(raw_events, default_observed_at, node_id_by_legacy_id)
    # v1 nodes carry no observation clock.  Keep the caller's explicit imputation
    # stable instead of moving a node when a later event first references it.
    node_observed_at = {node_id: default_observed_at for node_id in node_type_by_id}
    schemas, schema_version_at_time = _schema_timeline(
        node_type_by_id,
        node_observed_at,
        prepared,
        base_version=schema_version,
        default_observed_at=default_observed_at,
    )
    result_bundle_id = bundle_id or f"v2:bundle:{canonical_hash(source)}"

    evidence_observations: dict[str, list[datetime]] = defaultdict(list)
    for event in prepared:
        for ref in event.evidence_refs:
            evidence_observations[ref].append(event.observed_at)
    evidence_id_by_ref = {
        ref: f"v2:evidence:{canonical_hash({'legacy_ref': ref})}"
        for ref in sorted(evidence_observations)
    }
    evidence = [
        EvidenceRecord(
            evidence_id=evidence_id_by_ref[ref],
            observed_at=min(evidence_observations[ref]),
            kind="legacy-reference",
            source_uri=ref,
            content_hash=canonical_hash({"legacy_ref": ref}),
            payload={"legacy_ref": ref},
            observed_at_imputed=True,
        )
        for ref in sorted(evidence_observations)
    ]

    lineage: list[SourceLineage] = [
        SourceLineage(
            source_contract=str(manifest.get("contract_version") or "1.x"),
            source_kind="manifest",
            source_id="manifest",
            source_hash=canonical_hash(manifest),
            target_kind="bundle",
            target_id=result_bundle_id,
            converter=CONVERTER_ID,
            observed_at=schemas[0].observed_at,
        )
    ]
    for ref in sorted(evidence_observations):
        lineage.append(
            SourceLineage(
                source_contract="1.x",
                source_kind="evidence-reference",
                source_id=_lineage_source_id("evidence", ref),
                source_hash=canonical_hash({"legacy_ref": ref}),
                target_kind="evidence",
                target_id=evidence_id_by_ref[ref],
                converter=CONVERTER_ID,
                observed_at=min(evidence_observations[ref]),
            )
        )

    nodes_by_id: dict[str, NodeRecord] = {}
    for index, raw in enumerate(raw_nodes):
        legacy_node_id = str(raw.get("id") or "")
        node_id = node_id_by_legacy_id[legacy_node_id]
        node_hash = source_node_hashes[node_id]
        observed_at = node_observed_at[node_id]
        if node_id not in nodes_by_id:
            payload: dict[str, Any] = {
                "label": raw.get("label", ""),
                "attrs": raw.get("attrs") or {},
            }
            if node_id != legacy_node_id:
                payload["legacy_node_id"] = legacy_node_id
            nodes_by_id[node_id] = NodeRecord(
                node_id=node_id,
                node_type=str(raw.get("type") or ""),
                schema_version=schema_version_at_time[observed_at],
                observed_at=observed_at,
                payload_history=(
                    PayloadSnapshot(
                        observed_at=observed_at,
                        valid_from=observed_at,
                        payload=payload,
                        observed_at_imputed=True,
                    ),
                ),
                observed_at_imputed=True,
            )
        lineage.append(
            SourceLineage(
                source_contract="1.x",
                source_kind="node",
                source_id=_lineage_source_id("node", legacy_node_id, index),
                source_hash=node_hash,
                target_kind="node",
                target_id=node_id,
                converter=CONVERTER_ID,
                observed_at=observed_at,
            )
        )

    # v2 forbids simultaneously duplicated semantic assertions.  Group legacy
    # assertions deterministically, but preserve explicit lifecycle operations as
    # distinct records so their targets can be resolved and replayed.
    grouped: dict[tuple[Any, ...], list[_PreparedEvent]] = defaultdict(list)
    for event in prepared:
        if event.operation is Operation.ASSERT:
            grouped[(schema_version_at_time[event.observed_at], *event.assertion_key)].append(event)
    representatives: list[_PreparedEvent] = []
    target_by_source_index: dict[int, str] = {}
    targets_by_old_id: dict[str, set[str]] = defaultdict(set)
    for key in sorted(grouped, key=repr):
        group = sorted(
            grouped[key],
            key=lambda event: (
                event.observed_at,
                event.valid_from,
                event.source_hash,
                event.source_index,
            ),
        )
        representative = group[0]
        target_id = _stable_event_id(representative)
        representatives.append(representative)
        for source_event in group:
            target_by_source_index[source_event.source_index] = target_id
            targets_by_old_id[source_event.old_event_id].add(target_id)

    lifecycle_events = sorted(
        (event for event in prepared if event.operation is not Operation.ASSERT),
        key=lambda event: (event.observed_at, event.source_index),
    )
    for event in lifecycle_events:
        assert event.lifecycle_target_ref is not None
        targets = targets_by_old_id.get(event.lifecycle_target_ref)
        if not targets:
            raise V1ConversionError(
                f"event {event.source_id!r}: dangling lifecycle target "
                f"{event.lifecycle_target_ref!r}"
            )
        if len(targets) != 1:
            raise V1ConversionError(
                f"event {event.source_id!r}: lifecycle target "
                f"{event.lifecycle_target_ref!r} is ambiguous"
            )
        resolved = replace(
            event,
            payload={**event.payload, "supersedes_event_id": next(iter(targets))},
        )
        target_id = _stable_event_id(resolved)
        representatives.append(resolved)
        target_by_source_index[event.source_index] = target_id
        targets_by_old_id[event.old_event_id].add(target_id)

    ambiguous_derivation_ids = {
        old_id for old_id, target_ids in targets_by_old_id.items() if len(target_ids) > 1
    }
    events = _schema_change_events(schemas)
    for representative in representatives:
        resolved_derivation = []
        for old_id in representative.derivation_refs:
            if old_id in ambiguous_derivation_ids:
                raise V1ConversionError(
                    f"event {representative.source_id!r}: derivation ID {old_id!r} is ambiguous"
                )
            targets = targets_by_old_id.get(old_id)
            if not targets:
                raise V1ConversionError(
                    f"event {representative.source_id!r}: dangling derivation ID {old_id!r}"
                )
            resolved_derivation.append(next(iter(targets)))
        event_id = target_by_source_index[representative.source_index]
        events.append(
            GraphEventV2(
                event_id=event_id,
                observed_at=representative.observed_at,
                valid_from=representative.valid_from,
                valid_to=representative.valid_to,
                operation=representative.operation,
                schema_version=schema_version_at_time[representative.observed_at],
                relation=representative.relation,
                arguments=representative.arguments,
                payload=representative.payload,
                basis_refs=tuple(
                    evidence_id_by_ref[ref] for ref in sorted(set(representative.evidence_refs))
                ),
                derivation=DerivationRecord(
                    method=representative.derivation_method,
                    input_refs=tuple(sorted(set(resolved_derivation))),
                    parameters={"converter": CONVERTER_ID},
                ),
                provenance=ProvenanceRecord(
                    source="braid-v1",
                    source_record_id=representative.source_id,
                    license=None,
                    acquired_at=representative.observed_at,
                ),
                observed_at_imputed=representative.observed_at_imputed,
            )
        )
    for source_event in prepared:
        lineage.append(
            SourceLineage(
                source_contract="1.x",
                source_kind="event",
                source_id=source_event.source_id,
                source_hash=source_event.source_hash,
                target_kind="event",
                target_id=target_by_source_index[source_event.source_index],
                converter=CONVERTER_ID,
                observed_at=source_event.observed_at,
            )
        )

    # Legacy judgments had no causal clock or graph participants.  Preserve them as
    # explicitly typed evidence instead of fabricating edges or user negatives.
    for index, raw in enumerate(raw_judgments):
        if not isinstance(raw, Mapping):
            raise V1ConversionError(f"judgments[{index}]: expected an object")
        digest = canonical_hash(raw)
        judgment_id = f"v2:evidence:judgment:{canonical_hash({'index': index, 'row': raw})}"
        evidence.append(
            EvidenceRecord(
                evidence_id=judgment_id,
                observed_at=default_observed_at,
                kind="legacy-judgment",
                content_hash=digest,
                payload=dict(raw),
                observed_at_imputed=True,
            )
        )
        lineage.append(
            SourceLineage(
                source_contract="1.x",
                source_kind="judgment",
                source_id=_lineage_source_id("judgment", raw.get("id"), index),
                source_hash=digest,
                target_kind="evidence",
                target_id=judgment_id,
                converter=CONVERTER_ID,
                observed_at=default_observed_at,
            )
        )

    result = GraphBundleV2(
        bundle_id=result_bundle_id,
        schemas=schemas,
        nodes=tuple(sorted(nodes_by_id.values(), key=lambda node: node.node_id)),
        evidence=tuple(sorted(evidence, key=lambda record: record.evidence_id)),
        events=tuple(
            sorted(
                events,
                key=lambda event: (
                    event.observed_at,
                    event.operation is not Operation.SCHEMA_CHANGE,
                    event.event_id,
                ),
            )
        ),
        lineage=tuple(
            sorted(
                lineage,
                key=lambda item: (
                    item.source_kind,
                    item.source_id,
                    item.observed_at,
                    item.target_kind,
                    item.target_id,
                ),
            )
        ),
    )
    if canonical_hash(source) != source_hash_before:
        raise RuntimeError("v1 source was mutated during conversion")
    validate_bundle(result)
    return result
