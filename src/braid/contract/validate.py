"""Fail-closed semantic validation for Braid v2 records."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from .serde import canonical_json
from .types import (
    EvidenceRecord,
    ForecastDistribution,
    ForecastRequest,
    GraphBundleV2,
    GraphEventV2,
    NodeRecord,
    Operation,
    RelationDecl,
    SchemaDecl,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ContractValidationError(ValueError):
    """Raised when a record cannot safely cross the contract boundary."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"contract validation failed:\n{details}")


def _aware(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _require_clock(value: datetime | None, path: str, issues: list[ValidationIssue]) -> bool:
    if value is None:
        issues.append(ValidationIssue(path, "clock is required"))
        return False
    if not _aware(value):
        issues.append(ValidationIssue(path, "clock must include a UTC offset"))
        return False
    return True


def _require_id(value: str, path: str, issues: list[ValidationIssue]) -> None:
    if not value or value.strip() != value:
        issues.append(ValidationIssue(path, "must be a non-empty, unpadded string"))


def _duplicates(values: Iterable[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def _check_json(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append(ValidationIssue(path, "contains a non-finite float"))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                issues.append(ValidationIssue(path, "object keys must be strings"))
            _check_json(item, f"{path}.{key}", issues)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]", issues)
        return
    issues.append(ValidationIssue(path, f"unsupported payload value {type(value).__name__}"))


def _require_bool(value: Any, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, bool):
        issues.append(ValidationIssue(path, "must be a boolean"))


def _schema_issues(schema: SchemaDecl, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _require_id(schema.version, f"{path}.version", issues)
    _require_clock(schema.observed_at, f"{path}.observed_at", issues)
    _require_bool(schema.observed_at_imputed, f"{path}.observed_at_imputed", issues)
    _check_json(schema.constraints, f"{path}.constraints", issues)

    type_names = [node_type.name for node_type in schema.node_types]
    for duplicate in sorted(_duplicates(type_names)):
        issues.append(ValidationIssue(f"{path}.node_types", f"duplicate node type {duplicate!r}"))
    known_types = set(type_names)
    for index, node_type in enumerate(schema.node_types):
        _require_id(node_type.name, f"{path}.node_types[{index}].name", issues)
        _check_json(node_type.constraints, f"{path}.node_types[{index}].constraints", issues)

    relation_names = [relation.name for relation in schema.relations]
    for duplicate in sorted(_duplicates(relation_names)):
        issues.append(ValidationIssue(f"{path}.relations", f"duplicate relation {duplicate!r}"))
    known_relations = set(relation_names)
    relation_by_name = {relation.name: relation for relation in schema.relations}
    for relation_index, relation in enumerate(schema.relations):
        relation_path = f"{path}.relations[{relation_index}]"
        _require_id(relation.name, f"{relation_path}.name", issues)
        if not relation.roles:
            issues.append(
                ValidationIssue(f"{relation_path}.roles", "relation must declare at least one role")
            )
        role_names = [role.name for role in relation.roles]
        for duplicate in sorted(_duplicates(role_names)):
            issues.append(
                ValidationIssue(f"{relation_path}.roles", f"duplicate role {duplicate!r}")
            )
        for role_index, role in enumerate(relation.roles):
            role_path = f"{relation_path}.roles[{role_index}]"
            _require_id(role.name, f"{role_path}.name", issues)
            if not role.allowed_node_types:
                issues.append(ValidationIssue(f"{role_path}.allowed_node_types", "cannot be empty"))
            for node_type in role.allowed_node_types:
                if node_type not in known_types:
                    issues.append(
                        ValidationIssue(
                            f"{role_path}.allowed_node_types",
                            f"references unknown node type {node_type!r}",
                        )
                    )
            if role.min_count < 0:
                issues.append(ValidationIssue(f"{role_path}.min_count", "must be non-negative"))
            if role.max_count is not None and role.max_count < role.min_count:
                issues.append(
                    ValidationIssue(f"{role_path}.max_count", "must be at least min_count or null")
                )
        if relation.inverse is not None and relation.inverse not in known_relations:
            issues.append(
                ValidationIssue(f"{relation_path}.inverse", f"unknown inverse {relation.inverse!r}")
            )
        _check_json(relation.constraints, f"{relation_path}.constraints", issues)

    for index, example in enumerate(schema.support_examples):
        example_path = f"{path}.support_examples[{index}]"
        relation = relation_by_name.get(example.relation)
        if relation is None:
            issues.append(
                ValidationIssue(
                    f"{example_path}.relation", f"unknown relation {example.relation!r}"
                )
            )
            continue
        roles = {role.name: role for role in relation.roles}
        for role_name, node_type in example.role_types.items():
            role = roles.get(role_name)
            if role is None:
                issues.append(
                    ValidationIssue(f"{example_path}.role_types", f"unknown role {role_name!r}")
                )
            elif node_type not in role.allowed_node_types:
                issues.append(
                    ValidationIssue(
                        f"{example_path}.role_types.{role_name}",
                        f"node type {node_type!r} is not allowed",
                    )
                )
    return issues


def validate_schema(schema: SchemaDecl) -> None:
    issues = _schema_issues(schema, "schema")
    if issues:
        raise ContractValidationError(issues)


def _validate_evidence_graph(
    evidence_by_id: Mapping[str, EvidenceRecord], issues: list[ValidationIssue]
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(evidence_id: str, trail: tuple[str, ...]) -> None:
        if evidence_id in visiting:
            issues.append(
                ValidationIssue(
                    "bundle.evidence",
                    "evidence derivation cycle: " + " -> ".join((*trail, evidence_id)),
                )
            )
            return
        if evidence_id in visited:
            return
        visiting.add(evidence_id)
        evidence = evidence_by_id[evidence_id]
        for parent_id in evidence.parent_refs:
            if parent_id in evidence_by_id:
                visit(parent_id, (*trail, evidence_id))
        visiting.remove(evidence_id)
        visited.add(evidence_id)

    for evidence_id in evidence_by_id:
        visit(evidence_id, ())


def _relation_for_event(
    event: GraphEventV2,
    schema: SchemaDecl,
    path: str,
    issues: list[ValidationIssue],
) -> RelationDecl | None:
    relations = {relation.name: relation for relation in schema.relations}
    relation_operations = {Operation.ASSERT, Operation.RETRACT, Operation.SUPERSEDE}
    if event.operation in relation_operations and event.relation is None:
        issues.append(
            ValidationIssue(f"{path}.relation", f"{event.operation.value} requires a relation")
        )
        return None
    relationless_operations = {
        Operation.CREATE_NODE,
        Operation.UPDATE_NODE,
        Operation.SCHEMA_CHANGE,
    }
    if event.operation in relationless_operations and event.relation is not None:
        issues.append(
            ValidationIssue(f"{path}.relation", f"{event.operation.value} cannot carry a relation")
        )
        return None
    if event.relation is None:
        return None
    relation = relations.get(event.relation)
    if relation is None:
        issues.append(ValidationIssue(f"{path}.relation", f"unknown relation {event.relation!r}"))
    return relation


def _validate_event_roles(
    event: GraphEventV2,
    relation: RelationDecl | None,
    nodes_by_id: Mapping[str, NodeRecord],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    binding_pairs = [(binding.role, binding.node_id) for binding in event.arguments]
    for duplicate in sorted(_duplicates(f"{role}\0{node_id}" for role, node_id in binding_pairs)):
        role, node_id = duplicate.split("\0", 1)
        issues.append(
            ValidationIssue(f"{path}.arguments", f"duplicate binding ({role!r}, {node_id!r})")
        )

    for index, binding in enumerate(event.arguments):
        binding_path = f"{path}.arguments[{index}]"
        _require_id(binding.role, f"{binding_path}.role", issues)
        _require_id(binding.node_id, f"{binding_path}.node_id", issues)
        node = nodes_by_id.get(binding.node_id)
        if node is None:
            issues.append(
                ValidationIssue(
                    f"{binding_path}.node_id",
                    f"references unknown node {binding.node_id!r}",
                )
            )
        elif (
            _aware(node.observed_at)
            and _aware(event.observed_at)
            and node.observed_at > event.observed_at
        ):
            issues.append(ValidationIssue(binding_path, "references a node not yet observed"))

    if relation is not None:
        declared_roles = {role.name: role for role in relation.roles}
        counts = Counter(binding.role for binding in event.arguments)
        for binding in event.arguments:
            if binding.role not in declared_roles:
                issues.append(
                    ValidationIssue(f"{path}.arguments", f"role {binding.role!r} is not declared")
                )
        for role_name, role in declared_roles.items():
            count = counts[role_name]
            if count < role.min_count or (role.max_count is not None and count > role.max_count):
                maximum = "unbounded" if role.max_count is None else str(role.max_count)
                issues.append(
                    ValidationIssue(
                        f"{path}.arguments",
                        f"role {role_name!r} has {count} bindings; "
                        f"expected {role.min_count}..{maximum}",
                    )
                )
        for binding in event.arguments:
            node = nodes_by_id.get(binding.node_id)
            role = declared_roles.get(binding.role)
            if (
                node is not None
                and role is not None
                and node.node_type not in role.allowed_node_types
            ):
                issues.append(
                    ValidationIssue(
                        f"{path}.arguments",
                        f"role {binding.role!r} does not allow node type {node.node_type!r}",
                    )
                )
        return

    counts = Counter(binding.role for binding in event.arguments)
    if event.operation in {Operation.CREATE_NODE, Operation.UPDATE_NODE}:
        if counts != Counter({"node": 1}):
            issues.append(
                ValidationIssue(
                    f"{path}.arguments",
                    f"{event.operation.value} requires exactly one 'node'",
                )
            )
    elif event.operation is Operation.SCHEMA_CHANGE and event.arguments:
        issues.append(ValidationIssue(f"{path}.arguments", "SCHEMA_CHANGE cannot bind nodes"))
    elif event.operation is Operation.EXPOSE:
        if counts["candidate"] < 1 or counts["viewer"] > 1 or set(counts) - {"candidate", "viewer"}:
            issues.append(
                ValidationIssue(
                    f"{path}.arguments",
                    "relationless EXPOSE requires candidate(s) and at most one viewer",
                )
            )
    elif event.operation is Operation.JUDGE and (
        counts["subject"] != 1 or counts["judge"] > 1 or set(counts) - {"subject", "judge"}
    ):
        issues.append(
            ValidationIssue(
                f"{path}.arguments",
                "relationless JUDGE requires one subject and at most one judge",
            )
        )


def _schema_change_target(event: GraphEventV2) -> str | None:
    target = event.payload.get("new_schema_version")
    return target if isinstance(target, str) and target.strip() == target and target else None


_LIFECYCLE_TARGET_KEYS = frozenset({"target_event_id", "retracts_event_id", "supersedes_event_id"})
_MODEL_CONFIDENCE_KEYS = frozenset(
    {
        "calibrated_uncertainty",
        "confidence",
        "forecast_probability",
        "model_confidence",
        "probability",
    }
)


def _assertion_key(event: GraphEventV2) -> tuple[Any, ...] | None:
    if event.relation is None:
        return None
    return (
        event.schema_version,
        event.relation,
        tuple(sorted((binding.role, binding.node_id) for binding in event.arguments)),
    )


def _same_relation_assertion(left: GraphEventV2, right: GraphEventV2) -> bool:
    return left.relation == right.relation and tuple(
        sorted((binding.role, binding.node_id) for binding in left.arguments)
    ) == tuple(sorted((binding.role, binding.node_id) for binding in right.arguments))


def _lifecycle_target_id(
    event: GraphEventV2,
    path: str,
    issues: list[ValidationIssue],
) -> str | None:
    canonical_key = (
        "retracts_event_id" if event.operation is Operation.RETRACT else "supersedes_event_id"
    )
    allowed_keys = {canonical_key, "target_event_id"}
    present = [key for key in allowed_keys if key in event.payload]
    wrong_keys = (_LIFECYCLE_TARGET_KEYS - allowed_keys) & set(event.payload)
    if wrong_keys:
        issues.append(
            ValidationIssue(
                f"{path}.payload",
                f"{event.operation.value} cannot use target key(s) {sorted(wrong_keys)!r}",
            )
        )
    if len(present) != 1:
        issues.append(
            ValidationIssue(
                f"{path}.payload",
                f"{event.operation.value} requires exactly one of {sorted(allowed_keys)!r}",
            )
        )
        return None
    target = event.payload[present[0]]
    if not isinstance(target, str) or not target or target.strip() != target:
        issues.append(
            ValidationIssue(
                f"{path}.payload.{present[0]}",
                "must be a non-empty, unpadded event ID",
            )
        )
        return None
    return target


def _validate_confidence_literals(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    *,
    allow_v1_archive: bool = False,
) -> None:
    """Keep event-level model confidence out of the graph state.

    Imported v1 extraction metadata remains available beneath its explicitly
    namespaced archival object.  Generated proposals receive no such exception;
    their confidence belongs exclusively in ``ForecastDistribution``.
    """

    if not isinstance(value, Mapping):
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                _validate_confidence_literals(
                    item,
                    f"{path}[{index}]",
                    issues,
                    allow_v1_archive=allow_v1_archive,
                )
        return
    for key, item in value.items():
        normalized = str(key).lower().replace("-", "_")
        archival_subtree = allow_v1_archive and path.endswith(
            (".payload.attrs", ".payload.legacy_extraction")
        )
        if normalized in _MODEL_CONFIDENCE_KEYS and not archival_subtree:
            issues.append(
                ValidationIssue(
                    f"{path}.{key}",
                    "model confidence literals are forbidden; use the forecast distribution",
                )
            )
        _validate_confidence_literals(
            item,
            f"{path}.{key}",
            issues,
            allow_v1_archive=allow_v1_archive,
        )


def _validate_node_operation_payload(
    event: GraphEventV2,
    schema: SchemaDecl,
    nodes_by_id: Mapping[str, NodeRecord],
    path: str,
    issues: list[ValidationIssue],
    *,
    proposal: bool,
) -> tuple[str, str] | None:
    if event.operation not in {Operation.CREATE_NODE, Operation.UPDATE_NODE}:
        return None
    node_bindings = [binding for binding in event.arguments if binding.role == "node"]
    if len(node_bindings) != 1 or len(event.arguments) != 1:
        return None  # Role validation emits the canonical cardinality issue.
    bound_node_id = node_bindings[0].node_id
    payload_node_id = event.payload.get("node_id")
    payload_node_type = event.payload.get("node_type")
    if payload_node_id != bound_node_id:
        issues.append(
            ValidationIssue(
                f"{path}.payload.node_id",
                "must be present and match the 'node' binding",
            )
        )
    if not isinstance(payload_node_type, str) or not payload_node_type:
        issues.append(
            ValidationIssue(
                f"{path}.payload.node_type",
                "must be a non-empty declared node type",
            )
        )
        return None
    known_types = {item.name for item in schema.node_types}
    if payload_node_type not in known_types:
        issues.append(
            ValidationIssue(
                f"{path}.payload.node_type",
                f"unknown node type {payload_node_type!r}",
            )
        )
    existing = nodes_by_id.get(bound_node_id)
    if event.operation is Operation.CREATE_NODE:
        if proposal and existing is not None:
            issues.append(
                ValidationIssue(
                    f"{path}.arguments",
                    f"node {bound_node_id!r} already exists",
                )
            )
        elif not proposal:
            if existing is None:
                issues.append(
                    ValidationIssue(
                        f"{path}.arguments",
                        "CREATE_NODE must materialize a matching bundle node",
                    )
                )
            else:
                if existing.node_type != payload_node_type:
                    issues.append(
                        ValidationIssue(
                            f"{path}.payload.node_type",
                            "does not match the materialized node type",
                        )
                    )
                if existing.schema_version != event.schema_version:
                    issues.append(
                        ValidationIssue(
                            f"{path}.schema_version",
                            "does not match the materialized node schema",
                        )
                    )
                if existing.observed_at != event.observed_at:
                    issues.append(
                        ValidationIssue(
                            f"{path}.observed_at",
                            "must equal the materialized node observation time",
                        )
                    )
    else:
        if existing is None:
            issues.append(
                ValidationIssue(
                    f"{path}.arguments",
                    f"UPDATE_NODE target {bound_node_id!r} does not exist causally",
                )
            )
        elif existing.node_type != payload_node_type:
            issues.append(
                ValidationIssue(
                    f"{path}.payload.node_type",
                    "UPDATE_NODE cannot change the node type",
                )
            )
        if (
            existing is not None
            and existing.observed_at is not None
            and event.observed_at is not None
            and existing.observed_at >= event.observed_at
        ):
            issues.append(
                ValidationIssue(
                    f"{path}.arguments",
                    "UPDATE_NODE target must exist strictly before the update",
                )
            )
        if not (set(event.payload) - {"node_id", "node_type"}):
            issues.append(
                ValidationIssue(
                    f"{path}.payload",
                    "UPDATE_NODE requires a payload field beyond node identity and type",
                )
            )
    if payload_node_id != bound_node_id or payload_node_type not in known_types:
        return None
    return bound_node_id, payload_node_type


def _validate_bundle_assertion_lifecycle(
    bundle: GraphBundleV2,
    issues: list[ValidationIssue],
) -> None:
    """Replay assertion state in causal order without mutating the bundle."""

    active_key_by_event: dict[str, tuple[Any, ...]] = {}
    active_event_by_key: dict[tuple[Any, ...], str] = {}
    indexed_events = sorted(
        enumerate(bundle.events),
        key=lambda item: (
            item[1].observed_at is None,
            item[1].observed_at or datetime.max,
            item[0],
        ),
    )
    events_by_id = {event.event_id: event for event in bundle.events}
    for index, event in indexed_events:
        path = f"bundle.events[{index}]"
        if event.operation is Operation.ASSERT:
            key = _assertion_key(event)
            if key is not None:
                previous = active_event_by_key.get(key)
                if previous is not None:
                    issues.append(
                        ValidationIssue(
                            path,
                            f"duplicate assertion; active semantic duplicate of event {previous!r}",
                        )
                    )
                else:
                    active_event_by_key[key] = event.event_id
                    active_key_by_event[event.event_id] = key
            continue
        if event.operation not in {Operation.RETRACT, Operation.SUPERSEDE}:
            continue

        issue_count = len(issues)
        target_id = _lifecycle_target_id(event, path, issues)
        target = events_by_id.get(target_id or "")
        if target is None:
            issues.append(
                ValidationIssue(
                    f"{path}.payload",
                    "lifecycle target does not identify an event in this bundle",
                )
            )
        elif (
            event.observed_at is None
            or target.observed_at is None
            or target.observed_at >= event.observed_at
        ):
            issues.append(
                ValidationIssue(
                    f"{path}.payload",
                    "lifecycle target must be observed strictly before this event",
                )
            )
        target_key = active_key_by_event.get(target_id or "")
        if target is not None and target_key is None:
            issues.append(
                ValidationIssue(
                    f"{path}.payload",
                    "lifecycle target must be an active ASSERT or SUPERSEDE event",
                )
            )
        if (
            event.operation is Operation.RETRACT
            and target is not None
            and not _same_relation_assertion(event, target)
        ):
            issues.append(
                ValidationIssue(
                    path,
                    "RETRACT relation and arguments must match its target",
                )
            )
        if len(issues) != issue_count or target_key is None:
            continue

        replacement_key: tuple[Any, ...] | None = None
        if event.operation is Operation.SUPERSEDE:
            replacement_key = _assertion_key(event)
            if replacement_key is None:
                continue
            previous = active_event_by_key.get(replacement_key)
            if previous is not None and previous != target_id:
                issues.append(
                    ValidationIssue(
                        path,
                        f"supersession duplicates active assertion {previous!r}",
                    )
                )
                continue

        assert target_id is not None
        del active_key_by_event[target_id]
        active_event_by_key.pop(target_key, None)
        if replacement_key is not None:
            active_event_by_key[replacement_key] = event.event_id
            active_key_by_event[event.event_id] = replacement_key


def _validate_schema_evolution(bundle: GraphBundleV2, issues: list[ValidationIssue]) -> None:
    """Require an explicit, ordered proposal before every non-root schema is used."""

    if not bundle.schemas:
        return
    ordered = sorted(
        enumerate(bundle.schemas),
        key=lambda item: (
            item[1].observed_at is not None,
            item[1].observed_at or datetime.min,
            item[0],
        ),
    )
    declarations_by_clock: dict[datetime, list[str]] = defaultdict(list)
    for _, schema in ordered:
        if _aware(schema.observed_at):
            assert schema.observed_at is not None
            declarations_by_clock[schema.observed_at].append(schema.version)
    for observed_at, versions in declarations_by_clock.items():
        if len(versions) > 1:
            issues.append(
                ValidationIssue(
                    "bundle.schemas",
                    "distinct schema declarations cannot share the same observed_at "
                    f"({observed_at.isoformat()}): {versions!r}",
                )
            )

    schema_versions = {schema.version for schema in bundle.schemas}
    changes_by_target: dict[str, list[tuple[int, GraphEventV2]]] = defaultdict(list)
    activation_by_clock: dict[datetime, tuple[int, str]] = {}
    for event_index, event in enumerate(bundle.events):
        if event.operation is not Operation.SCHEMA_CHANGE:
            continue
        target = _schema_change_target(event)
        if target is None:
            issues.append(
                ValidationIssue(
                    f"bundle.events[{event_index}].payload.new_schema_version",
                    "SCHEMA_CHANGE requires a non-empty schema version",
                )
            )
            continue
        if _aware(event.observed_at):
            assert event.observed_at is not None
            previous_activation = activation_by_clock.get(event.observed_at)
            if previous_activation is not None and previous_activation[1] != target:
                issues.append(
                    ValidationIssue(
                        f"bundle.events[{event_index}].observed_at",
                        "distinct SCHEMA_CHANGE activations cannot share the same observed_at; "
                        f"already activates {previous_activation[1]!r} at event "
                        f"{previous_activation[0]}",
                    )
                )
            else:
                activation_by_clock[event.observed_at] = (event_index, target)
        if target not in schema_versions:
            issues.append(
                ValidationIssue(
                    f"bundle.events[{event_index}].payload.new_schema_version",
                    f"unknown proposed schema {target!r}",
                )
            )
            continue
        changes_by_target[target].append((event_index, event))

    root_schema = ordered[0][1]
    if changes_by_target.get(root_schema.version):
        issues.append(
            ValidationIssue(
                "bundle.events",
                f"root schema {root_schema.version!r} cannot be a SCHEMA_CHANGE target",
            )
        )

    previous_schema = root_schema
    for _, schema in ordered[1:]:
        candidates = changes_by_target.get(schema.version, [])
        if len(candidates) != 1:
            issues.append(
                ValidationIssue(
                    "bundle.schemas",
                    f"schema {schema.version!r} requires exactly one SCHEMA_CHANGE; "
                    f"found {len(candidates)}",
                )
            )
            previous_schema = schema
            continue
        change_index, change = candidates[0]
        payload_previous = change.payload.get("previous_schema_version")
        if payload_previous != previous_schema.version:
            issues.append(
                ValidationIssue(
                    f"bundle.events[{change_index}].payload.previous_schema_version",
                    f"must equal prior schema {previous_schema.version!r}",
                )
            )
        if change.schema_version != previous_schema.version:
            issues.append(
                ValidationIssue(
                    f"bundle.events[{change_index}].schema_version",
                    f"schema change to {schema.version!r} must use prior schema "
                    f"{previous_schema.version!r}",
                )
            )
        if change.observed_at != schema.observed_at:
            issues.append(
                ValidationIssue(
                    f"bundle.events[{change_index}].observed_at",
                    f"must equal declaration time for schema {schema.version!r}",
                )
            )
        schema_hash = change.payload.get("schema_hash")
        if schema_hash != schema.canonical_hash():
            issues.append(
                ValidationIssue(
                    f"bundle.events[{change_index}].payload.schema_hash",
                    "must equal the canonical hash of the target schema declaration",
                )
            )
        for use_index, event in enumerate(bundle.events):
            if event.schema_version != schema.version or event.operation is Operation.SCHEMA_CHANGE:
                continue
            if event.observed_at == change.observed_at and use_index < change_index:
                issues.append(
                    ValidationIssue(
                        f"bundle.events[{use_index}]",
                        f"uses schema {schema.version!r} before its SCHEMA_CHANGE",
                    )
                )
        previous_schema = schema


def collect_bundle_issues(bundle: GraphBundleV2) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    _require_id(bundle.bundle_id, "bundle.bundle_id", issues)
    if not bundle.schemas:
        issues.append(ValidationIssue("bundle.schemas", "at least one schema is required"))

    schema_versions = [schema.version for schema in bundle.schemas]
    for duplicate in sorted(_duplicates(schema_versions)):
        issues.append(ValidationIssue("bundle.schemas", f"duplicate schema version {duplicate!r}"))
    schemas_by_version = {schema.version: schema for schema in bundle.schemas}
    for index, schema in enumerate(bundle.schemas):
        issues.extend(_schema_issues(schema, f"bundle.schemas[{index}]"))

    node_ids = [node.node_id for node in bundle.nodes]
    evidence_ids = [record.evidence_id for record in bundle.evidence]
    event_ids = [event.event_id for event in bundle.events]
    for duplicate in sorted(_duplicates(node_ids)):
        issues.append(ValidationIssue("bundle.nodes", f"duplicate node ID {duplicate!r}"))
    for duplicate in sorted(_duplicates(evidence_ids)):
        issues.append(ValidationIssue("bundle.evidence", f"duplicate evidence ID {duplicate!r}"))
    for duplicate in sorted(_duplicates(event_ids)):
        issues.append(ValidationIssue("bundle.events", f"duplicate event ID {duplicate!r}"))

    nodes_by_id = {node.node_id: node for node in bundle.nodes}
    evidence_by_id = {record.evidence_id: record for record in bundle.evidence}
    events_by_id = {event.event_id: event for event in bundle.events}

    for index, record in enumerate(bundle.evidence):
        path = f"bundle.evidence[{index}]"
        _require_id(record.evidence_id, f"{path}.evidence_id", issues)
        clock_ok = _require_clock(record.observed_at, f"{path}.observed_at", issues)
        _require_bool(record.observed_at_imputed, f"{path}.observed_at_imputed", issues)
        _require_id(record.kind, f"{path}.kind", issues)
        if record.content_hash is not None and not _SHA256.fullmatch(record.content_hash):
            issues.append(
                ValidationIssue(f"{path}.content_hash", "must be a lowercase SHA-256 digest")
            )
        _check_json(record.payload, f"{path}.payload", issues)
        for parent_id in record.parent_refs:
            parent = evidence_by_id.get(parent_id)
            if parent is None:
                issues.append(
                    ValidationIssue(
                        f"{path}.parent_refs",
                        f"dangling evidence reference {parent_id!r}",
                    )
                )
            elif (
                clock_ok and _aware(parent.observed_at) and parent.observed_at > record.observed_at
            ):
                issues.append(ValidationIssue(f"{path}.parent_refs", "references future evidence"))
    _validate_evidence_graph(evidence_by_id, issues)

    for index, node in enumerate(bundle.nodes):
        path = f"bundle.nodes[{index}]"
        _require_id(node.node_id, f"{path}.node_id", issues)
        node_clock_ok = _require_clock(node.observed_at, f"{path}.observed_at", issues)
        _require_bool(node.observed_at_imputed, f"{path}.observed_at_imputed", issues)
        schema = schemas_by_version.get(node.schema_version)
        if schema is None:
            issues.append(
                ValidationIssue(f"{path}.schema_version", f"unknown schema {node.schema_version!r}")
            )
        else:
            known_types = {node_type.name for node_type in schema.node_types}
            if node.node_type not in known_types:
                issues.append(
                    ValidationIssue(f"{path}.node_type", f"unknown type {node.node_type!r}")
                )
            if (
                node_clock_ok
                and _aware(schema.observed_at)
                and schema.observed_at > node.observed_at
            ):
                issues.append(ValidationIssue(path, "node predates its schema declaration"))
        previous_observed: datetime | None = None
        seen_snapshot_clocks: set[datetime] = set()
        for snapshot_index, snapshot in enumerate(node.payload_history):
            snapshot_path = f"{path}.payload_history[{snapshot_index}]"
            snapshot_clock_ok = _require_clock(
                snapshot.observed_at, f"{snapshot_path}.observed_at", issues
            )
            _require_bool(
                snapshot.observed_at_imputed,
                f"{snapshot_path}.observed_at_imputed",
                issues,
            )
            if snapshot_clock_ok:
                assert snapshot.observed_at is not None
                if snapshot.observed_at in seen_snapshot_clocks:
                    issues.append(
                        ValidationIssue(snapshot_path, "duplicate payload observation clock")
                    )
                seen_snapshot_clocks.add(snapshot.observed_at)
                if previous_observed is not None and snapshot.observed_at < previous_observed:
                    issues.append(
                        ValidationIssue(snapshot_path, "payload history is not chronological")
                    )
                previous_observed = snapshot.observed_at
                if node_clock_ok and snapshot.observed_at < node.observed_at:
                    issues.append(ValidationIssue(snapshot_path, "payload predates its node"))
            if snapshot.valid_from is not None and not _aware(snapshot.valid_from):
                issues.append(
                    ValidationIssue(
                        f"{snapshot_path}.valid_from", "clock must include a UTC offset"
                    )
                )
            if snapshot.valid_to is not None and not _aware(snapshot.valid_to):
                issues.append(
                    ValidationIssue(f"{snapshot_path}.valid_to", "clock must include a UTC offset")
                )
            if (
                _aware(snapshot.valid_from)
                and _aware(snapshot.valid_to)
                and snapshot.valid_to < snapshot.valid_from
            ):
                issues.append(ValidationIssue(snapshot_path, "valid_to precedes valid_from"))
            _check_json(snapshot.payload, f"{snapshot_path}.payload", issues)
            for evidence_id in snapshot.basis_refs:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    issues.append(
                        ValidationIssue(
                            f"{snapshot_path}.basis_refs",
                            f"dangling evidence reference {evidence_id!r}",
                        )
                    )
                elif (
                    snapshot_clock_ok
                    and _aware(evidence.observed_at)
                    and evidence.observed_at > snapshot.observed_at
                ):
                    issues.append(
                        ValidationIssue(f"{snapshot_path}.basis_refs", "references future evidence")
                    )
        for evidence_id in node.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                issues.append(
                    ValidationIssue(
                        f"{path}.evidence_ids",
                        f"dangling evidence reference {evidence_id!r}",
                    )
                )
            elif (
                node_clock_ok
                and _aware(evidence.observed_at)
                and evidence.observed_at > node.observed_at
            ):
                issues.append(ValidationIssue(f"{path}.evidence_ids", "references future evidence"))

    created_node_ids: set[str] = set()
    for index, event in enumerate(bundle.events):
        path = f"bundle.events[{index}]"
        _require_id(event.event_id, f"{path}.event_id", issues)
        event_clock_ok = _require_clock(event.observed_at, f"{path}.observed_at", issues)
        _require_bool(event.observed_at_imputed, f"{path}.observed_at_imputed", issues)
        valid_clock_ok = _require_clock(event.valid_from, f"{path}.valid_from", issues)
        if event.valid_to is not None and not _aware(event.valid_to):
            issues.append(ValidationIssue(f"{path}.valid_to", "clock must include a UTC offset"))
        if valid_clock_ok and _aware(event.valid_to) and event.valid_to < event.valid_from:
            issues.append(ValidationIssue(path, "valid_to precedes valid_from"))
        schema = schemas_by_version.get(event.schema_version)
        if schema is None:
            issues.append(
                ValidationIssue(
                    f"{path}.schema_version", f"unknown schema {event.schema_version!r}"
                )
            )
            relation = None
        else:
            if (
                event_clock_ok
                and _aware(schema.observed_at)
                and schema.observed_at > event.observed_at
            ):
                issues.append(ValidationIssue(path, "event predates its schema declaration"))
            relation = _relation_for_event(event, schema, path, issues)
        _validate_event_roles(event, relation, nodes_by_id, path, issues)
        _check_json(event.payload, f"{path}.payload", issues)
        _validate_confidence_literals(
            event.payload,
            f"{path}.payload",
            issues,
            allow_v1_archive=event.provenance.source == "braid-v1",
        )
        if schema is not None:
            node_identity = _validate_node_operation_payload(
                event,
                schema,
                nodes_by_id,
                path,
                issues,
                proposal=False,
            )
            if event.operation is Operation.CREATE_NODE and node_identity is not None:
                node_id, _ = node_identity
                if node_id in created_node_ids:
                    issues.append(
                        ValidationIssue(
                            path,
                            f"node {node_id!r} has more than one CREATE_NODE event",
                        )
                    )
                else:
                    created_node_ids.add(node_id)
        if event.operation not in {
            Operation.RETRACT,
            Operation.SUPERSEDE,
        } and _LIFECYCLE_TARGET_KEYS & set(event.payload):
            issues.append(
                ValidationIssue(
                    f"{path}.payload",
                    f"{event.operation.value} cannot carry a lifecycle target",
                )
            )
        if not event.derivation.method:
            issues.append(ValidationIssue(f"{path}.derivation.method", "cannot be empty"))
        _check_json(event.derivation.parameters, f"{path}.derivation.parameters", issues)
        _validate_confidence_literals(
            event.derivation.parameters,
            f"{path}.derivation.parameters",
            issues,
        )
        _require_id(event.provenance.source, f"{path}.provenance.source", issues)
        acquired_ok = _require_clock(
            event.provenance.acquired_at, f"{path}.provenance.acquired_at", issues
        )
        if acquired_ok and event_clock_ok and event.provenance.acquired_at > event.observed_at:
            issues.append(
                ValidationIssue(f"{path}.provenance.acquired_at", "postdates observation")
            )
        for evidence_id in event.basis_refs:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                issues.append(
                    ValidationIssue(
                        f"{path}.basis_refs",
                        f"dangling evidence reference {evidence_id!r}",
                    )
                )
            elif (
                event_clock_ok
                and _aware(evidence.observed_at)
                and evidence.observed_at > event.observed_at
            ):
                issues.append(ValidationIssue(f"{path}.basis_refs", "references future evidence"))
        for input_ref in event.derivation.input_refs:
            referenced_clock: datetime | None = None
            if input_ref in evidence_by_id:
                referenced_clock = evidence_by_id[input_ref].observed_at
            elif input_ref in nodes_by_id:
                referenced_clock = nodes_by_id[input_ref].observed_at
            elif input_ref in events_by_id:
                referenced_clock = events_by_id[input_ref].observed_at
            else:
                issues.append(
                    ValidationIssue(
                        f"{path}.derivation.input_refs",
                        f"dangling input reference {input_ref!r}",
                    )
                )
            if event_clock_ok and _aware(referenced_clock) and referenced_clock > event.observed_at:
                issues.append(
                    ValidationIssue(
                        f"{path}.derivation.input_refs",
                        f"future input reference {input_ref!r}",
                    )
                )
    _validate_bundle_assertion_lifecycle(bundle, issues)
    _validate_schema_evolution(bundle, issues)

    target_ids = {
        "schema": set(schema_versions),
        "node": set(node_ids),
        "evidence": set(evidence_ids),
        "event": set(event_ids),
        "bundle": {bundle.bundle_id},
    }
    for index, lineage in enumerate(bundle.lineage):
        path = f"bundle.lineage[{index}]"
        _require_id(lineage.source_contract, f"{path}.source_contract", issues)
        _require_id(lineage.source_kind, f"{path}.source_kind", issues)
        _require_id(lineage.source_id, f"{path}.source_id", issues)
        _require_id(lineage.converter, f"{path}.converter", issues)
        _require_clock(lineage.observed_at, f"{path}.observed_at", issues)
        if not _SHA256.fullmatch(lineage.source_hash):
            issues.append(
                ValidationIssue(f"{path}.source_hash", "must be a lowercase SHA-256 digest")
            )
        if lineage.target_kind not in target_ids:
            issues.append(
                ValidationIssue(
                    f"{path}.target_kind",
                    f"unknown target kind {lineage.target_kind!r}",
                )
            )
        elif lineage.target_id not in target_ids[lineage.target_kind]:
            issues.append(
                ValidationIssue(
                    f"{path}.target_id",
                    f"unknown {lineage.target_kind} {lineage.target_id!r}",
                )
            )
    return tuple(issues)


def validate_bundle(bundle: GraphBundleV2) -> None:
    issues = collect_bundle_issues(bundle)
    if issues:
        raise ContractValidationError(issues)


def validate_forecast_request(request: ForecastRequest) -> None:
    validation_bundle = replace(
        request.prefix,
        events=(*request.prefix.events, *request.support_events),
    )
    # Support events cross the same contract boundary as prefix events.  Validating
    # the augmented bundle catches duplicate IDs/assertions, invalid schema/roles,
    # dangling nodes/evidence/derivations, and bad dual clocks before tensorization.
    issues = list(collect_bundle_issues(validation_bundle))
    cutoff_ok = _require_clock(request.cutoff, "request.cutoff", issues)
    if request.horizon.total_seconds() <= 0:
        issues.append(ValidationIssue("request.horizon", "must be positive"))
    try:
        active_prefix_schema = request.prefix.schema
    except LookupError:
        active_prefix_schema = None
    if (
        active_prefix_schema is None
        or active_prefix_schema.canonical_hash() != request.schema.canonical_hash()
    ):
        issues.append(
            ValidationIssue(
                "request.schema",
                "must equal the causally latest active prefix schema",
            )
        )
    node_ids = {node.node_id for node in request.prefix.nodes}
    relations = {relation.name for relation in request.schema.relations}
    for node_id in request.task.query_node_ids:
        if node_id not in node_ids:
            issues.append(
                ValidationIssue("request.task.query_node_ids", f"unknown node {node_id!r}")
            )
    for relation in request.task.target_relations:
        if relation not in relations:
            issues.append(
                ValidationIssue("request.task.target_relations", f"unknown relation {relation!r}")
            )
    for index, support in enumerate(request.support_events):
        if support.schema_version != request.schema.version:
            issues.append(
                ValidationIssue(
                    f"request.support_events[{index}].schema_version",
                    "must match the requested schema",
                )
            )
    if cutoff_ok:
        assert request.cutoff is not None
        observed_records = [
            *(
                (f"request.prefix.schemas[{index}].observed_at", schema.observed_at)
                for index, schema in enumerate(request.prefix.schemas)
            ),
            *(
                (f"request.prefix.nodes[{index}].observed_at", node.observed_at)
                for index, node in enumerate(request.prefix.nodes)
            ),
            *(
                (
                    f"request.prefix.nodes[{node_index}].payload_history[{snapshot_index}]"
                    ".observed_at",
                    snapshot.observed_at,
                )
                for node_index, node in enumerate(request.prefix.nodes)
                for snapshot_index, snapshot in enumerate(node.payload_history)
            ),
            *(
                (f"request.prefix.evidence[{index}].observed_at", evidence.observed_at)
                for index, evidence in enumerate(request.prefix.evidence)
            ),
            *(
                (f"request.prefix.events[{index}].observed_at", event.observed_at)
                for index, event in enumerate(request.prefix.events)
            ),
            *(
                (f"request.prefix.lineage[{index}].observed_at", lineage.observed_at)
                for index, lineage in enumerate(request.prefix.lineage)
            ),
            *(
                (f"request.support_events[{index}].observed_at", event.observed_at)
                for index, event in enumerate(request.support_events)
            ),
        ]
        for path, clock in observed_records:
            if _aware(clock) and clock > request.cutoff:
                issues.append(ValidationIssue(path, "record is not visible at cutoff"))

        imputed_paths = [
            *(
                f"request.prefix.schemas[{index}].observed_at"
                for index, schema in enumerate(request.prefix.schemas)
                if schema.observed_at_imputed
            ),
            *(
                f"request.prefix.nodes[{index}].observed_at"
                for index, node in enumerate(request.prefix.nodes)
                if node.observed_at_imputed
            ),
            *(
                f"request.prefix.nodes[{node_index}].payload_history[{snapshot_index}].observed_at"
                for node_index, node in enumerate(request.prefix.nodes)
                for snapshot_index, snapshot in enumerate(node.payload_history)
                if snapshot.observed_at_imputed
            ),
            *(
                f"request.prefix.evidence[{index}].observed_at"
                for index, evidence in enumerate(request.prefix.evidence)
                if evidence.observed_at_imputed
            ),
            *(
                f"request.prefix.events[{index}].observed_at"
                for index, event in enumerate(request.prefix.events)
                if event.observed_at_imputed
            ),
            *(
                f"request.support_events[{index}].observed_at"
                for index, event in enumerate(request.support_events)
                if event.observed_at_imputed
            ),
        ]
        issues.extend(
            ValidationIssue(path, "imputed observation time is illegal for temporal requests")
            for path in imputed_paths
        )
    if issues:
        raise ContractValidationError(issues)


def _active_assertion_state(
    events: Iterable[GraphEventV2],
) -> tuple[dict[str, tuple[Any, ...]], dict[tuple[Any, ...], str]]:
    """Replay an already-validated prefix into its current assertion state."""

    active_key_by_event: dict[str, tuple[Any, ...]] = {}
    active_event_by_key: dict[tuple[Any, ...], str] = {}
    for event in sorted(
        events,
        key=lambda item: (item.observed_at or datetime.max, item.event_id),
    ):
        if event.operation is Operation.ASSERT:
            key = _assertion_key(event)
            if key is not None:
                active_key_by_event[event.event_id] = key
                active_event_by_key[key] = event.event_id
        elif event.operation in {Operation.RETRACT, Operation.SUPERSEDE}:
            target_keys = (
                ("target_event_id", "retracts_event_id")
                if event.operation is Operation.RETRACT
                else ("target_event_id", "supersedes_event_id")
            )
            target_id = next(
                (
                    value
                    for key in target_keys
                    if isinstance((value := event.payload.get(key)), str)
                ),
                None,
            )
            target_key = active_key_by_event.pop(target_id or "", None)
            if target_key is not None:
                active_event_by_key.pop(target_key, None)
            if event.operation is Operation.SUPERSEDE:
                key = _assertion_key(event)
                if key is not None:
                    active_key_by_event[event.event_id] = key
                    active_event_by_key[key] = event.event_id
    return active_key_by_event, active_event_by_key


def _proposal_schema_decl(
    event: GraphEventV2,
    active_schema: SchemaDecl,
    known_schema_versions: set[str],
    path: str,
    issues: list[ValidationIssue],
) -> SchemaDecl | None:
    previous = event.payload.get("previous_schema_version")
    target = _schema_change_target(event)
    if previous != active_schema.version:
        issues.append(
            ValidationIssue(
                f"{path}.payload.previous_schema_version",
                f"must equal active schema {active_schema.version!r}",
            )
        )
    if target is None or target == active_schema.version or target in known_schema_versions:
        issues.append(
            ValidationIssue(
                f"{path}.payload.new_schema_version",
                "must name a distinct, previously unseen schema version",
            )
        )

    raw_decl = event.payload.get("schema_decl")
    if not isinstance(raw_decl, Mapping):
        issues.append(
            ValidationIssue(
                f"{path}.payload.schema_decl",
                "must contain a serialized SchemaDecl",
            )
        )
        return None
    try:
        # Payload freezing turns JSON arrays into tuples.  Canonical JSON restores
        # the wire representation before the tagged record decoder runs.
        declaration = SchemaDecl.from_json(canonical_json(raw_decl))
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                f"{path}.payload.schema_decl",
                f"is not a valid serialized SchemaDecl: {exc}",
            )
        )
        return None
    issues.extend(_schema_issues(declaration, f"{path}.payload.schema_decl"))
    if declaration.version != target:
        issues.append(
            ValidationIssue(
                f"{path}.payload.new_schema_version",
                "does not match the embedded schema declaration",
            )
        )
    if declaration.observed_at != event.observed_at:
        issues.append(
            ValidationIssue(
                f"{path}.payload.schema_decl.observed_at",
                "must equal the schema-change observation time",
            )
        )
    if declaration.observed_at_imputed:
        issues.append(
            ValidationIssue(
                f"{path}.payload.schema_decl.observed_at",
                "a proposed schema cannot use an imputed observation time",
            )
        )
    if event.payload.get("schema_hash") != declaration.canonical_hash():
        issues.append(
            ValidationIssue(
                f"{path}.payload.schema_hash",
                "must equal the canonical hash of the embedded schema declaration",
            )
        )
    return declaration


def collect_proposed_patch_issues(
    request: ForecastRequest,
    events: Iterable[GraphEventV2],
    *,
    path: str = "patch",
) -> tuple[ValidationIssue, ...]:
    """Validate a proposal against a private state derived from the request prefix."""

    issues: list[ValidationIssue] = []
    assert request.cutoff is not None
    horizon_end = request.cutoff + request.horizon
    nodes_by_id = {node.node_id: node for node in request.prefix.nodes}
    evidence_by_id = {record.evidence_id: record for record in request.prefix.evidence}
    events_by_id = {event.event_id: event for event in request.prefix.events}
    active_key_by_event, active_event_by_key = _active_assertion_state(request.prefix.events)
    active_schema = request.schema
    known_schema_versions = {schema.version for schema in request.prefix.schemas}
    last_observed = request.cutoff

    for index, event in enumerate(events):
        event_path = f"{path}.events[{index}]"
        event_issue_start = len(issues)
        _require_id(event.event_id, f"{event_path}.event_id", issues)
        observed_ok = _require_clock(event.observed_at, f"{event_path}.observed_at", issues)
        valid_ok = _require_clock(event.valid_from, f"{event_path}.valid_from", issues)
        _require_bool(
            event.observed_at_imputed,
            f"{event_path}.observed_at_imputed",
            issues,
        )
        if event.observed_at_imputed:
            issues.append(
                ValidationIssue(
                    f"{event_path}.observed_at",
                    "proposed events cannot use an imputed observation time",
                )
            )
        if event.event_id in events_by_id:
            issues.append(ValidationIssue(f"{event_path}.event_id", "duplicate event ID"))
        if observed_ok:
            assert event.observed_at is not None
            if not request.cutoff < event.observed_at <= horizon_end:
                issues.append(
                    ValidationIssue(
                        f"{event_path}.observed_at",
                        "must be after cutoff and inside the forecast horizon",
                    )
                )
            if event.observed_at < last_observed:
                issues.append(
                    ValidationIssue(f"{event_path}.observed_at", "patch is not chronological")
                )
        if event.valid_to is not None and not _aware(event.valid_to):
            issues.append(
                ValidationIssue(f"{event_path}.valid_to", "clock must include a UTC offset")
            )
        if valid_ok and _aware(event.valid_to) and event.valid_to < event.valid_from:
            issues.append(ValidationIssue(event_path, "valid_to precedes valid_from"))
        if event.schema_version != active_schema.version:
            issues.append(
                ValidationIssue(
                    f"{event_path}.schema_version",
                    f"must match active schema {active_schema.version!r}",
                )
            )

        relation = _relation_for_event(event, active_schema, event_path, issues)
        candidate_node: NodeRecord | None = None
        node_identity = _validate_node_operation_payload(
            event,
            active_schema,
            nodes_by_id,
            event_path,
            issues,
            proposal=True,
        )
        if (
            event.operation is Operation.CREATE_NODE
            and node_identity is not None
            and node_identity[0] not in nodes_by_id
            and event.observed_at is not None
        ):
            candidate_node = NodeRecord(
                node_id=node_identity[0],
                node_type=node_identity[1],
                schema_version=active_schema.version,
                observed_at=event.observed_at,
            )

        role_nodes = dict(nodes_by_id)
        if candidate_node is not None:
            role_nodes[candidate_node.node_id] = candidate_node
        _validate_event_roles(event, relation, role_nodes, event_path, issues)
        _check_json(event.payload, f"{event_path}.payload", issues)
        _validate_confidence_literals(event.payload, f"{event_path}.payload", issues)
        if event.operation not in {
            Operation.RETRACT,
            Operation.SUPERSEDE,
        } and _LIFECYCLE_TARGET_KEYS & set(event.payload):
            issues.append(
                ValidationIssue(
                    f"{event_path}.payload",
                    f"{event.operation.value} cannot carry a lifecycle target",
                )
            )
        if not event.derivation.method:
            issues.append(ValidationIssue(f"{event_path}.derivation.method", "cannot be empty"))
        _check_json(event.derivation.parameters, f"{event_path}.derivation.parameters", issues)
        _validate_confidence_literals(
            event.derivation.parameters,
            f"{event_path}.derivation.parameters",
            issues,
        )
        _require_id(event.provenance.source, f"{event_path}.provenance.source", issues)
        acquired_ok = _require_clock(
            event.provenance.acquired_at,
            f"{event_path}.provenance.acquired_at",
            issues,
        )
        if acquired_ok and observed_ok and event.provenance.acquired_at > event.observed_at:
            issues.append(
                ValidationIssue(f"{event_path}.provenance.acquired_at", "postdates observation")
            )

        for evidence_id in event.basis_refs:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                issues.append(
                    ValidationIssue(
                        f"{event_path}.basis_refs",
                        f"evidence {evidence_id!r} was not visible at cutoff",
                    )
                )
            elif evidence.observed_at is None or evidence.observed_at > request.cutoff:
                issues.append(
                    ValidationIssue(f"{event_path}.basis_refs", "references evidence after cutoff")
                )

        visible_input_ids = set(nodes_by_id) | set(evidence_by_id) | set(events_by_id)
        for input_ref in event.derivation.input_refs:
            if input_ref not in visible_input_ids:
                issues.append(
                    ValidationIssue(
                        f"{event_path}.derivation.input_refs",
                        f"dangling proposal input {input_ref!r}",
                    )
                )

        replacement_key: tuple[Any, ...] | None = None
        lifecycle_target_id: str | None = None
        lifecycle_target_key: tuple[Any, ...] | None = None
        proposed_schema: SchemaDecl | None = None
        if event.operation is Operation.ASSERT:
            replacement_key = _assertion_key(event)
            previous = (
                active_event_by_key.get(replacement_key) if replacement_key is not None else None
            )
            if previous is not None:
                issues.append(
                    ValidationIssue(
                        event_path,
                        f"duplicate assertion; active semantic duplicate of event {previous!r}",
                    )
                )
        if event.operation in {Operation.RETRACT, Operation.SUPERSEDE}:
            lifecycle_target_id = _lifecycle_target_id(event, event_path, issues)
            target = events_by_id.get(lifecycle_target_id or "")
            if target is None:
                issues.append(
                    ValidationIssue(
                        f"{event_path}.payload",
                        "operation requires a visible prefix or earlier patch event",
                    )
                )
            else:
                if (
                    target.observed_at is None
                    or event.observed_at is None
                    or target.observed_at >= event.observed_at
                ):
                    issues.append(
                        ValidationIssue(
                            f"{event_path}.payload",
                            "lifecycle target must be observed strictly before this event",
                        )
                    )
                lifecycle_target_key = active_key_by_event.get(lifecycle_target_id or "")
                if lifecycle_target_key is None:
                    issues.append(
                        ValidationIssue(
                            f"{event_path}.payload",
                            "lifecycle target must be an active ASSERT or SUPERSEDE event",
                        )
                    )
                if event.operation is Operation.RETRACT and not _same_relation_assertion(
                    event, target
                ):
                    issues.append(
                        ValidationIssue(
                            event_path,
                            "RETRACT relation and arguments must match its target",
                        )
                    )
            if event.operation is Operation.SUPERSEDE:
                replacement_key = _assertion_key(event)
                previous = (
                    active_event_by_key.get(replacement_key)
                    if replacement_key is not None
                    else None
                )
                if previous is not None and previous != lifecycle_target_id:
                    issues.append(
                        ValidationIssue(
                            event_path,
                            f"supersession duplicates active assertion {previous!r}",
                        )
                    )
        if event.operation is Operation.SCHEMA_CHANGE:
            proposed_schema = _proposal_schema_decl(
                event,
                active_schema,
                known_schema_versions,
                event_path,
                issues,
            )

        event_is_valid = len(issues) == event_issue_start
        # State advances only after checking the event.  This is a private proposal
        # view; the request prefix remains immutable.
        if event_is_valid:
            if lifecycle_target_key is not None and lifecycle_target_id is not None:
                active_key_by_event.pop(lifecycle_target_id, None)
                active_event_by_key.pop(lifecycle_target_key, None)
            if replacement_key is not None:
                active_key_by_event[event.event_id] = replacement_key
                active_event_by_key[replacement_key] = event.event_id
            if candidate_node is not None:
                nodes_by_id[candidate_node.node_id] = candidate_node
            events_by_id[event.event_id] = event
            if proposed_schema is not None:
                active_schema = proposed_schema
                known_schema_versions.add(proposed_schema.version)

        if observed_ok:
            assert event.observed_at is not None
            last_observed = max(last_observed, event.observed_at)
    return tuple(issues)


def validate_proposed_patch(request: ForecastRequest, events: Iterable[GraphEventV2]) -> None:
    validate_forecast_request(request)
    issues = collect_proposed_patch_issues(request, tuple(events))
    if issues:
        raise ContractValidationError(issues)


def _proposed_event_identity(event: GraphEventV2) -> str:
    """Hash proposal semantics independently of its transport/event identifier."""

    return replace(event, event_id="proposal:identity-placeholder").canonical_hash()


def validate_forecast_distribution(
    distribution: ForecastDistribution,
    request: ForecastRequest | None = None,
) -> None:
    if request is not None:
        validate_forecast_request(request)
    issues: list[ValidationIssue] = []
    if not 0.0 <= distribution.calibrated_uncertainty <= 1.0:
        issues.append(ValidationIssue("distribution.calibrated_uncertainty", "must be in [0, 1]"))
    if not 0.0 <= distribution.retrieval_coverage <= 1.0:
        issues.append(ValidationIssue("distribution.retrieval_coverage", "must be in [0, 1]"))
    window_probability_mass = math.fsum(
        window.probability for window in distribution.sampled_patch_windows
    )
    if window_probability_mass > 1.0:
        issues.append(
            ValidationIssue(
                "distribution.sampled_patch_windows",
                "window probability mass must not exceed 1",
            )
        )
    for index, window in enumerate(distribution.sampled_patch_windows):
        if not 0.0 <= window.probability <= 1.0:
            issues.append(
                ValidationIssue(
                    f"distribution.sampled_patch_windows[{index}].probability",
                    "must be in [0, 1]",
                )
            )
        if request is not None:
            issues.extend(
                collect_proposed_patch_issues(
                    request,
                    window.events,
                    path=f"distribution.sampled_patch_windows[{index}]",
                )
            )
    marginal_event_ids: set[str] = set()
    marginal_identities: set[str] = set()
    marginal_events_by_id: dict[str, str] = {}
    for index, marginal in enumerate(distribution.event_marginals):
        if not 0.0 <= marginal.probability <= 1.0:
            issues.append(
                ValidationIssue(
                    f"distribution.event_marginals[{index}].probability",
                    "must be in [0, 1]",
                )
            )
        event = marginal.event
        if event.event_id in marginal_event_ids:
            issues.append(
                ValidationIssue(
                    f"distribution.event_marginals[{index}].event.event_id",
                    "duplicate marginal event ID",
                )
            )
        else:
            marginal_event_ids.add(event.event_id)
        identity = _proposed_event_identity(event)
        if identity in marginal_identities:
            issues.append(
                ValidationIssue(
                    f"distribution.event_marginals[{index}].event",
                    "duplicate marginal event identity",
                )
            )
        else:
            marginal_identities.add(identity)
        event_hash = event.canonical_hash()
        previous = marginal_events_by_id.get(event.event_id)
        if previous is not None and previous != event_hash:
            issues.append(
                ValidationIssue(
                    f"distribution.event_marginals[{index}].event.event_id",
                    "event ID identifies conflicting proposals",
                )
            )
        else:
            marginal_events_by_id[event.event_id] = event_hash
        if request is not None:
            issues.extend(
                collect_proposed_patch_issues(
                    request,
                    (event,),
                    path=f"distribution.event_marginals[{index}]",
                )
            )
    if distribution.abstention_reason is not None and (
        distribution.sampled_patch_windows or distribution.event_marginals
    ):
        issues.append(
            ValidationIssue(
                "distribution.abstention_reason",
                "an abstention cannot contain proposed events",
            )
        )
    if issues:
        raise ContractValidationError(issues)
