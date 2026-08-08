"""Finite-state constraints for proposed EventGraph patches.

The grammar operates on episode-local integer handles.  It validates a temporary
decode state and never mutates the source graph supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class Operation(StrEnum):
    CREATE_NODE = "CREATE_NODE"
    UPDATE_NODE = "UPDATE_NODE"
    ASSERT = "ASSERT"
    RETRACT = "RETRACT"
    SUPERSEDE = "SUPERSEDE"
    SCHEMA_CHANGE = "SCHEMA_CHANGE"
    EXPOSE = "EXPOSE"
    JUDGE = "JUDGE"


@dataclass(frozen=True, slots=True)
class RoleRule:
    handle: int
    target_types: frozenset[int] | int
    min_count: int = 1
    max_count: int | None = 1

    def __post_init__(self) -> None:
        targets = (
            frozenset({self.target_types})
            if isinstance(self.target_types, int)
            else frozenset(self.target_types)
        )
        object.__setattr__(self, "target_types", targets)
        if self.handle < 0 or not targets or any(target < 0 for target in targets):
            raise ValueError("role and type handles must be non-negative")
        if self.min_count < 0:
            raise ValueError("min_count cannot be negative")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count cannot be less than min_count")


@dataclass(frozen=True, slots=True)
class RelationRule:
    handle: int
    roles: tuple[RoleRule, ...]

    def __post_init__(self) -> None:
        role_handles = [role.handle for role in self.roles]
        if len(role_handles) != len(set(role_handles)):
            raise ValueError("relation role handles must be unique")


@dataclass(frozen=True, slots=True)
class GrammarSchema:
    version: str
    type_handles: frozenset[int]
    relations: Mapping[int, RelationRule]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", MappingProxyType(dict(self.relations)))
        if set(self.relations) != {rule.handle for rule in self.relations.values()}:
            raise ValueError("relation mapping keys must match relation handles")
        for relation in self.relations.values():
            for role in relation.roles:
                if not role.target_types.issubset(self.type_handles):
                    raise ValueError(f"role {role.handle} references an unknown type")


@dataclass(frozen=True, slots=True)
class VisibleNode:
    handle: int
    type_handle: int
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class VisibleEvidence:
    handle: int
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class DecodeContext:
    cutoff: datetime
    horizon_end: datetime
    schema: GrammarSchema
    nodes: Mapping[int, VisibleNode]
    evidence: Mapping[int, VisibleEvidence]
    event_handles: frozenset[int] = frozenset()
    assertion_keys: frozenset[tuple[Any, ...]] = frozenset()

    def __post_init__(self) -> None:
        if self.horizon_end < self.cutoff:
            raise ValueError("horizon_end cannot precede cutoff")
        if any(node.observed_at > self.cutoff for node in self.nodes.values()):
            raise ValueError("decode context contains a future node")
        if any(item.observed_at > self.cutoff for item in self.evidence.values()):
            raise ValueError("decode context contains future evidence")


@dataclass(frozen=True, slots=True)
class PatchEvent:
    event_handle: int
    operation: Operation | str
    observed_at: datetime
    valid_from: datetime
    schema_version: str
    valid_to: datetime | None = None
    relation_handle: int | None = None
    node_handle: int | None = None
    node_type_handle: int | None = None
    arguments: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    basis_refs: tuple[int, ...] = ()
    target_event_handle: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", Operation(self.operation))
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType({int(key): tuple(value) for key, value in self.arguments.items()}),
        )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class GrammarViolation:
    code: str
    message: str
    event_index: int | None = None


class GraphPatchGrammar:
    """Validate complete patches and expose operation-specific decode fields."""

    _FIELDS: dict[Operation, tuple[str, ...]] = {
        Operation.CREATE_NODE: ("node_handle", "node_type_handle", "payload"),
        Operation.UPDATE_NODE: ("node_handle", "payload"),
        Operation.ASSERT: ("relation_handle", "arguments", "basis_refs"),
        Operation.RETRACT: ("target_event_handle", "basis_refs"),
        Operation.SUPERSEDE: ("target_event_handle", "relation_handle", "arguments", "basis_refs"),
        Operation.SCHEMA_CHANGE: ("payload", "basis_refs"),
        Operation.EXPOSE: ("node_handle", "payload"),
        Operation.JUDGE: ("node_handle", "payload", "basis_refs"),
    }

    def allowed_operations(self) -> tuple[Operation, ...]:
        return tuple(Operation)

    def fields_for(self, operation: Operation | str) -> tuple[str, ...]:
        return self._FIELDS[Operation(operation)]

    def validate_patch(
        self, context: DecodeContext, events: Iterable[PatchEvent]
    ) -> tuple[GrammarViolation, ...]:
        """Validate against a private decode state, leaving ``context`` unchanged."""

        nodes = dict(context.nodes)
        event_handles = set(context.event_handles)
        assertion_keys = set(context.assertion_keys)
        violations: list[GrammarViolation] = []
        last_observed = context.cutoff

        for index, event in enumerate(events):
            current = self._validate_event(
                context,
                event,
                nodes=nodes,
                event_handles=event_handles,
                assertion_keys=assertion_keys,
                last_observed=last_observed,
            )
            violations.extend(GrammarViolation(item.code, item.message, index) for item in current)
            if current:
                continue
            if event.operation is Operation.CREATE_NODE:
                assert event.node_handle is not None and event.node_type_handle is not None
                nodes[event.node_handle] = VisibleNode(
                    event.node_handle, event.node_type_handle, event.observed_at
                )
            if event.operation is Operation.ASSERT:
                assertion_keys.add(_assertion_key(event))
            event_handles.add(event.event_handle)
            last_observed = event.observed_at
        return tuple(violations)

    def require_valid(self, context: DecodeContext, events: Iterable[PatchEvent]) -> None:
        violations = self.validate_patch(context, events)
        if violations:
            summary = "; ".join(f"{item.code}: {item.message}" for item in violations)
            raise ValueError(f"invalid graph patch: {summary}")

    def _validate_event(
        self,
        context: DecodeContext,
        event: PatchEvent,
        *,
        nodes: Mapping[int, VisibleNode],
        event_handles: set[int],
        assertion_keys: set[tuple[Any, ...]],
        last_observed: datetime,
    ) -> tuple[GrammarViolation, ...]:
        errors: list[GrammarViolation] = []

        def reject(code: str, message: str) -> None:
            errors.append(GrammarViolation(code, message))

        if event.event_handle in event_handles:
            reject("duplicate_event", f"event handle {event.event_handle} already exists")
        if event.schema_version != context.schema.version:
            reject("schema_version", "event does not use the active schema version")
        if not context.cutoff < event.observed_at <= context.horizon_end:
            reject("observed_time", "observed_at must be inside the forecast horizon")
        if event.observed_at < last_observed:
            reject("event_order", "events must be nondecreasing by observed_at")
        if event.valid_to is not None and event.valid_to < event.valid_from:
            reject("valid_interval", "valid_to cannot precede valid_from")
        missing_evidence = [ref for ref in event.basis_refs if ref not in context.evidence]
        if missing_evidence:
            reject("evidence_handle", f"unknown cutoff-visible evidence: {missing_evidence}")
        else:
            future_basis = [
                ref
                for ref in event.basis_refs
                if context.evidence[ref].observed_at > context.cutoff
            ]
            if future_basis:
                reject("future_evidence", f"basis refs were not visible at cutoff: {future_basis}")

        if event.operation is Operation.CREATE_NODE:
            if event.node_handle is None or event.node_type_handle is None:
                reject("create_fields", "CREATE_NODE requires node and type handles")
            elif event.node_handle in nodes:
                reject("duplicate_node", f"node handle {event.node_handle} already exists")
            elif event.node_type_handle not in context.schema.type_handles:
                reject("node_type", f"unknown node type handle {event.node_type_handle}")
        elif event.operation is Operation.UPDATE_NODE:
            if event.node_handle not in nodes:
                reject("node_handle", "UPDATE_NODE requires an existing node")
            if not event.payload:
                reject("empty_update", "UPDATE_NODE requires a payload snapshot")
        elif event.operation in (Operation.ASSERT, Operation.SUPERSEDE):
            self._validate_assertion(context.schema, event, nodes, reject)
            if (
                event.operation is Operation.ASSERT
                and not errors
                and _assertion_key(event) in assertion_keys
            ):
                reject("duplicate_assertion", "the same assertion already exists")
            if event.operation is Operation.SUPERSEDE:
                self._validate_target(event, event_handles, reject)
        elif event.operation is Operation.RETRACT:
            self._validate_target(event, event_handles, reject)
        elif event.operation is Operation.SCHEMA_CHANGE:
            if not event.payload:
                reject("schema_payload", "SCHEMA_CHANGE requires a schema proposal payload")
        elif event.operation is Operation.EXPOSE:
            if event.node_handle not in nodes:
                reject("node_handle", "EXPOSE requires an existing node")
        elif event.operation is Operation.JUDGE:
            if event.node_handle not in nodes:
                reject("node_handle", "JUDGE requires an existing subject node")
            if not event.payload:
                reject("judge_payload", "JUDGE requires a judgment payload")
        return tuple(errors)

    @staticmethod
    def _validate_target(event: PatchEvent, event_handles: set[int], reject: Any) -> None:
        if event.target_event_handle not in event_handles:
            reject("target_event", "operation requires an existing target event")

    @staticmethod
    def _validate_assertion(
        schema: GrammarSchema,
        event: PatchEvent,
        nodes: Mapping[int, VisibleNode],
        reject: Any,
    ) -> None:
        relation = schema.relations.get(event.relation_handle)
        if relation is None:
            reject("relation_handle", f"unknown relation handle {event.relation_handle}")
            return
        rules = {role.handle: role for role in relation.roles}
        extras = set(event.arguments).difference(rules)
        if extras:
            reject("role_handle", f"roles do not belong to relation: {sorted(extras)}")
        for role_handle, rule in rules.items():
            arguments = event.arguments.get(role_handle, ())
            if len(arguments) < rule.min_count:
                reject("role_cardinality", f"role {role_handle} has too few arguments")
            if rule.max_count is not None and len(arguments) > rule.max_count:
                reject("role_cardinality", f"role {role_handle} has too many arguments")
            for node_handle in arguments:
                node = nodes.get(node_handle)
                if node is None:
                    reject("argument_handle", f"unknown node handle {node_handle}")
                elif node.type_handle not in rule.target_types:
                    reject(
                        "argument_type",
                        f"node {node_handle} has type {node.type_handle}, expected one of "
                        f"{sorted(rule.target_types)}",
                    )


def _assertion_key(event: PatchEvent) -> tuple[Any, ...]:
    arguments = tuple(
        sorted((role, tuple(sorted(nodes))) for role, nodes in event.arguments.items())
    )
    return (event.relation_handle, arguments)
