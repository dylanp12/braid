"""Deterministic marked-process generation with dual clocks and schema evolution."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from braid.contract import (
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
    validate_bundle,
)


@dataclass(frozen=True, slots=True)
class ProcessGeneratorConfig:
    seed: int = 0
    actors: int = 8
    work_items: int = 24
    decisions: int = 8
    relation_events: int = 96
    mean_interarrival_seconds: float = 3_600.0
    max_valid_lag_seconds: int = 86_400
    retraction_probability: float = 0.12
    supersede_probability: float = 0.08
    schema_change_fraction: float = 0.55
    tie_probability: float = 0.08
    censor_fraction: float = 0.8

    def __post_init__(self) -> None:
        if min(self.actors, self.work_items, self.decisions, self.relation_events) <= 0:
            raise ValueError("entity and event counts must be positive")
        if self.mean_interarrival_seconds <= 0:
            raise ValueError("mean_interarrival_seconds must be positive")
        for name in (
            "retraction_probability",
            "supersede_probability",
            "schema_change_fraction",
            "tie_probability",
            "censor_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GeneratedWorld:
    bundle: GraphBundleV2
    censor_at: datetime
    seed: int
    attempted_relation_events: int
    emitted_relation_events: int
    duplicate_assertions_skipped: int


def _id(seed: int, kind: str, counter: int, *parts: object) -> str:
    material = "\0".join(str(part) for part in (seed, kind, counter, *parts))
    suffix = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"syn:{seed}:{kind}:{counter}:{suffix}"


def _schema_v1(observed_at: datetime) -> SchemaDecl:
    return SchemaDecl(
        version="synthetic-v1",
        observed_at=observed_at,
        node_types=(
            NodeTypeDecl("Actor", "person or automation participating in work"),
            NodeTypeDecl("WorkItem", "issue, task, or problem"),
            NodeTypeDecl("Decision", "recorded choice or resolution"),
        ),
        relations=(
            RelationDecl(
                "works_on",
                (
                    RoleDecl("actor", ("Actor",)),
                    RoleDecl("item", ("WorkItem",)),
                ),
                "an actor performs work on an item",
            ),
            RelationDecl(
                "resolves",
                (
                    RoleDecl("decision", ("Decision",)),
                    RoleDecl("item", ("WorkItem",)),
                ),
                "a decision resolves a work item",
            ),
            RelationDecl(
                "depends_on",
                (
                    RoleDecl("source", ("WorkItem",)),
                    RoleDecl("target", ("WorkItem",)),
                ),
                "one work item depends on another",
            ),
        ),
        constraints={"generator": "marked-process-v2", "synthetic": True},
    )


def _schema_v2(observed_at: datetime, previous: SchemaDecl) -> SchemaDecl:
    return SchemaDecl(
        version="synthetic-v2",
        observed_at=observed_at,
        node_types=previous.node_types,
        relations=(
            *previous.relations,
            RelationDecl(
                "contradicts",
                (
                    RoleDecl("left", ("Decision",)),
                    RoleDecl("right", ("Decision",)),
                ),
                "one decision conflicts with another",
                constraints={"symmetric": True},
            ),
        ),
        constraints={
            "generator": "marked-process-v2",
            "synthetic": True,
            "previous": previous.version,
        },
    )


def _provenance(now: datetime, seed: int, source_record_id: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        source="braid.synthetic.marked-process-v2",
        source_record_id=source_record_id,
        license="CDLA-Permissive-2.0",
        acquired_at=now,
    )


def generate_world(
    config: ProcessGeneratorConfig,
    *,
    start: datetime | None = None,
) -> GeneratedWorld:
    """Generate and validate one deterministic synthetic world.

    Event IDs include a monotonic counter and both relation participants, so
    repeated observations cannot collide. Duplicate assertions are skipped
    before they reach the bundle and the skip count is exposed for audit.
    """

    rng = random.Random(config.seed)
    start = start or datetime(2025, 1, 1, tzinfo=UTC)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start must include a UTC offset")

    schema1 = _schema_v1(start)
    schemas: list[SchemaDecl] = [schema1]
    nodes: list[NodeRecord] = []
    evidence: list[EvidenceRecord] = []
    events: list[GraphEventV2] = []
    event_counter = 0

    node_specs = [
        *((f"actor-{i}", "Actor") for i in range(config.actors)),
        *((f"item-{i}", "WorkItem") for i in range(config.work_items)),
        *((f"decision-{i}", "Decision") for i in range(config.decisions)),
    ]
    for index, (node_id, node_type) in enumerate(node_specs):
        observed_at = start + timedelta(seconds=index + 1)
        evidence_id = _id(config.seed, "evidence", index, node_id)
        evidence.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                observed_at=observed_at,
                kind="synthetic-node-seed",
                content_hash=hashlib.sha256(node_id.encode()).hexdigest(),
                payload={"synthetic": True},
            )
        )
        nodes.append(
            NodeRecord(
                node_id=node_id,
                node_type=node_type,
                schema_version=schema1.version,
                observed_at=observed_at,
                payload_history=(
                    PayloadSnapshot(
                        observed_at=observed_at,
                        valid_from=observed_at,
                        payload={"title": f"Synthetic {node_type} {index}", "ordinal": index},
                        basis_refs=(evidence_id,),
                    ),
                ),
                evidence_ids=(evidence_id,),
            )
        )
        events.append(
            GraphEventV2(
                event_id=_id(
                    config.seed,
                    "event",
                    event_counter,
                    "CREATE_NODE",
                    node_id,
                    observed_at.isoformat(),
                ),
                observed_at=observed_at,
                valid_from=observed_at,
                valid_to=None,
                operation=Operation.CREATE_NODE,
                schema_version=schema1.version,
                relation=None,
                arguments=(RoleBinding("node", node_id),),
                payload={"node_id": node_id, "node_type": node_type},
                basis_refs=(evidence_id,),
                derivation=DerivationRecord(
                    "synthetic-process",
                    (evidence_id,),
                    {"seed": config.seed},
                ),
                provenance=_provenance(observed_at, config.seed, node_id),
            )
        )
        event_counter += 1

    current_time = max(node.observed_at for node in nodes if node.observed_at is not None)
    schema_change_index = max(1, int(config.relation_events * config.schema_change_fraction))
    current_schema = schema1
    asserted: dict[tuple[str, tuple[tuple[str, str], ...]], GraphEventV2] = {}
    retracted: set[str] = set()
    skipped = 0
    emitted_relations = 0
    actors = [node.node_id for node in nodes if node.node_type == "Actor"]
    items = [node.node_id for node in nodes if node.node_type == "WorkItem"]
    decisions = [node.node_id for node in nodes if node.node_type == "Decision"]

    for attempt in range(config.relation_events):
        if attempt == schema_change_index:
            current_time += timedelta(
                seconds=max(1.0, rng.expovariate(1 / config.mean_interarrival_seconds))
            )
            previous_schema = current_schema
            current_schema = _schema_v2(current_time, schema1)
            schemas.append(current_schema)
            events.append(
                GraphEventV2(
                    event_id=_id(
                        config.seed,
                        "event",
                        event_counter,
                        "SCHEMA_CHANGE",
                        current_time.isoformat(),
                    ),
                    observed_at=current_time,
                    valid_from=current_time,
                    valid_to=None,
                    operation=Operation.SCHEMA_CHANGE,
                    schema_version=previous_schema.version,
                    relation=None,
                    arguments=(),
                    payload={
                        "previous_schema_version": previous_schema.version,
                        "new_schema_version": current_schema.version,
                        "schema_hash": current_schema.canonical_hash(),
                        "added_relations": ["contradicts"],
                    },
                    basis_refs=(),
                    derivation=DerivationRecord("synthetic-schema-process"),
                    provenance=_provenance(current_time, config.seed, current_schema.version),
                )
            )
            event_counter += 1

        if rng.random() >= config.tie_probability or not events:
            current_time += timedelta(
                seconds=max(1.0, rng.expovariate(1 / config.mean_interarrival_seconds))
            )
        tie_group = (
            f"tie:{current_time.isoformat()}"
            if events and events[-1].observed_at == current_time
            else None
        )

        active_assertions = [
            event
            for event in asserted.values()
            if event.event_id not in retracted
            and event.observed_at is not None
            and event.observed_at < current_time
        ]
        draw = rng.random()
        if active_assertions and draw < config.retraction_probability:
            target = rng.choice(active_assertions)
            operation = Operation.RETRACT
            relation = target.relation
            arguments = target.arguments
            payload = {"retracts_event_id": target.event_id}
            retracted.add(target.event_id)
            derivation_inputs = (target.event_id,)
        elif (
            active_assertions
            and draw < config.retraction_probability + config.supersede_probability
        ):
            target = rng.choice(active_assertions)
            operation = Operation.SUPERSEDE
            relation = target.relation
            arguments = target.arguments
            payload = {"supersedes_event_id": target.event_id}
            retracted.add(target.event_id)
            derivation_inputs = (target.event_id,)
        else:
            operation = Operation.ASSERT
            choices = ["works_on", "resolves", "depends_on"]
            if current_schema.version == "synthetic-v2":
                choices.append("contradicts")
            relation = rng.choice(choices)
            if relation == "works_on":
                arguments = (
                    RoleBinding("actor", rng.choice(actors)),
                    RoleBinding("item", rng.choice(items)),
                )
            elif relation == "resolves":
                arguments = (
                    RoleBinding("decision", rng.choice(decisions)),
                    RoleBinding("item", rng.choice(items)),
                )
            elif relation == "depends_on":
                source, target_id = rng.sample(items, 2)
                arguments = (RoleBinding("source", source), RoleBinding("target", target_id))
            else:
                left, right = rng.sample(decisions, 2)
                arguments = (RoleBinding("left", left), RoleBinding("right", right))
            signature = (
                relation,
                tuple(sorted((binding.role, binding.node_id) for binding in arguments)),
            )
            if signature in asserted:
                skipped += 1
                continue
            payload = {"synthetic": True, "attempt": attempt}
            derivation_inputs = tuple(binding.node_id for binding in arguments)

        evidence_id = _id(
            config.seed,
            "evidence",
            len(evidence),
            relation,
            current_time.isoformat(),
        )
        evidence.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                observed_at=current_time,
                kind="synthetic-process-observation",
                content_hash=hashlib.sha256(f"{relation}:{attempt}".encode()).hexdigest(),
                payload={"attempt": attempt, "operation": operation.value},
            )
        )
        valid_lag = rng.randint(0, config.max_valid_lag_seconds)
        valid_from = current_time - timedelta(seconds=valid_lag)
        event = GraphEventV2(
            event_id=_id(
                config.seed,
                "event",
                event_counter,
                operation.value,
                relation,
                current_time.isoformat(),
                *(binding.node_id for binding in arguments),
            ),
            observed_at=current_time,
            valid_from=valid_from,
            valid_to=(
                current_time if operation in {Operation.RETRACT, Operation.SUPERSEDE} else None
            ),
            operation=operation,
            schema_version=current_schema.version,
            relation=relation,
            arguments=arguments,
            payload=payload,
            basis_refs=(evidence_id,),
            derivation=DerivationRecord(
                "synthetic-marked-process",
                derivation_inputs,
                {"seed": config.seed},
            ),
            provenance=_provenance(current_time, config.seed, f"attempt-{attempt}"),
            tie_group=tie_group,
        )
        events.append(event)
        event_counter += 1
        emitted_relations += 1
        if operation is Operation.ASSERT:
            key = (
                relation,
                tuple(sorted((binding.role, binding.node_id) for binding in arguments)),
            )
            asserted[key] = event

    events.sort(key=lambda event: (event.observed_at, event.tie_group or "", event.event_id))
    bundle = GraphBundleV2(
        bundle_id=f"synthetic-world-{config.seed}",
        schemas=tuple(schemas),
        nodes=tuple(nodes),
        evidence=tuple(evidence),
        events=tuple(events),
    )
    validate_bundle(bundle)
    final_observation = max(event.observed_at for event in events if event.observed_at is not None)
    censor_at = start + (final_observation - start) * config.censor_fraction
    return GeneratedWorld(
        bundle=bundle,
        censor_at=censor_at,
        seed=config.seed,
        attempted_relation_events=config.relation_events,
        emitted_relation_events=emitted_relations,
        duplicate_assertions_skipped=skipped,
    )
