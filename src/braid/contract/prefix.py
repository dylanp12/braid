"""Causal as-of slicing for graph bundles."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from .serde import canonical_hash
from .types import DerivationRecord, GraphBundleV2
from .validate import ContractValidationError, ValidationIssue, validate_bundle


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _cutoff_text(cutoff: datetime) -> str:
    return cutoff.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_as_of_prefix(
    bundle: GraphBundleV2,
    cutoff: datetime,
    *,
    bundle_id: str | None = None,
    validate: bool = True,
) -> GraphBundleV2:
    """Return only information causally observable at ``cutoff``.

    Slicing happens before validation on purpose.  A future append that is malformed
    or reuses an old identifier cannot influence an otherwise valid historical
    prefix.  The resulting prefix itself is then validated by default.
    """

    if not _is_aware(cutoff):
        raise ContractValidationError(
            (ValidationIssue("cutoff", "clock must include a UTC offset"),)
        )

    schemas = tuple(
        schema
        for schema in bundle.schemas
        if schema.observed_at is not None
        and _is_aware(schema.observed_at)
        and schema.observed_at <= cutoff
    )
    visible_schema_versions = {schema.version for schema in schemas}

    evidence = tuple(
        record
        for record in bundle.evidence
        if record.observed_at is not None
        and _is_aware(record.observed_at)
        and record.observed_at <= cutoff
    )
    visible_evidence_ids = {record.evidence_id for record in evidence}
    evidence = tuple(
        replace(
            record,
            parent_refs=tuple(ref for ref in record.parent_refs if ref in visible_evidence_ids),
        )
        for record in evidence
    )

    nodes = []
    for node in bundle.nodes:
        if (
            node.observed_at is None
            or not _is_aware(node.observed_at)
            or node.observed_at > cutoff
            or node.schema_version not in visible_schema_versions
        ):
            continue
        snapshots = tuple(
            replace(
                snapshot,
                basis_refs=tuple(ref for ref in snapshot.basis_refs if ref in visible_evidence_ids),
            )
            for snapshot in node.payload_history
            if snapshot.observed_at is not None
            and _is_aware(snapshot.observed_at)
            and snapshot.observed_at <= cutoff
        )
        nodes.append(
            replace(
                node,
                payload_history=snapshots,
                evidence_ids=tuple(ref for ref in node.evidence_ids if ref in visible_evidence_ids),
            )
        )
    visible_node_ids = {node.node_id for node in nodes}

    events = tuple(
        event
        for event in bundle.events
        if event.observed_at is not None
        and _is_aware(event.observed_at)
        and event.observed_at <= cutoff
        and event.schema_version in visible_schema_versions
        and all(binding.node_id in visible_node_ids for binding in event.arguments)
    )
    visible_event_ids = {event.event_id for event in events}
    visible_input_ids = visible_evidence_ids | visible_node_ids | visible_event_ids
    events = tuple(
        replace(
            event,
            basis_refs=tuple(ref for ref in event.basis_refs if ref in visible_evidence_ids),
            derivation=DerivationRecord(
                method=event.derivation.method,
                input_refs=tuple(
                    ref for ref in event.derivation.input_refs if ref in visible_input_ids
                ),
                parameters=event.derivation.parameters,
            ),
        )
        for event in events
    )

    visible_targets = (
        {("schema", schema.version) for schema in schemas}
        | {("node", node.node_id) for node in nodes}
        | {("evidence", record.evidence_id) for record in evidence}
        | {("event", event.event_id) for event in events}
    )
    lineage = tuple(
        lineage
        for lineage in bundle.lineage
        if lineage.observed_at is not None
        and _is_aware(lineage.observed_at)
        and lineage.observed_at <= cutoff
        and (
            (lineage.target_kind, lineage.target_id) in visible_targets
            or (lineage.target_kind == "bundle" and lineage.target_id == bundle.bundle_id)
        )
    )
    prefix_material = {
        "cutoff": _cutoff_text(cutoff),
        "schemas": schemas,
        "nodes": nodes,
        "evidence": evidence,
        "events": events,
        "lineage": tuple(item for item in lineage if item.target_kind != "bundle"),
    }
    prefix_id = bundle_id or f"v2:prefix:{canonical_hash(prefix_material)}"
    # A bundle-lineage target must point at the new bundle to remain referentially
    # valid, while retaining the source digest and converter identity.
    lineage = tuple(
        replace(lineage, target_id=prefix_id)
        if lineage.target_kind == "bundle" and lineage.target_id == bundle.bundle_id
        else lineage
        for lineage in lineage
    )

    prefix = GraphBundleV2(
        bundle_id=prefix_id,
        schemas=schemas,
        nodes=tuple(nodes),
        evidence=evidence,
        events=events,
        lineage=lineage,
    )
    if validate:
        validate_bundle(prefix)
    return prefix


def prefixes_are_invariant(left: GraphBundleV2, right: GraphBundleV2, cutoff: datetime) -> bool:
    """Compare two source bundles only through information visible at ``cutoff``."""

    left_prefix = build_as_of_prefix(left, cutoff, bundle_id="comparison-prefix")
    right_prefix = build_as_of_prefix(right, cutoff, bundle_id="comparison-prefix")
    return left_prefix.canonical_hash() == right_prefix.canonical_hash()
