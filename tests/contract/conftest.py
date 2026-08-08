from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

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
    SupportExample,
)


@pytest.fixture
def times() -> tuple[datetime, ...]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return tuple(start + timedelta(days=offset) for offset in range(8))


@pytest.fixture
def schema(times: tuple[datetime, ...]) -> SchemaDecl:
    return SchemaDecl(
        version="2.0.0",
        observed_at=times[0],
        node_types=(
            NodeTypeDecl("Person", description="A contributor"),
            NodeTypeDecl("Repository", description="A software repository"),
        ),
        relations=(
            RelationDecl(
                "CONTRIBUTES_TO",
                roles=(
                    RoleDecl("actor", ("Person",), min_count=1, max_count=1),
                    RoleDecl("target", ("Repository",), min_count=1, max_count=1),
                ),
                inverse="HAS_CONTRIBUTOR",
            ),
            RelationDecl(
                "HAS_CONTRIBUTOR",
                roles=(
                    RoleDecl("target", ("Repository",), min_count=1, max_count=1),
                    RoleDecl("actor", ("Person",), min_count=1, max_count=1),
                ),
                inverse="CONTRIBUTES_TO",
            ),
        ),
        constraints={"closed_world": False},
        support_examples=(
            SupportExample(
                relation="CONTRIBUTES_TO",
                role_types={"actor": "Person", "target": "Repository"},
                text="A person contributes to a repository.",
            ),
        ),
    )


@pytest.fixture
def valid_bundle(times: tuple[datetime, ...], schema: SchemaDecl) -> GraphBundleV2:
    evidence = EvidenceRecord(
        evidence_id="evidence:commit-1",
        observed_at=times[0],
        kind="commit",
        source_uri="https://example.test/commit/1",
        content_hash="a" * 64,
        payload={"summary": "Initial commit"},
    )
    person = NodeRecord(
        node_id="person:ada",
        node_type="Person",
        schema_version=schema.version,
        observed_at=times[0],
        payload_history=(
            PayloadSnapshot(
                observed_at=times[0],
                valid_from=times[0],
                payload={"name": "Ada"},
                basis_refs=(evidence.evidence_id,),
            ),
        ),
        evidence_ids=(evidence.evidence_id,),
    )
    repository = NodeRecord(
        node_id="repo:braid",
        node_type="Repository",
        schema_version=schema.version,
        observed_at=times[0],
        payload_history=(
            PayloadSnapshot(
                observed_at=times[0],
                valid_from=times[0],
                payload={"name": "braid"},
            ),
        ),
    )
    event = GraphEventV2(
        event_id="event:contribution-1",
        observed_at=times[1],
        valid_from=times[1],
        valid_to=None,
        operation=Operation.ASSERT,
        schema_version=schema.version,
        relation="CONTRIBUTES_TO",
        arguments=(
            RoleBinding("actor", person.node_id),
            RoleBinding("target", repository.node_id),
        ),
        payload={"commit_count": 1},
        basis_refs=(evidence.evidence_id,),
        derivation=DerivationRecord("github-import", input_refs=(evidence.evidence_id,)),
        provenance=ProvenanceRecord(
            source="github",
            source_record_id="commit-1",
            license="MIT",
            acquired_at=times[1],
        ),
        tie_group="day-1",
    )
    return GraphBundleV2(
        bundle_id="bundle:valid",
        schemas=(schema,),
        nodes=(person, repository),
        evidence=(evidence,),
        events=(event,),
    )
