from dataclasses import replace

import pytest

from braid.bench.data import (
    DatasetManifest,
    FitMarker,
    HierarchyLevel,
    RawObject,
    RedistributionPolicy,
    SourceLicense,
    SplitAssignment,
    SplitManifest,
    SplitName,
    SplitRecord,
    validate_split_hierarchy,
)
from braid.bench.leakage import AuditRecord, LeakageKind, audit_leakage
from braid.bench.specs import BenchmarkTrack, Population, validate_population

HASH = "a" * 64


def organizations(count: int = 10, each: int = 500) -> tuple[tuple[str, int], ...]:
    return tuple((f"org-{index}", each) for index in range(count))


def test_track_population_minimums_and_organization_cap() -> None:
    passing = validate_population(
        BenchmarkTrack.UNSEEN_ENTITIES,
        Population(5_000, repository_clusters=30, organization_counts=organizations()),
    )
    assert passing.passed
    assert passing.largest_organization_share == 0.1

    concentrated = validate_population(
        BenchmarkTrack.UNSEEN_ENTITIES,
        Population(
            5_000,
            repository_clusters=30,
            organization_counts=(("giant", 751), ("rest", 4_249)),
        ),
    )
    assert not concentrated.passed
    assert any("exceeds" in failure for failure in concentrated.failures)


@pytest.mark.parametrize(
    ("track", "population"),
    [
        (
            BenchmarkTrack.UNSEEN_ENTITIES,
            Population(5_000, repository_clusters=29, organization_counts=organizations()),
        ),
        (
            BenchmarkTrack.UNSEEN_RELATIONS,
            Population(5_000, held_out_relations=11, organization_counts=organizations()),
        ),
        (
            BenchmarkTrack.UNSEEN_SCHEMAS,
            Population(5_000, schema_families=5, organization_counts=organizations()),
        ),
        (
            BenchmarkTrack.TEMPORAL_TRANSFER,
            Population(
                4_999,
                repository_clusters=30,
                organization_counts=(("a", 4_999),),
            ),
        ),
    ],
)
def test_each_track_rejects_insufficient_population(track, population) -> None:
    assert not validate_population(track, population).passed


def record(record_id: str, repository: str, observed_at: str) -> SplitRecord:
    return SplitRecord(record_id, "schema", "org", f"copy-{repository}", repository, observed_at)


def test_categorical_split_prevents_repository_crossing() -> None:
    records = (
        record("a", "repo", "2025-01-01T00:00:00Z"),
        record("b", "repo", "2025-02-01T00:00:00Z"),
    )
    assignments = (
        SplitAssignment("a", SplitName.TRAIN),
        SplitAssignment("b", SplitName.PUBLIC_TEST),
    )
    result = validate_split_hierarchy(
        records, assignments, held_out_level=HierarchyLevel.REPOSITORY
    )
    assert not result.passed


def test_chronological_split_accepts_order_and_rejects_future_to_train() -> None:
    records = (
        record("a", "repo", "2025-01-01T00:00:00Z"),
        record("b", "repo", "2025-02-01T00:00:00Z"),
    )
    correct = (
        SplitAssignment("a", SplitName.TRAIN),
        SplitAssignment("b", SplitName.PUBLIC_TEST),
    )
    assert validate_split_hierarchy(
        records, correct, held_out_level=HierarchyLevel.CHRONOLOGICAL_BLOCK
    ).passed
    reversed_assignment = (
        SplitAssignment("a", SplitName.PUBLIC_TEST),
        SplitAssignment("b", SplitName.TRAIN),
    )
    assert not validate_split_hierarchy(
        records, reversed_assignment, held_out_level=HierarchyLevel.CHRONOLOGICAL_BLOCK
    ).passed


def test_split_manifest_is_content_addressed_and_validated() -> None:
    records = (
        record("a", "repo", "2025-01-01T00:00:00Z"),
        record("b", "repo", "2025-02-01T00:00:00Z"),
    )
    assignments = (
        SplitAssignment("a", SplitName.TRAIN),
        SplitAssignment("b", SplitName.PUBLIC_TEST),
    )
    manifest = SplitManifest(
        "dataset",
        "1",
        "2025-03-01T00:00:00Z",
        HierarchyLevel.CHRONOLOGICAL_BLOCK,
        records,
        assignments,
    )
    assert (
        manifest.sha256
        == SplitManifest(
            "dataset",
            "1",
            "2025-03-01T00:00:00Z",
            HierarchyLevel.CHRONOLOGICAL_BLOCK,
            records,
            assignments,
        ).sha256
    )


def audit_record(
    record_id: str,
    split: SplitName,
    *,
    content_hash: str,
    cluster: str,
    text: str,
    subgraph: str,
    relation: str | None = None,
    source: str | None = None,
    target: str | None = None,
) -> AuditRecord:
    return AuditRecord(
        record_id,
        split,
        content_hash,
        cluster,
        text,
        subgraph,
        relation,
        source,
        target,
    )


def test_leakage_audit_finds_every_required_cross_split_channel() -> None:
    train = audit_record(
        "train",
        SplitName.TRAIN,
        content_hash="same",
        cluster="forks",
        text="Fix THE parser!",
        subgraph="motif",
        relation="parent_of",
        source="a",
        target="b",
    )
    test = audit_record(
        "test",
        SplitName.PUBLIC_TEST,
        content_hash="same",
        cluster="forks",
        text="fix the parser",
        subgraph="motif",
        relation="child_of",
        source="b",
        target="a",
    )
    bad_marker = FitMarker(
        "tokenizer",
        HASH,
        HASH,
        (SplitName.TRAIN, SplitName.DEVELOPMENT),
        HASH,
    )
    report = audit_leakage(
        (train, test),
        inverse_relations={"parent_of": "child_of", "child_of": "parent_of"},
        fit_markers=(bad_marker,),
    )
    assert not report.passed
    assert {finding.kind for finding in report.findings} == set(LeakageKind)


def test_clean_audit_and_train_only_marker_pass() -> None:
    marker = FitMarker("tokenizer", HASH, HASH, (SplitName.TRAIN,), HASH)
    records = (
        audit_record("a", SplitName.TRAIN, content_hash="a", cluster="a", text="one", subgraph="a"),
        audit_record(
            "b",
            SplitName.PUBLIC_TEST,
            content_hash="b",
            cluster="b",
            text="two",
            subgraph="b",
        ),
    )
    assert audit_leakage(records, fit_markers=(marker,)).passed


def test_global_dataset_rejects_alluvia_private_data() -> None:
    license = SourceLicense(
        "Apache-2.0",
        "https://example.test/terms",
        "2025-01-01T00:00:00Z",
        RedistributionPolicy.FULL,
        "Example",
        HASH,
        ("machine-learning-research", "model-training"),
    )
    with pytest.raises(ValueError, match="Alluvia-private"):
        DatasetManifest(
            "dataset",
            "1",
            "2025-01-02T00:00:00Z",
            license,
            (RawObject("https://example.test/data", HASH, 1),),
            (),
            ("schema",),
            ("org",),
            ("repo",),
            HASH,
            "retain while licensed",
            "delete on revocation",
            "public-project metadata",
            True,
            True,
        )


def test_source_text_publication_requires_affirmative_rights() -> None:
    license = SourceLicense(
        "source-terms",
        "https://example.test/terms",
        "2025-01-01T00:00:00Z",
        RedistributionPolicy.MANIFEST_ONLY,
        "Example",
        HASH,
        ("machine-learning-research", "model-training"),
    )
    with pytest.raises(ValueError, match="redistribution rights"):
        DatasetManifest(
            "dataset",
            "1",
            "2025-01-02T00:00:00Z",
            license,
            (RawObject("https://example.test/data", HASH, 1),),
            (),
            ("schema",),
            ("org",),
            ("repo",),
            HASH,
            "retain while licensed",
            "delete on revocation",
            "public-project metadata",
            True,
            False,
            True,
        )


def test_source_license_requires_hashed_terms_and_training_rights() -> None:
    with pytest.raises(ValueError, match="model training"):
        SourceLicense(
            "research-only",
            "https://example.test/terms",
            "2025-01-01T00:00:00Z",
            RedistributionPolicy.MANIFEST_ONLY,
            "Example",
            HASH,
            ("machine-learning-research",),
        )


def test_manifest_schema_rejects_implicit_container_and_enum_coercions() -> None:
    with pytest.raises(TypeError, match="allowed_uses must be a tuple"):
        SourceLicense(
            "Apache-2.0",
            "https://example.test/terms",
            "2025-01-01T00:00:00Z",
            RedistributionPolicy.FULL,
            "Example",
            HASH,
            ["machine-learning-research", "model-training"],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="must be a SplitName"):
        SplitAssignment("record", "train")  # type: ignore[arg-type]

    valid_record = record("record", "repo", "2025-01-01T00:00:00Z")
    valid_assignment = SplitAssignment("record", SplitName.TRAIN)
    with pytest.raises(TypeError, match="tuple of SplitRecord"):
        SplitManifest(
            "dataset",
            "1",
            "2025-01-02T00:00:00Z",
            HierarchyLevel.REPOSITORY,
            [valid_record],  # type: ignore[arg-type]
            (valid_assignment,),
        )


def test_fit_marker_is_immutable() -> None:
    marker = FitMarker("tokenizer", HASH, HASH, (SplitName.TRAIN,), HASH)
    with pytest.raises(AttributeError):
        marker.artifact_name = "changed"
    assert replace(marker, artifact_name="index").artifact_name == "index"
