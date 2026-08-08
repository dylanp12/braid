import base64
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from braid.bench.baselines import (
    BaselineFamily,
    BaselineMetadata,
    QualificationState,
)
from braid.bench.data import (
    DatasetManifest,
    HierarchyLevel,
    RawObject,
    RedistributionPolicy,
    SourceLicense,
    SplitAssignment,
    SplitManifest,
    SplitName,
    SplitRecord,
)
from braid.bench.evidence import (
    REQUIRED_RESULT_ARTIFACTS,
    ArtifactRef,
    EvaluationSubject,
    EvidenceBundle,
    RecordedMetric,
    ResultBundleManifest,
    SealedEvaluationReceipt,
    TrustedCustodianKey,
    canonical_json,
    checkpoint_manifest_payload,
    content_hash,
    hash_bytes,
    metrics_artifact_payload,
    verify_evidence_material,
    verify_receipt_signature,
    verify_replay_outputs,
    verify_result_directory,
)
from braid.bench.gates import (
    CandidateEvidence,
    ClaimLevel,
    ClaimProhibitedError,
    EndpointEvidence,
    EvidenceDossier,
    SealedCohortEvidence,
    authorize_claim,
    candidate_gate_payload,
    cohort_gate_payload,
    evaluate_candidate_gate,
    evaluate_frontier_gate,
    strongest_permitted_claim,
    trusted_keyring_hash,
)
from braid.bench.specs import (
    TRACK_SPECS,
    BenchmarkTrack,
    MetricDirection,
    Population,
    PopulationQualification,
    TrackPopulation,
    required_metric_specs,
)
from braid.contract import ModelManifest

HASH_A = "a" * 64
CONTAINER_BYTES = b"fixed-evaluation-container-manifest"


def private_and_trusted(
    custodian: str,
    seed_byte: int,
) -> tuple[Ed25519PrivateKey, TrustedCustodianKey]:
    private = Ed25519PrivateKey.from_private_bytes(bytes([seed_byte]) * 32)
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return private, TrustedCustodianKey.from_public_key_bytes(custodian, public)


def signed_receipt(
    result: ResultBundleManifest,
    cohort_id: str,
    custodian: str,
    private: Ed25519PrivateKey,
    trusted: TrustedCustodianKey,
    *,
    passed: bool = True,
    egress_disabled: bool = True,
    evaluated_at: str = "2026-01-04T00:00:00Z",
) -> SealedEvaluationReceipt:
    values = dict(
        receipt_version="1",
        cohort_id=cohort_id,
        custodian=custodian,
        evaluated_at=evaluated_at,
        result_bundle_hash=result.sha256,
        egress_disabled=egress_disabled,
        passed=passed,
        signature_algorithm="ed25519",
        signer_key_id=trusted.signer_key_id,
    )
    signature = base64.b64encode(
        private.sign(canonical_json(SealedEvaluationReceipt.payload(**values)))
    ).decode("ascii")
    return SealedEvaluationReceipt.issue(**values, signature=signature)


def model_manifest(*, checkpoint_hash: str = HASH_A, code_hash: str = "c" * 64) -> ModelManifest:
    return ModelManifest(
        architecture_hash="0" * 64,
        tokenizer_hash="1" * 64,
        code_hash=code_hash,
        environment_hash="3" * 64,
        data_hash="4" * 64,
        split_hash="5" * 64,
        checkpoint_hash=checkpoint_hash,
        evaluation_hash="7" * 64,
        metadata={"training_steps": 1},
    )


def typed_population_manifests(
    label: str,
    *,
    population_material: str | None = None,
    split: SplitName = SplitName.SEALED_TEST,
) -> tuple[DatasetManifest, SplitManifest]:
    material = population_material or label
    record = SplitRecord(
        f"record-{label}",
        f"schema-{material}",
        f"org-{material}",
        f"copy-{material}",
        f"repo-{material}",
        "2026-01-01T00:00:00Z",
    )
    split_manifest = SplitManifest(
        f"dataset-{label}",
        "1",
        "2026-01-02T00:00:00Z",
        HierarchyLevel.REPOSITORY,
        (record,),
        (SplitAssignment(record.record_id, split),),
    )
    license = SourceLicense(
        "Apache-2.0",
        "https://example.test/terms",
        "2025-12-01T00:00:00Z",
        RedistributionPolicy.FULL,
        "Example",
        "e" * 64,
        ("machine-learning-research", "model-training"),
    )
    dataset = DatasetManifest(
        split_manifest.dataset_id,
        split_manifest.version,
        "2026-01-02T00:00:00Z",
        license,
        (
            RawObject(
                f"https://example.test/{label}",
                hash_bytes(f"raw-{material}".encode()),
                len(material),
            ),
        ),
        (),
        (record.schema_family,),
        (record.organization,),
        (record.repository,),
        split_manifest.sha256,
        "retain while licensed",
        "delete on revocation",
        "public-project metadata",
        False,
    )
    return dataset, split_manifest


def write_result(
    directory: Path,
    *,
    run_id: str,
    gate_bytes: bytes,
    dataset: DatasetManifest,
    split: SplitManifest,
    model: ModelManifest,
    metrics: tuple[RecordedMetric, ...],
    container_bytes: bytes = CONTAINER_BYTES,
) -> ResultBundleManifest:
    directory.mkdir()
    subject = EvaluationSubject(
        model.manifest_id,
        model.checkpoint_hash,
        "sha256:" + hash_bytes(container_bytes),
    )
    payloads = {name: b"{}" for name in REQUIRED_RESULT_ARTIFACTS}
    payloads.update(
        {
            "checkpoint_manifest": canonical_json(checkpoint_manifest_payload(subject)),
            "container_manifest": container_bytes,
            "dataset_manifest": canonical_json(dataset),
            "gate_evidence": gate_bytes,
            "metrics": canonical_json(metrics_artifact_payload(metrics)),
            "model_manifest": model.to_json().encode("utf-8"),
            "split_manifest": canonical_json(split),
        }
    )
    artifacts = tuple(
        ArtifactRef.from_bytes(name, "application/json", payloads[name])
        for name in sorted(payloads)
    )
    result = ResultBundleManifest(
        "1",
        run_id,
        "2026-01-03T00:00:00Z",
        model.manifest_id,
        dataset.sha256,
        split.sha256,
        model.checkpoint_hash,
        subject.container_digest,
        ("python -m braid.bench.run --config exact.json",),
        artifacts,
        metrics,
    )
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    return result


def raw_population(track: BenchmarkTrack) -> TrackPopulation:
    organizations = tuple((f"org-{index}", 500) for index in range(10))
    values = dict(queries_or_nonempty_windows=5_000, organization_counts=organizations)
    if track in (BenchmarkTrack.UNSEEN_ENTITIES, BenchmarkTrack.TEMPORAL_TRANSFER):
        values["repository_clusters"] = 30
    if track is BenchmarkTrack.UNSEEN_RELATIONS:
        values["held_out_relations"] = 12
    if track is BenchmarkTrack.UNSEEN_SCHEMAS:
        values["schema_families"] = 6
    return TrackPopulation(track, Population(**values))


def endpoint(track, metric) -> EndpointEvidence:
    if metric.direction is MetricDirection.HIGHER_IS_BETTER:
        braid_value, baseline_value = 0.53, 0.50
    else:
        braid_value, baseline_value = 0.94, 1.00
    return EndpointEvidence(
        track,
        metric.name,
        braid_value,
        "qualified-best",
        baseline_value,
        True,
        0.001,
        0.01,
        60,
        100,
        0.05,
        0.01,
        0.02,
    )


def cohort(
    root: Path,
    cohort_id: str,
    custodian: str,
    private: Ed25519PrivateKey,
    trusted: TrustedCustodianKey,
    *,
    model: ModelManifest | None = None,
    population_material: str | None = None,
    endpoint_overrides: tuple[EndpointEvidence, ...] | None = None,
    population_overrides: tuple[TrackPopulation, ...] | None = None,
    retrieval_recall: float = 0.995,
    container_bytes: bytes = CONTAINER_BYTES,
) -> SealedCohortEvidence:
    checked_model = model or model_manifest()
    dataset, split = typed_population_manifests(
        cohort_id,
        population_material=population_material,
    )
    endpoints = endpoint_overrides or tuple(
        endpoint(track, metric) for track, metric in required_metric_specs()
    )
    populations = population_overrides or tuple(raw_population(track) for track in BenchmarkTrack)
    gate_values = dict(
        endpoints=endpoints,
        populations=populations,
        retrieval_true_target_recall=retrieval_recall,
        retrieval_queries=5_000,
        leakage_audits_passed=True,
        metric_canaries_passed=True,
        official_baselines_reproduced=True,
    )
    gate_bytes = canonical_json(cohort_gate_payload(**gate_values))
    recorded = tuple(
        sorted(
            (
                RecordedMetric(
                    item.track,
                    item.metric,
                    item.braid_value,
                    item.strongest_baseline,
                    item.baseline_value,
                )
                for item in endpoints
            ),
            key=lambda item: (item.track.value, item.name),
        )
    )
    directory = root / cohort_id
    result = write_result(
        directory,
        run_id=f"run-{cohort_id}",
        gate_bytes=gate_bytes,
        dataset=dataset,
        split=split,
        model=checked_model,
        metrics=recorded,
        container_bytes=container_bytes,
    )
    receipt = signed_receipt(result, cohort_id, custodian, private, trusted)
    return SealedCohortEvidence(
        bundle=EvidenceBundle(result, (receipt,)),
        result_directory=directory,
        dataset_manifest=dataset,
        split_manifest=split,
        model_manifest=checked_model,
        **gate_values,
    )


def passing_dossier(tmp_path: Path):
    private_a, trusted_a = private_and_trusted("custodian-a", 1)
    private_b, trusted_b = private_and_trusted("custodian-b", 2)
    keys = (trusted_a, trusted_b)
    cohorts = (
        cohort(tmp_path, "cohort-1", "custodian-a", private_a, trusted_a),
        cohort(tmp_path, "cohort-2", "custodian-b", private_b, trusted_b),
    )
    dossier = EvidenceDossier(
        cohorts,
        trusted_keyring_hash(keys),
        cohorts[0].subject,
    )
    return dossier, keys


def qualified_baseline() -> BaselineMetadata:
    return BaselineMetadata(
        "ULTRA",
        BaselineFamily.EXTERNAL,
        frozenset({BenchmarkTrack.UNSEEN_ENTITIES}),
        "official adapter",
        pinned_version="verified-commit",
        state=QualificationState.QUALIFIED,
        uses_identical_inputs=True,
        within_inference_envelope=True,
        official_result_reproduced=True,
        reason="public development qualification",
    )


def candidate(tmp_path: Path) -> CandidateEvidence:
    baseline = qualified_baseline()
    gate_bytes = canonical_json(
        candidate_gate_payload(
            qualified_baselines=(baseline,),
            leakage_audits_passed=True,
            metric_canaries_passed=True,
        )
    )
    dataset, split = typed_population_manifests(
        "public-development",
        split=SplitName.DEVELOPMENT,
    )
    model = model_manifest()
    metrics = (
        RecordedMetric(
            BenchmarkTrack.UNSEEN_ENTITIES,
            "filtered_mrr",
            0.60,
            "ULTRA",
            0.55,
        ),
    )
    directory = tmp_path / "public-development"
    result = write_result(
        directory,
        run_id="public-development",
        gate_bytes=gate_bytes,
        dataset=dataset,
        split=split,
        model=model,
        metrics=metrics,
    )
    return CandidateEvidence(
        result,
        directory,
        dataset,
        split,
        model,
        (baseline,),
        True,
        True,
    )


def test_canonical_hashes_ignore_mapping_insertion_order() -> None:
    left = {"z": [3, 2, 1], "a": {"two": 2, "one": 1}}
    right = {"a": {"one": 1, "two": 2}, "z": [3, 2, 1]}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    assert content_hash(left) == "8edc2beecde317fb9cd37130dbd7efa5b53cd2a12155eecd7e88b730382299c4"


def test_evidence_numbers_reject_bool_and_int_float_substitution() -> None:
    with pytest.raises(TypeError, match="booleans are not counts"):
        ArtifactRef("artifact", "application/json", HASH_A, True)
    with pytest.raises(TypeError, match="must be a float"):
        RecordedMetric(BenchmarkTrack.UNSEEN_ENTITIES, "mrr", 1, "base", 0.5)
    with pytest.raises(TypeError, match="must be a boolean"):
        replace(endpoint(*required_metric_specs()[0]), baseline_qualified=1)
    with pytest.raises(TypeError, match="booleans are not counts"):
        Population(True)


def test_result_manifest_is_complete_deterministic_and_immutable(tmp_path: Path) -> None:
    evidence = candidate(tmp_path)
    first = evidence.public_development_result
    with pytest.raises(FrozenInstanceError):
        first.run_id = "changed"
    with pytest.raises(ValueError, match="missing required artifacts"):
        replace(first, artifacts=first.artifacts[:-1])
    with pytest.raises(ValueError, match="artifact name"):
        ArtifactRef("../escape", "application/octet-stream", HASH_A, 0)


def test_result_directory_typed_material_and_replay_are_byte_verified(tmp_path: Path) -> None:
    evidence = candidate(tmp_path)
    result = evidence.public_development_result
    assert verify_result_directory(result, evidence.result_directory).passed
    assert verify_replay_outputs(result, evidence.result_directory).passed
    assert verify_evidence_material(
        result,
        evidence.result_directory,
        dataset_manifest=evidence.dataset_manifest,
        split_manifest=evidence.split_manifest,
        model_manifest=evidence.model_manifest,
    ).passed

    (evidence.result_directory / "metrics").write_bytes(b"changed")
    report = verify_evidence_material(
        result,
        evidence.result_directory,
        dataset_manifest=evidence.dataset_manifest,
        split_manifest=evidence.split_manifest,
        model_manifest=evidence.model_manifest,
    )
    assert not report.passed
    assert any("metrics" in failure for failure in report.failures)


def test_result_directory_rejects_symlinked_artifacts(tmp_path: Path) -> None:
    evidence = candidate(tmp_path)
    predictions = evidence.result_directory / "predictions"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(predictions.read_bytes())
    predictions.unlink()
    predictions.symlink_to(replacement)

    report = verify_result_directory(
        evidence.public_development_result,
        evidence.result_directory,
    )
    assert not report.passed
    assert any("symbolic link" in failure for failure in report.failures)


def test_receipt_uses_real_signature_and_binds_exact_bundle(tmp_path: Path) -> None:
    private, trusted = private_and_trusted("custodian", 3)
    result = candidate(tmp_path).public_development_result
    receipt = signed_receipt(result, "cohort", "custodian", private, trusted)
    assert verify_receipt_signature(receipt, (trusted,))
    assert EvidenceBundle(result, (receipt,)).receipts == (receipt,)
    forged = replace(receipt, signature=base64.b64encode(bytes(64)).decode("ascii"))
    assert not verify_receipt_signature(forged, (trusted,))
    with pytest.raises(ValueError, match="exact result bundle"):
        EvidenceBundle(replace(result, run_id="other"), (receipt,))
    with pytest.raises(TypeError, match="boolean"):
        replace(receipt, egress_disabled=1)


def test_result_and_baseline_metadata_reject_type_confusion(tmp_path: Path) -> None:
    evidence = candidate(tmp_path)
    with pytest.raises(TypeError, match="format_version"):
        replace(evidence.public_development_result, format_version=1)
    with pytest.raises(TypeError, match="run_id"):
        replace(evidence.public_development_result, run_id=1)
    with pytest.raises(TypeError, match="baseline name"):
        replace(evidence.qualified_baselines[0], name=1)
    with pytest.raises(TypeError, match="baseline family"):
        replace(evidence.qualified_baselines[0], family="external")


def test_result_directory_and_receipt_identity_are_canonical(tmp_path: Path) -> None:
    evidence = candidate(tmp_path)
    linked = tmp_path / "linked-result"
    linked.symlink_to(evidence.result_directory, target_is_directory=True)
    verification = verify_result_directory(evidence.public_development_result, linked)
    assert not verification.passed
    assert "result directory must not be a symbolic link" in verification.failures

    private, trusted = private_and_trusted("custodian", 30)
    with pytest.raises(ValueError, match="canonical whitespace"):
        replace(trusted, custodian="custodian ")
    result = evidence.public_development_result
    with pytest.raises(ValueError, match="predate"):
        EvidenceBundle(
            result,
            (
                signed_receipt(
                    result,
                    "cohort",
                    "custodian",
                    private,
                    trusted,
                    evaluated_at="2026-01-02T00:00:00Z",
                ),
            ),
        )


def test_duplicate_raw_content_and_split_strategy_relabeling_do_not_create_cohorts(
    tmp_path: Path,
) -> None:
    dataset, split = typed_population_manifests("identity")
    duplicate = replace(
        dataset.raw_objects[0],
        source_uri="https://example.test/mirror",
        byte_size=dataset.raw_objects[0].byte_size + 1,
    )
    with pytest.raises(ValueError, match="duplicate raw content identities"):
        replace(dataset, raw_objects=(*dataset.raw_objects, duplicate))
    relabeled = replace(split, held_out_level=HierarchyLevel.SCHEMA_FAMILY)
    assert relabeled.population_identity == split.population_identity


def test_endpoint_metric_domains_are_diagnostic_failures(tmp_path: Path) -> None:
    private_a, trusted_a = private_and_trusted("custodian-a", 31)
    private_b, trusted_b = private_and_trusted("custodian-b", 32)
    endpoints = tuple(endpoint(track, metric) for track, metric in required_metric_specs())
    invalid = (replace(endpoints[0], braid_value=1.2), *endpoints[1:])
    first = cohort(
        tmp_path,
        "domain-1",
        "custodian-a",
        private_a,
        trusted_a,
        endpoint_overrides=invalid,
    )
    second = cohort(tmp_path, "domain-2", "custodian-b", private_b, trusted_b)
    keys = (trusted_a, trusted_b)
    dossier = EvidenceDossier((first, second), trusted_keyring_hash(keys), first.subject)
    report = evaluate_frontier_gate(dossier, subject=dossier.subject, trusted_keys=keys)
    assert not report.passed
    assert any(
        "metric and baseline values must be in [0, 1]" in failure
        for failure in report.cohorts[0].failures
    )


def test_frontier_authorization_stays_disabled_without_raw_recomputation(
    tmp_path: Path,
) -> None:
    dossier, keys = passing_dossier(tmp_path)
    assert len(dossier.cohorts[0].split_manifest.records) == 1
    for name in ("predictions", "baseline_trials", "bootstrap_draws"):
        assert (dossier.cohorts[0].result_directory / name).read_bytes() == b"{}"
    report = evaluate_frontier_gate(dossier, subject=dossier.subject, trusted_keys=keys)
    assert not report.passed
    assert any("non-engineering claims are disabled" in failure for failure in report.failures)
    assert len(report.cohorts) == 2
    assert all(
        len(item.endpoints) == sum(len(spec.metrics) for spec in TRACK_SPECS.values())
        for item in report.cohorts
    )
    assert report.sealed_cohorts == 2
    assert report.independent_custodians == 2
    with pytest.raises(ClaimProhibitedError, match="non-engineering claims are disabled"):
        authorize_claim(
            ClaimLevel.FRONTIER,
            dossier,
            subject=dossier.subject,
            trusted_keys=keys,
        )
    assert (
        strongest_permitted_claim(dossier, subject=dossier.subject, trusted_keys=keys)
        is ClaimLevel.ENGINEERING_BUILD
    )


def test_frontier_gate_fails_without_explicit_subject_or_external_trust(tmp_path: Path) -> None:
    dossier, _ = passing_dossier(tmp_path)
    report = evaluate_frontier_gate(dossier)
    assert not report.passed
    assert any("claim subject" in failure for failure in report.failures)
    assert any("trusted custodian keyring" in failure for failure in report.failures)
    with pytest.raises(ClaimProhibitedError, match="explicit model/checkpoint/container"):
        authorize_claim(ClaimLevel.FRONTIER, dossier)


def test_signed_gate_payload_cannot_be_changed_after_receipt(tmp_path: Path) -> None:
    dossier, _ = passing_dossier(tmp_path)
    first = dossier.cohorts[0]
    changed = replace(first.endpoints[0], shortcut_margin=0.0)
    with pytest.raises(ValueError, match="gate evidence"):
        replace(first, endpoints=(changed, *first.endpoints[1:]))


def test_actual_result_directory_is_required_at_gate_time(tmp_path: Path) -> None:
    dossier, keys = passing_dossier(tmp_path)
    first = dossier.cohorts[0]
    (first.result_directory / "checkpoint_manifest").write_bytes(b"substituted")
    report = evaluate_frontier_gate(dossier, subject=dossier.subject, trusted_keys=keys)
    assert not report.passed
    assert any("checkpoint_manifest" in failure for failure in report.failures)


def test_population_gate_is_recomputed_from_raw_signed_counts(tmp_path: Path) -> None:
    private_a, trusted_a = private_and_trusted("custodian-a", 4)
    private_b, trusted_b = private_and_trusted("custodian-b", 5)
    populations = tuple(raw_population(track) for track in BenchmarkTrack)
    weak = (
        replace(
            populations[0],
            population=replace(populations[0].population, repository_clusters=29),
        ),
        *populations[1:],
    )
    first = cohort(
        tmp_path,
        "cohort-1",
        "custodian-a",
        private_a,
        trusted_a,
        population_overrides=weak,
    )
    second = cohort(tmp_path, "cohort-2", "custodian-b", private_b, trusted_b)
    keys = (trusted_a, trusted_b)
    dossier = EvidenceDossier(
        (first, second),
        trusted_keyring_hash(keys),
        first.subject,
    )
    report = evaluate_frontier_gate(dossier, subject=dossier.subject, trusted_keys=keys)
    assert not report.passed
    assert any("requires at least 30 repository clusters" in failure for failure in report.failures)
    with pytest.raises(TypeError, match="raw TrackPopulation"):
        replace(
            second,
            populations=(
                PopulationQualification(
                    BenchmarkTrack.UNSEEN_ENTITIES,
                    True,
                    (),
                    0.1,
                ),
            ),
        )


def test_manifest_rewording_cannot_reuse_a_spent_population(tmp_path: Path) -> None:
    private_a, trusted_a = private_and_trusted("custodian-a", 6)
    private_b, trusted_b = private_and_trusted("custodian-b", 7)
    first = cohort(tmp_path, "cohort-1", "custodian-a", private_a, trusted_a)
    second = cohort(
        tmp_path,
        "cohort-2",
        "custodian-b",
        private_b,
        trusted_b,
        population_material="cohort-1",
    )
    assert first.bundle.result.dataset_manifest_hash != second.bundle.result.dataset_manifest_hash
    assert first.bundle.result.split_manifest_hash != second.bundle.result.split_manifest_hash
    assert first.dataset_manifest.raw_object_identity == second.dataset_manifest.raw_object_identity
    assert first.split_manifest.population_identity == second.split_manifest.population_identity
    keys = (trusted_a, trusted_b)
    dossier = EvidenceDossier(
        (first, second),
        trusted_keyring_hash(keys),
        first.subject,
    )
    report = evaluate_frontier_gate(dossier, subject=dossier.subject, trusted_keys=keys)
    assert not report.passed
    assert any("typed raw-object populations" in failure for failure in report.failures)
    assert any("typed split populations" in failure for failure in report.failures)


@pytest.mark.parametrize("difference", ["model", "checkpoint", "container"])
def test_mixed_evaluation_subjects_cannot_form_a_dossier(
    tmp_path: Path,
    difference: str,
) -> None:
    private_a, trusted_a = private_and_trusted("custodian-a", 8)
    private_b, trusted_b = private_and_trusted("custodian-b", 9)
    first = cohort(tmp_path, "cohort-1", "custodian-a", private_a, trusted_a)
    second_model = (
        model_manifest(code_hash="d" * 64)
        if difference == "model"
        else model_manifest(checkpoint_hash="b" * 64)
        if difference == "checkpoint"
        else model_manifest()
    )
    second = cohort(
        tmp_path,
        f"cohort-2-{difference}",
        "custodian-b",
        private_b,
        trusted_b,
        model=second_model,
        container_bytes=(b"different-container" if difference == "container" else CONTAINER_BYTES),
    )
    keys = (trusted_a, trusted_b)
    with pytest.raises(ValueError, match="one model/checkpoint/container subject"):
        EvidenceDossier((first, second), trusted_keyring_hash(keys), first.subject)


def test_no_evidence_emits_only_engineering_build() -> None:
    assert strongest_permitted_claim() is ClaimLevel.ENGINEERING_BUILD
    assert authorize_claim(ClaimLevel.ENGINEERING_BUILD) is ClaimLevel.ENGINEERING_BUILD
    with pytest.raises(ClaimProhibitedError):
        authorize_claim(ClaimLevel.FRONTIER)
    with pytest.raises(TypeError, match="ClaimLevel"):
        authorize_claim("frontier")  # type: ignore[arg-type]


def test_research_candidate_requires_typed_material_and_subject(tmp_path: Path) -> None:
    with pytest.raises(ClaimProhibitedError, match="explicit model/checkpoint/container"):
        authorize_claim(ClaimLevel.RESEARCH_CANDIDATE)
    evidence = candidate(tmp_path)
    report = evaluate_candidate_gate(evidence, subject=evidence.subject)
    assert not report.passed
    assert any("non-engineering claims are disabled" in failure for failure in report.failures)
    with pytest.raises(ClaimProhibitedError, match="non-engineering claims are disabled"):
        authorize_claim(
            ClaimLevel.RESEARCH_CANDIDATE,
            candidate_evidence=evidence,
            subject=evidence.subject,
        )
    assert (
        strongest_permitted_claim(candidate_evidence=evidence, subject=evidence.subject)
        is ClaimLevel.ENGINEERING_BUILD
    )

    tampered = replace(evidence, leakage_audits_passed=False)
    report = evaluate_candidate_gate(tampered, subject=tampered.subject)
    assert not report.passed
    assert any("not bound" in failure for failure in report.failures)


def test_candidate_rejects_invented_track_metric_endpoint(tmp_path: Path) -> None:
    evidence = candidate(tmp_path)
    invented = (
        RecordedMetric(
            BenchmarkTrack.UNSEEN_ENTITIES,
            "invented_score",
            0.60,
            "ULTRA",
            0.55,
        ),
    )
    directory = tmp_path / "invented-endpoint"
    result = write_result(
        directory,
        run_id="invented-endpoint",
        gate_bytes=(evidence.result_directory / "gate_evidence").read_bytes(),
        dataset=evidence.dataset_manifest,
        split=evidence.split_manifest,
        model=evidence.model_manifest,
        metrics=invented,
    )
    changed = replace(
        evidence,
        public_development_result=result,
        result_directory=directory,
    )
    report = evaluate_candidate_gate(changed, subject=changed.subject)
    assert not report.passed
    assert any("unknown endpoints: E/invented_score" in failure for failure in report.failures)


def test_foundation_language_uses_the_same_subject_and_frontier_gate(tmp_path: Path) -> None:
    dossier, keys = passing_dossier(tmp_path)
    with pytest.raises(ClaimProhibitedError, match="non-engineering claims are disabled"):
        authorize_claim(
            ClaimLevel.FOUNDATION_MODEL,
            dossier,
            subject=dossier.subject,
            trusted_keys=keys,
        )
    wrong = replace(dossier.subject, checkpoint_digest="f" * 64)
    with pytest.raises(ClaimProhibitedError, match="requested claim subject"):
        authorize_claim(
            ClaimLevel.FOUNDATION_MODEL,
            dossier,
            subject=wrong,
            trusted_keys=keys,
        )
