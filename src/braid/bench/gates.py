"""Hard, artifact-bound evidence gates controlling Braid capability vocabulary."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from braid.contract.types import ModelManifest

from .baselines import BaselineMetadata
from .data import DatasetManifest, SplitManifest
from .evidence import (
    EvaluationSubject,
    EvidenceBundle,
    ResultBundleManifest,
    TrustedCustodianKey,
    canonical_json,
    content_hash,
    hash_bytes,
    verify_evidence_material,
    verify_receipt_signature,
)
from .specs import (
    TRACK_SPECS,
    BenchmarkTrack,
    ImprovementRule,
    MetricDirection,
    TrackPopulation,
    required_metric_specs,
    validate_population,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# This engineering release can verify that declared files and signed receipts are
# internally consistent, but it cannot yet derive benchmark populations, endpoint
# values, or bootstrap decisions from typed raw prediction artifacts. Keep this
# blocker unconditional: callers cannot opt into stronger claim language.
_CLAIM_GRADE_ARTIFACT_DERIVATION_BLOCKER = (
    "non-engineering claims are disabled until typed prediction, baseline, and "
    "bootstrap artifacts deterministically reproduce every population and endpoint"
)


def _require_exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (booleans are not counts)")
    return value


def _require_exact_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float")
    return value


class ClaimLevel(StrEnum):
    ENGINEERING_BUILD = "engineering build"
    RESEARCH_CANDIDATE = "research candidate"
    FRONTIER = "frontier"
    FOUNDATION_MODEL = "foundation model"


class ClaimProhibitedError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EndpointEvidence:
    track: BenchmarkTrack
    metric: str
    braid_value: float
    strongest_baseline: str
    baseline_value: float
    baseline_qualified: bool
    adjusted_lower_bound: float
    holm_adjusted_p_value: float
    clusters_won: int
    clusters_total: int
    worst_schema_regression: float
    calibration_gap: float
    shortcut_margin: float

    def __post_init__(self) -> None:
        if type(self.track) is not BenchmarkTrack:
            raise TypeError("endpoint track must be a BenchmarkTrack")
        for name, value in (
            ("braid_value", self.braid_value),
            ("baseline_value", self.baseline_value),
            ("adjusted_lower_bound", self.adjusted_lower_bound),
            ("holm_adjusted_p_value", self.holm_adjusted_p_value),
            ("worst_schema_regression", self.worst_schema_regression),
            ("calibration_gap", self.calibration_gap),
            ("shortcut_margin", self.shortcut_margin),
        ):
            _require_exact_float(value, f"endpoint {name}")
        _require_exact_bool(self.baseline_qualified, "endpoint baseline_qualified")
        _require_exact_int(self.clusters_won, "endpoint clusters_won")
        _require_exact_int(self.clusters_total, "endpoint clusters_total")
        numeric = (
            self.braid_value,
            self.baseline_value,
            self.adjusted_lower_bound,
            self.holm_adjusted_p_value,
            self.worst_schema_regression,
            self.calibration_gap,
            self.shortcut_margin,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("endpoint evidence values must be finite")
        if not self.metric or not self.strongest_baseline:
            raise ValueError("metric and strongest baseline names are required")
        if not 0 <= self.holm_adjusted_p_value <= 1:
            raise ValueError("adjusted p-value must be in [0, 1]")
        if self.clusters_total <= 0 or not 0 <= self.clusters_won <= self.clusters_total:
            raise ValueError("cluster wins require a positive, valid denominator")
        if self.worst_schema_regression < 0 or self.calibration_gap < 0:
            raise ValueError("schema regression and calibration gap are non-negative magnitudes")

    @property
    def cluster_win_fraction(self) -> float:
        return self.clusters_won / self.clusters_total


def cohort_gate_payload(
    *,
    endpoints: tuple[EndpointEvidence, ...],
    populations: tuple[TrackPopulation, ...],
    retrieval_true_target_recall: float,
    retrieval_queries: int,
    leakage_audits_passed: bool,
    metric_canaries_passed: bool,
    official_baselines_reproduced: bool,
) -> dict[str, object]:
    """Return the exact gate material whose hash must be in a signed result bundle."""

    return {
        "endpoints": endpoints,
        "populations": populations,
        "retrieval_true_target_recall": retrieval_true_target_recall,
        "retrieval_queries": retrieval_queries,
        "leakage_audits_passed": leakage_audits_passed,
        "metric_canaries_passed": metric_canaries_passed,
        "official_baselines_reproduced": official_baselines_reproduced,
    }


@dataclass(frozen=True, slots=True)
class SealedCohortEvidence:
    """One custodian-signed result plus the gate values bound inside it."""

    bundle: EvidenceBundle
    result_directory: Path
    dataset_manifest: DatasetManifest
    split_manifest: SplitManifest
    model_manifest: ModelManifest
    endpoints: tuple[EndpointEvidence, ...]
    populations: tuple[TrackPopulation, ...]
    retrieval_true_target_recall: float
    retrieval_queries: int
    leakage_audits_passed: bool
    metric_canaries_passed: bool
    official_baselines_reproduced: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_directory", Path(self.result_directory))
        if len(self.bundle.receipts) != 1:
            raise ValueError("a sealed cohort must contain exactly one evaluator receipt")
        if type(self.dataset_manifest) is not DatasetManifest:
            raise TypeError("sealed dataset_manifest must be a DatasetManifest")
        if type(self.split_manifest) is not SplitManifest:
            raise TypeError("sealed split_manifest must be a SplitManifest")
        if type(self.model_manifest) is not ModelManifest:
            raise TypeError("sealed model_manifest must be a ModelManifest")
        if type(self.endpoints) is not tuple or any(
            type(endpoint) is not EndpointEvidence for endpoint in self.endpoints
        ):
            raise TypeError("sealed endpoints must be a tuple of EndpointEvidence values")
        if type(self.populations) is not tuple or any(
            type(population) is not TrackPopulation for population in self.populations
        ):
            raise TypeError("sealed populations must contain raw TrackPopulation values")
        _require_exact_float(
            self.retrieval_true_target_recall,
            "retrieval_true_target_recall",
        )
        _require_exact_int(self.retrieval_queries, "retrieval_queries")
        for name, value in (
            ("leakage_audits_passed", self.leakage_audits_passed),
            ("metric_canaries_passed", self.metric_canaries_passed),
            ("official_baselines_reproduced", self.official_baselines_reproduced),
        ):
            _require_exact_bool(value, name)
        if (
            not math.isfinite(self.retrieval_true_target_recall)
            or not 0 <= self.retrieval_true_target_recall <= 1
        ):
            raise ValueError("retrieval true-target recall must be finite and in [0, 1]")
        if self.retrieval_queries <= 0:
            raise ValueError("retrieval recall requires a positive query count")

        gate_bytes = canonical_json(self.gate_payload())
        artifacts = {artifact.name: artifact for artifact in self.bundle.result.artifacts}
        gate_artifact = artifacts.get("gate_evidence")
        if gate_artifact is None:
            raise ValueError("sealed result is missing gate_evidence")
        if gate_artifact.sha256 != hash_bytes(gate_bytes):
            raise ValueError("gate evidence does not match the signed result artifact")
        if gate_artifact.byte_size != len(gate_bytes):
            raise ValueError("gate evidence byte size does not match its canonical payload")

        recorded = {(metric.track, metric.name): metric for metric in self.bundle.result.metrics}
        declared = {(endpoint.track, endpoint.metric): endpoint for endpoint in self.endpoints}
        if set(recorded) != set(declared):
            raise ValueError("signed result metrics and endpoint evidence differ")
        for key, endpoint in declared.items():
            metric = recorded[key]
            if (
                metric.value != endpoint.braid_value
                or metric.baseline_name != endpoint.strongest_baseline
                or metric.baseline_value != endpoint.baseline_value
            ):
                raise ValueError(f"signed result metric does not match endpoint {key!r}")

    @property
    def receipt(self):
        return self.bundle.receipts[0]

    @property
    def cohort_id(self) -> str:
        return self.receipt.cohort_id

    @property
    def subject(self) -> EvaluationSubject:
        return EvaluationSubject.from_result(self.bundle.result)

    def gate_payload(self) -> dict[str, object]:
        return cohort_gate_payload(
            endpoints=self.endpoints,
            populations=self.populations,
            retrieval_true_target_recall=self.retrieval_true_target_recall,
            retrieval_queries=self.retrieval_queries,
            leakage_audits_passed=self.leakage_audits_passed,
            metric_canaries_passed=self.metric_canaries_passed,
            official_baselines_reproduced=self.official_baselines_reproduced,
        )


@dataclass(frozen=True, slots=True)
class EvidenceDossier:
    """Two or more independently sealed, artifact-bound confirmatory cohorts."""

    cohorts: tuple[SealedCohortEvidence, ...]
    trust_policy_hash: str
    subject: EvaluationSubject

    def __post_init__(self) -> None:
        if type(self.cohorts) is not tuple or any(
            type(cohort) is not SealedCohortEvidence for cohort in self.cohorts
        ):
            raise TypeError("dossier cohorts must be a tuple of SealedCohortEvidence values")
        if type(self.subject) is not EvaluationSubject:
            raise TypeError("dossier subject must be an EvaluationSubject")
        if not _SHA256.fullmatch(self.trust_policy_hash):
            raise ValueError("trust_policy_hash must be a lowercase SHA-256")
        cohort_ids = [cohort.cohort_id for cohort in self.cohorts]
        if len(cohort_ids) != len(set(cohort_ids)):
            raise ValueError("sealed cohort IDs must be unique")
        if any(cohort.subject != self.subject for cohort in self.cohorts):
            raise ValueError(
                "every sealed cohort must bind the dossier's one model/checkpoint/container subject"
            )


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Minimum evidence needed to describe a release as a research candidate."""

    public_development_result: ResultBundleManifest
    result_directory: Path
    dataset_manifest: DatasetManifest
    split_manifest: SplitManifest
    model_manifest: ModelManifest
    qualified_baselines: tuple[BaselineMetadata, ...]
    leakage_audits_passed: bool
    metric_canaries_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_directory", Path(self.result_directory))
        if type(self.public_development_result) is not ResultBundleManifest:
            raise TypeError("candidate result must be a ResultBundleManifest")
        if type(self.dataset_manifest) is not DatasetManifest:
            raise TypeError("candidate dataset_manifest must be a DatasetManifest")
        if type(self.split_manifest) is not SplitManifest:
            raise TypeError("candidate split_manifest must be a SplitManifest")
        if type(self.model_manifest) is not ModelManifest:
            raise TypeError("candidate model_manifest must be a ModelManifest")
        if type(self.qualified_baselines) is not tuple or any(
            type(baseline) is not BaselineMetadata for baseline in self.qualified_baselines
        ):
            raise TypeError("qualified_baselines must be a tuple of BaselineMetadata values")
        _require_exact_bool(self.leakage_audits_passed, "leakage_audits_passed")
        _require_exact_bool(self.metric_canaries_passed, "metric_canaries_passed")
        for baseline in self.qualified_baselines:
            for name, value in (
                ("uses_identical_inputs", baseline.uses_identical_inputs),
                ("within_inference_envelope", baseline.within_inference_envelope),
                ("official_result_reproduced", baseline.official_result_reproduced),
            ):
                _require_exact_bool(value, f"baseline {baseline.name} {name}")

    @property
    def subject(self) -> EvaluationSubject:
        return EvaluationSubject.from_result(self.public_development_result)


def candidate_gate_payload(
    *,
    qualified_baselines: tuple[BaselineMetadata, ...],
    leakage_audits_passed: bool,
    metric_canaries_passed: bool,
) -> dict[str, object]:
    """Return public-development gate material bound by ``gate_evidence``."""

    return {
        "qualified_baselines": tuple(
            sorted(qualified_baselines, key=lambda baseline: baseline.name)
        ),
        "leakage_audits_passed": leakage_audits_passed,
        "metric_canaries_passed": metric_canaries_passed,
    }


@dataclass(frozen=True, slots=True)
class CandidateGateReport:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EndpointGateDecision:
    track: BenchmarkTrack
    metric: str
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CohortGateDecision:
    cohort_id: str
    passed: bool
    failures: tuple[str, ...]
    endpoints: tuple[EndpointGateDecision, ...]


@dataclass(frozen=True, slots=True)
class FrontierGateReport:
    passed: bool
    failures: tuple[str, ...]
    cohorts: tuple[CohortGateDecision, ...]
    sealed_cohorts: int
    independent_custodians: int


def trusted_keyring_hash(keys: tuple[TrustedCustodianKey, ...]) -> str:
    ordered = tuple(sorted(keys, key=lambda key: (key.custodian, key.signer_key_id)))
    return content_hash(ordered)


def evaluate_candidate_gate(
    evidence: CandidateEvidence,
    *,
    subject: EvaluationSubject | None = None,
) -> CandidateGateReport:
    """Require complete public-development artifacts and qualified comparators."""

    failures: list[str] = [_CLAIM_GRADE_ARTIFACT_DERIVATION_BLOCKER]
    if subject is None:
        failures.append("an explicit model/checkpoint/container claim subject is required")
    elif type(subject) is not EvaluationSubject or subject != evidence.subject:
        failures.append("candidate evidence does not bind the requested claim subject")
    material = verify_evidence_material(
        evidence.public_development_result,
        evidence.result_directory,
        dataset_manifest=evidence.dataset_manifest,
        split_manifest=evidence.split_manifest,
        model_manifest=evidence.model_manifest,
    )
    failures.extend(f"result material failed: {failure}" for failure in material.failures)
    artifacts = {
        artifact.name: artifact for artifact in evidence.public_development_result.artifacts
    }
    gate_bytes = canonical_json(
        candidate_gate_payload(
            qualified_baselines=evidence.qualified_baselines,
            leakage_audits_passed=evidence.leakage_audits_passed,
            metric_canaries_passed=evidence.metric_canaries_passed,
        )
    )
    gate_artifact = artifacts.get("gate_evidence")
    if gate_artifact is None or (
        gate_artifact.sha256 != hash_bytes(gate_bytes) or gate_artifact.byte_size != len(gate_bytes)
    ):
        failures.append("candidate gate evidence is not bound to the public result")
    baseline_by_name = {
        baseline.name: baseline for baseline in evidence.qualified_baselines if baseline.qualified
    }
    baseline_names = set(baseline_by_name)
    if not baseline_names:
        failures.append("at least one named qualified baseline is required")
    if len(baseline_names) != len(evidence.qualified_baselines):
        failures.append("every declared candidate comparator must be fully qualified")
    used_baselines = {metric.baseline_name for metric in evidence.public_development_result.metrics}
    if not used_baselines:
        failures.append("the public development result contains no metrics")
    known_endpoints = {(track, metric.name) for track, metric in required_metric_specs()}
    unknown_endpoints = sorted(
        f"{metric.track.value}/{metric.name}"
        for metric in evidence.public_development_result.metrics
        if (metric.track, metric.name) not in known_endpoints
    )
    if unknown_endpoints:
        failures.append(
            "public development result contains unknown endpoints: " + ", ".join(unknown_endpoints)
        )
    unqualified = sorted(used_baselines - baseline_names)
    if unqualified:
        failures.append("result uses unqualified baselines: " + ", ".join(unqualified))
    wrong_track = sorted(
        f"{metric.baseline_name}/{metric.track.value}"
        for metric in evidence.public_development_result.metrics
        if metric.baseline_name in baseline_by_name
        and metric.track not in baseline_by_name[metric.baseline_name].tracks
    )
    if wrong_track:
        failures.append(
            "qualified baseline is not declared for result track: " + ", ".join(wrong_track)
        )
    if not evidence.leakage_audits_passed:
        failures.append("leakage audits did not pass")
    if not evidence.metric_canaries_passed:
        failures.append("metric orientation/tie canaries did not pass")
    return CandidateGateReport(not failures, tuple(failures))


def _metric_spec(evidence: EndpointEvidence):
    for metric in TRACK_SPECS[evidence.track].metrics:
        if metric.name == evidence.metric:
            return metric
    raise KeyError((evidence.track, evidence.metric))


def _endpoint_decision(evidence: EndpointEvidence) -> EndpointGateDecision:
    spec = _metric_spec(evidence)
    failures: list[str] = []
    unit_interval_metrics = {
        "filtered_mrr",
        "semantic_zero_shot_mrr",
        "symbol_renamed_eight_shot_mrr",
        "schema_mapping_macro_f1",
        "schema_link_mrr",
        "macro_auprc",
        "integrated_brier",
        "contract_validity",
        "matched_event_f1",
        "evidence_basis_accuracy",
        "distinct_window_rate",
        "memorization_rate",
    }
    if evidence.metric in unit_interval_metrics and not (
        0.0 <= evidence.braid_value <= 1.0 and 0.0 <= evidence.baseline_value <= 1.0
    ):
        failures.append("metric and baseline values must be in [0, 1]")
    if evidence.metric == "time_calibration_error" and (
        evidence.braid_value < 0.0 or evidence.baseline_value < 0.0
    ):
        failures.append("time calibration error cannot be negative")
    if not evidence.baseline_qualified:
        failures.append("strongest baseline is not qualified under identical inputs/envelope")
    improvement = (
        evidence.braid_value - evidence.baseline_value
        if spec.direction is MetricDirection.HIGHER_IS_BETTER
        else evidence.baseline_value - evidence.braid_value
    )
    if spec.improvement_rule is ImprovementRule.ABSOLUTE:
        if improvement + 1e-12 < spec.minimum_improvement:
            failures.append(
                f"absolute improvement {improvement:.6g} is below {spec.minimum_improvement:.6g}"
            )
    else:
        denominator = abs(evidence.baseline_value)
        relative = improvement / denominator if denominator else -math.inf
        if relative + 1e-12 < spec.minimum_improvement:
            failures.append(
                f"relative improvement {relative:.3%} is below {spec.minimum_improvement:.1%}"
            )
    if evidence.adjusted_lower_bound <= 0:
        failures.append("multiplicity-adjusted lower confidence bound must exceed zero")
    if evidence.holm_adjusted_p_value > 0.05:
        failures.append("Holm-adjusted p-value exceeds .05")
    if evidence.cluster_win_fraction + 1e-12 < 0.60:
        failures.append(f"cluster win fraction {evidence.cluster_win_fraction:.3f} is below .600")
    if evidence.worst_schema_regression > 0.05 + 1e-12:
        failures.append(
            f"worst sufficiently populated schema regression "
            f"{evidence.worst_schema_regression:.3f} exceeds .050"
        )
    if evidence.calibration_gap > 0.01 + 1e-12:
        failures.append(f"calibration gap {evidence.calibration_gap:.3f} exceeds .010")
    if evidence.shortcut_margin + 1e-12 < 0.02:
        failures.append(f"shortcut margin {evidence.shortcut_margin:.3f} is below .020")
    return EndpointGateDecision(evidence.track, evidence.metric, not failures, tuple(failures))


def _evaluate_cohort(
    cohort: SealedCohortEvidence,
    trusted_keys: tuple[TrustedCustodianKey, ...],
) -> CohortGateDecision:
    failures: list[str] = []
    material = verify_evidence_material(
        cohort.bundle.result,
        cohort.result_directory,
        dataset_manifest=cohort.dataset_manifest,
        split_manifest=cohort.split_manifest,
        model_manifest=cohort.model_manifest,
    )
    failures.extend(f"result material failed: {failure}" for failure in material.failures)
    receipt = cohort.receipt
    if not receipt.passed:
        failures.append("custodian receipt reports a failed evaluation")
    if not receipt.egress_disabled:
        failures.append("custodian did not attest to egress-disabled execution")
    if not verify_receipt_signature(receipt, trusted_keys):
        failures.append("receipt signature is not valid under the preregistered keyring")

    endpoint_keys = [(item.track, item.metric) for item in cohort.endpoints]
    if len(endpoint_keys) != len(set(endpoint_keys)):
        failures.append("endpoint evidence contains duplicates")
    evidence_by_key = {(item.track, item.metric): item for item in cohort.endpoints}
    required = tuple((track, metric.name) for track, metric in required_metric_specs())
    missing = [
        f"{track.value}/{metric}"
        for track, metric in required
        if (track, metric) not in evidence_by_key
    ]
    if missing:
        failures.append("missing required endpoints: " + ", ".join(missing))
    unknown = sorted(
        f"{track.value}/{metric}"
        for track, metric in evidence_by_key
        if (track, metric) not in set(required)
    )
    if unknown:
        failures.append("unknown endpoint evidence: " + ", ".join(unknown))

    decisions: list[EndpointGateDecision] = []
    for track, metric in required:
        evidence = evidence_by_key.get((track, metric))
        if evidence is None:
            continue
        decision = _endpoint_decision(evidence)
        decisions.append(decision)
        if not decision.passed:
            failures.append(f"{track.value}/{metric} failed: " + "; ".join(decision.failures))

    population_tracks = [population.track for population in cohort.populations]
    if len(population_tracks) != len(set(population_tracks)):
        failures.append("population qualifications contain duplicate tracks")
    by_track = {population.track: population for population in cohort.populations}
    for track in BenchmarkTrack:
        raw_population = by_track.get(track)
        if raw_population is None:
            failures.append(f"missing raw population for track {track.value}")
        else:
            population = validate_population(track, raw_population.population)
        if raw_population is not None and not population.passed:
            failures.append(
                f"track {track.value} population failed: " + "; ".join(population.failures)
            )

    if cohort.retrieval_true_target_recall + 1e-12 < 0.99:
        failures.append(
            f"true-target retrieval recall {cohort.retrieval_true_target_recall:.4f} is below .9900"
        )
    if not cohort.leakage_audits_passed:
        failures.append("leakage audits did not pass")
    if not cohort.metric_canaries_passed:
        failures.append("metric orientation/tie canaries did not pass")
    if not cohort.official_baselines_reproduced:
        failures.append("official baseline results were not reproduced")
    return CohortGateDecision(cohort.cohort_id, not failures, tuple(failures), tuple(decisions))


def evaluate_frontier_gate(
    dossier: EvidenceDossier,
    *,
    subject: EvaluationSubject | None = None,
    trusted_keys: tuple[TrustedCustodianKey, ...] = (),
) -> FrontierGateReport:
    """Evaluate every endpoint on every independently signed prospective cohort."""

    failures: list[str] = [_CLAIM_GRADE_ARTIFACT_DERIVATION_BLOCKER]
    if subject is None:
        failures.append("an explicit model/checkpoint/container claim subject is required")
    elif type(subject) is not EvaluationSubject or subject != dossier.subject:
        failures.append("sealed dossier does not bind the requested claim subject")
    key_pairs = [(key.custodian, key.signer_key_id) for key in trusted_keys]
    if len(key_pairs) != len(set(key_pairs)):
        failures.append("trusted custodian keyring contains duplicates")
    if not trusted_keys:
        failures.append("a preregistered trusted custodian keyring is required")
    elif trusted_keyring_hash(trusted_keys) != dossier.trust_policy_hash:
        failures.append("trusted custodian keyring does not match the preregistered policy hash")

    decisions = tuple(_evaluate_cohort(cohort, trusted_keys) for cohort in dossier.cohorts)
    for decision in decisions:
        if not decision.passed:
            failures.append(
                f"sealed cohort {decision.cohort_id} failed: " + " | ".join(decision.failures)
            )

    valid = tuple(
        cohort
        for cohort, decision in zip(dossier.cohorts, decisions, strict=True)
        if decision.passed
    )
    cohort_ids = {cohort.cohort_id for cohort in valid}
    custodians = {cohort.receipt.custodian for cohort in valid}
    signer_keys = {cohort.receipt.signer_key_id for cohort in valid}
    result_hashes = {cohort.bundle.result.sha256 for cohort in valid}
    raw_object_identities = {cohort.dataset_manifest.raw_object_identity for cohort in valid}
    split_population_identities = {cohort.split_manifest.population_identity for cohort in valid}
    if len(cohort_ids) < 2:
        failures.append("two successful prospective sealed cohorts are required")
    if len(custodians) < 2 or len(signer_keys) < 2:
        failures.append("sealed cohorts require two independent custodians and signing keys")
    if len(result_hashes) < 2:
        failures.append("sealed cohorts must bind distinct prospective result bundles")
    if len(raw_object_identities) < 2:
        failures.append("sealed cohorts must bind distinct typed raw-object populations")
    if len(split_population_identities) < 2:
        failures.append("sealed cohorts must bind distinct typed split populations")

    return FrontierGateReport(
        not failures,
        tuple(failures),
        decisions,
        len(cohort_ids),
        len(custodians),
    )


def authorize_claim(
    level: ClaimLevel,
    dossier: EvidenceDossier | None = None,
    *,
    subject: EvaluationSubject | None = None,
    candidate_evidence: CandidateEvidence | None = None,
    trusted_keys: tuple[TrustedCustodianKey, ...] = (),
) -> ClaimLevel:
    """Enforce public vocabulary at the point a model card or CLI emits a claim."""

    if type(level) is not ClaimLevel:
        raise TypeError("claim level must be a ClaimLevel")
    if level is ClaimLevel.ENGINEERING_BUILD:
        return level
    if subject is None:
        raise ClaimProhibitedError(
            f"'{level.value}' is prohibited without an explicit model/checkpoint/container subject"
        )
    if level is ClaimLevel.RESEARCH_CANDIDATE:
        if candidate_evidence is None:
            raise ClaimProhibitedError(
                "'research candidate' is prohibited without public development evidence"
            )
        report = evaluate_candidate_gate(candidate_evidence, subject=subject)
        if not report.passed:
            raise ClaimProhibitedError(
                "'research candidate' is prohibited: " + " | ".join(report.failures)
            )
        return level
    if dossier is None:
        raise ClaimProhibitedError(f"'{level.value}' is prohibited without a complete dossier")
    report = evaluate_frontier_gate(dossier, subject=subject, trusted_keys=trusted_keys)
    if not report.passed:
        raise ClaimProhibitedError(f"'{level.value}' is prohibited: " + " | ".join(report.failures))
    return level


def strongest_permitted_claim(
    dossier: EvidenceDossier | None = None,
    *,
    subject: EvaluationSubject | None = None,
    candidate_evidence: CandidateEvidence | None = None,
    trusted_keys: tuple[TrustedCustodianKey, ...] = (),
) -> ClaimLevel:
    if (
        dossier is not None
        and evaluate_frontier_gate(
            dossier,
            subject=subject,
            trusted_keys=trusted_keys,
        ).passed
    ):
        return ClaimLevel.FRONTIER
    if (
        candidate_evidence is not None
        and evaluate_candidate_gate(candidate_evidence, subject=subject).passed
    ):
        return ClaimLevel.RESEARCH_CANDIDATE
    return ClaimLevel.ENGINEERING_BUILD
