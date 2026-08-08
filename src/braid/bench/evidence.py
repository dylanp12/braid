"""Content-addressed result bundles and independently signed sealed receipts."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from braid.contract.types import ModelManifest

from .specs import BenchmarkTrack

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


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


def _require_exact_str(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _primitive(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _primitive(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _primitive(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (set, frozenset)):
        encoded = [_primitive(item) for item in value]
        return sorted(encoded, key=lambda item: canonical_json(item))
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence manifests cannot contain non-finite floats")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported manifest value {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Serialize an evidence object without platform- or insertion-order variance."""

    return json.dumps(
        _primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _parse_time(value: str, name: str) -> datetime:
    _require_exact_str(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def _require_time(value: str, name: str) -> None:
    _parse_time(value, name)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    name: str
    media_type: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        _require_exact_str(self.name, "artifact name")
        _require_exact_str(self.media_type, "artifact media_type")
        _require_exact_str(self.sha256, "artifact sha256")
        _require_exact_int(self.byte_size, "artifact byte_size")
        if not _ARTIFACT_NAME.fullmatch(self.name) or not self.media_type:
            raise ValueError("artifact name and media type are required")
        _require_sha256(self.sha256, "artifact sha256")
        if self.byte_size < 0:
            raise ValueError("artifact byte size cannot be negative")

    @classmethod
    def from_bytes(cls, name: str, media_type: str, data: bytes) -> ArtifactRef:
        return cls(name, media_type, hash_bytes(data), len(data))

    @classmethod
    def from_file(cls, path: str | Path, media_type: str) -> ArtifactRef:
        source = Path(path)
        return cls.from_bytes(source.name, media_type, source.read_bytes())


REQUIRED_RESULT_ARTIFACTS = frozenset(
    {
        "predictions",
        "baseline_trials",
        "bootstrap_draws",
        "checkpoint_manifest",
        "container_manifest",
        "dataset_manifest",
        "source_lock",
        "environment_lock",
        "exclusion_log",
        "sbom",
        "hardware_record",
        "gate_evidence",
        "model_manifest",
        "metrics",
        "split_manifest",
        "stderr",
        "stdout",
    }
)


@dataclass(frozen=True, slots=True)
class RecordedMetric:
    track: BenchmarkTrack
    name: str
    value: float
    baseline_name: str
    baseline_value: float

    def __post_init__(self) -> None:
        if type(self.track) is not BenchmarkTrack:
            raise TypeError("recorded metric track must be a BenchmarkTrack")
        _require_exact_str(self.name, "recorded metric name")
        _require_exact_str(self.baseline_name, "recorded baseline name")
        _require_exact_float(self.value, "recorded metric value")
        _require_exact_float(self.baseline_value, "recorded baseline value")
        if not self.name or not self.baseline_name:
            raise ValueError("metric and baseline names are required")
        if not math.isfinite(self.value) or not math.isfinite(self.baseline_value):
            raise ValueError("recorded metric values must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationSubject:
    """The one immutable model/checkpoint/container tuple a claim is about."""

    model_manifest_hash: str
    checkpoint_digest: str
    container_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.model_manifest_hash, "subject model manifest hash")
        _require_exact_str(self.checkpoint_digest, "subject checkpoint digest")
        _require_exact_str(self.container_digest, "subject container digest")
        _require_sha256(self.model_manifest_hash, "subject model manifest hash")
        _require_sha256(self.checkpoint_digest, "subject checkpoint digest")
        if not _OCI_DIGEST.fullmatch(self.container_digest):
            raise ValueError("subject container digest must use sha256:<lowercase digest>")

    @classmethod
    def from_result(cls, result: ResultBundleManifest) -> EvaluationSubject:
        return cls(
            result.model_manifest_hash,
            result.checkpoint_digest,
            result.container_digest,
        )

    @property
    def sha256(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ResultBundleManifest:
    """Immutable manifest for every public or sealed evaluation result."""

    format_version: str
    run_id: str
    created_at: str
    model_manifest_hash: str
    dataset_manifest_hash: str
    split_manifest_hash: str
    checkpoint_digest: str
    container_digest: str
    exact_commands: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]
    metrics: tuple[RecordedMetric, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("format_version", self.format_version),
            ("run_id", self.run_id),
            ("created_at", self.created_at),
            ("model_manifest_hash", self.model_manifest_hash),
            ("dataset_manifest_hash", self.dataset_manifest_hash),
            ("split_manifest_hash", self.split_manifest_hash),
            ("checkpoint_digest", self.checkpoint_digest),
            ("container_digest", self.container_digest),
        ):
            _require_exact_str(value, f"result {name}")
        if type(self.exact_commands) is not tuple:
            raise TypeError("exact_commands must be a tuple")
        if any(type(command) is not str for command in self.exact_commands):
            raise TypeError("exact_commands must contain only strings")
        if type(self.artifacts) is not tuple or any(
            type(artifact) is not ArtifactRef for artifact in self.artifacts
        ):
            raise TypeError("artifacts must be a tuple of ArtifactRef values")
        if type(self.metrics) is not tuple or any(
            type(metric) is not RecordedMetric for metric in self.metrics
        ):
            raise TypeError("metrics must be a tuple of RecordedMetric values")
        if self.format_version != "1":
            raise ValueError("unsupported result bundle format version")
        if not self.run_id:
            raise ValueError("result bundle run ID is required")
        _require_time(self.created_at, "created_at")
        for name, value in (
            ("model manifest hash", self.model_manifest_hash),
            ("dataset manifest hash", self.dataset_manifest_hash),
            ("split manifest hash", self.split_manifest_hash),
            ("checkpoint digest", self.checkpoint_digest),
        ):
            _require_sha256(value, name)
        if not _OCI_DIGEST.fullmatch(self.container_digest):
            raise ValueError("container digest must use sha256:<lowercase digest>")
        if not self.exact_commands or any(not command.strip() for command in self.exact_commands):
            raise ValueError("at least one exact non-empty command is required")
        artifact_names = [artifact.name for artifact in self.artifacts]
        if artifact_names != sorted(artifact_names) or len(artifact_names) != len(
            set(artifact_names)
        ):
            raise ValueError("artifacts must be uniquely named and sorted")
        missing = sorted(REQUIRED_RESULT_ARTIFACTS - set(artifact_names))
        if missing:
            raise ValueError(f"result bundle missing required artifacts: {', '.join(missing)}")
        by_name = {artifact.name: artifact for artifact in self.artifacts}
        for field_name, artifact_name, expected in (
            ("model manifest hash", "model_manifest", self.model_manifest_hash),
            ("dataset manifest hash", "dataset_manifest", self.dataset_manifest_hash),
            ("split manifest hash", "split_manifest", self.split_manifest_hash),
        ):
            if by_name[artifact_name].sha256 != expected:
                raise ValueError(f"{field_name} does not bind the {artifact_name} artifact")
        if "sha256:" + by_name["container_manifest"].sha256 != self.container_digest:
            raise ValueError("container digest does not bind the container_manifest artifact")
        metric_keys = [(metric.track.value, metric.name) for metric in self.metrics]
        if metric_keys != sorted(metric_keys) or len(metric_keys) != len(set(metric_keys)):
            raise ValueError("recorded metrics must be unique and sorted by track/name")

    @property
    def sha256(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class SealedEvaluationReceipt:
    """Custodian attestation bound to an exact result bundle.

    The public gate accepts only Ed25519 signatures verified against its
    preregistered custodian keyring. ``signed_payload_hash`` also makes the exact
    canonical payload independently reconstructible.
    """

    receipt_version: str
    cohort_id: str
    custodian: str
    evaluated_at: str
    result_bundle_hash: str
    egress_disabled: bool
    passed: bool
    signature_algorithm: str
    signer_key_id: str
    signed_payload_hash: str
    signature: str

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_version", self.receipt_version),
            ("cohort_id", self.cohort_id),
            ("custodian", self.custodian),
            ("evaluated_at", self.evaluated_at),
            ("result_bundle_hash", self.result_bundle_hash),
            ("signature_algorithm", self.signature_algorithm),
            ("signer_key_id", self.signer_key_id),
            ("signed_payload_hash", self.signed_payload_hash),
            ("signature", self.signature),
        ):
            _require_exact_str(value, f"receipt {name}")
        _require_exact_bool(self.egress_disabled, "receipt egress_disabled")
        _require_exact_bool(self.passed, "receipt passed")
        if self.receipt_version != "1":
            raise ValueError("unsupported sealed receipt version")
        for name, value in (
            ("cohort_id", self.cohort_id),
            ("custodian", self.custodian),
        ):
            if value != " ".join(value.split()):
                raise ValueError(f"receipt {name} must use canonical whitespace")
        if not all(
            value.strip()
            for value in (
                self.receipt_version,
                self.cohort_id,
                self.custodian,
                self.signature_algorithm,
                self.signer_key_id,
                self.signature,
            )
        ):
            raise ValueError("sealed receipts require complete custodian/signature metadata")
        _require_time(self.evaluated_at, "evaluated_at")
        _require_sha256(self.result_bundle_hash, "result bundle hash")
        _require_sha256(self.signed_payload_hash, "signed payload hash")

    @staticmethod
    def payload(
        *,
        receipt_version: str,
        cohort_id: str,
        custodian: str,
        evaluated_at: str,
        result_bundle_hash: str,
        egress_disabled: bool,
        passed: bool,
        signature_algorithm: str,
        signer_key_id: str,
    ) -> dict[str, object]:
        return {
            "receipt_version": receipt_version,
            "cohort_id": cohort_id,
            "custodian": custodian,
            "evaluated_at": evaluated_at,
            "result_bundle_hash": result_bundle_hash,
            "egress_disabled": egress_disabled,
            "passed": passed,
            "signature_algorithm": signature_algorithm,
            "signer_key_id": signer_key_id,
        }

    @classmethod
    def issue(
        cls,
        *,
        receipt_version: str,
        cohort_id: str,
        custodian: str,
        evaluated_at: str,
        result_bundle_hash: str,
        egress_disabled: bool,
        passed: bool,
        signature_algorithm: str,
        signer_key_id: str,
        signature: str,
    ) -> SealedEvaluationReceipt:
        payload = cls.payload(
            receipt_version=receipt_version,
            cohort_id=cohort_id,
            custodian=custodian,
            evaluated_at=evaluated_at,
            result_bundle_hash=result_bundle_hash,
            egress_disabled=egress_disabled,
            passed=passed,
            signature_algorithm=signature_algorithm,
            signer_key_id=signer_key_id,
        )
        return cls(
            **payload,
            signed_payload_hash=content_hash(payload),
            signature=signature,
        )

    @property
    def payload_is_bound(self) -> bool:
        expected = content_hash(
            self.payload(
                receipt_version=self.receipt_version,
                cohort_id=self.cohort_id,
                custodian=self.custodian,
                evaluated_at=self.evaluated_at,
                result_bundle_hash=self.result_bundle_hash,
                egress_disabled=self.egress_disabled,
                passed=self.passed,
                signature_algorithm=self.signature_algorithm,
                signer_key_id=self.signer_key_id,
            )
        )
        return expected == self.signed_payload_hash


@dataclass(frozen=True, slots=True)
class TrustedCustodianKey:
    """A preregistered Ed25519 verification key for one independent custodian."""

    custodian: str
    signer_key_id: str
    public_key_base64: str

    def __post_init__(self) -> None:
        _require_exact_str(self.custodian, "trusted key custodian")
        _require_exact_str(self.signer_key_id, "trusted key signer_key_id")
        _require_exact_str(self.public_key_base64, "trusted key public_key_base64")
        if not self.custodian.strip():
            raise ValueError("trusted key custodian is required")
        if self.custodian != " ".join(self.custodian.split()):
            raise ValueError("trusted key custodian must use canonical whitespace")
        _require_sha256(self.signer_key_id, "signer key ID")
        try:
            raw = base64.b64decode(self.public_key_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("public key must be canonical base64") from error
        if len(raw) != 32:
            raise ValueError("Ed25519 public keys must contain 32 bytes")
        if hashlib.sha256(raw).hexdigest() != self.signer_key_id:
            raise ValueError("signer key ID must equal the public-key SHA-256")

    @classmethod
    def from_public_key_bytes(cls, custodian: str, public_key: bytes) -> TrustedCustodianKey:
        return cls(
            custodian=custodian,
            signer_key_id=hashlib.sha256(public_key).hexdigest(),
            public_key_base64=base64.b64encode(public_key).decode("ascii"),
        )


def verify_receipt_signature(
    receipt: SealedEvaluationReceipt,
    trusted_keys: tuple[TrustedCustodianKey, ...],
) -> bool:
    """Verify a receipt against an explicitly preregistered custodian keyring."""

    if not receipt.payload_is_bound or receipt.signature_algorithm.casefold() != "ed25519":
        return False
    matches = tuple(
        key
        for key in trusted_keys
        if key.custodian == receipt.custodian and key.signer_key_id == receipt.signer_key_id
    )
    if len(matches) != 1:
        return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return False
    try:
        signature = base64.b64decode(receipt.signature, validate=True)
        public_key = base64.b64decode(matches[0].public_key_base64, validate=True)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            canonical_json(
                receipt.payload(
                    receipt_version=receipt.receipt_version,
                    cohort_id=receipt.cohort_id,
                    custodian=receipt.custodian,
                    evaluated_at=receipt.evaluated_at,
                    result_bundle_hash=receipt.result_bundle_hash,
                    egress_disabled=receipt.egress_disabled,
                    passed=receipt.passed,
                    signature_algorithm=receipt.signature_algorithm,
                    signer_key_id=receipt.signer_key_id,
                )
            ),
        )
    except (binascii.Error, ValueError, InvalidSignature):
        return False
    return True


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    result: ResultBundleManifest
    receipts: tuple[SealedEvaluationReceipt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.result) is not ResultBundleManifest:
            raise TypeError("evidence result must be a ResultBundleManifest")
        if type(self.receipts) is not tuple or any(
            type(receipt) is not SealedEvaluationReceipt for receipt in self.receipts
        ):
            raise TypeError("receipts must be a tuple of SealedEvaluationReceipt values")
        receipt_keys = [(receipt.cohort_id, receipt.custodian) for receipt in self.receipts]
        if receipt_keys != sorted(receipt_keys) or len(receipt_keys) != len(set(receipt_keys)):
            raise ValueError("receipts must be unique and sorted by cohort/custodian")
        if any(receipt.result_bundle_hash != self.result.sha256 for receipt in self.receipts):
            raise ValueError("every receipt must bind this exact result bundle")
        if any(not receipt.payload_is_bound for receipt in self.receipts):
            raise ValueError("receipt signed payload hash does not match its fields")
        result_created_at = _parse_time(self.result.created_at, "result created_at")
        if any(
            _parse_time(receipt.evaluated_at, "receipt evaluated_at") < result_created_at
            for receipt in self.receipts
        ):
            raise ValueError("receipt evaluation cannot predate the result bundle")

    @property
    def sha256(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    passed: bool
    failures: tuple[str, ...]


def checkpoint_manifest_payload(
    subject: ResultBundleManifest | EvaluationSubject,
) -> dict[str, str]:
    """Return the sole canonical checkpoint-manifest representation."""

    if type(subject) is ResultBundleManifest:
        checked = EvaluationSubject.from_result(subject)
    elif type(subject) is EvaluationSubject:
        checked = subject
    else:
        raise TypeError("checkpoint manifest subject must be a result or EvaluationSubject")
    return {
        "checkpoint_digest": checked.checkpoint_digest,
        "container_digest": checked.container_digest,
        "format_version": "1",
        "model_manifest_hash": checked.model_manifest_hash,
    }


def metrics_artifact_payload(
    metrics: tuple[RecordedMetric, ...],
) -> tuple[RecordedMetric, ...]:
    """Name the exact canonical metric payload retained in every result bundle."""

    return metrics


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_result_directory(
    result: ResultBundleManifest,
    directory: str | Path,
) -> ArtifactVerification:
    """Verify every declared artifact byte-for-byte without executing its contents."""

    root = Path(directory)
    failures: list[str] = []
    if root.is_symlink():
        return ArtifactVerification(False, ("result directory must not be a symbolic link",))
    if not root.is_dir():
        return ArtifactVerification(False, ("result directory does not exist",))
    for artifact in result.artifacts:
        path = root / artifact.name
        if path.is_symlink():
            failures.append(f"artifact {artifact.name} must not be a symbolic link")
            continue
        if not path.is_file():
            failures.append(f"missing artifact {artifact.name}")
            continue
        stat = path.stat()
        if stat.st_size != artifact.byte_size:
            failures.append(f"artifact {artifact.name} has the wrong byte size")
            continue
        if _file_hash(path) != artifact.sha256:
            failures.append(f"artifact {artifact.name} has the wrong SHA-256")
    return ArtifactVerification(not failures, tuple(failures))


def verify_evidence_material(
    result: ResultBundleManifest,
    directory: str | Path,
    *,
    dataset_manifest: object,
    split_manifest: object,
    model_manifest: ModelManifest,
) -> ArtifactVerification:
    """Verify files plus typed dataset, split, model, checkpoint, and metric claims.

    The imports for benchmark manifest classes stay local so the public evidence
    primitives remain importable without optional model or evaluation packages.
    """

    from .data import DatasetManifest, SplitManifest

    failures = list(verify_result_directory(result, directory).failures)
    if type(dataset_manifest) is not DatasetManifest:
        failures.append("dataset evidence is not a typed DatasetManifest")
    if type(split_manifest) is not SplitManifest:
        failures.append("split evidence is not a typed SplitManifest")
    if type(model_manifest) is not ModelManifest:
        failures.append("model evidence is not a typed ModelManifest")
    if failures and (
        type(dataset_manifest) is not DatasetManifest
        or type(split_manifest) is not SplitManifest
        or type(model_manifest) is not ModelManifest
    ):
        return ArtifactVerification(False, tuple(failures))

    assert isinstance(dataset_manifest, DatasetManifest)
    assert isinstance(split_manifest, SplitManifest)
    assert isinstance(model_manifest, ModelManifest)
    for name, value in (
        ("architecture_hash", model_manifest.architecture_hash),
        ("tokenizer_hash", model_manifest.tokenizer_hash),
        ("code_hash", model_manifest.code_hash),
        ("environment_hash", model_manifest.environment_hash),
        ("data_hash", model_manifest.data_hash),
        ("split_hash", model_manifest.split_hash),
        ("checkpoint_hash", model_manifest.checkpoint_hash),
        ("evaluation_hash", model_manifest.evaluation_hash),
    ):
        try:
            _require_sha256(value, f"model manifest {name}")
        except (TypeError, ValueError) as error:
            failures.append(str(error))

    if dataset_manifest.dataset_id != split_manifest.dataset_id or (
        dataset_manifest.version != split_manifest.version
    ):
        failures.append("dataset and split manifests identify different dataset versions")
    if dataset_manifest.split_manifest_hash != split_manifest.sha256:
        failures.append("dataset manifest does not bind the typed split manifest")
    if result.dataset_manifest_hash != dataset_manifest.sha256:
        failures.append("result does not bind the typed dataset manifest")
    if result.split_manifest_hash != split_manifest.sha256:
        failures.append("result does not bind the typed split manifest")
    if result.model_manifest_hash != model_manifest.manifest_id:
        failures.append("result does not bind the typed model manifest")
    if result.checkpoint_digest != model_manifest.checkpoint_hash:
        failures.append("result checkpoint digest differs from the typed model manifest")

    expected = {
        "dataset_manifest": canonical_json(dataset_manifest),
        "split_manifest": canonical_json(split_manifest),
        "model_manifest": model_manifest.to_json().encode("utf-8"),
        "checkpoint_manifest": canonical_json(checkpoint_manifest_payload(result)),
        "metrics": canonical_json(metrics_artifact_payload(result.metrics)),
    }
    root = Path(directory)
    for name, expected_bytes in expected.items():
        path = root / name
        if path.is_file() and path.read_bytes() != expected_bytes:
            failures.append(f"artifact {name} is not its declared canonical typed payload")
    return ArtifactVerification(not failures, tuple(failures))


def verify_replay_outputs(
    result: ResultBundleManifest,
    replay_directory: str | Path,
    *,
    output_names: tuple[str, ...] = ("predictions", "metrics"),
) -> ArtifactVerification:
    """Verify deterministic replay outputs against their signed result references.

    Container execution belongs to the egress-disabled custodian. This function is
    the safe public boundary used immediately afterward: only exact raw predictions
    and published metric bytes satisfy a replay.
    """

    references = {artifact.name: artifact for artifact in result.artifacts}
    failures: list[str] = []
    root = Path(replay_directory)
    for name in output_names:
        artifact = references.get(name)
        if artifact is None:
            failures.append(f"result manifest does not declare replay output {name}")
            continue
        path = root / name
        if path.is_symlink():
            failures.append(f"replay output {name} must not be a symbolic link")
            continue
        if not path.is_file():
            failures.append(f"missing replay output {name}")
            continue
        if path.stat().st_size != artifact.byte_size or _file_hash(path) != artifact.sha256:
            failures.append(f"replay output {name} differs from the signed result")
    return ArtifactVerification(not failures, tuple(failures))
