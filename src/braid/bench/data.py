"""Dataset governance metadata and hierarchical split declarations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (booleans are not counts)")
    return value


def _require_exact_str(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


class RedistributionPolicy(StrEnum):
    FULL = "full"
    MANIFEST_ONLY = "manifest_only"
    NON_RECONSTRUCTIVE_AGGREGATES = "non_reconstructive_aggregates"
    PROHIBITED = "prohibited"


class SplitName(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    PUBLIC_TEST = "public_test"
    SEALED_TEST = "sealed_test"


class HierarchyLevel(StrEnum):
    SCHEMA_FAMILY = "schema_family"
    ORGANIZATION = "organization"
    COPY_CLUSTER = "fork_mirror_code_copy_cluster"
    REPOSITORY = "repository"
    CHRONOLOGICAL_BLOCK = "chronological_block"


SPLIT_HIERARCHY = (
    HierarchyLevel.SCHEMA_FAMILY,
    HierarchyLevel.ORGANIZATION,
    HierarchyLevel.COPY_CLUSTER,
    HierarchyLevel.REPOSITORY,
    HierarchyLevel.CHRONOLOGICAL_BLOCK,
)


@dataclass(frozen=True, slots=True)
class SourceLicense:
    identifier: str
    terms_url: str
    acquired_at: str
    redistribution: RedistributionPolicy
    attribution: str
    terms_sha256: str
    allowed_uses: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("license identifier", self.identifier),
            ("source terms URL", self.terms_url),
            ("acquired_at", self.acquired_at),
            ("attribution", self.attribution),
            ("terms_sha256", self.terms_sha256),
        ):
            _require_exact_str(value, name)
        if type(self.redistribution) is not RedistributionPolicy:
            raise TypeError("redistribution must be a RedistributionPolicy")
        if type(self.allowed_uses) is not tuple or any(
            type(value) is not str for value in self.allowed_uses
        ):
            raise TypeError("allowed_uses must be a tuple of strings")
        if not self.identifier.strip() or not self.terms_url.startswith(("https://", "http://")):
            raise ValueError("license identifier and source terms URL are required")
        _parse_time(self.acquired_at, "acquired_at")
        if not self.attribution.strip():
            raise ValueError("source attribution is required")
        if not _SHA256.fullmatch(self.terms_sha256):
            raise ValueError("source terms snapshot must have a lowercase SHA-256")
        if (
            not self.allowed_uses
            or tuple(sorted(set(self.allowed_uses))) != self.allowed_uses
            or any(not value.strip() for value in self.allowed_uses)
        ):
            raise ValueError("allowed uses must be non-empty, unique, and sorted")
        required = {"machine-learning-research", "model-training"}
        if not required.issubset(self.allowed_uses):
            raise ValueError("source terms must affirm both ML research and model training")


@dataclass(frozen=True, slots=True)
class RawObject:
    source_uri: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        _require_exact_str(self.source_uri, "raw object source_uri")
        _require_exact_str(self.sha256, "raw object sha256")
        _require_exact_int(self.byte_size, "raw object byte_size")
        if not self.source_uri or not _SHA256.fullmatch(self.sha256):
            raise ValueError("raw objects require a URI and lowercase SHA-256")
        if self.byte_size < 0:
            raise ValueError("raw object byte size cannot be negative")


@dataclass(frozen=True, slots=True)
class Transformation:
    name: str
    implementation_hash: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_exact_str(self.name, "transformation name")
        _require_exact_str(self.implementation_hash, "transformation implementation_hash")
        if type(self.parameters) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str for value in item)
            for item in self.parameters
        ):
            raise TypeError("transformation parameters must be string-pair tuples")
        if not self.name or not _SHA256.fullmatch(self.implementation_hash):
            raise ValueError("transformations require a name and implementation SHA-256")
        if tuple(sorted(self.parameters)) != self.parameters:
            raise ValueError("transformation parameters must be sorted")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Complete provenance and governance declaration for a dataset version."""

    dataset_id: str
    version: str
    created_at: str
    license: SourceLicense
    raw_objects: tuple[RawObject, ...]
    transformations: tuple[Transformation, ...]
    schema_families: tuple[str, ...]
    organizations: tuple[str, ...]
    repository_clusters: tuple[str, ...]
    split_manifest_hash: str
    retention_policy: str
    deletion_policy: str
    personal_data_classification: str
    contains_source_text: bool
    alluvia_private_data: bool = False
    publishes_source_text: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("version", self.version),
            ("created_at", self.created_at),
            ("retention_policy", self.retention_policy),
            ("deletion_policy", self.deletion_policy),
            ("personal_data_classification", self.personal_data_classification),
            ("split_manifest_hash", self.split_manifest_hash),
        ):
            _require_exact_str(value, f"dataset {name}")
        if type(self.license) is not SourceLicense:
            raise TypeError("dataset license must be a SourceLicense")
        for name, values, item_type in (
            ("raw_objects", self.raw_objects, RawObject),
            ("transformations", self.transformations, Transformation),
        ):
            if type(values) is not tuple or any(type(value) is not item_type for value in values):
                raise TypeError(f"{name} must be a tuple of {item_type.__name__} values")
        for name, values in (
            ("schema_families", self.schema_families),
            ("organizations", self.organizations),
            ("repository_clusters", self.repository_clusters),
        ):
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise TypeError(f"{name} must be a tuple of strings")
        for name, value in (
            ("contains_source_text", self.contains_source_text),
            ("alluvia_private_data", self.alluvia_private_data),
            ("publishes_source_text", self.publishes_source_text),
        ):
            _require_exact_bool(value, name)
        if not self.dataset_id or not self.version:
            raise ValueError("dataset ID and version are required")
        _parse_time(self.created_at, "created_at")
        if not self.raw_objects:
            raise ValueError("a dataset must bind at least one raw object")
        object_keys = [(item.source_uri, item.sha256) for item in self.raw_objects]
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("raw objects must be unique")
        content_keys = [item.sha256 for item in self.raw_objects]
        if len(content_keys) != len(set(content_keys)):
            raise ValueError("duplicate raw content identities are prohibited")
        if not _SHA256.fullmatch(self.split_manifest_hash):
            raise ValueError("split_manifest_hash must be lowercase SHA-256")
        for name, values in (
            ("schema families", self.schema_families),
            ("organizations", self.organizations),
            ("repository clusters", self.repository_clusters),
        ):
            if not values or len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{name} must be non-empty and unique")
        if not self.retention_policy.strip() or not self.deletion_policy.strip():
            raise ValueError("retention and deletion policies are required")
        if not self.personal_data_classification.strip():
            raise ValueError("personal-data classification is required")
        if self.alluvia_private_data:
            raise ValueError("raw or derived Alluvia-private data cannot enter a global dataset")
        if (
            self.publishes_source_text
            and self.license.redistribution is not RedistributionPolicy.FULL
        ):
            raise ValueError(
                "source text publication requires affirmative full redistribution rights"
            )

    @property
    def sha256(self) -> str:
        from .evidence import content_hash

        return content_hash(self)

    @property
    def raw_object_identity(self) -> str:
        """Hash source content identities while ignoring mutable manifest prose/URIs."""

        from .evidence import content_hash

        identities = tuple(sorted(item.sha256 for item in self.raw_objects))
        return content_hash(identities)


@dataclass(frozen=True, slots=True)
class SplitRecord:
    record_id: str
    schema_family: str
    organization: str
    copy_cluster: str
    repository: str
    observed_at: str

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("schema_family", self.schema_family),
            ("organization", self.organization),
            ("copy_cluster", self.copy_cluster),
            ("repository", self.repository),
            ("observed_at", self.observed_at),
        ):
            _require_exact_str(value, name)
        if any(
            not value
            for value in (
                self.record_id,
                self.schema_family,
                self.organization,
                self.copy_cluster,
                self.repository,
            )
        ):
            raise ValueError("split record identifiers cannot be empty")
        _parse_time(self.observed_at, "observed_at")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    record_id: str
    split: SplitName

    def __post_init__(self) -> None:
        _require_exact_str(self.record_id, "split assignment record_id")
        if type(self.split) is not SplitName:
            raise TypeError("split assignment split must be a SplitName")


@dataclass(frozen=True, slots=True)
class SplitValidation:
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact_bool(self.passed, "split validation passed")
        if type(self.failures) is not tuple or any(
            type(failure) is not str for failure in self.failures
        ):
            raise TypeError("split validation failures must be a tuple of strings")


def _parse_time(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _group_key(record: SplitRecord, level: HierarchyLevel) -> tuple[str, ...]:
    values = (
        record.schema_family,
        record.organization,
        record.copy_cluster,
        record.repository,
    )
    if level is HierarchyLevel.CHRONOLOGICAL_BLOCK:
        return values
    return values[: SPLIT_HIERARCHY.index(level) + 1]


def validate_split_hierarchy(
    records: Iterable[SplitRecord],
    assignments: Iterable[SplitAssignment],
    *,
    held_out_level: HierarchyLevel,
) -> SplitValidation:
    """Validate one declared split boundary in the normative hierarchy.

    For categorical levels no group at or above the boundary may cross splits.
    A chronological boundary permits a repository in multiple splits but enforces
    train → development → public-test → sealed-test time ordering.
    """

    materialized = tuple(records)
    by_id = {record.record_id: record for record in materialized}
    if len(by_id) != len(materialized):
        return SplitValidation(False, ("duplicate split record IDs",))
    assigned_pairs = tuple(assignments)
    assigned = {item.record_id: item.split for item in assigned_pairs}
    failures: list[str] = []
    if len(assigned) != len(assigned_pairs):
        failures.append("duplicate split assignments")
    missing = sorted(set(by_id) - set(assigned))
    unknown = sorted(set(assigned) - set(by_id))
    if missing:
        failures.append(f"records without assignments: {', '.join(missing)}")
    if unknown:
        failures.append(f"assignments for unknown records: {', '.join(unknown)}")

    groups: dict[tuple[str, ...], dict[SplitName, list[datetime]]] = {}
    for record_id in sorted(set(by_id) & set(assigned)):
        record = by_id[record_id]
        groups.setdefault(_group_key(record, held_out_level), {}).setdefault(
            assigned[record_id], []
        ).append(_parse_time(record.observed_at, "observed_at"))

    if held_out_level is not HierarchyLevel.CHRONOLOGICAL_BLOCK:
        for key, split_times in sorted(groups.items()):
            if len(split_times) > 1:
                failures.append(f"{held_out_level.value} group {key!r} crosses splits")
    else:
        order = (
            SplitName.TRAIN,
            SplitName.DEVELOPMENT,
            SplitName.PUBLIC_TEST,
            SplitName.SEALED_TEST,
        )
        for key, split_times in sorted(groups.items()):
            previous_latest: datetime | None = None
            for split in order:
                times = split_times.get(split, [])
                if not times:
                    continue
                if previous_latest is not None and min(times) <= previous_latest:
                    failures.append(f"repository group {key!r} violates chronological order")
                    break
                previous_latest = max(times)
    return SplitValidation(not failures, tuple(failures))


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Content-addressed declaration of a single benchmark split boundary."""

    dataset_id: str
    version: str
    created_at: str
    held_out_level: HierarchyLevel
    records: tuple[SplitRecord, ...]
    assignments: tuple[SplitAssignment, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("version", self.version),
            ("created_at", self.created_at),
        ):
            _require_exact_str(value, f"split manifest {name}")
        if type(self.held_out_level) is not HierarchyLevel:
            raise TypeError("held_out_level must be a HierarchyLevel")
        if type(self.records) is not tuple or any(
            type(record) is not SplitRecord for record in self.records
        ):
            raise TypeError("split records must be a tuple of SplitRecord values")
        if type(self.assignments) is not tuple or any(
            type(assignment) is not SplitAssignment for assignment in self.assignments
        ):
            raise TypeError("split assignments must be a tuple of SplitAssignment values")
        if not self.dataset_id or not self.version:
            raise ValueError("split manifest dataset ID and version are required")
        _parse_time(self.created_at, "created_at")
        record_ids = [record.record_id for record in self.records]
        assignment_ids = [assignment.record_id for assignment in self.assignments]
        if record_ids != sorted(record_ids) or assignment_ids != sorted(assignment_ids):
            raise ValueError("split records and assignments must be sorted by record ID")
        validation = validate_split_hierarchy(
            self.records,
            self.assignments,
            held_out_level=self.held_out_level,
        )
        if not validation.passed:
            raise ValueError("invalid split manifest: " + "; ".join(validation.failures))

    @property
    def sha256(self) -> str:
        from .evidence import content_hash

        return content_hash(self)

    @property
    def population_identity(self) -> str:
        """Hash the actual typed population assignment, not manifest decoration.

        Dataset/version labels, creation clocks, and record IDs are intentionally
        omitted. Renaming those fields therefore cannot turn a spent population
        into a second prospective cohort.
        """

        from .evidence import content_hash

        assignment_by_id = {item.record_id: item.split for item in self.assignments}
        members = tuple(
            sorted(
                (
                    record.schema_family,
                    record.organization,
                    record.copy_cluster,
                    record.repository,
                    _parse_time(record.observed_at, "observed_at").isoformat(),
                    assignment_by_id[record.record_id].value,
                )
                for record in self.records
            )
        )
        return content_hash(members)


@dataclass(frozen=True, slots=True)
class FitMarker:
    """Proof that a fitted artifact saw only the training partition."""

    artifact_name: str
    artifact_hash: str
    split_manifest_hash: str
    fitted_splits: tuple[SplitName, ...]
    training_record_ids_hash: str

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_name", self.artifact_name),
            ("artifact_hash", self.artifact_hash),
            ("split_manifest_hash", self.split_manifest_hash),
            ("training_record_ids_hash", self.training_record_ids_hash),
        ):
            _require_exact_str(value, f"fit marker {name}")
        if type(self.fitted_splits) is not tuple or any(
            type(split) is not SplitName for split in self.fitted_splits
        ):
            raise TypeError("fitted_splits must be a tuple of SplitName values")
        if not self.artifact_name:
            raise ValueError("fitted artifact name is required")
        for value in (self.artifact_hash, self.split_manifest_hash, self.training_record_ids_hash):
            if not _SHA256.fullmatch(value):
                raise ValueError("fit markers require lowercase SHA-256 hashes")

    @property
    def train_only(self) -> bool:
        return self.fitted_splits == (SplitName.TRAIN,)
