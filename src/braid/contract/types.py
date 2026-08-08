"""Immutable public records for the Braid v2 causal graph contract.

The dataclasses in this module deliberately contain no model-framework types.  They
form the wire boundary shared by data preparation, training, evaluation, and
independent custodians.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias, TypeVar

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]

_RecordT = TypeVar("_RecordT", bound="ContractRecord")


def _freeze(value: Any) -> Any:
    """Recursively freeze caller-owned containers.

    Besides making records effectively immutable, this prevents a subtle source of
    manifest/hash drift: mutating the dict used to construct a frozen dataclass.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class ContractRecord:
    """Serialization and hashing helpers inherited by every public record."""

    RECORD_VERSION: ClassVar[str] = "2.0"

    def __post_init__(self) -> None:
        for record_field in fields(self):
            object.__setattr__(
                self,
                record_field.name,
                _freeze(getattr(self, record_field.name)),
            )

    def to_dict(self) -> dict[str, Any]:
        from .serde import to_primitive

        encoded = to_primitive(self)
        if not isinstance(encoded, dict):  # pragma: no cover - defensive invariant
            raise TypeError(f"{type(self).__name__} did not encode as an object")
        return encoded

    @classmethod
    def from_dict(cls: type[_RecordT], value: Mapping[str, Any]) -> _RecordT:
        from .serde import from_primitive

        decoded = from_primitive(value)
        if not isinstance(decoded, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(decoded).__name__}")
        return decoded

    def to_json(self) -> str:
        from .serde import canonical_json

        return canonical_json(self)

    @classmethod
    def from_json(cls: type[_RecordT], value: str | bytes) -> _RecordT:
        from .serde import loads

        decoded = loads(value)
        if not isinstance(decoded, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(decoded).__name__}")
        return decoded

    def canonical_hash(self) -> str:
        from .serde import canonical_hash

        return canonical_hash(self)


class Operation(StrEnum):
    """The complete and intentionally closed v2 event operation vocabulary."""

    CREATE_NODE = "CREATE_NODE"
    UPDATE_NODE = "UPDATE_NODE"
    ASSERT = "ASSERT"
    RETRACT = "RETRACT"
    SUPERSEDE = "SUPERSEDE"
    SCHEMA_CHANGE = "SCHEMA_CHANGE"
    EXPOSE = "EXPOSE"
    JUDGE = "JUDGE"


@dataclass(frozen=True, slots=True)
class NodeTypeDecl(ContractRecord):
    name: str
    description: str = ""
    constraints: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class RoleDecl(ContractRecord):
    name: str
    allowed_node_types: tuple[str, ...]
    min_count: int = 1
    max_count: int | None = 1


@dataclass(frozen=True, slots=True)
class RelationDecl(ContractRecord):
    name: str
    roles: tuple[RoleDecl, ...]
    description: str = ""
    inverse: str | None = None
    constraints: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class SupportExample(ContractRecord):
    """Schema-only example: role bindings contain types, never global entity IDs."""

    relation: str
    role_types: Mapping[str, str]
    text: str = ""


@dataclass(frozen=True, slots=True)
class SchemaDecl(ContractRecord):
    version: str
    observed_at: datetime | None
    node_types: tuple[NodeTypeDecl, ...]
    relations: tuple[RelationDecl, ...]
    constraints: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))
    support_examples: tuple[SupportExample, ...] = ()
    observed_at_imputed: bool = False


@dataclass(frozen=True, slots=True)
class PayloadSnapshot(ContractRecord):
    observed_at: datetime | None
    payload: Mapping[str, JsonValue]
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    basis_refs: tuple[str, ...] = ()
    observed_at_imputed: bool = False


@dataclass(frozen=True, slots=True)
class NodeRecord(ContractRecord):
    node_id: str
    node_type: str
    schema_version: str
    observed_at: datetime | None
    payload_history: tuple[PayloadSnapshot, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    observed_at_imputed: bool = False

    def payload_as_of(self, cutoff: datetime) -> Mapping[str, JsonValue]:
        """Return the newest causally visible payload snapshot."""

        visible = [
            snapshot
            for snapshot in self.payload_history
            if snapshot.observed_at is not None and snapshot.observed_at <= cutoff
        ]
        return visible[-1].payload if visible else MappingProxyType({})


@dataclass(frozen=True, slots=True)
class EvidenceRecord(ContractRecord):
    evidence_id: str
    observed_at: datetime | None
    kind: str
    source_uri: str | None = None
    content_hash: str | None = None
    payload: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))
    parent_refs: tuple[str, ...] = ()
    observed_at_imputed: bool = False


@dataclass(frozen=True, slots=True)
class RoleBinding(ContractRecord):
    role: str
    node_id: str


@dataclass(frozen=True, slots=True)
class DerivationRecord(ContractRecord):
    method: str
    input_refs: tuple[str, ...] = ()
    parameters: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class ProvenanceRecord(ContractRecord):
    source: str
    source_record_id: str | None
    license: str | None
    acquired_at: datetime | None


@dataclass(frozen=True, slots=True)
class GraphEventV2(ContractRecord):
    event_id: str
    observed_at: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    operation: Operation
    schema_version: str
    relation: str | None
    arguments: tuple[RoleBinding, ...]
    payload: Mapping[str, JsonValue]
    basis_refs: tuple[str, ...]
    derivation: DerivationRecord
    provenance: ProvenanceRecord
    tie_group: str | None = None
    observed_at_imputed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.operation, str):
            object.__setattr__(self, "operation", Operation(self.operation))
        # ``dataclass(slots=True)`` creates a replacement class, which makes
        # zero-argument ``super()`` unreliable on some supported Python builds.
        ContractRecord.__post_init__(self)


@dataclass(frozen=True, slots=True)
class SourceLineage(ContractRecord):
    source_contract: str
    source_kind: str
    source_id: str
    source_hash: str
    target_kind: str
    target_id: str
    converter: str
    observed_at: datetime | None


@dataclass(frozen=True, slots=True)
class GraphBundleV2(ContractRecord):
    bundle_id: str
    schemas: tuple[SchemaDecl, ...]
    nodes: tuple[NodeRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    events: tuple[GraphEventV2, ...]
    lineage: tuple[SourceLineage, ...] = ()

    @property
    def schema(self) -> SchemaDecl:
        """Return the latest declaration; validated bundles cannot tie schema clocks."""

        if not self.schemas:
            raise LookupError("bundle has no schema declarations")
        return max(
            self.schemas,
            key=lambda item: (
                item.observed_at is not None,
                item.observed_at or datetime.min,
                item.version,
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskDeclaration(ContractRecord):
    name: str
    query_node_ids: tuple[str, ...] = ()
    target_relations: tuple[str, ...] = ()
    options: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class ForecastRequest(ContractRecord):
    request_id: str
    schema: SchemaDecl
    prefix: GraphBundleV2
    cutoff: datetime | None
    horizon: timedelta
    task: TaskDeclaration
    support_events: tuple[GraphEventV2, ...] = ()


@dataclass(frozen=True, slots=True)
class PatchWindow(ContractRecord):
    events: tuple[GraphEventV2, ...]
    probability: float


@dataclass(frozen=True, slots=True)
class EventMarginal(ContractRecord):
    event: GraphEventV2
    probability: float


@dataclass(frozen=True, slots=True)
class ForecastDistribution(ContractRecord):
    sampled_patch_windows: tuple[PatchWindow, ...]
    event_marginals: tuple[EventMarginal, ...]
    calibrated_uncertainty: float
    abstention_reason: str | None
    retrieval_coverage: float
    model_manifest_id: str


@dataclass(frozen=True, slots=True)
class ModelManifest(ContractRecord):
    architecture_hash: str
    tokenizer_hash: str
    code_hash: str
    environment_hash: str
    data_hash: str
    split_hash: str
    checkpoint_hash: str
    evaluation_hash: str
    metadata: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def manifest_id(self) -> str:
        return self.canonical_hash()
