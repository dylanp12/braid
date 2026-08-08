"""Deterministic byte-fallback tokenization and episode-local handles."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class HandleKind(StrEnum):
    TYPE = "type"
    RELATION = "relation"
    ROLE = "role"
    NODE = "node"
    EVIDENCE = "evidence"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class LocalHandle:
    """An episode-local reference; ``external_id`` is never embedded."""

    kind: HandleKind
    index: int
    external_id: str


class DynamicHandleTable:
    """Assign dense handles in declaration/observation order.

    Renaming external identifiers while preserving schema and event order therefore
    leaves all numeric model inputs unchanged.
    """

    def __init__(self) -> None:
        self._by_kind: dict[HandleKind, dict[str, int]] = {kind: {} for kind in HandleKind}
        self._external: dict[HandleKind, list[str]] = {kind: [] for kind in HandleKind}

    def register(self, kind: HandleKind | str, external_id: str) -> LocalHandle:
        kind = HandleKind(kind)
        external_id = str(external_id)
        existing = self._by_kind[kind].get(external_id)
        if existing is None:
            existing = len(self._external[kind])
            self._by_kind[kind][external_id] = existing
            self._external[kind].append(external_id)
        return LocalHandle(kind, existing, external_id)

    def resolve(self, kind: HandleKind | str, external_id: str) -> LocalHandle:
        kind = HandleKind(kind)
        try:
            index = self._by_kind[kind][str(external_id)]
        except KeyError as exc:
            raise KeyError(f"unknown {kind.value} handle {external_id!r}") from exc
        return LocalHandle(kind, index, str(external_id))

    def external_id(self, kind: HandleKind | str, index: int) -> str:
        kind = HandleKind(kind)
        try:
            return self._external[kind][index]
        except IndexError as exc:
            raise KeyError(f"unknown {kind.value} index {index}") from exc

    def count(self, kind: HandleKind | str) -> int:
        return len(self._external[HandleKind(kind)])

    def model_view(self) -> dict[str, tuple[int, ...]]:
        """Return the identifier-free view that is safe to feed to a model."""

        return {kind.value: tuple(range(len(self._external[kind]))) for kind in HandleKind}


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    normalization: str = "NFC"
    byte_offset: int = 16
    version: int = 1


@dataclass(frozen=True, slots=True)
class EncodedSchemaItem:
    kind: HandleKind
    handle: int
    token_ids: tuple[int, ...]


class BraidTokenizer:
    """A fixed byte vocabulary with schema-aware framing.

    UTF-8 bytes provide deterministic fallback for every string.  Schema identifiers
    select local handles but are intentionally excluded from the encoded text.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    TYPE = 3
    RELATION = 4
    ROLE = 5
    DESCRIPTION = 6
    SEPARATOR = 7
    PAYLOAD = 8
    UNKNOWN = 9

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        if self.config.byte_offset < 16:
            raise ValueError("byte_offset must leave room for reserved tokens")

    @property
    def vocab_size(self) -> int:
        return self.config.byte_offset + 256

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "class": type(self).__name__,
                "config": {
                    "normalization": self.config.normalization,
                    "byte_offset": self.config.byte_offset,
                    "version": self.config.version,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def encode_text(self, text: str, *, framed: bool = True) -> tuple[int, ...]:
        normalized = unicodedata.normalize(self.config.normalization, str(text))
        encoded = tuple(self.config.byte_offset + byte for byte in normalized.encode("utf-8"))
        return (self.BOS, *encoded, self.EOS) if framed else encoded

    def decode_text(self, token_ids: Iterable[int], *, errors: str = "strict") -> str:
        values = []
        for token in token_ids:
            if token in (self.BOS, self.EOS, self.PAD):
                continue
            byte = token - self.config.byte_offset
            if not 0 <= byte <= 255:
                raise ValueError(f"token {token} is not a byte token")
            values.append(byte)
        return bytes(values).decode("utf-8", errors=errors)

    def encode_schema(
        self,
        *,
        types: Sequence[Any],
        relations: Sequence[Any],
        handles: DynamicHandleTable | None = None,
        schema_constraints: Mapping[str, Any] | None = None,
        support_examples: Sequence[Any] = (),
    ) -> tuple[DynamicHandleTable, tuple[EncodedSchemaItem, ...]]:
        """Encode schema declarations without embedding their identifiers.

        Values may be mappings or objects.  A declaration uses ``id``/``name`` as
        its external handle, while only description, constraints, and role semantics
        enter the token stream.
        """

        handles = handles or DynamicHandleTable()
        types = tuple(types)
        relations = tuple(relations)

        # Allocate every declaration handle before rendering any semantic text. This
        # lets constraints, inverses, and support examples refer to local handles
        # without exposing external identifier spellings to the model.
        for declaration in types:
            external_id = _field(declaration, "id", "type_id", "name")
            handles.register(HandleKind.TYPE, external_id)
        for relation in relations:
            external_id = _field(relation, "id", "relation_id", "name")
            handles.register(HandleKind.RELATION, external_id)
            for role in _field(relation, "roles", default=()):
                role_id = _field(role, "id", "role_id", "name")
                handles.register(HandleKind.ROLE, f"{external_id}:{role_id}")

        references = _schema_reference_markers(types, relations, handles)
        encoded_schema_constraints = _rename_schema_references(schema_constraints or {}, references)
        examples_by_relation = _encode_support_examples(support_examples, handles)

        items: list[EncodedSchemaItem] = []
        for declaration in types:
            external_id = _field(declaration, "id", "type_id", "name")
            handle = handles.resolve(HandleKind.TYPE, external_id)
            description = _field(declaration, "description", default="")
            constraints = _rename_schema_references(
                _field(declaration, "constraints", default={}), references
            )
            if encoded_schema_constraints:
                constraints = {
                    "declaration": constraints,
                    "schema": encoded_schema_constraints,
                }
            text = _semantic_text(description, constraints)
            token_ids = (
                self.BOS,
                self.TYPE,
                self.DESCRIPTION,
                *self.encode_text(text, framed=False),
                self.EOS,
            )
            items.append(EncodedSchemaItem(HandleKind.TYPE, handle.index, token_ids))

        for relation in relations:
            external_id = _field(relation, "id", "relation_id", "name")
            handle = handles.resolve(HandleKind.RELATION, external_id)
            description = _field(relation, "description", default="")
            relation_constraints = _rename_schema_references(
                _field(relation, "constraints", default={}), references
            )
            inverse = _field(relation, "inverse", default=None)
            inverse_handle = (
                None if inverse is None else handles.resolve(HandleKind.RELATION, inverse).index
            )
            relation_semantics: dict[str, Any] = {
                "constraints": relation_constraints,
                "inverse_relation_handle": inverse_handle,
                "support_examples": examples_by_relation.get(handle.index, ()),
            }
            if encoded_schema_constraints:
                relation_semantics["schema"] = encoded_schema_constraints
            pieces = [_semantic_text(description, relation_semantics)]
            for role in _field(relation, "roles", default=()):
                role_id = _field(role, "id", "role_id", "name")
                role_handle = handles.resolve(HandleKind.ROLE, f"{external_id}:{role_id}")
                role_description = _field(role, "description", default="")
                targets = _field(role, "allowed_node_types", default=())
                if not targets:
                    target = _field(role, "target_type", "type_id", "target", default="")
                    targets = (target,) if target else ()
                # Resolve for structural validation, but never encode the external ID.
                target_handles = tuple(
                    handles.resolve(HandleKind.TYPE, target).index for target in targets
                )
                cardinality = {
                    "min": _field(role, "min_count", "minimum", default=0),
                    "max": _field(role, "max_count", "maximum", default=None),
                    "target_type_handles": target_handles,
                }
                if encoded_schema_constraints:
                    cardinality["schema"] = encoded_schema_constraints
                role_text = _semantic_text(role_description, cardinality)
                role_tokens = (
                    self.BOS,
                    self.ROLE,
                    self.DESCRIPTION,
                    *self.encode_text(role_text, framed=False),
                    self.EOS,
                )
                items.append(EncodedSchemaItem(HandleKind.ROLE, role_handle.index, role_tokens))
                pieces.append(role_text)
            relation_text = " | ".join(pieces)
            relation_tokens = (
                self.BOS,
                self.RELATION,
                self.DESCRIPTION,
                *self.encode_text(relation_text, framed=False),
                self.EOS,
            )
            items.append(EncodedSchemaItem(HandleKind.RELATION, handle.index, relation_tokens))
        return handles, tuple(items)

    def encode_schema_decl(
        self, schema: Any, handles: DynamicHandleTable | None = None
    ) -> tuple[DynamicHandleTable, tuple[EncodedSchemaItem, ...]]:
        """Encode a :class:`braid.contract.SchemaDecl` without importing it eagerly."""

        return self.encode_schema(
            types=_field(schema, "node_types"),
            relations=_field(schema, "relations"),
            handles=handles,
            schema_constraints=_field(schema, "constraints", default={}),
            support_examples=_field(schema, "support_examples", default=()),
        )


def _schema_reference_markers(
    types: Sequence[Any],
    relations: Sequence[Any],
    handles: DynamicHandleTable,
) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}

    def register(external_id: Any, marker: str) -> None:
        candidates.setdefault(str(external_id), set()).add(marker)

    for declaration in types:
        external_id = _field(declaration, "id", "type_id", "name")
        handle = handles.resolve(HandleKind.TYPE, external_id).index
        register(external_id, f"$type:{handle}")
    for relation in relations:
        external_id = _field(relation, "id", "relation_id", "name")
        relation_handle = handles.resolve(HandleKind.RELATION, external_id).index
        register(external_id, f"$relation:{relation_handle}")
        for role in _field(relation, "roles", default=()):
            role_id = _field(role, "id", "role_id", "name")
            composite = f"{external_id}:{role_id}"
            role_handle = handles.resolve(HandleKind.ROLE, composite).index
            marker = f"$role:{role_handle}"
            register(composite, marker)
            register(role_id, marker)
    return {
        external_id: next(iter(markers))
        for external_id, markers in candidates.items()
        if len(markers) == 1
    }


def _rename_schema_references(value: Any, references: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            references.get(str(key), str(key)): _rename_schema_references(item, references)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_rename_schema_references(item, references) for item in value)
    if isinstance(value, str):
        return references.get(value, value)
    return value


def _encode_support_examples(
    examples: Sequence[Any], handles: DynamicHandleTable
) -> dict[int, tuple[dict[str, Any], ...]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for example in examples:
        relation = _field(example, "relation", "relation_id")
        relation_handle = handles.resolve(HandleKind.RELATION, relation).index
        role_types = _field(example, "role_types", default={})
        if not isinstance(role_types, Mapping):
            raise ValueError("support-example role_types must be a mapping")
        bindings = []
        for role, node_type in role_types.items():
            role_handle = handles.resolve(HandleKind.ROLE, f"{relation}:{role}").index
            type_handle = handles.resolve(HandleKind.TYPE, node_type).index
            bindings.append((role_handle, type_handle))
        grouped.setdefault(relation_handle, []).append(
            {
                "role_type_handles": tuple(sorted(bindings)),
                "text": str(_field(example, "text", default="")),
            }
        )
    return {handle: tuple(items) for handle, items in grouped.items()}


def _field(value: Any, *names: str, default: Any = ...) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not ...:
        return default
    raise ValueError(f"missing required field; expected one of {names}")


def _semantic_text(description: Any, constraints: Any) -> str:
    """Canonical semantic text, excluding declaration IDs."""

    suffix = json.dumps(constraints, sort_keys=True, separators=(",", ":"), default=str)
    return f"{str(description).strip()}\n{suffix}"
