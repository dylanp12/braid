"""Canonical, lossless JSON serialization for contract records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from . import types as contract_types

_RECORD_TAG = "$record"
_DATETIME_TAG = "$datetime"
_DURATION_TAG = "$duration_seconds"
_ENUM_TAG = "$enum"


def _record_types() -> dict[str, type[contract_types.ContractRecord]]:
    return {
        name: value
        for name, value in vars(contract_types).items()
        if isinstance(value, type)
        and issubclass(value, contract_types.ContractRecord)
        and value is not contract_types.ContractRecord
    }


def _enum_types() -> dict[str, type[Enum]]:
    return {"Operation": contract_types.Operation}


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("cannot serialize a naive datetime")
    normalized = value.astimezone(UTC)
    text = normalized.isoformat(timespec="microseconds")
    return text.removesuffix("+00:00") + "Z"


def to_primitive(value: Any) -> Any:
    """Convert records and supported standard-library values to JSON values."""

    if isinstance(value, datetime):
        return {_DATETIME_TAG: _format_datetime(value)}
    if isinstance(value, timedelta):
        return {_DURATION_TAG: value.total_seconds()}
    if isinstance(value, Enum):
        return {_ENUM_TAG: f"{type(value).__name__}.{value.name}"}
    if is_dataclass(value) and isinstance(value, contract_types.ContractRecord):
        result = {_RECORD_TAG: type(value).__name__}
        result.update(
            {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
        )
        return result
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinite floats are not valid contract values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def from_primitive(value: Any) -> Any:
    """Restore a value produced by :func:`to_primitive`."""

    if isinstance(value, list):
        return tuple(from_primitive(item) for item in value)
    if not isinstance(value, Mapping):
        return value
    if set(value) == {_DATETIME_TAG}:
        text = str(value[_DATETIME_TAG])
        suffix = "+00:00" if text.endswith("Z") else ""
        parsed = datetime.fromisoformat(text.removesuffix("Z") + suffix)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("serialized datetime is naive")
        return parsed
    if set(value) == {_DURATION_TAG}:
        return timedelta(seconds=float(value[_DURATION_TAG]))
    if set(value) == {_ENUM_TAG}:
        enum_name, member_name = str(value[_ENUM_TAG]).split(".", 1)
        try:
            return _enum_types()[enum_name][member_name]
        except KeyError as exc:
            raise ValueError(f"unknown contract enum {value[_ENUM_TAG]!r}") from exc
    if _RECORD_TAG in value:
        type_name = str(value[_RECORD_TAG])
        try:
            record_type = _record_types()[type_name]
        except KeyError as exc:
            raise ValueError(f"unknown contract record {type_name!r}") from exc
        unknown = set(value) - {_RECORD_TAG} - {field.name for field in fields(record_type)}
        if unknown:
            raise ValueError(f"unknown fields for {type_name}: {sorted(unknown)}")
        kwargs = {
            field.name: from_primitive(value[field.name])
            for field in fields(record_type)
            if field.name in value
        }
        return record_type(**kwargs)
    return {str(key): from_primitive(item) for key, item in value.items()}


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def loads(value: str | bytes) -> Any:
    return from_primitive(json.loads(value))
