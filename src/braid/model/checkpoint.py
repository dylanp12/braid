"""Hash-bound, weights-only EventGraph checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from braid.model.config import EventGraphConfig
from braid.model.model import EventGraphModel


class CheckpointCompatibilityError(ValueError):
    """A checkpoint does not match the requested research lineage."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    format_version: int
    created_at: str
    training_steps: int
    architecture_hash: str
    tokenizer_hash: str
    schema_hash: str
    code_hash: str
    environment_hash: str
    data_hash: str
    split_hash: str
    evaluation_hash: str
    checkpoint_hash: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.format_version, bool) or not isinstance(self.format_version, int):
            raise TypeError("format_version must be an integer")
        if self.format_version <= 0:
            raise ValueError("format_version must be positive")
        if isinstance(self.training_steps, bool) or not isinstance(self.training_steps, int):
            raise TypeError("training_steps must be an integer")
        if self.training_steps < 0:
            raise ValueError("training_steps cannot be negative")
        try:
            created_at = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("created_at must be an ISO-8601 datetime") from exc
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        for field in (
            "architecture_hash",
            "tokenizer_hash",
            "schema_hash",
            "code_hash",
            "environment_hash",
            "data_hash",
            "split_hash",
            "evaluation_hash",
        ):
            _digest(getattr(self, field), field)
        if self.checkpoint_hash:
            _digest(self.checkpoint_hash, "checkpoint_hash")

    @classmethod
    def for_model(
        cls,
        model: EventGraphModel,
        *,
        tokenizer_hash: str,
        schema_hash: str,
        code_hash: str,
        environment_hash: str,
        data_hash: str,
        split_hash: str,
        evaluation_hash: str,
        training_steps: int,
    ) -> CheckpointMetadata:
        if training_steps < 0:
            raise ValueError("training_steps cannot be negative")
        return cls(
            format_version=1,
            created_at=datetime.now(UTC).isoformat(),
            training_steps=training_steps,
            architecture_hash=model.architecture_fingerprint,
            tokenizer_hash=_digest(tokenizer_hash, "tokenizer_hash"),
            schema_hash=_digest(schema_hash, "schema_hash"),
            code_hash=_digest(code_hash, "code_hash"),
            environment_hash=_digest(environment_hash, "environment_hash"),
            data_hash=_digest(data_hash, "data_hash"),
            split_hash=_digest(split_hash, "split_hash"),
            evaluation_hash=_digest(evaluation_hash, "evaluation_hash"),
        )

    @property
    def manifest_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def checksum_bound(self) -> bool:
        """Whether metadata names an immutable serialized weight artifact."""

        return bool(self.checkpoint_hash)

    def to_contract_manifest(self) -> Any:
        """Return the public wire manifest while retaining schema/training metadata."""

        from braid.contract.types import ModelManifest

        return ModelManifest(
            architecture_hash=self.architecture_hash,
            tokenizer_hash=self.tokenizer_hash,
            code_hash=self.code_hash,
            environment_hash=self.environment_hash,
            data_hash=self.data_hash,
            split_hash=self.split_hash,
            checkpoint_hash=self.checkpoint_hash,
            evaluation_hash=self.evaluation_hash,
            metadata={
                "schema_hash": self.schema_hash,
                "training_steps": self.training_steps,
                "checkpoint_format": self.format_version,
            },
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: EventGraphModel
    metadata: CheckpointMetadata


def save_checkpoint(
    directory: str | os.PathLike[str],
    model: EventGraphModel,
    metadata: CheckpointMetadata,
) -> CheckpointMetadata:
    """Write SafeTensors and a canonical manifest without executable pickle objects."""

    if metadata.architecture_hash != model.architecture_fingerprint:
        raise CheckpointCompatibilityError("metadata architecture hash does not match model")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"checkpoint directory is not empty: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path, prefix="weights-", suffix=".safetensors.tmp", delete=False
    ) as tmp:
        temporary_weights = Path(tmp.name)
    try:
        state = {
            name: tensor.detach().contiguous().cpu() for name, tensor in model.state_dict().items()
        }
        save_file(state, str(temporary_weights))
        checkpoint_hash = _file_hash(temporary_weights)
        complete = replace(metadata, checkpoint_hash=checkpoint_hash)
        manifest = {
            "metadata": asdict(complete),
            "config": model.config.to_dict(),
            "tokenizer_vocab_size": model.tokenizer_vocab_size,
        }
        temporary_manifest = path / "manifest.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_weights, path / "weights.safetensors")
        os.replace(temporary_manifest, path / "manifest.json")
        return complete
    finally:
        temporary_weights.unlink(missing_ok=True)


def load_checkpoint(
    directory: str | os.PathLike[str],
    *,
    expected_hashes: Mapping[str, str] | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metadata = CheckpointMetadata(**manifest["metadata"])
    if metadata.format_version != 1:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format {metadata.format_version}"
        )
    allowed = {
        "architecture_hash",
        "tokenizer_hash",
        "schema_hash",
        "code_hash",
        "environment_hash",
        "data_hash",
        "split_hash",
        "evaluation_hash",
        "checkpoint_hash",
    }
    for name, expected in (expected_hashes or {}).items():
        if name not in allowed:
            raise KeyError(f"unknown checkpoint hash field {name!r}")
        expected = _digest(expected, name)
        actual = getattr(metadata, name)
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"{name} mismatch: expected {expected}, checkpoint has {actual}"
            )
    weights = path / "weights.safetensors"
    actual_weight_hash = _file_hash(weights)
    if actual_weight_hash != metadata.checkpoint_hash:
        raise CheckpointCompatibilityError("checkpoint tensor hash does not match manifest")
    config = EventGraphConfig.from_dict(manifest["config"])
    model = EventGraphModel(config, tokenizer_vocab_size=int(manifest["tokenizer_vocab_size"]))
    if model.architecture_fingerprint != metadata.architecture_hash:
        raise CheckpointCompatibilityError("manifest architecture does not match constructed model")
    device = torch.device(map_location)
    state = load_file(str(weights), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model._braid_verified_checkpoint_hash = metadata.checkpoint_hash
    model._braid_verified_manifest_id = metadata.manifest_id
    model._braid_verified_state_versions = _state_versions(model)
    return LoadedCheckpoint(model=model, metadata=metadata)


def checkpoint_is_bound(model: EventGraphModel, metadata: CheckpointMetadata) -> bool:
    """Return whether ``model`` came from the checksum-verified checkpoint loader.

    This is an accidental-mismatch guard, not a security boundary against Python code
    that deliberately mutates private attributes.  It prevents an initialized or
    separately trained model from being paired with plausible-looking metadata.
    """

    return (
        metadata.format_version == 1
        and metadata.checksum_bound
        and getattr(model, "_braid_verified_checkpoint_hash", None) == metadata.checkpoint_hash
        and getattr(model, "_braid_verified_manifest_id", None) == metadata.manifest_id
        and getattr(model, "_braid_verified_state_versions", None) == _state_versions(model)
    )


def _state_versions(model: EventGraphModel) -> tuple[tuple[str, int], ...]:
    state = (*model.named_parameters(), *model.named_buffers())
    return tuple((name, int(tensor._version)) for name, tensor in state)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value
