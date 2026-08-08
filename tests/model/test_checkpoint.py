import hashlib
from dataclasses import replace

import pytest
import torch

from braid.model.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from braid.model.config import TINY_CONFIG
from braid.model.model import EventGraphModel
from braid.model.tokenizer import BraidTokenizer


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_checkpoint_round_trip_binds_every_research_hash(tmp_path) -> None:
    torch.manual_seed(8)
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    metadata = CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=digest("schema"),
        code_hash=digest("code"),
        environment_hash=digest("environment"),
        data_hash=digest("data"),
        split_hash=digest("split"),
        evaluation_hash=digest("evaluation"),
        training_steps=12,
    )
    completed = save_checkpoint(tmp_path / "checkpoint", model, metadata)
    loaded = load_checkpoint(
        tmp_path / "checkpoint",
        expected_hashes={
            "tokenizer_hash": tokenizer.fingerprint,
            "schema_hash": digest("schema"),
            "code_hash": digest("code"),
            "data_hash": digest("data"),
        },
    )
    assert completed.checkpoint_hash
    assert loaded.metadata == completed
    for expected, actual in zip(model.parameters(), loaded.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_wrong_schema_lineage(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    metadata = CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=digest("schema"),
        code_hash=digest("code"),
        environment_hash=digest("environment"),
        data_hash=digest("data"),
        split_hash=digest("split"),
        evaluation_hash=digest("evaluation"),
        training_steps=1,
    )
    save_checkpoint(tmp_path / "checkpoint", model, metadata)
    with pytest.raises(CheckpointCompatibilityError, match="schema_hash mismatch"):
        load_checkpoint(tmp_path / "checkpoint", expected_hashes={"schema_hash": digest("other")})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("training_steps", -1, "cannot be negative"),
        ("created_at", "2026-01-01T00:00:00", "UTC offset"),
        ("data_hash", "NOT-A-DIGEST", "data_hash"),
        ("checkpoint_hash", "abc", "checkpoint_hash"),
    ),
)
def test_checkpoint_metadata_rejects_invalid_lineage_fields(field, value, message) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    metadata = CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=digest("schema"),
        code_hash=digest("code"),
        environment_hash=digest("environment"),
        data_hash=digest("data"),
        split_hash=digest("split"),
        evaluation_hash=digest("evaluation"),
        training_steps=1,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        replace(metadata, **{field: value})


def test_load_checkpoint_honors_requested_model_device(tmp_path, monkeypatch) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    metadata = CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=digest("schema"),
        code_hash=digest("code"),
        environment_hash=digest("environment"),
        data_hash=digest("data"),
        split_hash=digest("split"),
        evaluation_hash=digest("evaluation"),
        training_steps=1,
    )
    save_checkpoint(tmp_path / "checkpoint", model, metadata)

    moved_to: list[torch.device] = []
    original_to = EventGraphModel.to

    def recording_to(self, device, *args, **kwargs):
        moved_to.append(torch.device(device))
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(EventGraphModel, "to", recording_to)
    loaded = load_checkpoint(tmp_path / "checkpoint", map_location=torch.device("cpu"))

    assert moved_to == [torch.device("cpu")]
    assert next(loaded.model.parameters()).device == torch.device("cpu")
    assert loaded.model._braid_verified_checkpoint_hash == loaded.metadata.checkpoint_hash
    assert loaded.model._braid_verified_manifest_id == loaded.metadata.manifest_id


def test_load_checkpoint_rejects_tampered_weight_bytes(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    metadata = CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=digest("schema"),
        code_hash=digest("code"),
        environment_hash=digest("environment"),
        data_hash=digest("data"),
        split_hash=digest("split"),
        evaluation_hash=digest("evaluation"),
        training_steps=1,
    )
    directory = tmp_path / "checkpoint"
    save_checkpoint(directory, model, metadata)
    weights = directory / "weights.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(CheckpointCompatibilityError, match="tensor hash"):
        load_checkpoint(directory)
