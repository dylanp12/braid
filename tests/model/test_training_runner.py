from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from braid.model.checkpoint import load_checkpoint
from braid.model.config import SCALE_CONFIGS, TINY_CONFIG
from braid.model.tokenizer import BraidTokenizer
from braid.model.training_runner import (
    SNAPSHOT_KIND,
    DeterministicObjectiveIterator,
    SupervisedForecastExample,
    TrainingDatasetManifest,
    TrainingLineage,
    TrainingRunConfig,
    config_from_declaration,
    fit_tokenizer_train_only,
    load_training_examples,
    run_training,
    smoke_training_examples,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_training_example_round_trip_and_complete_censoring_contract(tmp_path) -> None:
    examples = smoke_training_examples()
    quiet = next(example for example in examples if example.is_censored)
    restored = SupervisedForecastExample.from_json(quiet.to_json())

    assert restored == quiet
    assert restored.outcome_complete is True
    assert restored.censor_at == restored.request.cutoff + restored.request.horizon

    source = tmp_path / "training.jsonl"
    source.write_text("\n".join(example.to_json() for example in examples) + "\n")
    assert load_training_examples(source) == examples


def test_training_example_fails_closed_on_incomplete_or_nontraining_data() -> None:
    example = smoke_training_examples()[0]
    with pytest.raises(ValueError, match="outcome_complete=true"):
        replace(example, outcome_complete=False)
    with pytest.raises(ValueError, match="train split only"):
        replace(example, split="development")
    with pytest.raises(ValueError, match="censor_at must equal"):
        replace(example, censor_at=example.censor_at.replace(year=2027))

    raw = json.loads(example.to_json())
    del raw["outcome_complete"]
    with pytest.raises(ValueError, match="missing fields"):
        SupervisedForecastExample.from_json(json.dumps(raw))


def test_train_only_tokenizer_guard_and_objective_iterator_are_deterministic() -> None:
    examples = smoke_training_examples()
    first = DeterministicObjectiveIterator(examples, seed=19)
    second = DeterministicObjectiveIterator(tuple(reversed(examples)), seed=19)

    assert {
        objective: example.example_id for objective, example in first.objective_batch(0).items()
    } == {objective: example.example_id for objective, example in second.objective_batch(0).items()}
    receipt = fit_tokenizer_train_only(BraidTokenizer(), examples)
    assert receipt.fitted_split == "train"
    assert len(receipt.receipt_hash) == 64


def test_smoke_runner_executes_exact_mixture_and_censored_survival() -> None:
    result = run_training(smoke_training_examples())

    assert result.optimizer_steps == 1
    assert result.objective_weights == {
        "patch": 0.65,
        "schema_episode": 0.15,
        "reconstruction": 0.10,
        "contrastive": 0.10,
    }
    assert result.objective_example_counts == {
        "patch": 1,
        "schema_episode": 1,
        "reconstruction": 1,
        "contrastive": 1,
    }
    assert result.censored_example_count == 1
    assert result.raw_encoded_tokens > 0
    assert result.diverse_training_tokens is None
    assert result.diverse_data_floor_passed is None
    assert result.mean_objective_losses["contrastive"] > 0
    assert result.artifact_kind == SNAPSHOT_KIND
    assert result.claim_level == "none"
    assert result.qualified is False
    assert result.cloud_authorized is False
    assert result.snapshots == ()


def test_snapshots_resume_optimizer_but_remain_unqualified(tmp_path) -> None:
    lineage = TrainingLineage(digest("code"), digest("environment"))
    output = tmp_path / "run"
    first = run_training(
        smoke_training_examples(),
        run_config=TrainingRunConfig(total_steps=1),
        lineage=lineage,
        output_directory=output,
    )
    snapshot = output / "step-00000001"
    loaded = load_checkpoint(snapshot / "model")

    assert first.snapshots == (str(snapshot),)
    assert loaded.metadata.training_steps == 0
    assert (snapshot / "optimizer.safetensors").is_file()
    assert (snapshot / "optimizer.json").is_file()

    resumed = run_training(
        smoke_training_examples(),
        run_config=TrainingRunConfig(total_steps=2),
        lineage=lineage,
        output_directory=output,
        resume_snapshot=snapshot,
    )
    assert resumed.start_step == 1
    assert resumed.optimizer_steps == 2
    assert resumed.snapshots == (str(output / "step-00000002"),)
    assert load_checkpoint(output / "step-00000002" / "model").metadata.training_steps == 0


def test_scale_and_data_gates_fail_before_allocation() -> None:
    with pytest.raises(ValueError, match="explicitly local scale"):
        run_training(smoke_training_examples(), model_config=SCALE_CONFIGS["300m"])

    local_but_underfed = replace(
        TINY_CONFIG,
        name="explicit-local-test",
        execution_policy="local explicit test",
        min_diverse_training_tokens=10**9,
    )
    receipt = fit_tokenizer_train_only(BraidTokenizer(), smoke_training_examples())
    dataset_manifest = TrainingDatasetManifest(
        data_hash=receipt.corpus_hash,
        split_manifest_hash=digest("split-manifest"),
        training_record_ids_hash=receipt.training_record_ids_hash,
        diversity_audit_hash=digest("diversity-audit"),
        diverse_training_tokens=10,
    )
    with pytest.raises(ValueError, match="diverse-token floor"):
        run_training(
            smoke_training_examples(),
            model_config=local_but_underfed,
            dataset_manifest=dataset_manifest,
        )

    declaration = SCALE_CONFIGS["300m"].to_dict() | {
        "promotion_policy": {"requires_explicit_cloud_approval": True}
    }
    with pytest.raises(ValueError, match="cannot authorize"):
        config_from_declaration(declaration)
