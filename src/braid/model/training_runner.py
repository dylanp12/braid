"""Small, deterministic, fail-closed EventGraph training runner.

This module is an engineering reference path, not a promotion mechanism.  It trains
the currently implemented single-next-event heads, writes resumable *unqualified*
snapshots, and deliberately refuses cloud-only scale declarations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from braid.contract.serde import canonical_json, from_primitive
from braid.contract.types import (
    DerivationRecord,
    EvidenceRecord,
    ForecastRequest,
    GraphBundleV2,
    GraphEventV2,
    NodeRecord,
    NodeTypeDecl,
    ProvenanceRecord,
    RelationDecl,
    RoleBinding,
    RoleDecl,
    SchemaDecl,
    TaskDeclaration,
)
from braid.contract.types import Operation as ContractOperation
from braid.contract.validate import validate_forecast_request, validate_proposed_patch
from braid.model.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from braid.model.config import TINY_CONFIG, EventGraphConfig
from braid.model.decoder import FactorizedEventDistribution
from braid.model.grammar import Operation as ModelOperation
from braid.model.model import EventGraphModel
from braid.model.tensorize import TensorizedForecast, tensorize_forecast_request
from braid.model.tokenizer import BraidTokenizer, HandleKind
from braid.model.training import (
    ObjectiveWeights,
    PatchLoss,
    PatchTargets,
    censored_exponential_nll,
    factorized_patch_loss,
)

TRAINING_EXAMPLE_FORMAT = 1
TRAINING_SNAPSHOT_FORMAT = 1
SNAPSHOT_KIND = "unqualified-training-snapshot"
_DIGEST_CHARS = frozenset("0123456789abcdef")
_CONFIG_FIELDS = frozenset(field.name for field in fields(EventGraphConfig))
OBJECTIVE_IMPLEMENTATION_NOTES = {
    "patch": "single-next-event factorized likelihood plus explicit quiet-window survival",
    "schema_episode": (
        "schema-conditioned next-event likelihood; held-out vocabulary membership must be "
        "audited outside this runner"
    ),
    "reconstruction": (
        "held-out next-target factor reconstruction, not masked visible-prefix reconstruction"
    ),
    "contrastive": (
        "node/evidence pointer classification, without explicit hard semantic or structural "
        "corruption generation"
    ),
}


class ObjectiveKind(StrEnum):
    PATCH = "patch"
    SCHEMA_EPISODE = "schema_episode"
    RECONSTRUCTION = "reconstruction"
    CONTRASTIVE = "contrastive"


@dataclass(frozen=True, slots=True)
class SupervisedForecastExample:
    """One causally valid next-event or right-censored training episode.

    ``outcome_complete`` is intentionally mandatory.  It records the data builder's
    assertion that the entire request window was observed; omitting it is not treated
    as a negative event.  The reference heads currently support at most one target
    event, one existing-node pointer, and one evidence pointer per episode.
    """

    example_id: str
    objective: ObjectiveKind
    request: ForecastRequest
    target_events: tuple[GraphEventV2, ...]
    censor_at: datetime
    outcome_complete: bool
    split: str = "train"
    format_version: int = TRAINING_EXAMPLE_FORMAT

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != TRAINING_EXAMPLE_FORMAT
        ):
            raise ValueError(f"unsupported training-example format {self.format_version}")
        if not isinstance(self.example_id, str):
            raise TypeError("training example_id must be a string")
        if not self.example_id.strip():
            raise ValueError("training example_id is required")
        if not isinstance(self.objective, ObjectiveKind):
            raise TypeError("training objective must be an ObjectiveKind")
        if not isinstance(self.request, ForecastRequest):
            raise TypeError("training request must be a ForecastRequest")
        if not isinstance(self.target_events, tuple) or any(
            not isinstance(event, GraphEventV2) for event in self.target_events
        ):
            raise TypeError("target_events must be a tuple of GraphEventV2 records")
        if not isinstance(self.censor_at, datetime):
            raise TypeError("censor_at must be a datetime")
        if not isinstance(self.split, str):
            raise TypeError("training split must be a string")
        if self.split != "train":
            raise ValueError("training examples must come from the train split only")
        if self.outcome_complete is not True:
            raise ValueError("outcome_complete=true is required for every training window")
        if self.censor_at.tzinfo is None or self.censor_at.utcoffset() is None:
            raise ValueError("censor_at must include a UTC offset")
        validate_forecast_request(self.request)
        assert self.request.cutoff is not None
        expected_censor = self.request.cutoff + self.request.horizon
        if self.censor_at != expected_censor:
            raise ValueError("censor_at must equal request.cutoff + request.horizon")
        if len(self.target_events) > 1:
            raise ValueError("the reference runner supports at most one target event")
        if self.objective is ObjectiveKind.SCHEMA_EPISODE and len(
            self.request.support_events
        ) not in {0, 1, 2, 4, 8}:
            raise ValueError("schema episodes require K in {0,1,2,4,8} support events")
        if not self.target_events:
            if self.objective in (ObjectiveKind.RECONSTRUCTION, ObjectiveKind.CONTRASTIVE):
                raise ValueError(f"{self.objective.value} episodes require a target event")
            return

        target = self.target_events[0]
        validate_proposed_patch(self.request, self.target_events)
        assert target.observed_at is not None
        if not self.request.cutoff < target.observed_at <= self.censor_at:
            raise ValueError("target event must be observed after cutoff and by censor_at")
        visible_nodes = {node.node_id for node in self.request.prefix.nodes}
        pointer_targets = {
            binding.node_id for binding in target.arguments if binding.node_id in visible_nodes
        }
        if len(pointer_targets) > 1:
            raise ValueError(
                "target has multiple existing-node pointers; the reference argument head "
                "requires one explicit pointer target"
            )
        if len(target.basis_refs) > 1:
            raise ValueError(
                "target has multiple evidence pointers; the reference evidence head "
                "requires one explicit pointer target"
            )
        if self.objective is ObjectiveKind.CONTRASTIVE and (
            len(pointer_targets) != 1 or len(target.basis_refs) != 1
        ):
            raise ValueError(
                "contrastive episodes require one visible node pointer and one evidence pointer"
            )

    @property
    def is_censored(self) -> bool:
        return not self.target_events

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return canonical_json(
            {
                "censor_at": self.censor_at,
                "example_id": self.example_id,
                "format_version": self.format_version,
                "objective": self.objective.value,
                "outcome_complete": self.outcome_complete,
                "request": self.request,
                "split": self.split,
                "target_events": self.target_events,
            }
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> SupervisedForecastExample:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise TypeError("training JSONL rows must be objects")
        required = {
            "censor_at",
            "example_id",
            "format_version",
            "objective",
            "outcome_complete",
            "request",
            "split",
            "target_events",
        }
        missing = required - set(raw)
        unknown = set(raw) - required
        if missing:
            raise ValueError(f"training example is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"training example has unknown fields: {sorted(unknown)}")
        for name in ("example_id", "objective", "split"):
            if not isinstance(raw[name], str):
                raise TypeError(f"{name} must be a string")
        if isinstance(raw["format_version"], bool) or not isinstance(raw["format_version"], int):
            raise TypeError("format_version must be an integer")
        request = from_primitive(raw["request"])
        target_events = from_primitive(raw["target_events"])
        censor_at = from_primitive(raw["censor_at"])
        if not isinstance(request, ForecastRequest):
            raise TypeError("training example request must be a ForecastRequest")
        if not isinstance(target_events, tuple) or any(
            not isinstance(event, GraphEventV2) for event in target_events
        ):
            raise TypeError("target_events must contain GraphEventV2 records")
        if not isinstance(censor_at, datetime):
            raise TypeError("censor_at must be a serialized datetime")
        outcome_complete = raw["outcome_complete"]
        if not isinstance(outcome_complete, bool):
            raise TypeError("outcome_complete must be a boolean")
        return cls(
            example_id=raw["example_id"],
            objective=ObjectiveKind(raw["objective"]),
            request=request,
            target_events=target_events,
            censor_at=censor_at,
            outcome_complete=outcome_complete,
            split=raw["split"],
            format_version=raw["format_version"],
        )


@dataclass(frozen=True, slots=True)
class TokenizerFitReceipt:
    tokenizer_hash: str
    corpus_hash: str
    training_record_ids_hash: str
    fitted_split: str = "train"

    def __post_init__(self) -> None:
        if self.fitted_split != "train":
            raise ValueError("tokenizer fitting is permitted on the train split only")
        for name in ("tokenizer_hash", "corpus_hash", "training_record_ids_hash"):
            _require_digest(getattr(self, name), name)

    @property
    def receipt_hash(self) -> str:
        return _hash_json(asdict(self))


@dataclass(frozen=True, slots=True)
class TrainingDatasetManifest:
    """External train-split and diversity attestation for non-tiny runs."""

    data_hash: str
    split_manifest_hash: str
    training_record_ids_hash: str
    diversity_audit_hash: str
    diverse_training_tokens: int
    fitted_split: str = "train"
    format_version: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != 1
        ):
            raise ValueError("unsupported training dataset manifest format")
        if self.fitted_split != "train":
            raise ValueError("training dataset manifest must bind only the train split")
        for name in (
            "data_hash",
            "split_manifest_hash",
            "training_record_ids_hash",
            "diversity_audit_hash",
        ):
            _require_digest(getattr(self, name), name)
        if (
            isinstance(self.diverse_training_tokens, bool)
            or not isinstance(self.diverse_training_tokens, int)
            or self.diverse_training_tokens < 0
        ):
            raise ValueError("diverse_training_tokens must be a non-negative integer")

    @property
    def manifest_hash(self) -> str:
        return _hash_json(asdict(self))

    @classmethod
    def from_json(cls, value: str | bytes) -> TrainingDatasetManifest:
        raw = json.loads(value)
        expected = {field.name for field in fields(cls)}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"dataset manifest must contain exactly {sorted(expected)}")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class TrainingLineage:
    """Caller-supplied hashes unavailable to the public runner itself."""

    code_hash: str
    environment_hash: str

    def __post_init__(self) -> None:
        _require_digest(self.code_hash, "code_hash")
        _require_digest(self.environment_hash, "environment_hash")

    @classmethod
    def from_json(cls, value: str | bytes) -> TrainingLineage:
        raw = json.loads(value)
        if not isinstance(raw, dict) or set(raw) != {"code_hash", "environment_hash"}:
            raise ValueError("lineage must contain exactly code_hash and environment_hash")
        return cls(code_hash=str(raw["code_hash"]), environment_hash=str(raw["environment_hash"]))


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    total_steps: int = 1
    save_every_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_steps, bool)
            or not isinstance(self.total_steps, int)
            or self.total_steps <= 0
        ):
            raise ValueError("total_steps must be positive")
        if (
            isinstance(self.save_every_steps, bool)
            or not isinstance(self.save_every_steps, int)
            or self.save_every_steps <= 0
        ):
            raise ValueError("save_every_steps must be positive")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.device:
            raise ValueError("device is required")


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    optimizer_steps: int
    start_step: int
    objective_weights: Mapping[str, float]
    objective_example_counts: Mapping[str, int]
    mean_objective_losses: Mapping[str, float]
    mean_weighted_loss: float
    tokenizer_fit_receipt_hash: str
    training_record_ids_hash: str
    corpus_hash: str
    split_hash: str
    raw_encoded_tokens: int
    diverse_training_tokens: int | None
    diverse_data_floor_passed: bool | None
    dataset_manifest_hash: str | None
    censored_example_count: int
    objective_implementation: Mapping[str, str]
    model_state_hash: str
    snapshots: tuple[str, ...]
    artifact_kind: str = SNAPSHOT_KIND
    claim_level: str = "none"
    qualified: bool = False
    cloud_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicObjectiveIterator:
    """Order objective-specific episodes by stable seeded hashes per epoch."""

    def __init__(
        self,
        examples: Sequence[SupervisedForecastExample],
        *,
        seed: int,
        weights: ObjectiveWeights | None = None,
    ) -> None:
        self.weights = weights or ObjectiveWeights()
        self.seed = seed
        pools: dict[ObjectiveKind, list[SupervisedForecastExample]] = {
            objective: [] for objective in ObjectiveKind
        }
        seen: set[str] = set()
        for example in examples:
            if example.example_id in seen:
                raise ValueError(f"duplicate training example ID {example.example_id!r}")
            seen.add(example.example_id)
            pools[example.objective].append(example)
        for objective, weight in _objective_weight_map(self.weights).items():
            if weight > 0 and not pools[objective]:
                raise ValueError(f"training corpus has no {objective.value} objective examples")
        self.pools = {
            objective: tuple(sorted(items, key=lambda item: item.example_id))
            for objective, items in pools.items()
        }

    def objective_batch(self, step: int) -> dict[ObjectiveKind, SupervisedForecastExample]:
        if step < 0:
            raise ValueError("step cannot be negative")
        result: dict[ObjectiveKind, SupervisedForecastExample] = {}
        for objective, weight in _objective_weight_map(self.weights).items():
            if weight <= 0:
                continue
            pool = self.pools[objective]
            epoch, offset = divmod(step, len(pool))
            ranked = sorted(
                pool,
                key=lambda item: hashlib.sha256(
                    f"{self.seed}:{objective.value}:{epoch}:{item.example_id}".encode()
                ).digest(),
            )
            result[objective] = ranked[offset]
        return result


def load_training_examples(path: str | os.PathLike[str]) -> tuple[SupervisedForecastExample, ...]:
    """Load canonical training JSONL, optionally gzip-compressed."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"}
    examples: list[SupervisedForecastExample] = []
    with opener(source, **kwargs) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(SupervisedForecastExample.from_json(line))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid training example at {source}:{line_number}: {exc}"
                ) from exc
    if not examples:
        raise ValueError("training corpus is empty")
    DeterministicObjectiveIterator(examples, seed=0)
    return tuple(examples)


def training_corpus_hash(examples: Sequence[SupervisedForecastExample]) -> str:
    ordered = sorted((example.example_id, example.canonical_hash) for example in examples)
    return _hash_json(ordered)


def fit_tokenizer_train_only(
    tokenizer: BraidTokenizer,
    examples: Sequence[SupervisedForecastExample],
) -> TokenizerFitReceipt:
    """Exercise the tokenizer fit boundary using training records only.

    The released byte tokenizer has no learned vocabulary.  This guard still forces
    all preprocessing through the train partition and records the exact corpus so a
    future fitted tokenizer cannot silently consume development or test examples.
    """

    if not examples:
        raise ValueError("tokenizer fit requires training examples")
    if any(example.split != "train" for example in examples):
        raise ValueError("tokenizer fitting is permitted on the train split only")
    before = tokenizer.fingerprint
    for example in sorted(examples, key=lambda item: item.example_id):
        tokenizer.encode_schema_decl(example.request.schema)
        for evidence in example.request.prefix.evidence:
            tokenizer.encode_text(
                canonical_json({"kind": evidence.kind, "payload": evidence.payload})
            )
        for event in (*example.request.prefix.events, *example.target_events):
            tokenizer.encode_text(canonical_json(event.payload))
    if tokenizer.fingerprint != before:
        raise RuntimeError("tokenizer fingerprint changed during train-only fitting")
    record_ids_hash = _hash_json(sorted(example.example_id for example in examples))
    return TokenizerFitReceipt(
        tokenizer_hash=before,
        corpus_hash=training_corpus_hash(examples),
        training_record_ids_hash=record_ids_hash,
    )


def config_from_declaration(value: Mapping[str, Any]) -> EventGraphConfig:
    """Extract an executable architecture from a checked-in scale declaration."""

    missing = _CONFIG_FIELDS - set(value)
    if missing:
        raise ValueError(f"training config is missing architecture fields: {sorted(missing)}")
    config = EventGraphConfig.from_dict({name: value[name] for name in _CONFIG_FIELDS})
    promotion = value.get("promotion_policy")
    if isinstance(promotion, Mapping) and bool(promotion.get("requires_explicit_cloud_approval")):
        raise ValueError("public training runner cannot authorize a cloud-gated configuration")
    _validate_local_config(config)
    return config


def run_training(
    examples: Sequence[SupervisedForecastExample],
    *,
    model_config: EventGraphConfig = TINY_CONFIG,
    run_config: TrainingRunConfig | None = None,
    lineage: TrainingLineage | None = None,
    dataset_manifest: TrainingDatasetManifest | None = None,
    output_directory: str | os.PathLike[str] | None = None,
    resume_snapshot: str | os.PathLike[str] | None = None,
) -> TrainingRunResult:
    """Execute deterministic optimizer steps over the exact objective mixture.

    Saved model manifests intentionally retain ``training_steps=0``.  Optimizer step
    counts live in the trainer state, so an engineering snapshot cannot satisfy the
    inference wrapper's trained-model gate without a separate qualification process.
    """

    run_config = run_config or TrainingRunConfig()
    _validate_local_config(model_config)
    if not examples:
        raise ValueError("training requires at least one example")
    iterator = DeterministicObjectiveIterator(examples, seed=run_config.seed)
    tokenizer = BraidTokenizer()
    fit_receipt = fit_tokenizer_train_only(tokenizer, examples)
    raw_encoded_tokens = sum(len(tokenizer.encode_text(example.to_json())) for example in examples)
    if dataset_manifest is not None:
        if dataset_manifest.data_hash != fit_receipt.corpus_hash:
            raise ValueError("dataset manifest data_hash does not match training JSONL")
        if dataset_manifest.training_record_ids_hash != fit_receipt.training_record_ids_hash:
            raise ValueError("dataset manifest training record IDs do not match training JSONL")
    if model_config.name != "tiny" and dataset_manifest is None:
        raise ValueError("non-tiny execution requires a validated dataset/diversity manifest")
    if (
        dataset_manifest is not None
        and dataset_manifest.diverse_training_tokens < model_config.min_diverse_training_tokens
    ):
        raise ValueError(
            "training corpus is below the declared minimum diverse-token floor: "
            f"manifest records {dataset_manifest.diverse_training_tokens}, required "
            f"{model_config.min_diverse_training_tokens}"
        )
    schema_hash = _hash_json(
        sorted({example.request.schema.canonical_hash() for example in examples})
    )
    split_hash = (
        fit_receipt.training_record_ids_hash
        if dataset_manifest is None
        else dataset_manifest.split_manifest_hash
    )
    evaluation_hash = hashlib.sha256(f"unevaluated:{fit_receipt.corpus_hash}".encode()).hexdigest()
    output = Path(output_directory) if output_directory is not None else None
    if output is not None and lineage is None:
        raise ValueError("saved training snapshots require explicit code/environment lineage")
    if resume_snapshot is not None and output is None:
        raise ValueError("resume requires an output directory for subsequent snapshots")
    device = torch.device(run_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is not available")

    torch.manual_seed(run_config.seed)
    if resume_snapshot is None:
        if output is not None and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"new training output directory is not empty: {output}")
        model = EventGraphModel(
            model_config,
            tokenizer_vocab_size=tokenizer.vocab_size,
        ).to(device)
        start_step = 0
    else:
        assert lineage is not None
        loaded, trainer_state = _load_training_snapshot(
            Path(resume_snapshot),
            map_location=device,
            expected_hashes={
                "code_hash": lineage.code_hash,
                "data_hash": fit_receipt.corpus_hash,
                "environment_hash": lineage.environment_hash,
                "evaluation_hash": evaluation_hash,
                "schema_hash": schema_hash,
                "split_hash": split_hash,
                "tokenizer_hash": fit_receipt.tokenizer_hash,
            },
        )
        model = loaded.model
        if model.config != model_config or model.tokenizer_vocab_size != tokenizer.vocab_size:
            raise ValueError("resume snapshot architecture does not match requested config")
        start_step = _validated_resume_step(
            trainer_state,
            run_config,
            fit_receipt,
            dataset_manifest,
        )
    if start_step >= run_config.total_steps:
        raise ValueError("total_steps must exceed the resume snapshot step")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    if resume_snapshot is not None:
        _restore_optimizer(optimizer, Path(resume_snapshot))

    weights = ObjectiveWeights()
    weight_map = _objective_weight_map(weights)
    component_sums = {objective: 0.0 for objective in ObjectiveKind}
    counts = {objective: 0 for objective in ObjectiveKind}
    weighted_sum = 0.0
    snapshots: list[str] = []
    model.train()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for zero_based_step in range(start_step, run_config.total_steps):
            torch.manual_seed(run_config.seed + zero_based_step)
            optimizer.zero_grad(set_to_none=True)
            component_losses: dict[ObjectiveKind, Tensor] = {}
            for objective, example in iterator.objective_batch(zero_based_step).items():
                component_losses[objective] = _objective_loss(
                    model,
                    tokenizer,
                    example,
                )
            total = weights.combine(
                patch=component_losses[ObjectiveKind.PATCH],
                schema_episode=component_losses[ObjectiveKind.SCHEMA_EPISODE],
                reconstruction=component_losses[ObjectiveKind.RECONSTRUCTION],
                contrastive=component_losses[ObjectiveKind.CONTRASTIVE],
            )
            if not torch.isfinite(total):
                raise ValueError("training loss became non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.max_grad_norm)
            optimizer.step()
            completed_step = zero_based_step + 1
            weighted_sum += float(total.detach().cpu().item())
            for objective, loss in component_losses.items():
                component_sums[objective] += float(loss.detach().cpu().item())
                counts[objective] += 1
            if output is not None and (
                completed_step % run_config.save_every_steps == 0
                or completed_step == run_config.total_steps
            ):
                assert lineage is not None
                snapshot = _save_training_snapshot(
                    output,
                    model,
                    optimizer,
                    optimizer_steps=completed_step,
                    run_config=run_config,
                    fit_receipt=fit_receipt,
                    dataset_manifest=dataset_manifest,
                    schema_hash=schema_hash,
                    split_hash=split_hash,
                    evaluation_hash=evaluation_hash,
                    lineage=lineage,
                )
                snapshots.append(str(snapshot))
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    executed = run_config.total_steps - start_step
    return TrainingRunResult(
        optimizer_steps=run_config.total_steps,
        start_step=start_step,
        objective_weights={objective.value: weight for objective, weight in weight_map.items()},
        objective_example_counts={
            objective.value: counts[objective] for objective in ObjectiveKind
        },
        mean_objective_losses={
            objective.value: component_sums[objective] / counts[objective]
            for objective in ObjectiveKind
        },
        mean_weighted_loss=weighted_sum / executed,
        tokenizer_fit_receipt_hash=fit_receipt.receipt_hash,
        training_record_ids_hash=fit_receipt.training_record_ids_hash,
        corpus_hash=fit_receipt.corpus_hash,
        split_hash=split_hash,
        raw_encoded_tokens=raw_encoded_tokens,
        diverse_training_tokens=(
            None if dataset_manifest is None else dataset_manifest.diverse_training_tokens
        ),
        diverse_data_floor_passed=(
            None
            if dataset_manifest is None
            else dataset_manifest.diverse_training_tokens
            >= model_config.min_diverse_training_tokens
        ),
        dataset_manifest_hash=(
            None if dataset_manifest is None else dataset_manifest.manifest_hash
        ),
        censored_example_count=sum(example.is_censored for example in examples),
        objective_implementation=OBJECTIVE_IMPLEMENTATION_NOTES,
        model_state_hash=_model_state_hash(model),
        snapshots=tuple(snapshots),
    )


def smoke_training_examples() -> tuple[SupervisedForecastExample, ...]:
    """Return a tiny, wholly synthetic four-objective smoke corpus."""

    cutoff = datetime(2026, 1, 2, tzinfo=UTC)
    schema = SchemaDecl(
        version="smoke-v2",
        observed_at=cutoff - timedelta(days=2),
        node_types=(NodeTypeDecl("actor", "synthetic actor"),),
        relations=(
            RelationDecl("observes", (RoleDecl("actor", ("actor",)),), "synthetic relation"),
        ),
        constraints={"synthetic": True},
    )
    evidence = EvidenceRecord(
        evidence_id="evidence:smoke",
        observed_at=cutoff - timedelta(hours=2),
        kind="synthetic",
        payload={"fixture": True},
    )
    distractor_evidence = EvidenceRecord(
        evidence_id="evidence:smoke-distractor",
        observed_at=cutoff - timedelta(hours=3),
        kind="synthetic",
        payload={"fixture": "distractor"},
    )
    visible_event_time = cutoff - timedelta(hours=1)
    prefix_event = GraphEventV2(
        event_id="event:visible-smoke",
        observed_at=visible_event_time,
        valid_from=visible_event_time,
        valid_to=None,
        operation=ContractOperation.EXPOSE,
        schema_version=schema.version,
        relation=None,
        arguments=(RoleBinding("candidate", "node:smoke"),),
        payload={"channel": "fixture"},
        basis_refs=(evidence.evidence_id,),
        derivation=DerivationRecord("synthetic-smoke", (evidence.evidence_id,)),
        provenance=ProvenanceRecord("synthetic-smoke", None, "CC0-1.0", visible_event_time),
    )
    prefix = GraphBundleV2(
        bundle_id="bundle:smoke-prefix",
        schemas=(schema,),
        nodes=(
            NodeRecord("node:smoke", "actor", schema.version, cutoff - timedelta(days=1)),
            NodeRecord(
                "node:smoke-distractor",
                "actor",
                schema.version,
                cutoff - timedelta(days=1),
            ),
        ),
        evidence=(evidence, distractor_evidence),
        events=(prefix_event,),
    )
    target_time = cutoff + timedelta(hours=1)
    target = GraphEventV2(
        event_id="event:target-smoke",
        observed_at=target_time,
        valid_from=target_time,
        valid_to=None,
        operation=ContractOperation.EXPOSE,
        schema_version=schema.version,
        relation=None,
        arguments=(RoleBinding("candidate", "node:smoke"),),
        payload={"channel": "smoke"},
        basis_refs=(evidence.evidence_id,),
        derivation=DerivationRecord("synthetic-smoke", (evidence.evidence_id,)),
        provenance=ProvenanceRecord("synthetic-smoke", None, "CC0-1.0", target_time),
    )
    horizon = timedelta(days=1)
    examples = []
    for objective in ObjectiveKind:
        request = ForecastRequest(
            request_id=f"request:smoke:{objective.value}",
            schema=schema,
            prefix=prefix,
            cutoff=cutoff,
            horizon=horizon,
            task=TaskDeclaration(
                "training-smoke",
                query_node_ids=("node:smoke",),
                target_relations=("observes",),
            ),
        )
        examples.append(
            SupervisedForecastExample(
                example_id=f"smoke:{objective.value}",
                objective=objective,
                request=request,
                target_events=(target,),
                censor_at=cutoff + horizon,
                outcome_complete=True,
            )
        )
    quiet_request = ForecastRequest(
        request_id="request:smoke:patch-quiet",
        schema=schema,
        prefix=prefix,
        cutoff=cutoff,
        horizon=horizon,
        task=TaskDeclaration(
            "training-smoke",
            query_node_ids=("node:smoke",),
            target_relations=("observes",),
        ),
    )
    examples.append(
        SupervisedForecastExample(
            example_id="smoke:patch-quiet",
            objective=ObjectiveKind.PATCH,
            request=quiet_request,
            target_events=(),
            censor_at=cutoff + horizon,
            outcome_complete=True,
        )
    )
    return tuple(examples)


def _objective_loss(
    model: EventGraphModel,
    tokenizer: BraidTokenizer,
    example: SupervisedForecastExample,
) -> Tensor:
    episode = tensorize_forecast_request(
        example.request,
        tokenizer,
        time_unit_seconds=model.config.time_unit_seconds,
        device=next(model.parameters()).device,
    )
    output = model(episode.schema_batch, episode.event_batch)
    distribution = _final_distribution(output.distribution)
    assert example.request.cutoff is not None
    if example.is_censored:
        duration = (example.censor_at - example.request.cutoff).total_seconds()
        return censored_exponential_nll(
            distribution.delta_log_rate,
            torch.full_like(distribution.delta_log_rate, duration),
            torch.zeros_like(distribution.delta_log_rate, dtype=torch.bool),
            duration_scale=model.config.time_unit_seconds,
        )

    targets = _event_targets(example, episode, distribution)
    patch = factorized_patch_loss(
        distribution,
        targets,
        time_scale=model.config.time_unit_seconds,
    )
    if example.objective in (ObjectiveKind.PATCH, ObjectiveKind.SCHEMA_EPISODE):
        return patch.total
    if example.objective is ObjectiveKind.RECONSTRUCTION:
        return _reconstruction_loss(patch, targets)
    if example.objective is ObjectiveKind.CONTRASTIVE:
        return (patch.argument + patch.evidence) / 2
    raise AssertionError(f"unhandled objective {example.objective}")


def _event_targets(
    example: SupervisedForecastExample,
    episode: TensorizedForecast,
    distribution: FactorizedEventDistribution,
) -> PatchTargets:
    event = example.target_events[0]
    assert event.observed_at is not None
    assert event.valid_from is not None
    assert example.request.cutoff is not None
    shape = distribution.event_mask.shape
    device = distribution.event_mask.device
    relation = 0
    relation_active = event.relation is not None
    if relation_active:
        relation = episode.handles.resolve(HandleKind.RELATION, event.relation).index
    visible_nodes = {node_id: index for index, node_id in enumerate(episode.node_ids)}
    pointer_targets = sorted(
        {
            visible_nodes[binding.node_id]
            for binding in event.arguments
            if binding.node_id in visible_nodes
        }
    )
    argument_active = bool(pointer_targets)
    argument = pointer_targets[0] if pointer_targets else 0
    evidence_active = bool(event.basis_refs)
    evidence = (
        episode.handles.resolve(HandleKind.EVIDENCE, event.basis_refs[0]).index
        if evidence_active
        else 0
    )
    payload_tokens = BraidTokenizer().encode_text(canonical_json(event.payload))
    payload = payload_tokens[1] if len(payload_tokens) > 2 else BraidTokenizer.EOS
    operation_order = tuple(item.value for item in ModelOperation)
    operation = operation_order.index(event.operation.value)
    return PatchTargets(
        delta_durations=torch.full(
            shape,
            (event.observed_at - example.request.cutoff).total_seconds(),
            dtype=distribution.delta_log_rate.dtype,
            device=device,
        ),
        delta_observed=torch.ones(shape, dtype=torch.bool, device=device),
        valid_lags=torch.full(
            shape,
            (event.valid_from - event.observed_at).total_seconds(),
            dtype=distribution.valid_lag_mean.dtype,
            device=device,
        ),
        operations=torch.full(shape, operation, dtype=torch.long, device=device),
        relations=torch.full(shape, relation, dtype=torch.long, device=device),
        arguments=torch.full(shape, argument, dtype=torch.long, device=device),
        payload_tokens=torch.full(shape, payload, dtype=torch.long, device=device),
        evidence=torch.full(shape, evidence, dtype=torch.long, device=device),
        relation_mask=torch.full(shape, relation_active, dtype=torch.bool, device=device),
        argument_mask=torch.full(shape, argument_active, dtype=torch.bool, device=device),
        evidence_mask=torch.full(shape, evidence_active, dtype=torch.bool, device=device),
    )


def _final_distribution(
    distribution: FactorizedEventDistribution,
) -> FactorizedEventDistribution:
    final = int(distribution.event_mask[0].sum().item()) - 1

    def select(value: Tensor) -> Tensor:
        return value[:, final : final + 1]

    return FactorizedEventDistribution(
        hidden_states=select(distribution.hidden_states),
        delta_log_rate=select(distribution.delta_log_rate),
        valid_lag_mean=select(distribution.valid_lag_mean),
        valid_lag_log_scale=select(distribution.valid_lag_log_scale),
        operation_logits=select(distribution.operation_logits),
        relation_logits=select(distribution.relation_logits),
        argument_logits=select(distribution.argument_logits),
        payload_logits=select(distribution.payload_logits),
        evidence_logits=select(distribution.evidence_logits),
        event_mask=torch.ones_like(select(distribution.event_mask)),
    )


def _reconstruction_loss(patch: PatchLoss, targets: PatchTargets) -> Tensor:
    factors = [patch.operation, patch.payload]
    if targets.relation_mask is not None and targets.relation_mask.any():
        factors.append(patch.relation)
    if targets.argument_mask is not None and targets.argument_mask.any():
        factors.append(patch.argument)
    if targets.evidence_mask is not None and targets.evidence_mask.any():
        factors.append(patch.evidence)
    return torch.stack(factors).mean()


def _validate_local_config(config: EventGraphConfig) -> None:
    policy = config.execution_policy.lower()
    if config.name != "tiny" and not policy.startswith("local "):
        raise ValueError(
            "public training runner permits only tiny or explicitly local scale configurations"
        )
    if "cloud" in policy or "multi-gpu" in policy:
        raise ValueError("public training runner cannot authorize cloud or multi-GPU execution")


def _objective_weight_map(weights: ObjectiveWeights) -> dict[ObjectiveKind, float]:
    return {
        ObjectiveKind.PATCH: weights.patch,
        ObjectiveKind.SCHEMA_EPISODE: weights.schema_episode,
        ObjectiveKind.RECONSTRUCTION: weights.reconstruction,
        ObjectiveKind.CONTRASTIVE: weights.contrastive,
    }


def _save_training_snapshot(
    output: Path,
    model: EventGraphModel,
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_steps: int,
    run_config: TrainingRunConfig,
    fit_receipt: TokenizerFitReceipt,
    dataset_manifest: TrainingDatasetManifest | None,
    schema_hash: str,
    split_hash: str,
    evaluation_hash: str,
    lineage: TrainingLineage,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"step-{optimizer_steps:08d}"
    if destination.exists():
        raise FileExistsError(f"training snapshot already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=output))
    try:
        metadata = CheckpointMetadata.for_model(
            model,
            tokenizer_hash=fit_receipt.tokenizer_hash,
            schema_hash=schema_hash,
            code_hash=lineage.code_hash,
            environment_hash=lineage.environment_hash,
            data_hash=fit_receipt.corpus_hash,
            split_hash=split_hash,
            evaluation_hash=evaluation_hash,
            # Positive values are reserved for separately qualified artifacts.
            training_steps=0,
        )
        completed = save_checkpoint(temporary / "model", model, metadata)
        optimizer_hash = _save_optimizer(optimizer, temporary)
        state = {
            "artifact_kind": SNAPSHOT_KIND,
            "checkpoint_manifest_id": completed.manifest_id,
            "config_fingerprint": model.architecture_fingerprint,
            "dataset_manifest_hash": (
                None if dataset_manifest is None else dataset_manifest.manifest_hash
            ),
            "fit_receipt_hash": fit_receipt.receipt_hash,
            "format_version": TRAINING_SNAPSHOT_FORMAT,
            "optimizer_hash": optimizer_hash,
            "optimizer_steps": optimizer_steps,
            "run_fingerprint": _run_fingerprint(run_config),
        }
        (temporary / "trainer.json").write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _load_training_snapshot(
    snapshot: Path,
    *,
    map_location: torch.device,
    expected_hashes: Mapping[str, str],
) -> tuple[Any, dict[str, Any]]:
    state_path = snapshot / "trainer.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    required = {
        "artifact_kind",
        "checkpoint_manifest_id",
        "config_fingerprint",
        "dataset_manifest_hash",
        "fit_receipt_hash",
        "format_version",
        "optimizer_hash",
        "optimizer_steps",
        "run_fingerprint",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("training snapshot state has an invalid schema")
    if state["format_version"] != TRAINING_SNAPSHOT_FORMAT:
        raise ValueError("unsupported training snapshot format")
    if state["artifact_kind"] != SNAPSHOT_KIND:
        raise ValueError("resume source is not an unqualified training snapshot")
    if _optimizer_artifact_hash(snapshot) != state["optimizer_hash"]:
        raise ValueError("optimizer artifact hash does not match trainer state")
    loaded = load_checkpoint(
        snapshot / "model",
        expected_hashes=expected_hashes,
        map_location=map_location,
    )
    if loaded.metadata.manifest_id != state["checkpoint_manifest_id"]:
        raise ValueError("model manifest does not match trainer state")
    if loaded.metadata.training_steps != 0:
        raise ValueError("training snapshots must remain unqualified")
    return loaded, state


def _validated_resume_step(
    state: Mapping[str, Any],
    run_config: TrainingRunConfig,
    fit_receipt: TokenizerFitReceipt,
    dataset_manifest: TrainingDatasetManifest | None,
) -> int:
    if state["run_fingerprint"] != _run_fingerprint(run_config):
        raise ValueError("resume run configuration does not match snapshot")
    if state["fit_receipt_hash"] != fit_receipt.receipt_hash:
        raise ValueError("resume tokenizer fit receipt does not match training corpus")
    expected_dataset_manifest = None if dataset_manifest is None else dataset_manifest.manifest_hash
    if state["dataset_manifest_hash"] != expected_dataset_manifest:
        raise ValueError("resume dataset manifest does not match snapshot")
    step = state["optimizer_steps"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("resume optimizer_steps must be a positive integer")
    return step


def _save_optimizer(optimizer: torch.optim.Optimizer, directory: Path) -> str:
    state = optimizer.state_dict()
    tensors: dict[str, Tensor] = {}
    scalar_state: dict[str, dict[str, Any]] = {}
    for parameter_id, values in state["state"].items():
        for name, value in values.items():
            if isinstance(value, Tensor):
                tensors[f"state.{parameter_id}.{name}"] = value.detach().contiguous().cpu()
            elif value is None or isinstance(value, (bool, int, float, str)):
                scalar_state.setdefault(str(parameter_id), {})[name] = value
            else:
                raise TypeError(f"unsupported optimizer state value {name}: {type(value).__name__}")
    if not tensors:
        tensors["__empty__"] = torch.empty(0)
    tensor_path = directory / "optimizer.safetensors"
    save_file(tensors, str(tensor_path))
    metadata = {
        "optimizer_class": type(optimizer).__name__,
        "param_groups": state["param_groups"],
        "scalar_state": scalar_state,
    }
    (directory / "optimizer.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return _optimizer_artifact_hash(directory)


def _restore_optimizer(optimizer: torch.optim.Optimizer, snapshot: Path) -> None:
    metadata = json.loads((snapshot / "optimizer.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or set(metadata) != {
        "optimizer_class",
        "param_groups",
        "scalar_state",
    }:
        raise ValueError("optimizer metadata has an invalid schema")
    if metadata["optimizer_class"] != type(optimizer).__name__:
        raise ValueError("optimizer class does not match resume snapshot")
    tensors = load_file(str(snapshot / "optimizer.safetensors"), device="cpu")
    state: dict[int, dict[str, Any]] = {}
    for key, value in tensors.items():
        if key == "__empty__":
            continue
        prefix, parameter_id, name = key.split(".", 2)
        if prefix != "state":
            raise ValueError(f"unknown optimizer tensor key {key!r}")
        state.setdefault(int(parameter_id), {})[name] = value
    scalar_state = metadata["scalar_state"]
    if not isinstance(scalar_state, dict):
        raise ValueError("optimizer scalar state must be an object")
    for parameter_id, values in scalar_state.items():
        if not isinstance(values, dict):
            raise ValueError("optimizer parameter state must be an object")
        state.setdefault(int(parameter_id), {}).update(values)
    optimizer.load_state_dict({"state": state, "param_groups": metadata["param_groups"]})


def _run_fingerprint(config: TrainingRunConfig) -> str:
    # Total steps and save frequency may increase/change on a valid resume.  Every
    # optimization-semantic field remains bound.
    return _hash_json(
        {
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "optimizer": "AdamW",
            "objective_weights": asdict(ObjectiveWeights()),
            "seed": config.seed,
            "weight_decay": config.weight_decay,
        }
    )


def _model_state_hash(model: EventGraphModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_artifact_hash(directory: Path) -> str:
    return _hash_json(
        {
            "metadata": _file_hash(directory / "optimizer.json"),
            "tensors": _file_hash(directory / "optimizer.safetensors"),
        }
    )


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _DIGEST_CHARS:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
