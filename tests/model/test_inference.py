import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import torch

from braid.contract.types import (
    ForecastRequest,
    GraphBundleV2,
    NodeRecord,
    NodeTypeDecl,
    RelationDecl,
    RoleDecl,
    SchemaDecl,
    TaskDeclaration,
)
from braid.model.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from braid.model.config import TINY_CONFIG
from braid.model.grammar import Operation as ModelOperation
from braid.model.inference import CalibrationRecord, EventGraphForecaster
from braid.model.model import EventGraphModel
from braid.model.retrieval import NodeCandidate, RetrievalResult, TypedRetriever
from braid.model.tensorize import tensorize_forecast_request
from braid.model.tokenizer import BraidTokenizer


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def request() -> ForecastRequest:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    schema = SchemaDecl(
        version="v2",
        observed_at=now - timedelta(days=2),
        node_types=(NodeTypeDecl("person", "a person"),),
        relations=(
            RelationDecl(
                "knows",
                (RoleDecl("subject", ("person",)), RoleDecl("object", ("person",))),
                "social connection",
            ),
        ),
    )
    bundle = GraphBundleV2(
        bundle_id="prefix",
        schemas=(schema,),
        nodes=(
            NodeRecord("n1", "person", "v2", now - timedelta(days=1)),
            NodeRecord("n2", "person", "v2", now - timedelta(hours=1)),
        ),
        evidence=(),
        events=(),
    )
    return ForecastRequest(
        request_id="request-1",
        schema=schema,
        prefix=bundle,
        cutoff=now,
        horizon=timedelta(days=1),
        task=TaskDeclaration("patch", query_node_ids=("n1",), target_relations=("knows",)),
    )


def metadata(
    model: EventGraphModel, tokenizer: BraidTokenizer, *, steps: int
) -> CheckpointMetadata:
    return CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=digest("schema"),
        code_hash=digest("code"),
        environment_hash=digest("environment"),
        data_hash=digest("data"),
        split_hash=digest("split"),
        evaluation_hash=digest("evaluation"),
        training_steps=steps,
    )


def test_untrained_weights_fail_closed() -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    forecaster = EventGraphForecaster(
        model,
        tokenizer,
        metadata(model, tokenizer, steps=0),
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )
    result = forecaster.forecast(request())
    assert result.abstention_reason == "UNTRAINED_MODEL"
    assert result.sampled_patch_windows == ()


def test_mismatched_calibration_lineage_fails_closed(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    save_checkpoint(
        tmp_path / "checkpoint",
        model,
        metadata(model, tokenizer, steps=1),
    )
    loaded = load_checkpoint(tmp_path / "checkpoint")
    forecaster = EventGraphForecaster(
        loaded.model,
        tokenizer,
        loaded.metadata,
        calibration=CalibrationRecord(1.0, digest("some-other-evaluation")),
    )
    result = forecaster.forecast(request())
    assert result.abstention_reason == "CALIBRATION_LINEAGE_MISMATCH"


def test_positive_step_metadata_without_verified_weights_fails_closed() -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    forecaster = EventGraphForecaster(
        model,
        tokenizer,
        metadata(model, tokenizer, steps=1),
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )

    result = forecaster.forecast(request())

    assert result.abstention_reason == "UNVERIFIED_CHECKPOINT"
    assert result.sampled_patch_windows == ()


def test_verified_metadata_cannot_be_attached_to_a_different_model(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    saved_model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    completed = save_checkpoint(
        tmp_path / "checkpoint",
        saved_model,
        metadata(saved_model, tokenizer, steps=1),
    )
    other_model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    forecaster = EventGraphForecaster(
        other_model,
        tokenizer,
        completed,
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )

    assert forecaster.forecast(request()).abstention_reason == "UNVERIFIED_CHECKPOINT"


def test_verified_checkpoint_is_invalidated_by_weight_mutation(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    save_checkpoint(
        tmp_path / "checkpoint",
        model,
        metadata(model, tokenizer, steps=1),
    )
    loaded = load_checkpoint(tmp_path / "checkpoint")
    with torch.no_grad():
        next(loaded.model.parameters()).add_(1.0)
    forecaster = EventGraphForecaster(
        loaded.model,
        tokenizer,
        loaded.metadata,
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )

    assert forecaster.forecast(request()).abstention_reason == "UNVERIFIED_CHECKPOINT"


def test_verified_weights_cannot_be_paired_with_forged_lineage(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    save_checkpoint(
        tmp_path / "checkpoint",
        model,
        metadata(model, tokenizer, steps=1),
    )
    loaded = load_checkpoint(tmp_path / "checkpoint")
    forged = replace(
        loaded.metadata,
        training_steps=999,
        data_hash=digest("different-data"),
    )
    forecaster = EventGraphForecaster(
        loaded.model,
        tokenizer,
        forged,
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )

    assert forecaster.forecast(request()).abstention_reason == "UNVERIFIED_CHECKPOINT"


def test_schema_only_prefix_tensorizes_to_explicit_start_state() -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size).eval()
    episode = tensorize_forecast_request(request(), tokenizer)
    assert episode.event_batch.start_mask.tolist() == [[True]]
    output = model(episode.schema_batch, episode.event_batch)
    assert output.distribution.operation_logits.shape == (1, 1, 8)


def test_empty_schema_abstains_before_tensorization(tmp_path) -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    save_checkpoint(
        tmp_path / "checkpoint",
        model,
        metadata(model, tokenizer, steps=1),
    )
    loaded = load_checkpoint(tmp_path / "checkpoint")
    forecast_request = request()
    empty_schema = replace(forecast_request.schema, node_types=(), relations=())
    empty_request = replace(
        forecast_request,
        schema=empty_schema,
        prefix=replace(forecast_request.prefix, schemas=(empty_schema,), nodes=()),
        task=replace(
            forecast_request.task,
            query_node_ids=(),
            target_relations=(),
        ),
    )
    forecaster = EventGraphForecaster(
        loaded.model,
        tokenizer,
        loaded.metadata,
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )

    result = forecaster.forecast(empty_request)

    assert result.abstention_reason == "NO_DECODABLE_RELATIONS"
    assert result.sampled_patch_windows == ()


def test_argument_decoding_is_restricted_to_retrieved_nodes() -> None:
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    forecaster = EventGraphForecaster(
        model,
        tokenizer,
        metadata(model, tokenizer, steps=0),
        calibration=CalibrationRecord(1.0, digest("evaluation")),
    )
    forecast_request = request()
    episode = tensorize_forecast_request(forecast_request, tokenizer)
    operation_logits = torch.full((1, 1, len(ModelOperation)), -100.0)
    operation_logits[0, 0, tuple(ModelOperation).index(ModelOperation.ASSERT)] = 100.0
    distribution = SimpleNamespace(
        event_mask=torch.tensor([[True]]),
        operation_logits=operation_logits,
        relation_logits=torch.zeros((1, 1, 1)),
        # The non-retrieved n2 has the larger raw score.
        argument_logits=torch.tensor([[[0.0, 100.0]]]),
        evidence_logits=torch.zeros((1, 1, 1)),
        delta_log_rate=torch.zeros((1, 1)),
        valid_lag_mean=torch.zeros((1, 1)),
    )
    retrieved = RetrievalResult(
        candidates=(NodeCandidate(0, 0, forecast_request.prefix.nodes[0].observed_at),),
        coverage=1.0,
        type_coverage=1.0,
        true_target_recall=1.0,
    )

    decoded = forecaster._decode_one(forecast_request, episode, distribution, retrieved)

    assert decoded is not None
    event, _, _ = decoded
    assert {binding.node_id for binding in event.arguments} == {"n1"}


def test_claim_grade_mode_abstains_without_99_percent_target_recall(tmp_path) -> None:
    class WeakRecallRetriever:
        def retrieve(self, nodes, **kwargs):
            return TypedRetriever().retrieve(
                nodes,
                **kwargs,
                true_target_handles=(0, 999),
            )

    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    save_checkpoint(
        tmp_path / "checkpoint",
        model,
        metadata(model, tokenizer, steps=1),
    )
    loaded = load_checkpoint(tmp_path / "checkpoint")
    forecaster = EventGraphForecaster(
        loaded.model,
        tokenizer,
        loaded.metadata,
        calibration=CalibrationRecord(1.0, digest("evaluation")),
        retriever=WeakRecallRetriever(),
        require_claim_grade_retrieval=True,
    )

    result = forecaster.forecast(request())

    assert result.abstention_reason == "RETRIEVAL_RECALL_NOT_CLAIM_GRADE"
