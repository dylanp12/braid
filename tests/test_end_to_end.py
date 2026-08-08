import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from braid.bench.gates import (
    ClaimLevel,
    ClaimProhibitedError,
    authorize_claim,
    strongest_permitted_claim,
)
from braid.contract import ForecastRequest, TaskDeclaration, build_as_of_prefix
from braid.model.checkpoint import CheckpointMetadata
from braid.model.config import TINY_CONFIG
from braid.model.inference import EventGraphForecaster
from braid.model.model import EventGraphModel
from braid.model.tokenizer import BraidTokenizer
from braid.synthetic import ProcessGeneratorConfig, generate_world


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_causal_bundle_to_fail_closed_forecast_and_claim_gate() -> None:
    world = generate_world(ProcessGeneratorConfig(seed=23, relation_events=24))
    source_hash = world.bundle.canonical_hash()
    prefix = build_as_of_prefix(world.bundle, world.censor_at)
    request = ForecastRequest(
        request_id="engineering-canary",
        schema=prefix.schema,
        prefix=prefix,
        cutoff=world.censor_at,
        horizon=timedelta(days=1),
        task=TaskDeclaration(
            "graph-patch",
            query_node_ids=(prefix.nodes[0].node_id,),
            target_relations=(prefix.schema.relations[0].name,),
        ),
    )
    tokenizer = BraidTokenizer()
    model = EventGraphModel(TINY_CONFIG, tokenizer_vocab_size=tokenizer.vocab_size)
    checkpoint = CheckpointMetadata.for_model(
        model,
        tokenizer_hash=tokenizer.fingerprint,
        schema_hash=prefix.schema.canonical_hash(),
        code_hash=_digest("test-code"),
        environment_hash=_digest("test-environment"),
        data_hash=world.bundle.canonical_hash(),
        split_hash=_digest("test-split"),
        evaluation_hash=_digest("no-evaluation"),
        training_steps=0,
    )
    distribution = EventGraphForecaster(
        model,
        tokenizer,
        checkpoint,
        calibration=None,
    ).forecast(request)

    assert distribution.abstention_reason == "UNTRAINED_MODEL"
    assert distribution.sampled_patch_windows == ()
    assert world.bundle.canonical_hash() == source_hash
    assert strongest_permitted_claim() is ClaimLevel.ENGINEERING_BUILD
    with pytest.raises(ClaimProhibitedError):
        authorize_claim(ClaimLevel.FRONTIER)


def test_checked_in_scale_declarations_match_executable_configs() -> None:
    from braid.model.config import SCALE_CONFIGS

    root = Path(__file__).resolve().parents[1]
    for key, model_config in SCALE_CONFIGS.items():
        declaration = json.loads((root / "configs" / f"{model_config.name}.json").read_text())
        assert declaration["context_events"] == model_config.context_events
        assert (
            declaration["min_diverse_training_tokens"] == model_config.min_diverse_training_tokens
        )
        assert declaration["name"] == model_config.name
        assert key in {"10m", "20m", "40m", "300m", "1.3b", "7.2b"}
