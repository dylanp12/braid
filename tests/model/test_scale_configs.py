from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from braid.contract import Operation
from braid.model.config import SCALE_CONFIGS, EventGraphConfig
from braid.model.model import EventGraphModel
from braid.model.tokenizer import BraidTokenizer
from braid.model.training import ObjectiveWeights

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = ROOT / "configs"
MODEL_FIELDS = tuple(EventGraphConfig.__dataclass_fields__)
TOP_LEVEL_FIELDS = {
    "format_version",
    "scale_key",
    "name",
    "claim_level",
    "parameter_label",
    "target_parameters",
    "expected_parameters",
    "parameter_tolerance_fraction",
    *MODEL_FIELDS,
    "tokenizer",
    "training_policy",
    "promotion_policy",
}


def _declarations() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(CONFIG_DIRECTORY.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = str(value["scale_key"])
        if key in result:
            raise AssertionError(f"duplicate scale declaration {key!r}")
        result[key] = value
    return result


def test_json_declarations_are_complete_and_match_executable_configs() -> None:
    declarations = _declarations()
    assert set(declarations) == set(SCALE_CONFIGS)

    tokenizer = BraidTokenizer()
    expected_tokenizer = {
        "class": type(tokenizer).__name__,
        "version": tokenizer.config.version,
        "normalization": tokenizer.config.normalization,
        "byte_offset": tokenizer.config.byte_offset,
        "vocab_size": tokenizer.vocab_size,
    }
    expected_training = {
        "objective_weights": asdict(ObjectiveWeights()),
        "support_events_k": [0, 1, 2, 4, 8],
        "operation_vocabulary": [operation.value for operation in Operation],
        "right_censoring_survival_likelihood": True,
        "patch_factor_weighting": "uniform-mean",
        "balanced_patch_factors": [
            "time",
            "valid_lag",
            "operation",
            "relation",
            "argument",
            "payload",
            "evidence",
        ],
        "contrastive_corruptions": ["hard-semantic", "hard-structural"],
        "reconstruction_visibility": "causal-visible-prefix",
        "sequence_order": "observed_at",
        "valid_time_encoding": "signed-lag",
        "random_episode_handle_renaming": True,
        "same_timestamp_policy": "marked-tie-groups-with-randomized-training-order",
        "fit_preprocessors_on": "train",
        "alluvia_value_head": False,
    }

    for key, model_config in SCALE_CONFIGS.items():
        declared = declarations[key]
        assert set(declared) == TOP_LEVEL_FIELDS
        assert declared["format_version"] == 1
        assert declared["claim_level"] == "engineering-build"
        assert declared["tokenizer"] == expected_tokenizer
        assert declared["training_policy"] == expected_training
        assert declared["parameter_label"] == model_config.parameter_label
        for field_name, expected in model_config.to_dict().items():
            assert declared[field_name] == expected, f"{key}.{field_name} drifted"

        promotion = declared["promotion_policy"]
        assert isinstance(promotion, dict)
        assert promotion == {
            "requires_data_floor": True,
            "requires_scaling_gate": key in {"300m", "1.3b", "7.2b"},
            "requires_shortcut_gate": key in {"300m", "1.3b", "7.2b"},
            "requires_explicit_cloud_approval": key in {"300m", "1.3b", "7.2b"},
            "cloud_authorized": False,
        }


@pytest.mark.parametrize("scale_key", tuple(SCALE_CONFIGS))
def test_declared_parameter_count_matches_meta_model(scale_key: str) -> None:
    declaration = _declarations()[scale_key]
    tokenizer = BraidTokenizer()
    with torch.device("meta"):
        model = EventGraphModel(
            SCALE_CONFIGS[scale_key],
            tokenizer_vocab_size=tokenizer.vocab_size,
        )
    actual = model.parameter_count
    target = int(declaration["target_parameters"])
    tolerance = float(declaration["parameter_tolerance_fraction"])

    assert actual == declaration["expected_parameters"]
    assert abs(actual - target) / target <= tolerance
    del model
    gc.collect()
