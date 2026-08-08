from __future__ import annotations

import builtins
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from braid.cli import main
from braid.contract import GraphBundleV2, ModelManifest, validate_bundle
from braid.synthetic import ProcessGeneratorConfig, generate_world


def _write_bundle(path: Path, bundle: GraphBundleV2) -> None:
    path.write_text(bundle.to_json() + "\n", encoding="utf-8")


def _legacy_source() -> dict[str, object]:
    return {
        "manifest": {"contract_version": "0.1.0", "generator": "cli-test"},
        "nodes": [
            {"id": "person:ada", "type": "Person", "label": "Ada", "attrs": {}},
            {
                "id": "repo:braid",
                "type": "Repository",
                "label": "Braid",
                "attrs": {},
            },
        ],
        "events": [
            {
                "event_id": "event:legacy-1",
                "event_type": "inference",
                "relation": "CONTRIBUTES_TO",
                "participants": [["actor", "person:ada"], ["target", "repo:braid"]],
                "event_time": ["2025-01-02T00:00:00Z", None],
                "derived_at": None,
                "evidence": ["https://example.test/commit/1"],
                "derivation": [],
                "attrs": {"source": "fixture"},
                "extraction": {"pipeline_version": 1},
                "human_confirmed": False,
            }
        ],
        "judgments": [],
    }


def test_status_uses_only_current_claim_wording(capsys) -> None:
    assert main(["status"]) == 0

    output = capsys.readouterr().out
    status = json.loads(output)
    assert status["claim_level"] == "engineering-build"
    assert status["program_status"] == "pre-candidate-engineering"
    assert status["gates"]["raw_artifact_recomputation"] == "not-implemented"
    assert status["gates"]["benchmark_tracks"] == {
        "E": "not-submitted",
        "P": "not-submitted",
        "R": "not-submitted",
        "S": "not-submitted",
        "T": "not-submitted",
    }
    assert "frontier" not in output.lower()
    assert "foundation" not in output.lower()
    canonical = json.dumps(status, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert output == canonical + "\n"


def test_validate_reports_invalid_bundle_as_a_clear_nonzero_error(tmp_path: Path, capsys) -> None:
    bundle = generate_world(ProcessGeneratorConfig(seed=8, relation_events=20)).bundle
    broken = replace(bundle, events=(*bundle.events, bundle.events[0]))
    source = tmp_path / "invalid.json"
    _write_bundle(source, broken)

    assert main(["validate", str(source)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: validate: contract validation failed:")
    assert "duplicate event ID" in captured.err


def test_validate_and_prefix_emit_canonical_contract_json(tmp_path: Path, capsys) -> None:
    world = generate_world(ProcessGeneratorConfig(seed=4, relation_events=30))
    source = tmp_path / "bundle.json"
    _write_bundle(source, world.bundle)

    assert main(["validate", str(source)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["valid"] is True
    assert receipt["sha256"] == world.bundle.canonical_hash()

    cutoff = world.censor_at.isoformat()
    assert main(["prefix", str(source), "--cutoff", cutoff]) == 0
    encoded = capsys.readouterr().out
    prefix = GraphBundleV2.from_json(encoded)
    validate_bundle(prefix)
    assert all(event.observed_at <= world.censor_at for event in prefix.events)
    assert encoded == prefix.to_json() + "\n"


def test_convert_v1_writes_valid_lineage_bundle_atomically(tmp_path: Path, capsys) -> None:
    source = tmp_path / "legacy.json"
    output = tmp_path / "converted.json"
    source.write_text(json.dumps(_legacy_source()), encoding="utf-8")

    assert (
        main(
            [
                "convert-v1",
                str(source),
                "--default-observed-at",
                "2025-01-01T00:00:00Z",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    converted = GraphBundleV2.from_json(output.read_text(encoding="utf-8"))
    validate_bundle(converted)
    assert receipt["sha256"] == converted.canonical_hash()
    assert converted.lineage
    assert output.read_text(encoding="utf-8") == converted.to_json() + "\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_failed_conversion_does_not_replace_existing_output(tmp_path: Path, capsys) -> None:
    source = tmp_path / "invalid-legacy.json"
    output = tmp_path / "keep.json"
    source.write_text('{"manifest":{},"nodes":[],"events":[null]}', encoding="utf-8")
    output.write_text("keep-me\n", encoding="utf-8")

    assert (
        main(
            [
                "convert-v1",
                str(source),
                "--default-observed-at",
                "2025-01-01T00:00:00Z",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert output.read_text(encoding="utf-8") == "keep-me\n"
    assert "error: convert-v1:" in capsys.readouterr().err


def test_synthetic_output_is_byte_deterministic(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert main(["synthetic", "--seed", "17", "--events", "40", "--output", str(first)]) == 0
    capsys.readouterr()
    assert main(["synthetic", "--seed", "17", "--events", "40", "--output", str(second)]) == 0
    capsys.readouterr()

    assert first.read_bytes() == second.read_bytes()
    bundle = GraphBundleV2.from_json(first.read_text(encoding="utf-8"))
    validate_bundle(bundle)


def test_scale_reads_declaration_without_importing_or_constructing_model(
    monkeypatch, capsys
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"braid.model.model", "torch"}:
            raise AssertionError(f"scale command imported executable model dependency {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert main(["scale", "probe-40m"]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["name"] == "probe-40m"
    assert config["parameter_label"] == "~40M"
    assert config["execution_policy"] == "local RTX 4090; publish only after gates"


def test_manifest_id_is_the_canonical_contract_hash(tmp_path: Path, capsys) -> None:
    manifest = ModelManifest(
        architecture_hash="0" * 64,
        tokenizer_hash="1" * 64,
        code_hash="2" * 64,
        environment_hash="3" * 64,
        data_hash="4" * 64,
        split_hash="5" * 64,
        checkpoint_hash="6" * 64,
        evaluation_hash="7" * 64,
        metadata={"created_at": datetime(2025, 1, 1, tzinfo=UTC).isoformat()},
    )
    source = tmp_path / "manifest.json"
    source.write_text(manifest.to_json() + "\n", encoding="utf-8")

    assert main(["manifest-id", str(source)]) == 0
    assert json.loads(capsys.readouterr().out) == {"manifest_id": manifest.manifest_id}


def test_train_reports_missing_optional_model_dependencies(monkeypatch, capsys) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "braid.model.training":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert main(["train", "--dry-run"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "install 'braid-eventgraph[models]'" in captured.err


def test_train_dry_run_is_allocation_free_and_does_not_certify_diversity(
    monkeypatch, capsys
) -> None:
    from braid.model.model import EventGraphModel

    def reject_allocation(*args, **kwargs):
        raise AssertionError("dry-run allocated an EventGraph model")

    monkeypatch.setattr(EventGraphModel, "__init__", reject_allocation)
    assert main(["train", "--dry-run", "--smoke"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["allocation_performed"] is False
    assert plan["execute_required"] is True
    assert plan["config"]["name"] == "tiny"
    assert plan["raw_encoded_tokens"] > 0
    assert plan["diverse_training_tokens"] is None
    assert plan["diverse_data_floor_passed"] is None
    assert plan["qualified"] is False
    assert "held-out vocabulary" in plan["objective_implementation"]["schema_episode"]


def test_train_requires_execute_and_smoke_writes_no_snapshot(monkeypatch, capsys) -> None:
    import braid.model.training_runner as training_runner

    assert main(["train", "--smoke"]) == 1
    assert "deliberate --execute" in capsys.readouterr().err

    calls = []

    def fake_run(examples, **kwargs):
        calls.append((examples, kwargs))
        return SimpleNamespace(
            to_dict=lambda: {
                "artifact_kind": "unqualified-training-snapshot",
                "claim_level": "none",
                "qualified": False,
                "snapshots": [],
            }
        )

    monkeypatch.setattr(training_runner, "run_training", fake_run)
    assert main(["train", "--execute", "--smoke"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["qualified"] is False
    assert receipt["snapshots"] == []
    assert len(calls) == 1
    assert calls[0][1]["output_directory"] is None
