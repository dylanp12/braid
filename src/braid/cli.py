"""Dependency-light command line interface for the public Braid package."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _emit(value: Any) -> None:
    """Write one canonical JSON value to stdout."""

    from braid.contract import canonical_json

    sys.stdout.write(canonical_json(value) + "\n")


def _atomic_write(path: Path, value: Any) -> None:
    """Atomically replace ``path`` with one canonical JSON document."""

    from braid.contract import canonical_json

    destination = path.expanduser()
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {parent}")

    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_contract(path: Path, expected_type: type[Any]) -> Any:
    value = expected_type.from_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, expected_type):  # pragma: no cover - guarded by from_json
        raise TypeError(f"expected {expected_type.__name__}, got {type(value).__name__}")
    return value


def _parse_clock(value: str, option: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{option} must be a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{option} must include a UTC offset")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    opener: Callable[..., Any] = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"}
    with opener(path, **kwargs) as stream:
        rows = []
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def _find_source_file(directory: Path, stem: str, *, required: bool) -> Path | None:
    candidates = (directory / f"{stem}.jsonl.gz", directory / f"{stem}.jsonl")
    found = [path for path in candidates if path.is_file()]
    if len(found) > 1:
        raise ValueError(f"ambiguous v1 source: both {found[0].name} and {found[1].name} exist")
    if found:
        return found[0]
    if required:
        names = " or ".join(path.name for path in candidates)
        raise FileNotFoundError(f"v1 source is missing {names}: {directory}")
    return None


def _read_v1_source(path: Path) -> Mapping[str, Any]:
    """Read either a single migration JSON file or a frozen v1 bundle directory."""

    source = path.expanduser()
    if source.is_file():
        return _read_json_object(source)
    if not source.is_dir():
        raise FileNotFoundError(f"v1 source does not exist: {source}")

    manifest_candidates = (source / "contract.json", source / "manifest.json")
    manifests = [candidate for candidate in manifest_candidates if candidate.is_file()]
    if len(manifests) != 1:
        expected = "exactly one of contract.json or manifest.json"
        raise ValueError(f"v1 source must contain {expected}: {source}")
    nodes_path = _find_source_file(source, "nodes", required=True)
    events_path = _find_source_file(source, "events", required=True)
    judgments_path = _find_source_file(source, "judgments", required=False)
    assert nodes_path is not None and events_path is not None
    return {
        "manifest": _read_json_object(manifests[0]),
        "nodes": _read_json_lines(nodes_path),
        "events": _read_json_lines(events_path),
        "judgments": _read_json_lines(judgments_path) if judgments_path else [],
    }


def _status(_args: argparse.Namespace) -> int:
    from braid.bench import BenchmarkTrack

    _emit(
        {
            "claim_level": "engineering-build",
            "gates": {
                "benchmark_tracks": {track.value: "not-submitted" for track in BenchmarkTrack},
                "independent_sealed_cohorts": {"required": 2, "verified": 0},
                "raw_artifact_recomputation": "not-implemented",
                "leakage_audits": "not-submitted",
                "metric_canaries": "not-submitted",
                "official_baselines": "not-submitted",
            },
            "program": "Braid 2",
            "program_status": "pre-candidate-engineering",
        }
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    from braid.contract import GraphBundleV2, validate_bundle

    bundle = _read_contract(args.bundle, GraphBundleV2)
    validate_bundle(bundle)
    _emit(
        {
            "bundle_id": bundle.bundle_id,
            "counts": {
                "events": len(bundle.events),
                "evidence": len(bundle.evidence),
                "nodes": len(bundle.nodes),
                "schemas": len(bundle.schemas),
            },
            "sha256": bundle.canonical_hash(),
            "valid": True,
        }
    )
    return 0


def _prefix(args: argparse.Namespace) -> int:
    from braid.contract import GraphBundleV2, build_as_of_prefix

    bundle = _read_contract(args.bundle, GraphBundleV2)
    cutoff = _parse_clock(args.cutoff, "--cutoff")
    prefix = build_as_of_prefix(bundle, cutoff)
    _emit(prefix)
    return 0


def _convert_v1(args: argparse.Namespace) -> int:
    from braid.contract import convert_v1_bundle

    fallback = _parse_clock(args.default_observed_at, "--default-observed-at")
    bundle = convert_v1_bundle(_read_v1_source(args.source), default_observed_at=fallback)
    _atomic_write(args.output, bundle)
    _emit(
        {
            "bundle_id": bundle.bundle_id,
            "output": str(args.output),
            "sha256": bundle.canonical_hash(),
        }
    )
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    from braid.synthetic import ProcessGeneratorConfig, generate_world

    world = generate_world(ProcessGeneratorConfig(seed=args.seed, relation_events=args.events))
    _atomic_write(args.output, world.bundle)
    _emit(
        {
            "attempted_relation_events": world.attempted_relation_events,
            "bundle_id": world.bundle.bundle_id,
            "censor_at": world.censor_at,
            "duplicate_assertions_skipped": world.duplicate_assertions_skipped,
            "emitted_relation_events": world.emitted_relation_events,
            "output": str(args.output),
            "seed": world.seed,
            "sha256": world.bundle.canonical_hash(),
        }
    )
    return 0


def _scale(args: argparse.Namespace) -> int:
    # Deliberately local: contract-only commands must not import the model package,
    # and this command only reads a declaration rather than allocating weights.
    from braid.model.config import SCALE_CONFIGS, TINY_CONFIG, get_scale_config

    aliases = {config.name: name for name, config in SCALE_CONFIGS.items()}
    aliases[TINY_CONFIG.name] = "tiny"
    config = get_scale_config(aliases.get(args.name, args.name))
    _emit(config.to_dict())
    return 0


def _manifest_id(args: argparse.Namespace) -> int:
    from braid.contract import ModelManifest

    manifest = _read_contract(args.manifest, ModelManifest)
    _emit({"manifest_id": manifest.manifest_id})
    return 0


def _train(args: argparse.Namespace) -> int:
    try:
        from braid.model.config import TINY_CONFIG
        from braid.model.tokenizer import BraidTokenizer
        from braid.model.training import ObjectiveWeights
        from braid.model.training_runner import (
            OBJECTIVE_IMPLEMENTATION_NOTES,
            SNAPSHOT_KIND,
            DeterministicObjectiveIterator,
            TrainingDatasetManifest,
            TrainingLineage,
            TrainingRunConfig,
            config_from_declaration,
            fit_tokenizer_train_only,
            load_training_examples,
            run_training,
            smoke_training_examples,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "safetensors", "torch"}:
            raise
        raise ValueError(
            "training requires the optional model dependencies; install 'braid-eventgraph[models]'"
        ) from None

    if not args.dry_run and not args.execute:
        raise ValueError("training requires a deliberate --execute or allocation-free --dry-run")
    if args.dry_run and args.execute:
        raise ValueError("--dry-run and --execute are mutually exclusive")

    if args.config is None:
        config = TINY_CONFIG
    else:
        config = config_from_declaration(_read_json_object(args.config))

    examples = None
    if args.smoke:
        if args.data is not None:
            raise ValueError("--smoke uses its embedded corpus and cannot accept --data")
        if args.config is not None:
            raise ValueError("--smoke is fixed to the tiny config")
        if args.dataset_manifest is not None:
            raise ValueError("--smoke cannot accept an external dataset manifest")
        examples = smoke_training_examples()
    elif args.data is not None:
        examples = load_training_examples(args.data)
    dataset_manifest = (
        None
        if args.dataset_manifest is None
        else TrainingDatasetManifest.from_json(args.dataset_manifest.read_text(encoding="utf-8"))
    )

    if args.dry_run:
        tokenizer = BraidTokenizer()
        receipt = fit_tokenizer_train_only(tokenizer, examples) if examples is not None else None
        counts = None
        raw_encoded_tokens = None
        if examples is not None:
            iterator = DeterministicObjectiveIterator(examples, seed=args.seed)
            counts = {
                objective.value: len(iterator.pools[objective]) for objective in iterator.pools
            }
            raw_encoded_tokens = sum(
                len(tokenizer.encode_text(example.to_json())) for example in examples
            )
        if dataset_manifest is not None:
            if receipt is None:
                raise ValueError("--dataset-manifest requires --data")
            if dataset_manifest.data_hash != receipt.corpus_hash:
                raise ValueError("dataset manifest data_hash does not match training JSONL")
            if dataset_manifest.training_record_ids_hash != receipt.training_record_ids_hash:
                raise ValueError("dataset manifest training record IDs do not match training JSONL")
        _emit(
            {
                "allocation_performed": False,
                "artifact_kind": SNAPSHOT_KIND,
                "claim_level": "none",
                "cloud_authorized": False,
                "config": config.to_dict(),
                "corpus_hash": None if receipt is None else receipt.corpus_hash,
                "dataset_manifest_hash": (
                    None if dataset_manifest is None else dataset_manifest.manifest_hash
                ),
                "diverse_data_floor_passed": (
                    None
                    if dataset_manifest is None
                    else dataset_manifest.diverse_training_tokens
                    >= config.min_diverse_training_tokens
                ),
                "diverse_training_tokens": (
                    None if dataset_manifest is None else dataset_manifest.diverse_training_tokens
                ),
                "execute_required": True,
                "mode": "dry-run",
                "raw_encoded_tokens": raw_encoded_tokens,
                "objective_example_counts": counts,
                "objective_implementation": OBJECTIVE_IMPLEMENTATION_NOTES,
                "objective_weights": asdict(ObjectiveWeights()),
                "qualified": False,
                "tokenizer_fit_receipt_hash": (None if receipt is None else receipt.receipt_hash),
                "training_record_ids_hash": (
                    None if receipt is None else receipt.training_record_ids_hash
                ),
            }
        )
        return 0

    if args.smoke:
        if any(
            value is not None
            for value in (args.output, args.resume, args.lineage, args.dataset_manifest)
        ):
            raise ValueError("smoke execution never reads or writes training snapshots")
    else:
        missing = [
            option
            for option, value in (
                ("--data", args.data),
                ("--lineage", args.lineage),
                ("--output", args.output),
            )
            if value is None
        ]
        if missing:
            raise ValueError("executed training requires " + ", ".join(missing))
    assert examples is not None
    lineage = (
        None
        if args.lineage is None
        else TrainingLineage.from_json(args.lineage.read_text(encoding="utf-8"))
    )
    result = run_training(
        examples,
        model_config=config,
        run_config=TrainingRunConfig(
            total_steps=args.steps,
            save_every_steps=args.save_every,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
        ),
        lineage=lineage,
        dataset_manifest=dataset_manifest,
        output_directory=args.output,
        resume_snapshot=args.resume,
    )
    _emit(result.to_dict())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="braid",
        description="Braid 2 contract, evidence, and declared-scale utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show the current claim and evidence gates")
    status.set_defaults(handler=_status)

    validate = subparsers.add_parser("validate", help="validate a canonical v2 bundle")
    validate.add_argument("bundle", type=Path)
    validate.set_defaults(handler=_validate)

    prefix = subparsers.add_parser("prefix", help="emit a causally visible bundle prefix")
    prefix.add_argument("bundle", type=Path)
    prefix.add_argument("--cutoff", required=True, help="aware ISO-8601 observation cutoff")
    prefix.set_defaults(handler=_prefix)

    convert = subparsers.add_parser(
        "convert-v1", help="convert a frozen v1 JSON file or bundle directory"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument(
        "--default-observed-at",
        required=True,
        help="aware ISO-8601 fallback for records without observation time",
    )
    convert.add_argument("--output", required=True, type=Path)
    convert.set_defaults(handler=_convert_v1)

    synthetic = subparsers.add_parser("synthetic", help="write an audited synthetic bundle")
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--events", type=_positive_int, default=96)
    synthetic.add_argument("--output", required=True, type=Path)
    synthetic.set_defaults(handler=_synthetic)

    scale = subparsers.add_parser("scale", help="show a scale declaration without allocation")
    scale.add_argument("name")
    scale.set_defaults(handler=_scale)

    manifest = subparsers.add_parser("manifest-id", help="hash a canonical model manifest")
    manifest.add_argument("manifest", type=Path)
    manifest.set_defaults(handler=_manifest_id)

    train = subparsers.add_parser(
        "train",
        help="dry-run or explicitly execute the local unqualified training runner",
    )
    train.add_argument("--dry-run", action="store_true", help="validate without model allocation")
    train.add_argument(
        "--execute",
        action="store_true",
        help="deliberately permit local model allocation and optimizer steps",
    )
    train.add_argument("--smoke", action="store_true", help="use the embedded tiny corpus")
    train.add_argument("--data", type=Path, help="explicit train-only ForecastRequest JSONL")
    train.add_argument("--config", type=Path, help="explicit local scale declaration JSON")
    train.add_argument("--lineage", type=Path, help="code/environment hash JSON")
    train.add_argument(
        "--dataset-manifest",
        type=Path,
        help="hash-bound train split/diversity manifest required for non-tiny runs",
    )
    train.add_argument("--output", type=Path, help="directory for unqualified snapshots")
    train.add_argument("--resume", type=Path, help="unqualified snapshot to resume")
    train.add_argument("--steps", type=_positive_int, default=1, help="total optimizer steps")
    train.add_argument(
        "--save-every",
        type=_positive_int,
        default=1,
        help="snapshot interval in optimizer steps",
    )
    train.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu", help="explicit local torch device")
    train.set_defaults(handler=_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"error: {args.command}: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
