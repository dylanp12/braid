# Reference training runner

`braid train` is a small, local engineering path for exercising the current
single-next-event EventGraph heads. It is deliberately not a promotion tool and does
not claim that the planned research training system or any trained checkpoint exists.

## Safety modes

No model allocation occurs unless `--execute` is present. The allocation-free default
inspection path is:

```bash
braid train --dry-run
braid train --dry-run --smoke
```

The embedded smoke corpus is synthetic, uses the test-only tiny configuration, includes
an explicitly censored quiet window, and never writes weights:

```bash
braid train --execute --smoke
```

An ordinary tiny local run requires explicit data, lineage, and output paths:

```bash
braid train --execute \
  --data /local/path/train.jsonl \
  --lineage /local/path/lineage.json \
  --output /local/path/run \
  --steps 20 \
  --save-every 5
```

The default executable architecture is always `tiny`. A declared local probe must be
selected with an explicit `--config` path and must also provide a hash-bound
`--dataset-manifest` whose audited diverse-token count reaches that declaration's
floor. Cloud-only and multi-GPU declarations are rejected. Checked-in configuration
fields never constitute cloud authorization.

## Training JSONL contract

Every line is one canonical object with exactly these fields:

- `format_version`: currently `1`.
- `example_id`: unique training-record identifier.
- `split`: exactly `train`.
- `objective`: `patch`, `schema_episode`, `reconstruction`, or `contrastive`.
- `request`: a serialized v2 `ForecastRequest` containing only the causal prefix.
- `target_events`: zero or one serialized v2 `GraphEventV2`.
- `censor_at`: an aware timestamp exactly equal to `request.cutoff + request.horizon`.
- `outcome_complete`: explicitly `true`, asserting that the entire outcome window was
  observed.

Zero target events means a right-censored quiet window and contributes only the
survival likelihood. It is never inferred from a missing field. The current public
heads support at most one future event, one existing-node pointer, and one evidence
pointer. Records outside that boundary fail before optimization.

All four objective pools must be present. Each optimizer step consumes one
deterministically selected episode from each pool and combines their losses at exactly
65/15/10/10. This is accounting for the implemented reference objectives, not a claim
that the complete research objectives exist:

- `patch` is single-next-event factorized likelihood plus explicit quiet-window
  survival.
- `schema_episode` uses schema-conditioned next-event likelihood; this runner does not
  audit whether relation/type vocabularies are genuinely held out.
- `reconstruction` reconstructs factors of an explicitly held-out next target; it is
  not masked visible-prefix field reconstruction.
- `contrastive` is node/evidence pointer classification; it does not yet generate hard
  semantic or structural corruptions.

## Fit and dataset manifests

The fixed byte tokenizer has no learned vocabulary, but every preprocessing pass still
goes through a train-only guard. Its receipt binds the tokenizer fingerprint, canonical
corpus hash, and training-record-ID hash. Development, public-test, and sealed-test rows
are rejected.

Raw encoded JSONL token counts are diagnostic only and never satisfy a diverse-data
gate. Non-tiny execution requires a dataset manifest with exactly:

```json
{
  "data_hash": "<sha256 of canonical training examples>",
  "diverse_training_tokens": 800000000,
  "diversity_audit_hash": "<sha256>",
  "fitted_split": "train",
  "format_version": 1,
  "split_manifest_hash": "<sha256>",
  "training_record_ids_hash": "<sha256>"
}
```

The public runner validates the binding but does not create or certify the diversity
audit. An allocation-free dry run reports the canonical `corpus_hash` and
`training_record_ids_hash` needed to construct this manifest. The lineage file
similarly contains exactly caller-supplied `code_hash` and `environment_hash` values.

## Snapshots and resume

The save schedule writes atomic `step-NNNNNNNN` directories containing SafeTensors
model and AdamW optimizer state plus canonical JSON metadata. Resume verifies the
model, optimizer, corpus, split, tokenizer-fit receipt, run configuration, and optional
dataset-manifest binding before continuing.

Every artifact is labelled `unqualified-training-snapshot`. Its public checkpoint
metadata deliberately records `training_steps=0`, while the separate trainer state
records optimizer steps. Consequently these snapshots remain behind the inference
wrapper's untrained-model gate. Qualification, calibration, benchmark comparison, and
promotion are separate evidence-gated processes not implemented by this command.
