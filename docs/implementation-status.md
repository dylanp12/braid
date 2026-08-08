# Implementation status

This document is the release boundary for the first public Braid 2 engineering
build. It separates executable code from work that the evidence policy intentionally
prevents us from claiming or running.

## Implemented and testable

- Immutable v2 records for schemas, graph events, forecast requests and
  distributions, and model manifests.
- Fail-closed validation for dual clocks, identifiers, roles, types, evidence,
  lineage, duplicate assertions, and causal references.
- Deterministic v1-to-v2 conversion that records source hashes without rewriting
  the frozen input. Every clock synthesized from a migration fallback is marked
  as imputed; temporal requests reject those records until an independently
  justified observation clock is supplied.
- As-of prefix construction that removes future schemas, nodes, payload snapshots,
  events, evidence, and judgments before tensorization.
- An audited synthetic marked-process generator with unique event identities,
  retractions, supersessions, schema evolution, tie groups, and censoring.
- Benchmark track and population specifications, metric canaries, runnable shortcut
  baselines, split/leakage audits, hierarchical statistics, result manifests,
  byte-for-byte result/replay verification, cryptographically verified sealed
  receipts, non-composite diagnostic gate reports, and an unconditional block on every
  non-engineering claim. Patch validity is recomputed from native request/distribution
  pairs rather than accepted as a reported flag.
- EventGraph model components: schema and motif encoders, causal decoder,
  role-aware node memory, typed retrieval, finite-state graph-patch grammar,
  factorized losses, right-censoring likelihood, and SafeTensors checkpoints.
- A tiny-by-default, explicitly executed local reference training runner with
  deterministic objective-pool iteration, exact 65/15/10/10 accounting, train-only
  tokenizer preprocessing receipts, explicit censoring, AdamW steps, and resumable
  SafeTensors snapshots. Snapshots remain inference-ineligible and unqualified.
- Declarative 10M, 20M, 40M, 300M, 1.3B, and 7.2B candidate configurations with
  exact architecture, tokenizer, objective-mixture, data-floor, and promotion-policy
  fields. Meta-device tests bind each rounded scale label to its actual parameter
  count without allocating weights.
- Public command-line tools for status, validation, causal prefixing, v1 conversion,
  synthetic fixtures, scale declarations, manifest identifiers, and allocation-free
  training dry runs or deliberately executed local smoke runs.

The released inference wrapper fails closed for untrained, uncalibrated, incompatible,
or low-retrieval-coverage checkpoints. Its current engineering path proposes one
grammar-valid event per sampled window. The grammar and training heads cover all eight
contract operations; autoregressive multi-event rollouts and structured decoding for
`UPDATE_NODE`, `SCHEMA_CHANGE`, and `JUDGE` remain implementation gates.

## Evidence-gated and not yet performed

- No licensed 0.8B-token diverse corpus has been approved for the 40M probe.
- The reference runner does not yet implement the full planned objective construction:
  schema episodes do not audit held-out vocabularies, reconstruction is held-out
  next-target factor reconstruction rather than masked visible-prefix reconstruction,
  and contrastive loss is pointer classification without explicit hard corruptions.
  The current argument target remains single-pointer.
- No 10M/20M/40M IsoFLOP run, 40M training run, scaling curve, or cloud run has been
  executed or authorized by this release.
- No true-target retrieval measurement has reached the 99% comparison gate.
- External baseline adapters are declared but not qualified; official anchor results
  have not been reproduced.
- No E/R/S/T/P population has a Braid 2 result bundle.
- The evidence controller does not yet derive population counts, endpoint metrics,
  baseline comparisons, or bootstrap decisions from typed raw artifacts. Candidate,
  frontier, and foundation-model authorization therefore remains hard-disabled even
  for an internally consistent signed dossier.
- No independent custodian has evaluated a signed checkpoint, and no prospective
  sealed cohort has been spent.
- No public model weights are included. Randomly initialized test weights are never
  presented as a checkpoint.
- The optional Alluvia adapter is absent by design until the research gate passes.

Consequently this release is a **pre-candidate engineering build**.
It is not a trained model release, a production system, or evidence for a capability
claim.
