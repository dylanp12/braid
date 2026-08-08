# Evidence and claim policy

## Claim vocabulary

- **Engineering build:** contract, model, and benchmark code execute correctly.
- **Research candidate:** public development results exist with complete raw
  artifacts and qualified baselines.
- **Competitive:** preregistered confirmatory endpoints are non-inferior to the
  strongest qualified comparator.
- **Frontier:** prohibited unless every E/R/S/T/P track passes the superiority
  gate on two independently administered prospective cohorts.

There is no composite frontier score. A failed track cannot be averaged away.

## Required tracks

- E: unseen entities in entirely held-out repository/organization clusters.
- R: unseen relations in semantic zero-shot and symbol-renamed eight-shot modes.
- S: unseen raw schemas, including schema-mapping accuracy.
- T: future-event discrimination and calibrated survival forecasts.
- P: complete graph-patch distribution, validity, matched event-set quality,
  timing, evidence basis, diversity, and memorization.

Every comparison uses the identical population, cutoff, candidates, censoring,
and permitted features. Sampled negatives are frozen and matched when complete
type-compatible ranking is infeasible. Ties receive average rank.

## Confirmatory decision rule

For every endpoint, compare against the strongest qualified baseline inside a
hierarchical bootstrap resampling schema, organization, repository, time block,
and query. Holm-adjust the confirmatory family at alpha 0.05.

A track passes only when:

- the adjusted lower 95% confidence bound is above zero;
- the point improvement is at least 0.02 for MRR, macro-F1, or AUPRC, or a 5%
  relative reduction for Brier score or normalized NLL;
- Braid wins on at least 60% of top-level clusters;
- no populated schema family regresses by more than 0.05;
- calibration trails the best comparator by no more than 0.01; and
- the strongest learned baseline beats every relevant shortcut probe by 0.02
  on public development data.

## Sealed evaluation

Before prospective data exists, freeze hypotheses, exclusions, checkpoints,
containers, metric code, and baseline configurations. A conflict-free external
custodian runs the signed container without network egress. No intermediate
score or error example returns to model developers. After disclosure, the
cohort is retired and cannot support a fresh claim. Frontier status requires a
second prospective cohort.

Every result must retain query-level predictions, baseline trials, exclusion
logs, bootstrap draws, published metric bytes, manifests, environment and hardware records, checkpoint
digests, stdout/stderr, and the signed evaluator receipt. Reports are generated
from those artifacts.

The public claim controller in this engineering release is deliberately
**non-authorizing**: it always rejects research-candidate, frontier, and foundation-model
claims. It diagnoses whether declared files, typed manifests, signed receipts, and
reported gates are internally consistent, but it does not yet recompute population
counts, endpoint values, baseline trials, or bootstrap decisions from typed raw
prediction artifacts. Empty or semantically unrelated files therefore cannot unlock
claim vocabulary in this release.

A future claim-grade controller must deterministically reconstruct every population and
endpoint from signed raw records before this unconditional block may be removed. The
intended format canonically encodes each cohort's endpoint values, raw population counts,
retrieval audit, leakage result, metric canaries, and baseline-reproduction status as
`gate_evidence` inside that cohort's result manifest. The receipt must bind that exact
result hash and verify as Ed25519 under a custodian key registered in the pre-evaluation
trust policy. Free-form signatures, unregistered keys, unrelated result hashes, or
missing artifacts already fail the diagnostic checks; passing those checks still cannot
authorize a claim.

Candidate and sealed evidence must name an actual local result directory. The
controller re-reads every declared file, rejects symbolic links, verifies bytes
and sizes, and compares canonical typed `DatasetManifest`, `SplitManifest`, and
`ModelManifest` values with their artifacts. It also reconstructs the only valid
checkpoint-manifest and metrics payloads from the declared result. Population
qualification is recomputed from signed `TrackPopulation` counts; a caller-supplied
pass flag is never accepted.

Every non-engineering claim names one explicit evaluation subject: the exact model
manifest, checkpoint digest, and container digest. Both sealed cohorts must bind
that same subject. Raw-object content identities and typed split membership are
compared independently of dataset labels, creation timestamps, record IDs, and
manifest prose. Observation timestamps remain part of chronological membership.
These checks diagnose obvious reuse but are not yet sufficient to authorize a claim.

Every endpoint and population gate is evaluated independently on each of the two
prospective cohorts. The cohorts must have distinct IDs, custodians, signing keys,
result hashes, dataset manifests, and split manifests, and both receipts must attest
to egress-disabled execution. Two reruns of one spent population do not constitute
two prospective cohorts.

## External trust boundary

Two facts remain outside what repository code can prove. Custodian identities,
Ed25519 keys, and the hash of their trust policy must be anchored by an independent
authority before either cohort is collected; a keyring supplied after seeing results
is not preregistration. In addition, semantic independence between cohorts still
depends on the custodian's signed leakage audit and source-governance records. Byte
identity detects replay and superficial manifest changes, but cannot by itself prove
that differently encoded corpora are unrelated. These are explicit external trust
assumptions, not outputs of the claim controller.
