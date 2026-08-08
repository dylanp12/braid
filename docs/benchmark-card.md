# Braid benchmark card

## Purpose

The benchmark measures five claims independently. It has no composite score.
Engineering tests and synthetic fixtures validate the harness but cannot
establish real-world capability.

| Track | Held-out capability | Confirmatory endpoints |
|---|---|---|
| E | Entities and complete repository clusters | Macro-repository filtered MRR |
| R | Relations in semantic zero-shot and symbolized eight-shot modes | Macro-relation MRR in both modes |
| S | Genuinely different raw source schemas | Schema-mapping macro-F1 and link MRR |
| T | Future events under right censoring | Macro-repository AUPRC and integrated Brier score |
| P | Complete future graph-patch distributions | Validity, normalized NLL, matched event-set F1, time calibration, evidence basis, conditional diversity, and memorization |

Normalized window NLL is the negative sum of discrete-event and continuous-time
log densities divided by the token count. It must be finite but is not artificially
clamped non-negative, because a continuous density may exceed one in its chosen units.

## Population floors

Entity and temporal tracks require 30 repository clusters. Relation transfer
requires 12 target relations. Schema transfer requires six source-schema
families. Applicable tracks require at least 5,000 confirmatory queries or
non-empty windows. No organization may contribute more than 15% of a track.

## Baseline policy

The baseline registry includes shortcut, classical, relational, temporal,
textual, external foundation-model, and legacy comparators. A baseline is
qualified only when its code/checkpoint, permitted inputs, candidate population,
tuning envelope, environment, and official-result reproduction are frozen.

Every comparison uses identical cutoffs, candidates, labels, censoring, and
feature availability. Average rank is used for ties. Shortcut probes are a
benchmark-sensitivity requirement, not weak comparators to celebrate beating.

## Evidence artifacts

Result manifests require complete predictions, baseline trials, bootstrap
draws, source and environment locks, an SBOM, and hardware records. Prospective
receipts bind the result hash, egress status, custodian, signing key, and cohort.
Patch validity is recomputed from each native `ForecastDistribution` against its
own `ForecastRequest`; evaluators never accept a model- or caller-supplied validity
flag. Diversity is computed across repeated samples conditioned on the same request,
not across unrelated queries.
