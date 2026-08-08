# Braid EventGraph model card

## Release status

**Claim level:** engineering build
**Program status:** pre-candidate engineering phase
**Trained public checkpoint:** none
**Production authorization:** none
**Automatic graph mutation:** prohibited

This source release implements the Braid EventGraph architecture, training
objectives, constrained inference, checkpoint lineage, and diagnostic evidence gates. It
does not contain a trained 40M probe and makes no capability claim.

## Intended use

- Research on zero/few-shot temporal transfer across unseen entities,
  relations, and raw schemas.
- Reproducible comparison of temporal-graph, relational, and shortcut baselines.
- Contract, causal-slicing, constrained-decoding, and checkpoint-integrity tests.

## Out-of-scope use

Production ranking, trust/integrity decisions, personnel evaluation, automated
forecast execution, safety-sensitive decisions, or public frontier/foundation
marketing.

## Architecture

The model combines a schema encoder, relation-motif encoder, causal event
decoder, and role-aware persistent node memory. It predicts field-factorized
future graph patches and can abstain when untrained, uncalibrated, incompatible,
or below the configured retrieval-coverage threshold.

All entity/relation/type handles are episode-local. Temporal input is sorted by
observation time and validity is a signed lag. Constrained decoding validates
types, roles, cardinalities, evidence visibility, graph handles, and horizon.

## Evidence status

Braid 1 results are not inherited. No E/R/S/T/P benchmark population currently
has a Braid 2 model result. `braid status` therefore reports an engineering
build in a pre-candidate research program. All non-engineering claim vocabulary is
hard-disabled until endpoint and population evidence is reproduced from typed raw
artifacts. Frontier language additionally requires every endpoint, all population and
leakage gates, official baseline reproductions, and two independent prospective sealed
cohorts.

## Privacy

Global training excludes raw Alluvia notes, judgments, embeddings, and derived
examples. Future product adapters remain local or organization-local and are
not part of this release.
