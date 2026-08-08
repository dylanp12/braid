# Synthetic marked-process data card

## Purpose

`braid.synthetic` generates deterministic mechanism-test worlds containing
dual clocks, typed role bindings, evidence, valid-time lag, ties, retractions,
supersessions, right-censoring cutoffs, and schema evolution.

## Integrity properties

- IDs include the seed, record kind, monotonic counter, and content digest.
- Semantic duplicate assertions are skipped and reported.
- Every generated bundle passes the strict v2 validator before return.
- Seeds reproduce identical canonical bundle hashes.
- Censoring produces a causal prefix; it does not erase the held-out suffix.

## Limitations

The generator is not a model of human software work or any other real domain.
Its output may diagnose architecture and contract behavior, but cannot support
trust, usefulness, temporal-transfer, foundation, or frontier claims. Repeated
or trivially varied generated records do not count as diverse scaling tokens.

Project-owned generated fixtures may be distributed under
[CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/), with a manifest that
identifies generator code, configuration, seed, and canonical bundle hash.
