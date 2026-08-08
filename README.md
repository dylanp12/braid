# Braid

Braid is an evidence-gated research platform for schema-conditioned temporal
graph forecasting. Its model is designed to generate constrained future graph
patches over previously unseen entities, relations, and schemas.

This source release is an **engineering build** in Braid 2's pre-candidate research
phase. It is not a trained model release and makes no capability claim. Its public
claim controller categorically rejects every non-engineering label until metrics and
populations can be recomputed from typed raw artifacts.
Frontier and foundation-model labels are prohibited until every public benchmark
track and two independently administered sealed evaluations pass the gates encoded
in this repository.

## Repository boundary

This public repository contains the v2 contract, validator, benchmark,
reference baselines, EventGraph model code, training objectives, exact candidate
declarations, a tiny-by-default local reference training runner, and reproducibility
formats. Raw licensed corpora, prospective holdouts, and strongest weights live outside
this repository. Private Alluvia data is never a global training input.

## Quick start

```bash
uv sync --frozen --extra dev
uv run pytest
uv run braid --help
uv run braid train --dry-run --smoke
```

See `docs/architecture.md`, `docs/evidence-policy.md`, and
`docs/data-governance.md` for the normative research rules.
Candidate declarations and their parameter-count contract are documented in
`configs/README.md`.
The runner's fail-closed JSONL format, objective limitations, data-manifest gate, and
unqualified snapshot semantics are documented in `docs/training.md`.

The exact boundary between implemented code and evidence-gated future work is
recorded in `docs/implementation-status.md`.
