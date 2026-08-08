# Contributing

Contributions must preserve causal slicing, public/private data boundaries, and
the claim vocabulary in `docs/evidence-policy.md`.

Before submitting a change, run:

```bash
uv sync --frozen --extra dev
uv run python tools/check_public_boundary.py
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Model changes require an ablation against the simplest affected baseline.
Metric changes require orientation and tie canaries. Dataset changes require a
license manifest and a fresh leakage report. Reports must be generated from raw
machine-readable artifacts.
