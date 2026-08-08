"""Architecture and evidence-gated scale configurations for EventGraph.

The large configurations are declarations, not released or trained models.  Keeping
them here makes a proposed scaling run reviewable without allocating any weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_TIME_UNIT_SECONDS = 86_400.0
MAX_NORMALIZED_TIME = 1_000_000.0


@dataclass(frozen=True, slots=True)
class EventGraphConfig:
    """Shape of an EventGraph Transformer.

    ``parameter_label`` is deliberately descriptive.  The exact parameter count
    depends on the tokenizer vocabulary and schema supplied to a run and must be
    recorded in that run's manifest.
    """

    name: str
    parameter_label: str
    d_model: int
    n_heads: int
    schema_layers: int
    decoder_layers: int
    d_ff: int
    context_events: int
    memory_slots: int
    dropout: float = 0.1
    time_unit_seconds: float = DEFAULT_TIME_UNIT_SECONDS
    min_diverse_training_tokens: int = 0
    execution_policy: str = "tests only"

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.schema_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("encoder and decoder need at least one layer")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff must be at least d_model")
        if self.context_events <= 0 or self.memory_slots <= 0:
            raise ValueError("context_events and memory_slots must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.time_unit_seconds) or self.time_unit_seconds <= 0:
            raise ValueError("time_unit_seconds must be positive and finite")
        if self.min_diverse_training_tokens < 0:
            raise ValueError("min_diverse_training_tokens cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EventGraphConfig:
        return cls(**value)


TINY_CONFIG = EventGraphConfig(
    name="tiny",
    parameter_label="test-only",
    d_model=32,
    n_heads=4,
    schema_layers=1,
    decoder_layers=1,
    d_ff=64,
    context_events=64,
    memory_slots=128,
    dropout=0.0,
)


SCALE_CONFIGS: dict[str, EventGraphConfig] = {
    "10m": EventGraphConfig(
        name="probe-10m",
        parameter_label="~10M",
        d_model=256,
        n_heads=8,
        schema_layers=3,
        decoder_layers=7,
        d_ff=1024,
        context_events=8_192,
        memory_slots=65_536,
        min_diverse_training_tokens=200_000_000,
        execution_policy="local IsoFLOP probe",
    ),
    "20m": EventGraphConfig(
        name="probe-20m",
        parameter_label="~20M",
        d_model=384,
        n_heads=8,
        schema_layers=3,
        decoder_layers=6,
        d_ff=1536,
        context_events=8_192,
        memory_slots=65_536,
        min_diverse_training_tokens=400_000_000,
        execution_policy="local IsoFLOP probe",
    ),
    "40m": EventGraphConfig(
        name="probe-40m",
        parameter_label="~40M",
        d_model=512,
        n_heads=8,
        schema_layers=3,
        decoder_layers=7,
        d_ff=2048,
        context_events=8_192,
        memory_slots=131_072,
        min_diverse_training_tokens=800_000_000,
        execution_policy="local RTX 4090; publish only after gates",
    ),
    "300m": EventGraphConfig(
        name="base-300m",
        parameter_label="~300M",
        d_model=1024,
        n_heads=16,
        schema_layers=6,
        decoder_layers=15,
        d_ff=4096,
        context_events=16_384,
        memory_slots=262_144,
        min_diverse_training_tokens=6_000_000_000,
        execution_policy="cloud only after probe and benchmark gates",
    ),
    "1.3b": EventGraphConfig(
        name="large-1.3b",
        parameter_label="~1.3B",
        d_model=2048,
        n_heads=32,
        schema_layers=6,
        decoder_layers=17,
        d_ff=8192,
        context_events=32_768,
        memory_slots=524_288,
        min_diverse_training_tokens=26_000_000_000,
        execution_policy="requires positive scaling and transfer curves",
    ),
    "7.2b": EventGraphConfig(
        name="frontier-candidate-7.2b",
        parameter_label="~7.2B",
        d_model=4096,
        n_heads=32,
        schema_layers=8,
        decoder_layers=25,
        d_ff=16_384,
        context_events=65_536,
        memory_slots=1_048_576,
        min_diverse_training_tokens=140_000_000_000,
        execution_policy="explicitly authorized multi-GPU run after every gate",
    ),
}


def get_scale_config(name: str) -> EventGraphConfig:
    """Return a declared scale without constructing or allocating the model."""

    if name == "tiny":
        return TINY_CONFIG
    try:
        return SCALE_CONFIGS[name]
    except KeyError as exc:
        choices = ", ".join(("tiny", *SCALE_CONFIGS))
        raise KeyError(f"unknown scale {name!r}; choose one of {choices}") from exc
