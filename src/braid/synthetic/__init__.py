"""Audited synthetic process generators for mechanism tests.

Synthetic worlds are never claim-grade evidence. They exist to exercise causal
contracts, schema transfer, censoring, and model-training paths.
"""

from .process import GeneratedWorld, ProcessGeneratorConfig, generate_world

__all__ = ["GeneratedWorld", "ProcessGeneratorConfig", "generate_world"]
