"""EventGraph model package.

Tokenization, grammar, retrieval, and scale declarations remain importable without
PyTorch.  Neural components require the optional ``models`` dependency.
"""

from braid.model.config import SCALE_CONFIGS, TINY_CONFIG, EventGraphConfig, get_scale_config
from braid.model.grammar import GraphPatchGrammar, Operation
from braid.model.retrieval import TypedRetriever
from braid.model.tokenizer import BraidTokenizer, DynamicHandleTable, HandleKind

__all__ = [
    "BraidTokenizer",
    "DynamicHandleTable",
    "EventGraphConfig",
    "GraphPatchGrammar",
    "HandleKind",
    "Operation",
    "SCALE_CONFIGS",
    "TINY_CONFIG",
    "TypedRetriever",
    "get_scale_config",
]
