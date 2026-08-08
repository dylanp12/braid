"""Public benchmark and evidence policy for Braid.

The package deliberately keeps measurement, evidence capture, and claim policy
separate from model code. This engineering release exposes diagnostic dossier
checks but cannot authorize any non-engineering claim until raw artifact replay
is implemented.
"""

from .specs import TRACK_SPECS, BenchmarkTrack, MetricDirection, validate_population

__all__ = [
    "TRACK_SPECS",
    "BenchmarkTrack",
    "MetricDirection",
    "validate_population",
]
