"""Probing tests for MedVLM-Probe"""

from .base import BaseProbe
from .classification import ClassificationProbe
from .consistency import ConsistencyProbe
from .swap import SwapProbe
from .grounding import GroundingProbe

__all__ = [
    "BaseProbe",
    "ClassificationProbe", 
    "ConsistencyProbe",
    "SwapProbe",
    "GroundingProbe"
]

PROBE_REGISTRY = {
    "classification": ClassificationProbe,
    "consistency": ConsistencyProbe,
    "swap": SwapProbe,
    "grounding": GroundingProbe,
}


def get_probe(probe_name: str) -> type:
    """Get probe class by name"""
    if probe_name not in PROBE_REGISTRY:
        raise ValueError(f"Unknown probe: {probe_name}. Available: {list(PROBE_REGISTRY.keys())}")
    return PROBE_REGISTRY[probe_name]
