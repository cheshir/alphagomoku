"""Threat-Space Search (TSS) module for tactical Gomoku analysis."""

from .position import Position
from .threat_detector import ThreatDetector, ThreatType
from .tss_config import TSSConfig, get_default_config, set_default_config
from .tss_search import TSSResult, tss_search

__all__ = [
    "tss_search",
    "TSSResult",
    "ThreatDetector",
    "ThreatType",
    "Position",
    "TSSConfig",
    "get_default_config",
    "set_default_config",
]
