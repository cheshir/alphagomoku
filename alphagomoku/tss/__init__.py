"""Threat-Space Search (TSS) module for tactical Gomoku analysis."""

from .position import Position
from .threat_detector import ThreatDetector, ThreatType
from .tss_search import TSSResult, tss_search

__all__ = ["tss_search", "TSSResult", "ThreatDetector", "ThreatType", "Position"]
