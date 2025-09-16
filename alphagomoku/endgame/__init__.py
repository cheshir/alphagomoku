"""Endgame solver module for exact Gomoku analysis."""

from .endgame_solver import endgame_search, EndgameResult, should_use_endgame_solver
from .position import EndgamePosition

__all__ = ['endgame_search', 'EndgameResult', 'EndgamePosition', 'should_use_endgame_solver']