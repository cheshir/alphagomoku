"""Data models for the API."""

from typing import Optional, List, Tuple, Literal
from pydantic import BaseModel, Field
import time


class MoveRequest(BaseModel):
    """Request to make a move."""
    row: int = Field(..., ge=0, lt=15)
    col: int = Field(..., ge=0, lt=15)


class CreateGameRequest(BaseModel):
    """Request to create a new game."""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    player_color: Literal[1, -1] = 1  # 1 = black (first), -1 = white


class Move(BaseModel):
    """Represents a move on the board."""
    row: int
    col: int
    player: int


class TopMove(BaseModel):
    """Top move candidate from MCTS."""
    row: int
    col: int
    probability: float
    visits: int


class DebugInfo(BaseModel):
    """Debug information about AI decision."""
    policy_distribution: Optional[List[List[float]]] = None
    value_estimate: float
    simulations: int
    thinking_time_ms: float
    top_moves: List[TopMove]
    search_depth: int
    nodes_explored: int


class GameState(BaseModel):
    """Current state of a game."""
    game_id: str
    board: List[List[int]]
    board_size: int
    difficulty: str
    player_color: int
    current_player: int
    status: Literal["in_progress", "player_won", "ai_won", "draw"]
    last_move: Optional[Move] = None
    move_count: int
    player_time: float = 0.0  # seconds
    ai_time: float = 0.0  # seconds


class MoveResponse(BaseModel):
    """Response after making a move."""
    player_move: Move
    ai_move: Optional[Move] = None
    board: List[List[int]]
    current_player: int
    status: Literal["in_progress", "player_won", "ai_won", "draw"]
    move_count: int
    player_time: float
    ai_time: float
    debug_info: Optional[DebugInfo] = None


class CreateGameResponse(BaseModel):
    """Response when creating a game."""
    game_id: str
    board_size: int
    difficulty: str
    player_color: int
    current_player: int
    status: str
    ai_move: Optional[Move] = None  # If AI goes first
    debug_info: Optional[DebugInfo] = None


class ResignResponse(BaseModel):
    """Response when player resigns."""
    game_id: str
    status: str
    winner: int