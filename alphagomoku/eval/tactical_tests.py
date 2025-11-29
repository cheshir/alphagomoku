"""Tactical test suite for evaluating model's tactical awareness.

This module contains a curated set of tactical positions that test
the model's ability to find forced wins, defend threats, and execute
double attacks.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TacticalPosition:
    """A single tactical test position"""
    board: np.ndarray  # 15x15 board state
    solution: Tuple[int, int]  # (row, col) of best move
    current_player: int  # 1 or -1
    category: str  # Type of tactic
    description: str  # Human-readable description
    difficulty: str  # "easy", "medium", "hard"


def _create_board(moves: List[Tuple[int, int, int]]) -> np.ndarray:
    """Create board from list of (row, col, player) tuples"""
    board = np.zeros((15, 15), dtype=np.int8)
    for row, col, player in moves:
        board[row, col] = player
    return board


# ============================================================================
# Easy Tactics (Win in 1, Obvious Defense)
# ============================================================================

EASY_TACTICS = [
    TacticalPosition(
        board=_create_board([
            (7, 6, 1), (7, 7, 1), (7, 8, 1), (7, 9, 1),  # Four in a row
            (6, 7, -1), (8, 7, -1),  # Opponent stones
        ]),
        solution=(7, 5),  # Complete the five
        current_player=1,
        category="win_in_1",
        description="Win in 1: Complete horizontal five",
        difficulty="easy",
    ),
    TacticalPosition(
        board=_create_board([
            (7, 7, 1), (7, 8, 1), (7, 9, 1), (7, 10, 1),  # Opponent's four
            (6, 7, 1), (8, 7, 1),  # Our stones
        ]),
        solution=(7, 6),  # Block the five
        current_player=-1,
        category="defend_immediate",
        description="Defend: Block immediate threat",
        difficulty="easy",
    ),
    TacticalPosition(
        board=_create_board([
            (7, 7, 1), (8, 8, 1), (9, 9, 1), (10, 10, 1),  # Diagonal four
        ]),
        solution=(6, 6),  # Complete diagonal
        current_player=1,
        category="win_in_1",
        description="Win in 1: Diagonal five",
        difficulty="easy",
    ),
]


# ============================================================================
# Medium Tactics (Open Four, Double Three, Win in 2)
# ============================================================================

MEDIUM_TACTICS = [
    TacticalPosition(
        board=_create_board([
            (7, 6, 1), (7, 7, 1), (7, 9, 1),  # Open four threat
            (6, 8, -1), (8, 8, -1),  # Opponent
        ]),
        solution=(7, 8),  # Create unstoppable threat
        current_player=1,
        category="open_four",
        description="Create open four (unstoppable)",
        difficulty="medium",
    ),
    TacticalPosition(
        board=_create_board([
            (7, 6, 1), (7, 7, 1), (7, 8, 1),  # Horizontal three
            (9, 8, 1), (10, 8, 1), (11, 8, 1),  # Vertical three
            (6, 7, -1),  # Opponent
        ]),
        solution=(7, 9),  # Create double threat
        current_player=1,
        category="double_threat",
        description="Double three attack",
        difficulty="medium",
    ),
    TacticalPosition(
        board=_create_board([
            (7, 5, 1), (7, 6, 1), (7, 7, 1),  # Three in a row
            (8, 9, 1),  # Supporting stone
            (6, 6, -1), (8, 6, -1),  # Opponent
        ]),
        solution=(7, 8),  # Extend to create threats
        current_player=1,
        category="win_in_2",
        description="Win in 2 moves",
        difficulty="medium",
    ),
]


# ============================================================================
# Hard Tactics (Complex Combinations, Win in 3+)
# ============================================================================

HARD_TACTICS = [
    TacticalPosition(
        board=_create_board([
            (7, 5, 1), (7, 6, 1), (7, 8, 1),  # Broken four
            (5, 7, 1), (6, 7, 1),  # Vertical threat
            (9, 9, 1), (10, 10, 1),  # Diagonal possibility
            (7, 7, -1),  # Opponent blocks one threat
            (6, 8, -1), (8, 8, -1),  # Opponent defense
        ]),
        solution=(8, 7),  # Multi-threat creation
        current_player=1,
        category="complex_combination",
        description="Complex multi-threat position",
        difficulty="hard",
    ),
    TacticalPosition(
        board=_create_board([
            (7, 7, 1), (7, 8, 1), (7, 9, 1),  # Three
            (9, 7, 1), (10, 7, 1),  # Vertical two
            (9, 9, 1),  # Supporting
            (6, 7, -1), (6, 8, -1),  # Opponent
        ]),
        solution=(8, 7),  # Key move that creates multiple threats
        current_player=1,
        category="win_in_3",
        description="Win in 3: Multiple threat combinations",
        difficulty="hard",
    ),
]


# ============================================================================
# Complete Test Suite
# ============================================================================

ALL_TACTICAL_POSITIONS = EASY_TACTICS + MEDIUM_TACTICS + HARD_TACTICS


def get_tactical_positions(difficulty: Optional[str] = None) -> List[TacticalPosition]:
    """Get tactical positions filtered by difficulty.

    Args:
        difficulty: Filter by "easy", "medium", or "hard". None returns all.

    Returns:
        List of tactical positions
    """
    if difficulty is None:
        return ALL_TACTICAL_POSITIONS

    return [pos for pos in ALL_TACTICAL_POSITIONS if pos.difficulty == difficulty]


def evaluate_tactical_suite(
    model,
    env,
    mcts_simulations: int = 400,
    difficulty: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate model on tactical test suite.

    Args:
        model: GomokuNet model
        env: GomokuEnv environment
        mcts_simulations: MCTS simulations per position
        difficulty: Filter positions by difficulty
        verbose: Print detailed results

    Returns:
        Dictionary with evaluation results
    """
    from ..mcts.mcts import MCTS
    from ..mcts.config import MCTSConfig

    positions = get_tactical_positions(difficulty)
    config = MCTSConfig(num_simulations=mcts_simulations)
    mcts = MCTS(model, env, config)

    correct = 0
    results_by_category = {}

    for pos in positions:
        # Set board state
        env.board = pos.board.copy()
        env.current_player = pos.current_player
        env.game_over = False

        # Get MCTS prediction
        policy, value = mcts.search(env.board, temperature=0.0)
        predicted_action = np.argmax(policy)
        predicted_row, predicted_col = divmod(predicted_action, 15)

        # Check if correct
        is_correct = (predicted_row, predicted_col) == pos.solution
        if is_correct:
            correct += 1

        # Track by category
        if pos.category not in results_by_category:
            results_by_category[pos.category] = {"correct": 0, "total": 0}
        results_by_category[pos.category]["total"] += 1
        if is_correct:
            results_by_category[pos.category]["correct"] += 1

        if verbose:
            status = "✓" if is_correct else "✗"
            print(
                f"{status} {pos.description}: "
                f"Expected {pos.solution}, Got ({predicted_row}, {predicted_col})"
            )

    accuracy = correct / len(positions) if positions else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(positions),
        "by_category": results_by_category,
    }
