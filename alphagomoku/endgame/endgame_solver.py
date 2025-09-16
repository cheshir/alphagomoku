"""Alpha-beta endgame solver for Gomoku."""

import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import numpy as np

from .position import EndgamePosition


@dataclass
class EndgameResult:
    """Result of endgame search."""

    best_move: Optional[Tuple[int, int]]  # Best move found, None if no solution
    is_win: bool  # True if current player can force a win
    is_loss: bool  # True if current player will lose with optimal play
    evaluation: int  # Exact evaluation (-1: loss, 0: draw, 1: win)
    depth_to_mate: Optional[int]  # Plies to mate (positive for win, negative for loss)
    search_stats: Dict[str, int]  # Search statistics


class EndgameSolver:
    """Alpha-beta endgame solver for exact analysis of Gomoku positions."""

    def __init__(self):
        self.transposition_table: Dict[str, Tuple[int, int, Optional[Tuple[int, int]]]] = {}
        self.nodes_searched = 0
        self.tt_hits = 0
        self.start_time = 0.0

    def _position_hash(self, position: EndgamePosition) -> str:
        """Generate hash key for position."""
        # Simple hash using board state and current player
        board_hash = hash(position.board.tobytes())
        return f"{board_hash}_{position.current_player}"

    def _evaluate_position(self, position: EndgamePosition) -> Tuple[int, Optional[int]]:
        """Evaluate terminal position. Returns (value, depth_to_mate)."""
        is_terminal, winner = position.is_terminal()

        if not is_terminal:
            return 0, None

        if winner == 0:
            return 0, 0  # Draw
        elif winner == position.current_player:
            return 1, 1  # Current player won
        else:
            return -1, -1  # Current player lost

    def _alphabeta(self, position: EndgamePosition, depth: int, alpha: int, beta: int,
                  maximizing: bool, time_limit: float) -> Tuple[int, Optional[Tuple[int, int]], Optional[int]]:
        """
        Alpha-beta search with transposition table.
        Returns (evaluation, best_move, depth_to_mate).
        """
        self.nodes_searched += 1

        # Check time limit
        if time.time() - self.start_time > time_limit:
            return 0, None, None

        # Check terminal condition
        terminal_eval, terminal_depth = self._evaluate_position(position)
        if terminal_depth is not None:
            return terminal_eval, None, terminal_depth

        # Check transposition table
        pos_hash = self._position_hash(position)
        if pos_hash in self.transposition_table:
            self.tt_hits += 1
            stored_eval, stored_depth, stored_move = self.transposition_table[pos_hash]
            if stored_depth >= depth:
                return stored_eval, stored_move, None

        # Generate moves (prioritize critical moves in endgame)
        legal_moves = position.get_critical_moves()
        if not legal_moves:
            legal_moves = position.get_legal_moves()

        if not legal_moves:
            return 0, None, 0  # Draw

        # Sort moves to try winning moves first
        def move_priority(move):
            row, col = move
            # Try the move and see if it wins immediately
            temp_pos = position.make_move(row, col)
            is_term, winner = temp_pos.is_terminal()
            if is_term and winner == position.current_player:
                return 1000  # Immediate win
            elif is_term and winner == -position.current_player:
                return -1000  # Immediate loss
            else:
                return 0

        legal_moves.sort(key=move_priority, reverse=True)

        best_move = None
        best_eval = -2 if maximizing else 2
        best_mate_depth = None

        for move in legal_moves:
            row, col = move
            new_position = position.make_move(row, col)

            eval_score, _, mate_depth = self._alphabeta(
                new_position, depth - 1, alpha, beta, not maximizing, time_limit
            )

            # Adjust mate depth
            if mate_depth is not None:
                mate_depth = mate_depth + 1 if mate_depth > 0 else mate_depth - 1

            if maximizing:
                if eval_score > best_eval or (eval_score == best_eval and
                    mate_depth is not None and (best_mate_depth is None or mate_depth < best_mate_depth)):
                    best_eval = eval_score
                    best_move = move
                    best_mate_depth = mate_depth
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            else:
                if eval_score < best_eval or (eval_score == best_eval and
                    mate_depth is not None and (best_mate_depth is None or mate_depth > best_mate_depth)):
                    best_eval = eval_score
                    best_move = move
                    best_mate_depth = mate_depth
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff

        # Store in transposition table
        self.transposition_table[pos_hash] = (best_eval, depth, best_move)

        return best_eval, best_move, best_mate_depth

    def search(self, position: EndgamePosition, max_depth: int = 20,
               time_limit: float = 5.0) -> EndgameResult:
        """
        Perform endgame search on position.

        Args:
            position: Position to analyze
            max_depth: Maximum search depth
            time_limit: Time limit in seconds

        Returns:
            EndgameResult with analysis results
        """
        self.nodes_searched = 0
        self.tt_hits = 0
        self.start_time = time.time()

        # Quick terminal check
        terminal_eval, terminal_depth = self._evaluate_position(position)
        if terminal_depth is not None:
            return EndgameResult(
                best_move=None,
                is_win=terminal_eval == 1,
                is_loss=terminal_eval == -1,
                evaluation=terminal_eval,
                depth_to_mate=terminal_depth,
                search_stats={
                    'nodes_searched': 1,
                    'tt_hits': 0,
                    'time_ms': int((time.time() - self.start_time) * 1000),
                    'depth_reached': 0
                }
            )

        # Check for immediate winning moves
        for move in position.get_critical_moves() or position.get_legal_moves():
            row, col = move
            temp_pos = position.make_move(row, col)
            is_term, winner = temp_pos.is_terminal()
            if is_term and winner == position.current_player:
                return EndgameResult(
                    best_move=move,
                    is_win=True,
                    is_loss=False,
                    evaluation=1,
                    depth_to_mate=1,
                    search_stats={
                        'nodes_searched': 1,
                        'tt_hits': 0,
                        'time_ms': int((time.time() - self.start_time) * 1000),
                        'depth_reached': 1
                    }
                )

        # Iterative deepening search
        best_result = None
        for depth in range(1, max_depth + 1):
            if time.time() - self.start_time > time_limit:
                break

            evaluation, best_move, mate_depth = self._alphabeta(
                position, depth, -2, 2, True, time_limit
            )

            best_result = EndgameResult(
                best_move=best_move,
                is_win=evaluation == 1,
                is_loss=evaluation == -1,
                evaluation=evaluation,
                depth_to_mate=mate_depth,
                search_stats={
                    'nodes_searched': self.nodes_searched,
                    'tt_hits': self.tt_hits,
                    'time_ms': int((time.time() - self.start_time) * 1000),
                    'depth_reached': depth
                }
            )

            # Stop if we found a definitive result
            if evaluation != 0:
                break

        return best_result or EndgameResult(
            best_move=None,
            is_win=False,
            is_loss=False,
            evaluation=0,
            depth_to_mate=None,
            search_stats={
                'nodes_searched': self.nodes_searched,
                'tt_hits': self.tt_hits,
                'time_ms': int((time.time() - self.start_time) * 1000),
                'depth_reached': 0
            }
        )


def endgame_search(position: EndgamePosition, max_depth: int = 20,
                  time_limit: float = 5.0) -> EndgameResult:
    """
    Convenience function for endgame search.

    Args:
        position: Position to analyze
        max_depth: Maximum search depth
        time_limit: Time limit in seconds

    Returns:
        EndgameResult with analysis results
    """
    solver = EndgameSolver()
    return solver.search(position, max_depth, time_limit)


def should_use_endgame_solver(position: EndgamePosition, difficulty: str = 'medium') -> bool:
    """
    Determine if endgame solver should be used based on position and difficulty.

    Args:
        position: Current position
        difficulty: Difficulty level ('easy', 'medium', 'strong')

    Returns:
        True if endgame solver should be used
    """
    empty_count = position.get_empty_count()

    thresholds = {
        'easy': 0,  # Disabled
        'medium': 14,
        'strong': 20
    }

    threshold = thresholds.get(difficulty, 14)
    return empty_count <= threshold and empty_count > 0