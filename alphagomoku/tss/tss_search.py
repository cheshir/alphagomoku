"""Main TSS search implementation."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .position import Position
from .threat_detector import ThreatDetector, ThreatType


@dataclass
class TSSResult:
    """Result of TSS search."""

    forced_move: Optional[Tuple[int, int]] = None
    is_forced_win: bool = False
    is_forced_defense: bool = False
    search_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.search_stats is None:
            self.search_stats = {}


class TSSSearcher:
    """Threat-Space Search implementation."""

    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.threat_detector = ThreatDetector(board_size)
        self.nodes_visited = 0
        self.start_time = 0
        self.time_cap_ms = 0

    def search(self, position: Position, depth: int, time_cap_ms: int) -> TSSResult:
        """Perform TSS search."""
        if depth is None or depth < 0:
            raise ValueError("depth must be >= 0")
        if time_cap_ms is None or time_cap_ms <= 0:
            raise ValueError("time_cap_ms must be > 0")
        # Validate board values
        import numpy as np
        if not np.all(np.isin(position.board, [-1, 0, 1])):
            raise ValueError("Invalid board values in TSS position")
        self.nodes_visited = 0
        self.start_time = time.time() * 1000
        self.time_cap_ms = time_cap_ms

        # CRITICAL: Check for immediate WIN first (complete 5-in-a-row)
        immediate_win = self._check_immediate_win(position, position.current_player)
        if immediate_win:
            return TSSResult(
                forced_move=immediate_win,
                is_forced_win=True,
                search_stats={
                    "nodes_visited": 1,
                    "time_ms": time.time() * 1000 - self.start_time,
                    "reason": "immediate_win",
                },
            )

        # Check if opponent has immediate win threat (5-in-a-row next move)
        defense_moves = self.threat_detector.must_defend(
            position, position.current_player
        )

        # If both we and opponent have winning moves, check who wins first
        if defense_moves:
            # Check if we have a winning move too
            our_win = self._check_immediate_win(position, position.current_player)
            if our_win:
                # We both have immediate wins - we move first, so we win!
                return TSSResult(
                    forced_move=our_win,
                    is_forced_win=True,
                    search_stats={
                        "nodes_visited": 2,
                        "time_ms": time.time() * 1000 - self.start_time,
                        "reason": "win_over_defense",
                    },
                )

            # Only opponent has immediate win, must defend
            return TSSResult(
                forced_move=defense_moves[0],
                is_forced_defense=True,
                search_stats={
                    "nodes_visited": 2,
                    "time_ms": time.time() * 1000 - self.start_time,
                    "reason": "immediate_defense",
                },
            )

        # Search for forced win sequence (multi-move)
        best_move = self._search_forced_win(position, depth, position.current_player)
        if best_move:
            return TSSResult(
                forced_move=best_move,
                is_forced_win=True,
                search_stats={
                    "nodes_visited": self.nodes_visited,
                    "time_ms": time.time() * 1000 - self.start_time,
                    "reason": "forced_win",
                },
            )

        # No forced sequence found
        return TSSResult(
            search_stats={
                "nodes_visited": self.nodes_visited,
                "time_ms": time.time() * 1000 - self.start_time,
                "reason": "no_forced_sequence",
            }
        )

    def _check_immediate_win(
        self, position: Position, player: int
    ) -> Optional[Tuple[int, int]]:
        """Check if player has a move that immediately wins (completes 5-in-a-row)."""
        # Check all empty cells for immediate winning moves
        for r in range(self.board_size):
            for c in range(self.board_size):
                if position.board[r, c] == 0:
                    # Try placing stone here
                    test_board = position.board.copy()
                    test_board[r, c] = player

                    # Check if this creates 5 in a row
                    if self.threat_detector._creates_five_in_row(test_board, r, c, player):
                        return (r, c)
        return None

    def _search_forced_win(
        self, position: Position, depth: int, player: int
    ) -> Optional[Tuple[int, int]]:
        """Search for forced win sequence."""
        if depth <= 0 or self._time_exceeded():
            return None

        self.nodes_visited += 1

        # Check terminal
        is_terminal, winner = position.is_terminal()
        if is_terminal:
            return None if winner != player else True  # Found win

        # Get threat moves (prioritize high-value threats)
        threat_moves = self._get_prioritized_moves(position, player)

        for move in threat_moves:
            row, col = move
            new_position = position.make_move(row, col)

            # Check if this move wins immediately
            is_terminal, winner = new_position.is_terminal()
            if is_terminal and winner == player:
                return move

            # Check if this creates a winning threat
            if self.threat_detector.is_winning_threat(position, row, col, player):
                # Verify opponent cannot defend
                if self._opponent_cannot_defend(new_position, depth - 1, -player):
                    return move

        return None

    def _opponent_cannot_defend(
        self, position: Position, depth: int, opponent: int
    ) -> bool:
        """Check if opponent has no adequate defense."""
        if depth <= 0 or self._time_exceeded():
            return False

        self.nodes_visited += 1

        # Get opponent's best defensive moves
        defense_moves = self.threat_detector.must_defend(position, opponent)
        if not defense_moves:
            # No immediate threats to defend, opponent can play freely
            return False

        # Check if any defense works
        for move in defense_moves:
            row, col = move
            new_position = position.make_move(row, col)

            # After defense, can we still force a win?
            if not self._search_forced_win(new_position, depth - 1, -opponent):
                return False  # Defense works

        return True  # No defense works

    def _get_prioritized_moves(
        self, position: Position, player: int
    ) -> List[Tuple[int, int]]:
        """Get moves prioritized by threat value."""
        threat_moves = list(self.threat_detector.get_threat_moves(position, player))

        # Sort by threat priority
        def threat_priority(move):
            row, col = move
            threats = self.threat_detector._analyze_cell_threats(
                position, row, col, player
            )
            priority = 0
            for threat in threats:
                if threat == ThreatType.OPEN_FOUR:
                    priority += 1000
                elif threat == ThreatType.DOUBLE_FOUR:
                    priority += 900
                elif threat == ThreatType.DOUBLE_THREE:
                    priority += 800
                elif threat == ThreatType.BROKEN_FOUR:
                    priority += 700
                elif threat == ThreatType.OPEN_THREE:
                    priority += 600
            return priority

        threat_moves.sort(key=threat_priority, reverse=True)

        # Limit to top moves to control branching
        return threat_moves[:10]

    def _time_exceeded(self) -> bool:
        """Check if time limit exceeded."""
        current_time = time.time() * 1000
        return (current_time - self.start_time) > self.time_cap_ms


def tss_search(position: Position, depth: int, time_cap_ms: int) -> TSSResult:
    """
    Perform Threat-Space Search on the given position.

    Args:
        position (Position): Current board state and player to move.
        depth (int): Maximum search depth in plies.
        time_cap_ms (int): Time cap in milliseconds.

    Returns:
        TSSResult: Result object containing:
            - forced_move (Optional[Tuple[int, int]]): Move to play if forced win/defense found.
            - is_forced_win (bool): True if forced win line detected.
            - is_forced_defense (bool): True if forced defense detected.
            - search_stats (dict): Nodes visited, time used, etc.
    """
    searcher = TSSSearcher(position.board_size)
    return searcher.search(position, depth, time_cap_ms)
