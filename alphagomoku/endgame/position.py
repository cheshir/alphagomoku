"""Position representation for endgame solver."""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class EndgamePosition:
    """Position class optimized for endgame analysis."""

    board: np.ndarray  # 15x15 board state (-1, 0, 1)
    current_player: int  # 1 or -1
    last_move: Optional[Tuple[int, int]] = None
    board_size: int = 15

    def __post_init__(self):
        if self.board.shape != (self.board_size, self.board_size):
            raise ValueError(f"Board must be {self.board_size}x{self.board_size}")
        if self.current_player not in [-1, 1]:
            raise ValueError("Current player must be 1 or -1")

    def copy(self) -> 'EndgamePosition':
        """Create a copy of this position."""
        return EndgamePosition(
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            board_size=self.board_size
        )

    def make_move(self, row: int, col: int) -> 'EndgamePosition':
        """Create new position after making a move."""
        new_pos = self.copy()
        new_pos.board[row, col] = self.current_player
        new_pos.current_player = -self.current_player
        new_pos.last_move = (row, col)
        return new_pos

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get list of legal moves as (row, col) tuples."""
        return [(r, c) for r in range(self.board_size)
                for c in range(self.board_size) if self.board[r, c] == 0]

    def get_empty_count(self) -> int:
        """Get number of empty cells."""
        return int(np.sum(self.board == 0))

    def is_terminal(self) -> Tuple[bool, int]:
        """Check if position is terminal. Returns (is_terminal, winner)."""
        if self.last_move is None:
            return False, 0

        r, c = self.last_move
        player = self.board[r, c]

        # Check for 5 in a row
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Check positive direction
            rr, cc = r + dr, c + dc
            while (0 <= rr < self.board_size and 0 <= cc < self.board_size
                   and self.board[rr, cc] == player):
                count += 1
                rr += dr
                cc += dc
            # Check negative direction
            rr, cc = r - dr, c - dc
            while (0 <= rr < self.board_size and 0 <= cc < self.board_size
                   and self.board[rr, cc] == player):
                count += 1
                rr -= dr
                cc -= dc
            if count >= 5:
                return True, player

        # Check for draw
        if np.sum(self.board == 0) == 0:
            return True, 0

        return False, 0

    def get_critical_moves(self) -> List[Tuple[int, int]]:
        """Get moves near existing stones for more efficient search."""
        if self.get_empty_count() == self.board_size * self.board_size:
            # Empty board - return center
            center = self.board_size // 2
            return [(center, center)]

        critical_moves = set()

        # Find all stones on the board
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != 0:
                    # Add adjacent empty cells
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.board_size and
                                0 <= nc < self.board_size and
                                self.board[nr, nc] == 0):
                                critical_moves.add((nr, nc))

        return list(critical_moves)