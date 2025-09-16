"""Threat pattern detection for Gomoku TSS."""

from enum import Enum
from typing import List, Set, Tuple

import numpy as np

from .position import Position


class ThreatType(Enum):
    """Types of threats in Gomoku."""

    OPEN_THREE = "open_three"
    OPEN_FOUR = "open_four"
    BROKEN_FOUR = "broken_four"
    DOUBLE_THREE = "double_three"
    DOUBLE_FOUR = "double_four"


class ThreatDetector:
    """Detects tactical threats in Gomoku positions."""

    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def detect_threats(
        self, position: Position, player: int
    ) -> List[Tuple[int, int, ThreatType]]:
        """Detect all threats for a player. Returns list of (row, col, threat_type)."""
        threats = []

        for r in range(self.board_size):
            for c in range(self.board_size):
                if position.board[r, c] == 0:  # Empty cell
                    threat_types = self._analyze_cell_threats(position, r, c, player)
                    for threat_type in threat_types:
                        threats.append((r, c, threat_type))

        return threats

    def _analyze_cell_threats(
        self, position: Position, row: int, col: int, player: int
    ) -> List[ThreatType]:
        """Analyze what threats would be created by placing a stone at (row, col)."""
        threats = []

        # Temporarily place stone
        test_board = position.board.copy()
        test_board[row, col] = player

        # Check each direction for patterns
        for dr, dc in self.directions:
            pattern = self._get_line_pattern(test_board, row, col, dr, dc, player)
            threat_type = self._classify_pattern(pattern)
            if threat_type:
                threats.append(threat_type)

        # Check for double threats
        if len(threats) >= 2:
            if ThreatType.OPEN_THREE in threats:
                threats.append(ThreatType.DOUBLE_THREE)
            if ThreatType.OPEN_FOUR in threats or ThreatType.BROKEN_FOUR in threats:
                threats.append(ThreatType.DOUBLE_FOUR)

        return list(set(threats))  # Remove duplicates

    def _get_line_pattern(
        self, board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int
    ) -> str:
        """Extract pattern string along a direction."""
        pattern = ""

        # Go backwards first (up to 5 positions)
        r, c = row - dr, col - dc
        back_stones = []
        empty_count = 0
        while (
            0 <= r < self.board_size
            and 0 <= c < self.board_size
            and len(back_stones) < 5
        ):
            if board[r, c] == player:
                back_stones.append("X")
            elif board[r, c] == -player:
                break
            else:
                back_stones.append(".")
                empty_count += 1
                if empty_count > 1:  # Stop after 2 empty cells
                    break
            r -= dr
            c -= dc

        # Reverse and add to pattern
        pattern = "".join(reversed(back_stones))

        # Add center stone
        pattern += "X"

        # Go forwards (up to 5 positions)
        r, c = row + dr, col + dc
        empty_count = 0
        forward_count = 0
        while (
            0 <= r < self.board_size and 0 <= c < self.board_size and forward_count < 5
        ):
            if board[r, c] == player:
                pattern += "X"
            elif board[r, c] == -player:
                break
            else:
                pattern += "."
                empty_count += 1
                if empty_count > 1:  # Stop after 2 empty cells
                    break
            r += dr
            c += dc
            forward_count += 1

        return pattern

    def _classify_pattern(self, pattern: str) -> ThreatType:
        """Classify a pattern string into threat type."""
        # Count consecutive X's around the center
        x_count = pattern.count("X")

        # Open four: .XXXX. or XXXX with open end
        if ".XXXX." in pattern:
            return ThreatType.OPEN_FOUR
        elif x_count == 4 and ("XXXX." in pattern or ".XXXX" in pattern):
            return ThreatType.OPEN_FOUR

        # Broken four: X.XXX, XXX.X, XX.XX with open space
        broken_four_patterns = ["X.XXX", "XXX.X", "XX.XX"]
        for bp in broken_four_patterns:
            if bp in pattern:
                # Check if there's space to complete the five
                if "." in pattern and (
                    pattern.startswith(".") or pattern.endswith(".")
                ):
                    return ThreatType.BROKEN_FOUR

        # Open three: .XXX. or XXX with open ends
        if ".XXX." in pattern:
            return ThreatType.OPEN_THREE
        elif x_count == 3 and pattern.count(".") >= 2:
            # Check for open three patterns
            if "XXX." in pattern or ".XXX" in pattern:
                return ThreatType.OPEN_THREE

        return None

    def get_threat_moves(self, position: Position, player: int) -> Set[Tuple[int, int]]:
        """Get all moves that create or block threats."""
        threat_moves = set()

        # Moves that create threats for player
        for r, c, _ in self.detect_threats(position, player):
            threat_moves.add((r, c))

        # Moves that block opponent threats
        for r, c, _ in self.detect_threats(position, -player):
            threat_moves.add((r, c))

        return threat_moves

    def is_winning_threat(
        self, position: Position, row: int, col: int, player: int
    ) -> bool:
        """Check if move creates an immediate winning threat."""
        threats = self._analyze_cell_threats(position, row, col, player)
        return (
            ThreatType.OPEN_FOUR in threats
            or ThreatType.DOUBLE_FOUR in threats
            or ThreatType.DOUBLE_THREE in threats
        )

    def must_defend(self, position: Position, player: int) -> List[Tuple[int, int]]:
        """Get moves that must be played to defend against immediate threats."""
        opponent = -player
        defense_moves = []

        # Check for opponent's immediate winning threats
        # Look for patterns where opponent has 4 in a row with open ends
        for r in range(self.board_size):
            for c in range(self.board_size):
                if position.board[r, c] == 0:  # Empty cell
                    # Check if placing opponent stone here creates 5 in a row
                    test_board = position.board.copy()
                    test_board[r, c] = opponent

                    # Check if this completes 5 in a row
                    if self._creates_five_in_row(test_board, r, c, opponent):
                        defense_moves.append((r, c))

        return defense_moves

    def _creates_five_in_row(
        self, board: np.ndarray, row: int, col: int, player: int
    ) -> bool:
        """Check if placing a stone creates 5 in a row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Count the placed stone

            # Check positive direction
            r, c = row + dr, col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == player
            ):
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r, c] == player
            ):
                count += 1
                r, c = r - dr, c - dc

            if count >= 5:
                return True

        return False
