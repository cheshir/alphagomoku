"""Threat pattern detection for Gomoku TSS."""

from enum import Enum
from typing import List, Optional, Set, Tuple

import numpy as np

from .position import Position
from .tss_config import TSSConfig, get_default_config


class ThreatType(Enum):
    """Types of threats in Gomoku."""

    OPEN_THREE = "open_three"
    OPEN_FOUR = "open_four"
    BROKEN_FOUR = "broken_four"
    DOUBLE_THREE = "double_three"
    DOUBLE_FOUR = "double_four"


class ThreatDetector:
    """Detects tactical threats in Gomoku positions."""

    def __init__(self, board_size: int = 15, config: Optional[TSSConfig] = None):
        self.board_size = board_size
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.config = config if config is not None else get_default_config()

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
        """Classify a pattern string into threat type.

        Terminology:
        - OPEN FOUR (.XXXX.): Both ends open, unstoppable, guaranteed win
        - SEMI-OPEN FOUR (XXXX. or .XXXX): One end open, can be blocked
        - BROKEN FOUR (X.XXX, etc): Four stones with gap
        """
        # Count consecutive X's around the center
        x_count = pattern.count("X")

        # Open four: .XXXX. - BOTH ends must be open
        # This is the ONLY truly unstoppable four
        if ".XXXX." in pattern:
            return ThreatType.OPEN_FOUR

        # Broken four: X.XXX, XXX.X, XX.XX with open space
        # These are semi-open fours (4 stones with gap, need one more move)
        broken_four_patterns = ["X.XXX", "XXX.X", "XX.XX"]
        for bp in broken_four_patterns:
            if bp in pattern:
                # Check if there's space to complete the five
                if "." in pattern and (
                    pattern.startswith(".") or pattern.endswith(".")
                ):
                    return ThreatType.BROKEN_FOUR

        # Semi-open four: XXXX. or .XXXX (one end open)
        # Treated as broken four since it can be blocked
        if x_count == 4:
            if "XXXX." in pattern or ".XXXX" in pattern:
                # Only classify if it's not already classified as open four
                if ".XXXX." not in pattern:
                    return ThreatType.BROKEN_FOUR

        # Open three: .XXX. with space on both sides to extend to open four
        # This means the pattern must have potential to create .XXXX. on next move
        if ".XXX." in pattern:
            # Additional check: ensure there's room to extend to open four
            # The pattern should have at least one more empty cell on each side
            # Look for patterns like ..XXX..
            if pattern.count(".") >= 2:
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
        """Get moves that must be played to defend against immediate threats.

        Defense priority is configurable via TSSConfig to allow progressive
        learning - initially TSS helps with all tactics, but as the model
        learns, we disable hard-coded defenses.

        Configurable defenses:
        - defend_immediate_five: Always on (game rules)
        - defend_open_four: Can disable after ~100 epochs
        - defend_broken_four: Can disable after ~50 epochs
        - defend_open_three: Already disabled (MCTS learns this)
        """
        opponent = -player
        defense_moves = []

        # Priority 1: Immediate 5-in-a-row (always enabled - game rules)
        if self.config.defend_immediate_five:
            immediate_wins = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if position.board[r, c] == 0:
                        test_board = position.board.copy()
                        test_board[r, c] = opponent
                        if self._creates_five_in_row(test_board, r, c, opponent):
                            immediate_wins.append((r, c))

            if immediate_wins:
                return immediate_wins

        # Priority 2: Open-four threats (configurable)
        if self.config.defend_open_four:
            open_four_defenses = []
            opponent_threats = self.detect_threats(position, opponent)
            for r, c, threat_type in opponent_threats:
                if threat_type == ThreatType.OPEN_FOUR:
                    open_four_defenses.append((r, c))

            if open_four_defenses:
                return open_four_defenses

        # Priority 3: Broken-four threats (configurable)
        if self.config.defend_broken_four:
            broken_four_defenses = []
            opponent_threats = self.detect_threats(position, opponent)
            for r, c, threat_type in opponent_threats:
                if threat_type == ThreatType.BROKEN_FOUR:
                    broken_four_defenses.append((r, c))

            if broken_four_defenses:
                return broken_four_defenses

        # NOTE: Open-three is controlled by config.defend_open_three
        # Currently disabled - MCTS learns this pattern
        if self.config.defend_open_three:
            open_three_defenses = []
            opponent_threats = self.detect_threats(position, opponent)
            for r, c, threat_type in opponent_threats:
                if threat_type == ThreatType.OPEN_THREE:
                    open_three_defenses.append((r, c))

            if open_three_defenses:
                return open_three_defenses

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
