"""Adaptive simulation scheduling for MCTS"""

from typing import Tuple

import numpy as np


class AdaptiveSimulator:
    """Manages adaptive simulation count based on game phase and confidence"""

    def __init__(
        self,
        early_sims: Tuple[int, int] = (48, 130),
        mid_sims: Tuple[int, int] = (160, 280),
        late_sims: Tuple[int, int] = (40, 90),
        early_moves: int = 10,
        late_moves: int = 180,
    ):
        self.early_sims = early_sims
        self.mid_sims = mid_sims
        self.late_sims = late_sims
        self.early_moves = early_moves
        self.late_moves = late_moves

    def get_simulations(
        self,
        move_count: int,
        board_state: np.ndarray,
        confidence: float | None = None,
    ) -> int:
        """Get adaptive simulation count based on game phase and confidence"""

        confidence = 0.0 if confidence is None else float(confidence)

        # Game phase detection
        if move_count < self.early_moves:
            # Early game: fewer simulations
            base_sims = np.random.randint(*self.early_sims)
        elif move_count > self.late_moves:
            # Late game: fewer simulations
            base_sims = np.random.randint(*self.late_sims)
        else:
            # Mid game: more simulations
            base_sims = np.random.randint(*self.mid_sims)

        # Reduce search effort when the policy distribution is confident
        if confidence >= 0.9:
            base_sims = max(int(base_sims * 0.45), self.early_sims[0])
        elif confidence >= 0.75:
            base_sims = int(base_sims * 0.65)
        elif confidence >= 0.6:
            base_sims = int(base_sims * 0.8)

        # Encourage quicker resolution when only a handful of moves remain
        empty_cells = int(np.count_nonzero(board_state == 0))
        if empty_cells < 20:
            base_sims = max(int(base_sims * 0.8), self.late_sims[0])

        return max(base_sims, 32)

    def get_confidence(self, policy: np.ndarray) -> float:
        """Estimate confidence from policy distribution"""
        # Higher entropy = lower confidence
        entropy = -np.sum(policy * np.log(policy + 1e-8))
        max_entropy = np.log(len(policy))
        return 1.0 - (entropy / max_entropy)
