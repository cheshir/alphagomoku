"""Adaptive simulation scheduling for MCTS"""

from typing import Tuple

import numpy as np


class AdaptiveSimulator:
    """Manages adaptive simulation count based on game phase and confidence"""

    def __init__(
        self,
        early_sims: Tuple[int, int] = (50, 150),
        mid_sims: Tuple[int, int] = (200, 400),
        late_sims: Tuple[int, int] = (50, 100),
        early_moves: int = 10,
        late_moves: int = 180,
    ):
        self.early_sims = early_sims
        self.mid_sims = mid_sims
        self.late_sims = late_sims
        self.early_moves = early_moves
        self.late_moves = late_moves

    def get_simulations(
        self, move_count: int, board_state: np.ndarray, confidence: float = 0.0
    ) -> int:
        """Get adaptive simulation count based on game phase and confidence"""

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

        # Adjust based on confidence (high confidence = fewer sims needed)
        if confidence > 0.8:
            base_sims = int(base_sims * 0.5)
        elif confidence > 0.6:
            base_sims = int(base_sims * 0.7)

        return max(base_sims, 25)  # Minimum 25 simulations

    def get_confidence(self, policy: np.ndarray) -> float:
        """Estimate confidence from policy distribution"""
        # Higher entropy = lower confidence
        entropy = -np.sum(policy * np.log(policy + 1e-8))
        max_entropy = np.log(len(policy))
        return 1.0 - (entropy / max_entropy)
