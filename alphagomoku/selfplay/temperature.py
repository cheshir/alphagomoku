"""Temperature scheduling for self-play move selection.

Phase 3: Temperature Scheduler
Generates cleaner training data by reducing exploration noise over the course of a game.
"""

import numpy as np
from typing import Optional


class TemperatureScheduler:
    """Schedules temperature for move selection during self-play.

    Temperature controls exploration vs exploitation:
    - temp = 1.0: Sample proportionally (high exploration)
    - temp = 0.5: Sharper distribution (moderate exploration)
    - temp = 0.0: Always pick best move (no exploration)

    Strategy:
    - Early game (moves 0-8): Higher temperature for opening diversity
    - Mid game (moves 8-20): Gradual decay
    - Late game (moves 20+): Deterministic play
    - Critical positions: Override to 0.0 for forced moves
    """

    def __init__(
        self,
        early_temp: float = 0.8,
        early_moves: int = 3,
        mid_temp: float = 0.5,
        mid_moves: int = 6,
        late_temp: float = 0.0,
        epoch: int = 0,
    ):
        """Initialize temperature scheduler.

        Args:
            early_temp: Temperature for early moves (default 0.8)
            early_moves: Number of early moves (default 3)
            mid_temp: Temperature for mid moves (default 0.5)
            mid_moves: Number of mid moves (default 6)
            late_temp: Temperature for late moves (default 0.0)
            epoch: Current training epoch (for adaptive scheduling)
        """
        self.early_temp = early_temp
        self.early_moves = early_moves
        self.mid_temp = mid_temp
        self.mid_moves = mid_moves
        self.late_temp = late_temp
        self.epoch = epoch

        # Adaptive scheduling: reduce temperature as training progresses
        if epoch < 30:
            # Early training: More exploration
            self.early_temp = 1.0
            self.mid_temp = 0.8
        elif epoch < 60:
            # Mid training: Moderate exploration
            self.early_temp = 0.8
            self.mid_temp = 0.5
        else:
            # Late training: Minimal exploration
            self.early_temp = 0.5
            self.mid_temp = 0.3

    def get_temperature(
        self,
        move_number: int,
        is_critical: bool = False,
        policy_entropy: Optional[float] = None,
    ) -> float:
        """Get temperature for current move.

        Args:
            move_number: Move number in the game (0-indexed)
            is_critical: Whether this is a critical/forced position
            policy_entropy: Optional policy entropy for adaptive scheduling

        Returns:
            Temperature value [0.0, 1.0]
        """
        # Critical positions: always deterministic
        if is_critical:
            return 0.0

        # Phase-based temperature
        if move_number < self.early_moves:
            temp = self.early_temp
        elif move_number < self.early_moves + self.mid_moves:
            # Linear interpolation from early_temp to mid_temp
            progress = (move_number - self.early_moves) / self.mid_moves
            temp = self.early_temp + progress * (self.mid_temp - self.early_temp)
        else:
            temp = self.late_temp

        # Optional: Adapt based on policy entropy
        # High entropy = model uncertain → increase temperature
        # Low entropy = model confident → decrease temperature
        if policy_entropy is not None:
            # Entropy adjustment factor
            # Entropy typically in [0, 5] for 225 actions
            # High entropy (>3) = uncertain, boost temp by up to 0.2
            # Low entropy (<1) = confident, reduce temp by up to 0.2
            max_entropy = np.log(225)  # ~5.4 for 15x15 board
            normalized_entropy = policy_entropy / max_entropy

            if normalized_entropy > 0.5:
                # Uncertain: slightly increase exploration
                temp = min(1.0, temp + 0.1)
            elif normalized_entropy < 0.2:
                # Confident: slightly decrease exploration
                temp = max(0.0, temp - 0.1)

        return temp

    def should_sample(self, temperature: float) -> bool:
        """Whether to sample from policy (True) or pick argmax (False).

        Args:
            temperature: Temperature value

        Returns:
            True if should sample, False if should pick best move
        """
        return temperature > 1e-3

    def apply_temperature(self, policy: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature to policy distribution.

        Args:
            policy: Policy distribution (action probabilities)
            temperature: Temperature value

        Returns:
            Temperature-adjusted policy
        """
        if temperature < 1e-3:
            # Deterministic: return one-hot on best action
            best_action = np.argmax(policy)
            new_policy = np.zeros_like(policy)
            new_policy[best_action] = 1.0
            return new_policy

        # Apply temperature via softmax
        # policy = softmax(logits), so logits = log(policy)
        log_policy = np.log(policy + 1e-10)  # Add epsilon for stability
        adjusted_logits = log_policy / temperature

        # Re-normalize
        adjusted_logits -= adjusted_logits.max()  # For numerical stability
        exp_logits = np.exp(adjusted_logits)
        new_policy = exp_logits / exp_logits.sum()

        return new_policy

    def compute_entropy(self, policy: np.ndarray) -> float:
        """Compute entropy of policy distribution.

        Args:
            policy: Policy distribution

        Returns:
            Entropy value
        """
        # H(p) = -sum(p * log(p))
        # Higher entropy = more uncertain/uniform
        return -np.sum(policy * np.log(policy + 1e-10))


# Convenience function for backward compatibility
def get_temperature(move_number: int, epoch: int = 0, is_critical: bool = False) -> float:
    """Simple temperature function for backward compatibility.

    Args:
        move_number: Move number in game
        epoch: Training epoch
        is_critical: Whether position is critical

    Returns:
        Temperature value
    """
    scheduler = TemperatureScheduler(epoch=epoch)
    return scheduler.get_temperature(move_number, is_critical=is_critical)
