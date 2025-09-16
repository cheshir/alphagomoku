from dataclasses import dataclass
from typing import Optional


@dataclass
class MCTSConfig:
    """Configuration for MCTS parameters."""

    # Core MCTS parameters
    cpuct: float = 1.8
    num_simulations: int = 800
    batch_size: int = 32

    # Neural network evaluation
    policy_epsilon: float = 1e-8  # Small value to prevent division by zero

    # UCT scoring
    virtual_loss_penalty: float = -1e9  # Penalty for in-flight nodes

    # Value normalization
    initial_value: float = 0.0
    win_value: float = 1.0
    loss_value: float = -1.0
    draw_value: float = 0.0

    # Temperature settings
    default_temperature: float = 1.0
    zero_temperature: float = 0.0
    deterministic_threshold: float = 0.0  # Temperature below which selection is deterministic

    # Tree reuse
    enable_tree_reuse: bool = True

    # Terminal detection
    winning_sequence_length: int = 5

    # Virtual loss for batching
    enable_virtual_loss: bool = True

    @classmethod
    def create_easy(cls) -> 'MCTSConfig':
        """Create configuration for easy difficulty."""
        return cls(
            num_simulations=48,
            cpuct=1.0,
            default_temperature=1.2
        )

    @classmethod
    def create_medium(cls) -> 'MCTSConfig':
        """Create configuration for medium difficulty."""
        return cls(
            num_simulations=384,
            cpuct=1.4,
            default_temperature=0.8
        )

    @classmethod
    def create_strong(cls) -> 'MCTSConfig':
        """Create configuration for strong difficulty."""
        return cls(
            num_simulations=1600,
            cpuct=1.8,
            default_temperature=0.5
        )