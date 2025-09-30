"""TSS configuration for progressive learning."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TSSConfig:
    """Configuration for TSS threat detection and defense.

    This allows progressive disabling of hard-coded tactical rules
    as the neural network learns to handle these patterns naturally.
    """

    # Always enabled - these are game rules, not learned tactics
    defend_immediate_five: bool = True  # Always block immediate 5-in-a-row

    # Can be disabled after ~50-100 epochs when model learns tactics
    defend_open_four: bool = True      # Block .XXXX. (true open four)
    defend_broken_four: bool = True    # Block XXXX. or X.XXX patterns
    defend_open_three: bool = False    # Currently disabled - MCTS learns this

    # Win search settings
    search_forced_wins: bool = True    # Search for multi-move forced wins
    max_search_depth: int = 6          # Max depth for forced win search

    @classmethod
    def for_training_epoch(cls, epoch: int) -> "TSSConfig":
        """Get TSS config appropriate for training epoch.

        Progressive disabling schedule:
        - Epoch 0-50: Full TSS assistance (all defenses on)
        - Epoch 50-100: Disable broken-four defense (model learns semi-open patterns)
        - Epoch 100+: Disable open-four defense (model learns all tactical patterns)
        - Always keep immediate-five defense (game rules)

        Args:
            epoch: Current training epoch

        Returns:
            TSSConfig with appropriate settings
        """
        if epoch < 50:
            # Early training: Full TSS assistance
            return cls(
                defend_immediate_five=True,
                defend_open_four=True,
                defend_broken_four=True,
                defend_open_three=False,  # Let model learn this from start
            )
        elif epoch < 100:
            # Mid training: Disable broken-four defense
            return cls(
                defend_immediate_five=True,
                defend_open_four=True,
                defend_broken_four=False,  # Model should learn this now
                defend_open_three=False,
            )
        else:
            # Late training: Only keep game-rule defenses
            return cls(
                defend_immediate_five=True,
                defend_open_four=False,    # Model should learn this now
                defend_broken_four=False,
                defend_open_three=False,
            )

    @classmethod
    def for_inference(cls, difficulty: str) -> "TSSConfig":
        """Get TSS config for inference at given difficulty.

        For inference, we can use full TSS to maximize strength,
        or disable it to see pure learned behavior.

        Args:
            difficulty: "easy", "medium", or "hard"

        Returns:
            TSSConfig with appropriate settings
        """
        if difficulty == "easy":
            # Easy: Minimal TSS, more natural play
            return cls(
                defend_immediate_five=True,
                defend_open_four=False,
                defend_broken_four=False,
                defend_open_three=False,
            )
        elif difficulty == "medium":
            # Medium: Some TSS assistance
            return cls(
                defend_immediate_five=True,
                defend_open_four=True,
                defend_broken_four=False,
                defend_open_three=False,
            )
        else:  # hard
            # Hard: Full TSS for maximum strength
            return cls(
                defend_immediate_five=True,
                defend_open_four=True,
                defend_broken_four=True,
                defend_open_three=False,
            )


# Global default config (can be overridden)
DEFAULT_CONFIG = TSSConfig()


def set_default_config(config: TSSConfig):
    """Set the global default TSS config."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


def get_default_config() -> TSSConfig:
    """Get the global default TSS config."""
    return DEFAULT_CONFIG
