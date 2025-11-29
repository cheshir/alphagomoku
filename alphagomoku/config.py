"""Centralized configuration management for AlphaGomoku.

This module provides a single source of truth for all hyperparameters,
model architectures, and training configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal


# ============================================================================
# Model Architecture Presets
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    num_blocks: int
    channels: int
    board_size: int = 15
    use_checkpoint: bool = False
    description: str = ""

    @property
    def estimated_params(self) -> float:
        """Estimate number of parameters in millions"""
        # Rough estimation based on DW-ResNet-SE architecture
        # Input conv: 5 * channels * 9 + channels
        # Each block: ~2 * channels^2 + channels
        # Policy head: channels * 32 * 9 + 32 * 225 * 225
        # Value head: channels * 16 * 9 + 16 * 225 * 256 + 256
        input_params = 5 * self.channels * 9 + self.channels
        block_params = self.num_blocks * (2 * self.channels**2 + self.channels)
        policy_params = self.channels * 32 * 9 + 32 * self.board_size**2
        value_params = (
            self.channels * 16 * 9
            + 16 * self.board_size**2 * 256
            + 256
        )
        total = input_params + block_params + policy_params + value_params
        return total / 1_000_000

    def get_description(self) -> str:
        """Get human-readable description"""
        if self.description:
            return self.description
        return f"{self.estimated_params:.1f}M params ({self.num_blocks} blocks, {self.channels} ch)"


# Predefined model presets
MODEL_PRESETS: Dict[str, ModelConfig] = {
    "small": ModelConfig(
        num_blocks=10,
        channels=96,
        use_checkpoint=False,
        description="Fast iteration (1.2M params): 80% strength, 3-5x faster training"
    ),
    "medium": ModelConfig(
        num_blocks=14,
        channels=128,
        use_checkpoint=False,
        description="Balanced (2.5M params): 90% strength, 2x faster training"
    ),
    "large": ModelConfig(
        num_blocks=18,
        channels=192,
        use_checkpoint=False,
        description="Maximum strength (5M params): 100% strength, slowest training"
    ),
}


# ============================================================================
# Training Configuration Presets
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    # Model
    model_preset: str = "small"

    # Self-play
    selfplay_games: int = 100
    mcts_simulations: int = 400
    adaptive_sims: bool = True
    parallel_workers: int = 4
    batch_size_mcts: int = 64
    difficulty: str = "medium"  # For TSS/endgame during training

    # Training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 0.001
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"

    # Data
    buffer_max_size: int = 5_000_000
    map_size_gb: int = 32

    # Evaluation
    eval_frequency: int = 5
    eval_games: int = 50
    eval_baseline: str = "mcts_400"

    # System
    device: str = "auto"
    debug_memory: bool = False

    @property
    def description(self) -> str:
        """Human-readable description"""
        model_desc = MODEL_PRESETS[self.model_preset].description
        return f"{self.model_preset} model, {self.epochs} epochs, {self.selfplay_games} games/epoch"


# Predefined training presets
TRAINING_PRESETS: Dict[str, TrainingConfig] = {
    "fast_iteration": TrainingConfig(
        model_preset="small",
        selfplay_games=50,
        mcts_simulations=200,
        epochs=50,
        batch_size=256,
        parallel_workers=4,
        batch_size_mcts=32,
        eval_frequency=5,
        buffer_max_size=2_000_000,
        map_size_gb=16,
    ),
    "balanced": TrainingConfig(
        model_preset="small",
        selfplay_games=100,
        mcts_simulations=400,
        epochs=200,
        batch_size=256,
        parallel_workers=4,
        batch_size_mcts=64,
        eval_frequency=5,
        buffer_max_size=5_000_000,
        map_size_gb=32,
    ),
    "production": TrainingConfig(
        model_preset="medium",
        selfplay_games=200,
        mcts_simulations=600,
        epochs=200,
        batch_size=512,
        parallel_workers=1,
        batch_size_mcts=96,
        eval_frequency=10,
        buffer_max_size=5_000_000,
        map_size_gb=32,
    ),
}


# ============================================================================
# Inference Configuration
# ============================================================================

@dataclass
class InferenceConfig:
    """Inference settings by difficulty level"""
    mcts_simulations: int
    cpuct: float = 1.8
    temperature: float = 0.0
    batch_size: int = 64
    use_tss: bool = True
    use_endgame: bool = True
    tss_depth: int = 4
    tss_time_cap_ms: int = 100
    endgame_threshold: int = 20  # Empty cells


# Inference presets by difficulty
INFERENCE_PRESETS: Dict[str, InferenceConfig] = {
    "easy": InferenceConfig(
        mcts_simulations=64,
        use_tss=False,
        use_endgame=False,
        temperature=1.2,
    ),
    "medium": InferenceConfig(
        mcts_simulations=128,
        use_tss=True,
        use_endgame=True,
        tss_depth=4,
        tss_time_cap_ms=100,
        endgame_threshold=14,
    ),
    "hard": InferenceConfig(
        mcts_simulations=256,
        use_tss=True,
        use_endgame=True,
        tss_depth=6,
        tss_time_cap_ms=300,
        endgame_threshold=20,
    ),
    "strong": InferenceConfig(
        mcts_simulations=800,
        use_tss=True,
        use_endgame=True,
        tss_depth=7,
        tss_time_cap_ms=500,
        endgame_threshold=20,
    ),
}


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    # Baseline opponents
    baseline_mcts_sims: int = 400
    evaluation_games: int = 50

    # Tactical test suite
    tactical_test_timeout: float = 10.0  # seconds per position

    # Elo tracking
    initial_elo: int = 1500
    k_factor: int = 32


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_config(preset: str = "small") -> ModelConfig:
    """Get model configuration by preset name"""
    if preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset: {preset}. "
            f"Available: {list(MODEL_PRESETS.keys())}"
        )
    return MODEL_PRESETS[preset]


def get_training_config(preset: str = "balanced") -> TrainingConfig:
    """Get training configuration by preset name"""
    if preset not in TRAINING_PRESETS:
        raise ValueError(
            f"Unknown training preset: {preset}. "
            f"Available: {list(TRAINING_PRESETS.keys())}"
        )
    return TRAINING_PRESETS[preset]


def get_inference_config(difficulty: str = "medium") -> InferenceConfig:
    """Get inference configuration by difficulty"""
    if difficulty not in INFERENCE_PRESETS:
        raise ValueError(
            f"Unknown difficulty: {difficulty}. "
            f"Available: {list(INFERENCE_PRESETS.keys())}"
        )
    return INFERENCE_PRESETS[difficulty]


def print_config_summary(config: TrainingConfig):
    """Print training configuration summary"""
    model_cfg = MODEL_PRESETS[config.model_preset]

    print("=" * 60)
    print("AlphaGomoku Training Configuration")
    print("=" * 60)
    print(f"\n📐 Model: {config.model_preset}")
    print(f"   {model_cfg.get_description()}")
    print(f"   Blocks: {model_cfg.num_blocks}, Channels: {model_cfg.channels}")
    print(f"   Gradient checkpointing: {model_cfg.use_checkpoint}")

    print(f"\n🎮 Self-Play:")
    print(f"   Games per epoch: {config.selfplay_games}")
    print(f"   MCTS simulations: {config.mcts_simulations}")
    print(f"   Adaptive sims: {config.adaptive_sims}")
    print(f"   Parallel workers: {config.parallel_workers}")
    print(f"   MCTS batch size: {config.batch_size_mcts}")

    print(f"\n🏋️  Training:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.lr} → {config.min_lr}")
    print(f"   LR schedule: {config.lr_schedule}")
    print(f"   Warmup epochs: {config.warmup_epochs}")
    print(f"   Weight decay: {config.weight_decay}")

    print(f"\n💾 Data:")
    print(f"   Buffer max size: {config.buffer_max_size:,}")
    print(f"   LMDB map size: {config.map_size_gb} GB")

    print(f"\n📊 Evaluation:")
    print(f"   Frequency: Every {config.eval_frequency} epochs")
    print(f"   Eval games: {config.eval_games}")
    print(f"   Baseline: {config.eval_baseline}")

    print(f"\n⚙️  System:")
    print(f"   Device: {config.device}")
    print(f"   Debug memory: {config.debug_memory}")
    print("=" * 60)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "MODEL_PRESETS",
    "TRAINING_PRESETS",
    "INFERENCE_PRESETS",
    "get_model_config",
    "get_training_config",
    "get_inference_config",
    "print_config_summary",
]
