"""Dynamic hardware configuration for optimal training settings.

This module automatically detects available hardware (CPU, RAM, GPU) and
recommends optimal training configurations.
"""

import platform
import psutil
import torch
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HardwareInfo:
    """Hardware information"""
    device_type: str  # "mps", "cuda", "cpu"
    device_name: str
    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    gpu_memory_gb: Optional[float] = None


@dataclass
class RecommendedConfig:
    """Recommended training configuration for detected hardware"""
    model_preset: str
    parallel_workers: int
    batch_size: int
    batch_size_mcts: int
    selfplay_games: int
    expected_memory_gb: float
    notes: str
    # Training hyperparameters (with defaults)
    difficulty: str = "easy"  # Always easy for training (AlphaZero style)
    mcts_simulations: int = 400  # Default MCTS simulations
    map_size_gb: int = 32  # LMDB map size
    buffer_max_size: int = 5_000_000  # Replay buffer size


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and return info.

    Returns:
        HardwareInfo object with detected hardware
    """
    # RAM
    vm = psutil.virtual_memory()
    total_ram = vm.total / (1024**3)
    available_ram = vm.available / (1024**3)

    # CPU
    cpu_count = psutil.cpu_count(logical=False) or 4

    # Device detection
    device_type = "cpu"
    device_name = platform.processor() or "Unknown CPU"
    gpu_memory = None

    if torch.backends.mps.is_available():
        device_type = "mps"
        device_name = "Apple Silicon (MPS)"
        # MPS shares system RAM
        gpu_memory = total_ram  # Shared with system
    elif torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return HardwareInfo(
        device_type=device_type,
        device_name=device_name,
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        cpu_count=cpu_count,
        gpu_memory_gb=gpu_memory,
    )


def get_recommended_config(
    hardware: Optional[HardwareInfo] = None,
    prefer_speed: bool = False,
    prefer_strength: bool = False,
) -> RecommendedConfig:
    """Get recommended training configuration for hardware.

    Args:
        hardware: HardwareInfo object (auto-detected if None)
        prefer_speed: Optimize for training speed
        prefer_strength: Optimize for model strength

    Returns:
        RecommendedConfig with optimal settings
    """
    if hardware is None:
        hardware = detect_hardware()

    # Determine model preset based on RAM
    if hardware.total_ram_gb >= 32:
        # Plenty of RAM - can use medium or large
        if prefer_strength:
            model_preset = "medium"
        else:
            model_preset = "small"  # Still fast even with plenty of RAM
    elif hardware.total_ram_gb >= 16:
        # 16-24 GB - medium is fine, small is faster
        if prefer_strength:
            model_preset = "medium"
        else:
            model_preset = "small"
    else:
        # <16 GB - stick to small
        model_preset = "small"

    # Configure based on device and model
    if hardware.device_type == "cuda":
        # NVIDIA GPU
        gpu_mem = hardware.gpu_memory_gb or 8
        if gpu_mem >= 24:
            # High-end GPU (A100, 4090, etc.)
            return RecommendedConfig(
                model_preset="medium" if prefer_strength else "small",
                parallel_workers=1,
                batch_size=1024,
                batch_size_mcts=128,
                selfplay_games=200 if prefer_strength else 100,
                expected_memory_gb=8.0,
                notes="High-end CUDA GPU - optimized for throughput",
            )
        elif gpu_mem >= 16:
            # Mid-high GPU (3090, A6000, etc.)
            return RecommendedConfig(
                model_preset=model_preset,
                parallel_workers=1,
                batch_size=512,
                batch_size_mcts=96,
                selfplay_games=150 if prefer_strength else 100,
                expected_memory_gb=6.0,
                notes="Mid-high CUDA GPU - good balance",
            )
        else:
            # Lower-end GPU (3060, T4, etc.)
            return RecommendedConfig(
                model_preset="small",
                parallel_workers=1,
                batch_size=256,
                batch_size_mcts=64,
                selfplay_games=100,
                expected_memory_gb=4.0,
                notes="Entry-level CUDA GPU - small model recommended",
            )

    elif hardware.device_type == "mps":
        # Apple Silicon
        if model_preset == "medium":
            # Medium model on MPS
            workers = 1 if hardware.cpu_count < 6 else 2
            return RecommendedConfig(
                model_preset="medium",
                parallel_workers=workers,
                batch_size=512,
                batch_size_mcts=96,
                selfplay_games=150 if prefer_strength else 100,
                expected_memory_gb=8.0,
                notes="Medium model on Apple Silicon - 1-2 workers optimal",
            )
        else:
            # Small model on MPS - can use more parallelism
            workers = min(hardware.cpu_count - 2, 4)
            workers = max(workers, 2)
            return RecommendedConfig(
                model_preset="small",
                parallel_workers=workers,
                batch_size=256,
                batch_size_mcts=64,
                selfplay_games=100,
                expected_memory_gb=6.0,
                notes=f"Small model on Apple Silicon - {workers} workers for speed",
            )

    else:
        # CPU only
        workers = min(hardware.cpu_count, 8)
        return RecommendedConfig(
            model_preset="small",
            parallel_workers=workers,
            batch_size=128,
            batch_size_mcts=32,
            selfplay_games=50,
            expected_memory_gb=4.0,
            notes=f"CPU training - using {workers} workers (will be slow)",
        )


def print_hardware_info(hardware: Optional[HardwareInfo] = None):
    """Print detected hardware information.

    Args:
        hardware: HardwareInfo object (auto-detected if None)
    """
    if hardware is None:
        hardware = detect_hardware()

    print("=" * 70)
    print("DETECTED HARDWARE")
    print("=" * 70)
    print(f"Device: {hardware.device_name}")
    print(f"Type: {hardware.device_type.upper()}")
    print(f"CPU Cores: {hardware.cpu_count}")
    print(f"Total RAM: {hardware.total_ram_gb:.1f} GB")
    print(f"Available RAM: {hardware.available_ram_gb:.1f} GB")
    if hardware.gpu_memory_gb:
        if hardware.device_type == "mps":
            print(f"GPU Memory: Shared with system RAM")
        else:
            print(f"GPU Memory: {hardware.gpu_memory_gb:.1f} GB")
    print("=" * 70)


def print_recommended_config(
    config: RecommendedConfig,
    show_command: bool = True
):
    """Print recommended configuration.

    Args:
        config: RecommendedConfig object
        show_command: Whether to show the make/python command
    """
    print("\n" + "=" * 70)
    print("RECOMMENDED TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model Preset: {config.model_preset}")
    print(f"Parallel Workers: {config.parallel_workers}")
    print(f"Training Batch Size: {config.batch_size}")
    print(f"MCTS Batch Size: {config.batch_size_mcts}")
    print(f"Self-Play Games/Epoch: {config.selfplay_games}")
    print(f"MCTS Simulations: {config.mcts_simulations}")
    print(f"Difficulty: {config.difficulty} (pure MCTS, no TSS - AlphaZero style)")
    print(f"Map Size: {config.map_size_gb} GB")
    print(f"Buffer Max Size: {config.buffer_max_size:,}")
    print(f"Expected Memory Usage: ~{config.expected_memory_gb:.1f} GB")
    print(f"\nNotes: {config.notes}")
    print(f"\nℹ️  Difficulty 'easy' = Pure MCTS (fast, GPU-accelerated)")
    print(f"   TSS is used during inference/evaluation, not training")
    print(f"   See docs/TRAINING_PHILOSOPHY.md for details")

    if show_command:
        print("\n" + "-" * 70)
        print("COMMAND TO RUN:")
        print("-" * 70)
        print(f"python scripts/train.py \\")
        print(f"    --model-preset {config.model_preset} \\")
        print(f"    --parallel-workers {config.parallel_workers} \\")
        print(f"    --batch-size {config.batch_size} \\")
        print(f"    --batch-size-mcts {config.batch_size_mcts} \\")
        print(f"    --selfplay-games {config.selfplay_games} \\")
        print(f"    --mcts-simulations {config.mcts_simulations} \\")
        print(f"    --difficulty {config.difficulty} \\")
        print(f"    --map-size-gb {config.map_size_gb} \\")
        print(f"    --buffer-max-size {config.buffer_max_size} \\")
        print(f"    --epochs 200 \\")
        print(f"    --lr 1e-3 \\")
        print(f"    --min-lr 1e-6 \\")
        print(f"    --warmup-epochs 10 \\")
        print(f"    --lr-schedule cosine \\")
        print(f"    --device auto \\")
        print(f"    --resume auto")

    print("=" * 70)


def get_config_dict(config: RecommendedConfig) -> Dict[str, any]:
    """Convert RecommendedConfig to dictionary for programmatic use.

    Args:
        config: RecommendedConfig object

    Returns:
        Dictionary with configuration
    """
    return {
        "model_preset": config.model_preset,
        "parallel_workers": config.parallel_workers,
        "batch_size": config.batch_size,
        "batch_size_mcts": config.batch_size_mcts,
        "selfplay_games": config.selfplay_games,
        "expected_memory_gb": config.expected_memory_gb,
        "notes": config.notes,
    }


def check_memory_sufficient(
    config: RecommendedConfig,
    hardware: Optional[HardwareInfo] = None
) -> tuple[bool, str]:
    """Check if system has enough memory for config.

    Args:
        config: RecommendedConfig to check
        hardware: HardwareInfo (auto-detected if None)

    Returns:
        (is_sufficient, message)
    """
    if hardware is None:
        hardware = detect_hardware()

    available = hardware.available_ram_gb
    required = config.expected_memory_gb

    if available >= required * 1.2:  # 20% safety margin
        return True, f"✅ Sufficient memory: {available:.1f} GB available, {required:.1f} GB required"
    elif available >= required:
        return True, f"⚠️  Tight on memory: {available:.1f} GB available, {required:.1f} GB required (may swap)"
    else:
        return False, f"❌ Insufficient memory: {available:.1f} GB available, {required:.1f} GB required"


if __name__ == "__main__":
    # Demo: show detected hardware and recommendations
    hw = detect_hardware()
    print_hardware_info(hw)

    print("\n")
    config_fast = get_recommended_config(hw, prefer_speed=True)
    print("\n📊 OPTIMIZED FOR SPEED:")
    print_recommended_config(config_fast)

    print("\n")
    config_strong = get_recommended_config(hw, prefer_strength=True)
    print("\n💪 OPTIMIZED FOR STRENGTH:")
    print_recommended_config(config_strong)

    # Check memory
    print("\n")
    sufficient, msg = check_memory_sufficient(config_strong, hw)
    print(msg)
