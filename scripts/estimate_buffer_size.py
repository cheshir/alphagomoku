#!/usr/bin/env python3
"""Estimate LMDB buffer size requirements for training."""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.selfplay.selfplay import SelfPlayData


def estimate_position_size():
    """Estimate the size of a single position when pickled."""
    # Create a sample position
    state = np.random.randint(-1, 2, (5, 15, 15)).astype(np.float32)
    policy = np.random.random(225).astype(np.float32)
    value = float(np.random.random())
    
    sample_data = SelfPlayData(state=state, policy=policy, value=value)
    pickled_size = len(pickle.dumps(sample_data))
    
    return pickled_size


def main():
    parser = argparse.ArgumentParser(description='Estimate buffer size requirements')
    parser.add_argument('--games-per-epoch', type=int, default=200, help='Games per epoch')
    parser.add_argument('--positions-per-game', type=int, default=50, help='Average positions per game')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--buffer-max-positions', type=int, default=2_000_000, help='Max positions in buffer')
    
    args = parser.parse_args()
    
    print("AlphaGomoku Buffer Size Estimator")
    print("=" * 40)
    
    # Estimate position size
    position_size = estimate_position_size()
    print(f"Estimated size per position: {position_size:,} bytes ({position_size/1024:.1f} KB)")
    
    # Calculate per-epoch data
    positions_per_epoch = args.games_per_epoch * args.positions_per_game
    augmented_positions_per_epoch = positions_per_epoch * 8  # 8-fold augmentation
    
    print(f"\\nPer-epoch data:")
    print(f"  Raw positions: {positions_per_epoch:,}")
    print(f"  Augmented positions: {augmented_positions_per_epoch:,}")
    print(f"  Storage per epoch: {augmented_positions_per_epoch * position_size / 1024**2:.1f} MB")
    
    # Calculate buffer requirements
    buffer_size_bytes = args.buffer_max_positions * position_size
    buffer_size_gb = buffer_size_bytes / 1024**3
    
    print(f"\\nBuffer requirements:")
    print(f"  Max positions: {args.buffer_max_positions:,}")
    print(f"  Estimated storage: {buffer_size_gb:.1f} GB")
    
    # Calculate epochs to fill buffer
    epochs_to_fill = args.buffer_max_positions / augmented_positions_per_epoch
    
    print(f"\\nBuffer dynamics:")
    print(f"  Epochs to fill buffer: {epochs_to_fill:.1f}")
    if epochs_to_fill < args.epochs:
        print(f"  Buffer will be full after epoch {int(epochs_to_fill)}")
        print(f"  Remaining {args.epochs - int(epochs_to_fill)} epochs will use circular buffer")
    else:
        print(f"  Buffer will not fill completely in {args.epochs} epochs")
    
    # Recommendations
    print(f"\\nRecommendations:")
    
    # Map size recommendation (add 50% overhead)
    recommended_map_size = buffer_size_gb * 1.5
    print(f"  --map-size-gb: {max(4, int(recommended_map_size) + 1)} (minimum 4GB)")
    
    # Buffer size recommendations
    if epochs_to_fill < 10:
        smaller_buffer = int(augmented_positions_per_epoch * 20)  # 20 epochs worth
        print(f"  Consider smaller --buffer-max-size: {smaller_buffer:,} (20 epochs worth)")
    
    if buffer_size_gb > 16:
        print(f"  Warning: Large buffer size ({buffer_size_gb:.1f}GB) may cause memory issues")
        print(f"  Consider reducing --games-per-epoch or --buffer-max-size")
    
    # Memory usage estimate
    print(f"\\nMemory usage estimate:")
    print(f"  LMDB buffer: {buffer_size_gb:.1f} GB")
    print(f"  Model + training: ~2-4 GB")
    print(f"  Total estimated: {buffer_size_gb + 3:.1f} GB")
    
    if buffer_size_gb + 3 > 16:
        print(f"  ⚠️  May exceed typical system memory (16GB)")
    elif buffer_size_gb + 3 > 8:
        print(f"  ⚠️  Requires substantial memory (8GB+)")
    else:
        print(f"  ✅ Should fit in typical system memory")


if __name__ == "__main__":
    main()