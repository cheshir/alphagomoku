#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayData

def create_sample_data(count: int = 1000) -> list:
    """Create sample training data"""
    data = []
    board_size = 15

    for i in range(count):
        # Random state (5 channels x 15x15)
        state = np.random.rand(5, board_size, board_size).astype(np.float32)
        # Random policy (15x15 flattened)
        policy = np.random.rand(board_size * board_size).astype(np.float32)
        policy = policy / policy.sum()  # Normalize
        # Random value
        value = np.random.uniform(-1, 1)

        data.append(SelfPlayData(state=state, policy=policy, value=value))

    return data

def test_memory_usage():
    """Test memory usage with traditional vs lazy augmentation"""
    print("🧠 Testing Memory Usage: Traditional vs Lazy Augmentation")
    print("=" * 60)

    sample_data = create_sample_data(1000)
    print(f"Created {len(sample_data)} sample training examples")

    # Test traditional augmentation (8x memory usage)
    print("\n1. Traditional Augmentation (Pre-generates all 8 augmentations):")
    temp_dir_traditional = tempfile.mkdtemp()

    try:
        buffer_traditional = DataBuffer(
            db_path=temp_dir_traditional,
            lazy_augmentation=False,
            max_size=100_000
        )

        start_time = time.time()
        buffer_traditional.add_data(sample_data)
        add_time_traditional = time.time() - start_time

        traditional_size = len(buffer_traditional)
        print(f"   ✓ Data stored: {traditional_size:,} examples")
        print(f"   ✓ Storage time: {add_time_traditional:.3f}s")

        # Get directory size
        traditional_disk_size = sum(
            os.path.getsize(os.path.join(temp_dir_traditional, f))
            for f in os.listdir(temp_dir_traditional)
        ) / (1024 * 1024)  # Convert to MB
        print(f"   ✓ Disk usage: {traditional_disk_size:.1f} MB")

        # Test sampling
        start_time = time.time()
        batch = buffer_traditional.sample_batch(512)
        sample_time_traditional = time.time() - start_time
        print(f"   ✓ Batch sampling (512): {sample_time_traditional*1000:.1f}ms")

    finally:
        shutil.rmtree(temp_dir_traditional)

    # Test lazy augmentation (1x storage, augment on-demand)
    print("\n2. Lazy Augmentation (On-demand augmentation):")
    temp_dir_lazy = tempfile.mkdtemp()

    try:
        buffer_lazy = DataBuffer(
            db_path=temp_dir_lazy,
            lazy_augmentation=True,
            max_size=100_000
        )

        start_time = time.time()
        buffer_lazy.add_data(sample_data)
        add_time_lazy = time.time() - start_time

        lazy_size = len(buffer_lazy)
        print(f"   ✓ Data stored: {lazy_size:,} examples")
        print(f"   ✓ Storage time: {add_time_lazy:.3f}s")

        # Get directory size
        lazy_disk_size = sum(
            os.path.getsize(os.path.join(temp_dir_lazy, f))
            for f in os.listdir(temp_dir_lazy)
        ) / (1024 * 1024)  # Convert to MB
        print(f"   ✓ Disk usage: {lazy_disk_size:.1f} MB")

        # Test sampling (applies random augmentation)
        start_time = time.time()
        batch = buffer_lazy.sample_batch(512)
        sample_time_lazy = time.time() - start_time
        print(f"   ✓ Batch sampling (512): {sample_time_lazy*1000:.1f}ms")

        # Verify augmentation is working
        original_states = set()
        augmented_states = set()
        for _ in range(100):
            batch = buffer_lazy.sample_batch(10)
            for example in batch:
                state_hash = hash(example.state.tobytes())
                augmented_states.add(state_hash)

        # Check original data
        for example in sample_data[:10]:
            original_states.add(hash(example.state.tobytes()))

        unique_augmented = len(augmented_states)
        print(f"   ✓ Augmentation diversity: {unique_augmented} unique states from sampling")

    finally:
        shutil.rmtree(temp_dir_lazy)

    # Performance comparison
    print("\n📊 Performance Comparison:")
    memory_savings = (traditional_disk_size - lazy_disk_size) / traditional_disk_size * 100
    print(f"   🎯 Memory savings: {memory_savings:.1f}% ({traditional_disk_size:.1f}MB → {lazy_disk_size:.1f}MB)")

    storage_speedup = add_time_traditional / add_time_lazy
    print(f"   🚀 Storage speedup: {storage_speedup:.1f}x faster")

    sampling_overhead = (sample_time_lazy / sample_time_traditional - 1) * 100
    print(f"   ⏱️  Sampling overhead: {sampling_overhead:+.1f}% ({sample_time_traditional*1000:.1f}ms → {sample_time_lazy*1000:.1f}ms)")

    print(f"\n✅ Lazy augmentation reduces storage by ~{memory_savings:.0f}% with minimal sampling overhead!")

if __name__ == '__main__':
    test_memory_usage()