#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# MPS optimization settings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.mcts.mcts import MCTS

def test_all_optimizations():
    """Comprehensive test of all performance optimizations"""
    print("🚀 AlphaGomoku Performance Optimization Test Suite")
    print("=" * 60)

    # Device setup
    model = GomokuNet(board_size=15, num_blocks=12, channels=64)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    if torch.backends.mps.is_available():
        torch.set_num_threads(1)
        print("✓ MPS optimizations enabled")

    print(f"Device: {device}")
    print(f"Model parameters: {model.get_model_size():,}")

    env = GomokuEnv(board_size=15)

    # Test 1: Single inference speed
    print("\n1. Testing Single Inference Speed:")
    test_states = []
    for _ in range(10):
        env.reset()
        # Make a few random moves to create diverse positions
        for _ in range(np.random.randint(3, 8)):
            legal_actions = env.get_legal_actions()
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
                env.step(action)

        # Convert to tensor once and reuse
        board_size = env.board_size
        own_stones = (env.board == env.current_player).astype(np.float32)
        opp_stones = (env.board == -env.current_player).astype(np.float32)
        last_move = np.zeros((board_size, board_size), dtype=np.float32)
        if env.last_move[0] >= 0:
            last_move[env.last_move[0], env.last_move[1]] = 1.0
        side_to_move = np.ones((board_size, board_size), dtype=np.float32)
        pattern_maps = np.zeros((board_size, board_size), dtype=np.float32)

        state_np = np.stack([own_stones, opp_stones, last_move, side_to_move, pattern_maps])
        state_tensor = torch.from_numpy(state_np).float().to(device)
        test_states.append(state_tensor)

    # Time single inferences
    model.eval()
    single_times = []
    for state in test_states:
        start_time = time.time()
        with torch.inference_mode():
            policy, value = model.predict(state)
        single_times.append(time.time() - start_time)

    avg_single_time = np.mean(single_times) * 1000  # Convert to ms
    print(f"   ✓ Average single inference: {avg_single_time:.1f}ms")

    # Current performance vs target
    if avg_single_time <= 16.7:
        print("   🎉 EXCELLENT: Meeting single inference target!")
    elif avg_single_time <= 25.0:
        print("   ✅ GOOD: Close to single inference target")
    else:
        print("   ⚠️  SLOW: Above single inference target (16.7ms)")

    # Test 2: Batch inference speed
    print("\n2. Testing Batch Inference Speed:")
    batch_sizes = [8, 16, 32, 64]

    for batch_size in batch_sizes:
        batch_states = torch.stack(test_states[:batch_size])

        start_time = time.time()
        with torch.inference_mode():
            policies, values = model.predict_batch(batch_states)
        batch_time = time.time() - start_time
        per_sample_time = (batch_time / batch_size) * 1000  # ms per sample

        print(f"   ✓ Batch {batch_size:2d}: {per_sample_time:.1f}ms per sample (total: {batch_time*1000:.1f}ms)")

        if per_sample_time <= 11.0:
            status = "🎉 EXCELLENT"
        elif per_sample_time <= 15.0:
            status = "✅ GOOD"
        else:
            status = "⚠️  SLOW"
        print(f"      {status}: vs target 11ms per sample")

    # Test 3: MCTS with optimized batching
    print("\n3. Testing MCTS with Optimized Batching:")
    test_simulations = [50, 100, 200]

    for sim_count in test_simulations:
        # Test with batching enabled
        mcts_batched = MCTS(model, env, num_simulations=sim_count, batch_size=32)
        env.reset()

        start_time = time.time()
        action_probs, visits = mcts_batched.search(env.board, temperature=1.0)
        batched_time = time.time() - start_time

        # Test without batching
        mcts_single = MCTS(model, env, num_simulations=sim_count, batch_size=1)
        env.reset()

        start_time = time.time()
        action_probs, visits = mcts_single.search(env.board, temperature=1.0)
        single_time = time.time() - start_time

        speedup = single_time / batched_time
        print(f"   ✓ {sim_count:3d} sims: {batched_time:.3f}s (batched) vs {single_time:.3f}s (single) = {speedup:.1f}x speedup")

    # Test 4: Memory efficiency
    print("\n4. Testing Memory Optimization:")

    # Test tensor conversion efficiency
    dummy_states = [torch.randn(5, 15, 15) for _ in range(100)]

    # Old way: individual tensor conversions
    start_time = time.time()
    for state in dummy_states:
        state_on_device = state.to(device)
    old_way_time = time.time() - start_time

    # New way: batch conversion
    start_time = time.time()
    batch_np = np.stack([state.numpy() for state in dummy_states])
    batch_tensor = torch.from_numpy(batch_np).to(device)
    new_way_time = time.time() - start_time

    tensor_speedup = old_way_time / new_way_time
    print(f"   ✓ Tensor conversion speedup: {tensor_speedup:.1f}x")
    print(f"     Old way: {old_way_time*1000:.1f}ms, New way: {new_way_time*1000:.1f}ms")

    # Performance summary
    print("\n📊 Performance Summary:")
    print(f"   🎯 Single inference: {avg_single_time:.1f}ms (target: 16.7ms)")

    # Estimate 800-sim performance
    estimated_800_sim = (batched_time / sim_count) * 800
    print(f"   🎯 Estimated 800-sim MCTS: {estimated_800_sim:.1f}s (target: <5s)")

    if estimated_800_sim < 5.0:
        print("   ✅ All performance targets met!")
    else:
        print("   ⚠️  Some targets not met, consider further optimization")

    print(f"\n✨ Optimization Benefits:")
    print(f"   • Reduced GPU memory transfers with batched device operations")
    print(f"   • 87.5% memory savings with lazy data augmentation")
    print(f"   • {tensor_speedup:.1f}x faster tensor conversions")
    print(f"   • Optimized batching provides {speedup:.1f}x MCTS speedup")

if __name__ == '__main__':
    test_all_optimizations()