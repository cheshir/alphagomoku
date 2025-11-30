#!/usr/bin/env python3
"""Test GPU utilization during MCTS self-play."""

import torch
import time
import numpy as np
from alphagomoku.model.network import GomokuNet
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.selfplay.selfplay import SelfPlayWorker

def test_mcts_gpu():
    """Test pure MCTS GPU utilization (no UnifiedSearch)."""
    print("=" * 70)
    print("Testing Pure MCTS GPU Utilization")
    print("=" * 70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🎯 Device: {device}")

    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = GomokuNet.from_preset('small', device=device)
    print(f"   Model: {model.get_model_size():,} parameters")

    env = GomokuEnv(board_size=15)
    mcts = MCTS(model=model, env=env, num_simulations=100, batch_size=64)

    print("\n🎲 Running 5 self-play games with pure MCTS...")
    print("   (Watch GPU utilization with nvidia-smi)")

    start_time = time.time()

    for game_num in range(5):
        game_start = time.time()
        state = env.reset()
        done = False
        moves = 0

        while not done and moves < 225:
            action_probs, _ = mcts.search(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            state, _, done, _ = env.step(action)
            moves += 1

        game_time = time.time() - game_start
        print(f"   Game {game_num + 1}/5: {moves} moves in {game_time:.1f}s ({game_time/moves:.2f}s/move)")

        if device == 'cuda':
            print(f"      GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    total_time = time.time() - start_time
    print(f"\n✅ Total time: {total_time:.1f}s ({total_time/5:.1f}s/game)")
    print("\nIf GPU utilization was 0%, the problem is in MCTS batching.")
    print("If GPU utilization was high, the problem is TSS/UnifiedSearch.")


def test_unified_search_gpu():
    """Test UnifiedSearch with difficulty=easy."""
    print("\n" + "=" * 70)
    print("Testing UnifiedSearch (difficulty=easy) GPU Utilization")
    print("=" * 70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🎯 Device: {device}")

    model = GomokuNet.from_preset('small', device=device)
    env = GomokuEnv(board_size=15)

    # Use SelfPlayWorker which uses UnifiedSearch
    worker = SelfPlayWorker(
        model=model,
        mcts_simulations=100,
        batch_size=64,
        difficulty='easy',  # Pure MCTS, no TSS
        epoch=0
    )

    print("\n🎲 Running 3 self-play games with UnifiedSearch (easy)...")
    print("   (Watch GPU utilization with nvidia-smi)")

    start_time = time.time()

    data = worker.generate_batch(num_games=3)

    total_time = time.time() - start_time
    print(f"\n✅ Generated {len(data)} positions in {total_time:.1f}s ({total_time/3:.1f}s/game)")
    print("\nIf GPU utilization was 0%, the problem is in UnifiedSearch.")
    print("If GPU utilization was high, difficulty=easy is working correctly.")


if __name__ == "__main__":
    print("\nThis script tests GPU utilization during self-play.")
    print("Run 'watch -n 0.5 nvidia-smi' in another terminal to monitor GPU usage.\n")

    # Test 1: Pure MCTS
    test_mcts_gpu()

    # Test 2: UnifiedSearch with easy
    test_unified_search_gpu()

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
