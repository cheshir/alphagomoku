#!/usr/bin/env python3

import os
import sys
import time
import torch
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

def test_performance():
    """Test the performance optimizations"""
    print("🚀 Testing AlphaGomoku Performance Optimizations")
    print("=" * 50)
    
    # Initialize components
    model = GomokuNet(board_size=15, num_blocks=12, channels=64)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    
    if torch.backends.mps.is_available():
        torch.set_num_threads(1)
        print("✓ MPS optimizations enabled")
    
    print(f"Device: {device}")
    print(f"Model parameters: {model.get_model_size():,}")
    
    env = GomokuEnv(board_size=15)
    test_sims = 100  # Reduced for testing
    mcts = MCTS(model, env, num_simulations=test_sims)
    
    # Test 1: Action mask vectorization
    print("\n1. Testing action mask vectorization...")
    start_time = time.time()
    for _ in range(1000):
        mask = env._get_action_mask()
    mask_time = time.time() - start_time
    print(f"   ✓ 1000 action masks in {mask_time:.3f}s ({mask_time*1000:.1f}ms)")
    
    # Test 2: MCTS search speed
    print("\n2. Testing MCTS search speed...")
    env.reset()
    
    start_time = time.time()
    action_probs, visits = mcts.search(env.board, temperature=1.0)
    search_time = time.time() - start_time
    
    print(f"   ✓ MCTS search ({test_sims} sims) in {search_time:.3f}s")
    print(f"   ✓ Expanded {len(mcts.root.children)} children")
    
    # Test 3: Multiple searches
    print("\n3. Testing multiple MCTS searches...")
    total_time = 0
    num_searches = 5
    
    for i in range(num_searches):
        env.reset()
        start_time = time.time()
        action_probs, visits = mcts.search(env.board, temperature=1.0)
        search_time = time.time() - start_time
        total_time += search_time
        print(f"   Search {i+1}: {search_time:.3f}s")
    
    avg_time = total_time / num_searches
    print(f"   ✓ Average search time: {avg_time:.3f}s")
    
    # Performance targets
    print("\n📊 Performance Analysis:")
    if avg_time < 1.0:
        print("   🎉 EXCELLENT: Sub-second MCTS searches achieved!")
    elif avg_time < 3.0:
        print("   ✅ GOOD: Fast MCTS searches")
    else:
        print("   ⚠️  SLOW: Consider reducing simulations or checking device")
    
    print(f"\n🎯 Target for production: ~800 simulations in <5s on M1 Pro")
    estimated_800_time = avg_time * (800 / test_sims)
    print(f"   Estimated 800-sim time: {estimated_800_time:.1f}s")
    
    if estimated_800_time < 5.0:
        print("   ✅ Should meet production targets!")
    else:
        print("   ⚠️  May need further optimization")

if __name__ == '__main__':
    test_performance()