#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# MPS optimization settings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.selfplay.parallel import ParallelSelfPlay


def benchmark_mcts_performance():
    """Benchmark MCTS performance with different configurations"""
    print("🔬 MCTS Performance Benchmark")
    print("=" * 50)
    
    # Initialize model and environment
    model = GomokuNet(board_size=15, num_blocks=12, channels=64)
    # Move to MPS/CUDA if available for fair batched speedups
    if torch.backends.mps.is_available():
        model.to('mps')
    elif torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    env = GomokuEnv(board_size=15)
    
    # Test configurations
    configs = [
        {"sims": 100, "batch": 1, "name": "Baseline (100 sims, no batch)"},
        {"sims": 100, "batch": 16, "name": "Batched (100 sims, batch=16)"},
        {"sims": 100, "batch": 32, "name": "Batched (100 sims, batch=32)"},
        {"sims": 100, "batch": 64, "name": "Batched (100 sims, batch=64)"},
        {"sims": 400, "batch": 1, "name": "Baseline (400 sims, no batch)"},
        {"sims": 400, "batch": 32, "name": "Batched (400 sims, batch=32)"},
        {"sims": 400, "batch": 64, "name": "Batched (400 sims, batch=64)"},
        {"sims": 800, "batch": 64, "name": "Batched (800 sims, batch=64)"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        
        # Create MCTS instance
        mcts = MCTS(
            model=model, 
            env=env, 
            num_simulations=config["sims"],
            batch_size=config["batch"]
        )
        
        # Warm up
        env.reset()
        mcts.search(env.board, temperature=0.0)
        
        # Benchmark
        times = []
        for i in range(5):
            env.reset()
            start_time = time.time()
            policy, _ = mcts.search(env.board, temperature=0.0)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"   Run {i+1}: {elapsed:.3f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results.append({
            "config": config["name"],
            "avg_time": avg_time,
            "std_time": std_time,
            "sims_per_sec": config["sims"] / avg_time
        })
        
        print(f"   Average: {avg_time:.3f}±{std_time:.3f}s ({config['sims']/avg_time:.0f} sims/sec)")
    
    # Summary
    print(f"\n📈 Performance Summary:")
    print("-" * 70)
    print(f"{'Configuration':<35} {'Time (s)':<12} {'Sims/sec':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['config']:<35} {result['avg_time']:.3f}±{result['std_time']:.3f}   {result['sims_per_sec']:.0f}")
    
    return results


def benchmark_selfplay_performance():
    """Benchmark self-play performance with different configurations"""
    print("\n\n🎮 Self-Play Performance Benchmark")
    print("=" * 50)
    
    model = GomokuNet(board_size=15, num_blocks=12, channels=64)
    if torch.backends.mps.is_available():
        model.to('mps')
    elif torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    
    # Test configurations
    configs = [
        {"adaptive": False, "batch": 1, "parallel": 1, "name": "Baseline"},
        {"adaptive": True, "batch": 32, "parallel": 1, "name": "Optimized Single"},
        {"adaptive": True, "batch": 32, "parallel": 2, "name": "Parallel (2 workers)"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        
        if config["parallel"] > 1:
            worker = ParallelSelfPlay(
                model=model,
                mcts_simulations=200,  # Reduced for faster testing
                adaptive_sims=config["adaptive"],
                batch_size=config["batch"],
                num_workers=config["parallel"]
            )
        else:
            worker = SelfPlayWorker(
                model=model,
                mcts_simulations=200,
                adaptive_sims=config["adaptive"],
                batch_size=config["batch"]
            )
        
        # Benchmark game generation
        start_time = time.time()
        data = worker.generate_batch(3)  # Generate 3 games for testing
        elapsed = time.time() - start_time
        
        games_per_hour = (3 / elapsed) * 3600
        positions_per_hour = (len(data) / elapsed) * 3600
        
        results.append({
            "config": config["name"],
            "time": elapsed,
            "games_per_hour": games_per_hour,
            "positions_per_hour": positions_per_hour,
            "positions": len(data)
        })
        
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Games/hour: {games_per_hour:.0f}")
        print(f"   Positions/hour: {positions_per_hour:.0f}")
        print(f"   Positions generated: {len(data)}")
    
    # Summary
    print(f"\n📈 Self-Play Summary:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Games/hr':<10} {'Positions/hr':<12}")
    print("-" * 80)
    for result in results:
        print(f"{result['config']:<20} {result['time']:.1f}      {result['games_per_hour']:.0f}      {result['positions_per_hour']:.0f}")
    
    return results


def test_adaptive_simulations():
    """Test adaptive simulation scheduling"""
    print("\n\n🧠 Adaptive Simulation Test")
    print("=" * 50)
    
    from alphagomoku.mcts.adaptive import AdaptiveSimulator
    
    adaptive = AdaptiveSimulator()
    board = np.zeros((15, 15))
    
    # Test different game phases
    phases = [
        (5, "Early game"),
        (50, "Mid game"),
        (200, "Late game")
    ]
    
    for move_count, phase_name in phases:
        sims = []
        for _ in range(10):
            sim_count = adaptive.get_simulations(move_count, board)
            sims.append(sim_count)
        
        print(f"{phase_name} (move {move_count}): {np.mean(sims):.0f}±{np.std(sims):.0f} sims")
        print(f"   Range: {min(sims)}-{max(sims)} simulations")


def main():
    """Run all benchmarks"""
    print("🚀 AlphaGomoku Optimization Benchmarks")
    print("=" * 60)
    
    # Check device
    if torch.backends.mps.is_available():
        print("✓ MPS acceleration available")
    elif torch.cuda.is_available():
        print("✓ CUDA acceleration available")
    else:
        print("⚠️  No accelerator; CPU-only (batching may be slower)")
    
    # Run benchmarks
    mcts_results = benchmark_mcts_performance()
    selfplay_results = benchmark_selfplay_performance()
    test_adaptive_simulations()
    
    # Overall summary
    print("\n\n🎯 Optimization Impact Summary")
    print("=" * 60)
    
    if len(mcts_results) >= 2:
        baseline_time = mcts_results[0]["avg_time"]
        optimized_time = mcts_results[1]["avg_time"]
        speedup = baseline_time / optimized_time
        print(f"MCTS Speedup (100 sims): {speedup:.1f}x faster")
    
    if len(selfplay_results) >= 2:
        baseline_games = selfplay_results[0]["games_per_hour"]
        optimized_games = selfplay_results[1]["games_per_hour"]
        speedup = optimized_games / baseline_games
        print(f"Self-Play Speedup: {speedup:.1f}x more games/hour")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
