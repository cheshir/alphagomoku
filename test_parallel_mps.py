#!/usr/bin/env python3
"""Test that parallel workers actually use MPS device"""

import torch
import time
import sys
from alphagomoku.model.network import GomokuNet
from alphagomoku.train.trainer import Trainer
from alphagomoku.selfplay.parallel import ParallelSelfPlay

def test_parallel_with_mps(num_workers=2, num_games=8):
    """Test parallel self-play with MPS-enabled workers"""

    print("=" * 70)
    print(f"Testing Parallel Self-Play with {num_workers} workers")
    print("=" * 70)

    # Create model and move to MPS (simulating train.py)
    print("\n1. Creating and initializing model...")
    model = GomokuNet(board_size=15, num_blocks=30, channels=192)
    trainer = Trainer(model, lr=1e-3)
    print(f"   Main process model device: {next(model.parameters()).device}")

    # Create parallel self-play worker
    print(f"\n2. Creating ParallelSelfPlay with {num_workers} workers...")
    parallel_worker = ParallelSelfPlay(
        model=model,
        board_size=15,
        mcts_simulations=100,
        adaptive_sims=False,
        batch_size=96,  # Optimal batch size
        num_workers=num_workers,
        difficulty='medium'
    )

    print(f"\n3. Generating {num_games} games...")
    print("   Watch for '[Worker subprocess]' messages to confirm MPS usage\n")
    print("-" * 70)

    start = time.time()
    data = parallel_worker.generate_batch(num_games, debug=False)
    elapsed = time.time() - start

    print("-" * 70)
    print(f"\n4. Results:")
    print(f"   Total games: {num_games}")
    print(f"   Total positions: {len(data)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    print(f"   Games/hour: {num_games / (elapsed / 3600):.0f}")
    print(f"   Avg time/game: {elapsed / num_games:.1f}s")

    return elapsed, len(data)

def compare_configurations():
    """Compare different worker configurations"""

    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    configs = [
        (1, "Single worker (MPS)"),
        (2, "Two workers (MPS each)"),
        (3, "Three workers (MPS each)"),
    ]

    results = []

    for num_workers, description in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {description}")
        print(f"{'='*70}")

        try:
            elapsed, positions = test_parallel_with_mps(num_workers=num_workers, num_games=8)
            games_per_hour = 8 / (elapsed / 3600)
            results.append({
                'workers': num_workers,
                'description': description,
                'elapsed': elapsed,
                'positions': positions,
                'games_per_hour': games_per_hour,
                'status': 'success'
            })
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                'workers': num_workers,
                'description': description,
                'status': 'failed',
                'error': str(e)
            })

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<30} {'Time(s)':<12} {'Games/hr':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline_gph = None
    for r in results:
        if r['status'] == 'success':
            if baseline_gph is None:
                baseline_gph = r['games_per_hour']
                speedup = 1.0
            else:
                speedup = r['games_per_hour'] / baseline_gph

            print(f"{r['description']:<30} {r['elapsed']:<12.1f} {r['games_per_hour']:<12.0f} {speedup:<10.2f}x")
        else:
            print(f"{r['description']:<30} FAILED: {r['error'][:30]}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if len([r for r in results if r['status'] == 'success']) >= 2:
        successful = [r for r in results if r['status'] == 'success']
        best = max(successful, key=lambda x: x['games_per_hour'])

        print(f"\n✅ Best configuration: {best['description']}")
        print(f"   Games/hour: {best['games_per_hour']:.0f}")
        print(f"   Speedup: {best['games_per_hour'] / baseline_gph:.2f}x vs single worker")

        # Calculate epoch time
        games_per_epoch = 128
        epoch_selfplay_minutes = (games_per_epoch / best['games_per_hour']) * 60
        print(f"\n   Expected self-play time for 128 games: {epoch_selfplay_minutes:.0f} minutes")
        print(f"   Estimated total epoch time: ~{epoch_selfplay_minutes + 15:.0f} minutes")
        print(f"   Epochs per day: ~{24*60 / (epoch_selfplay_minutes + 15):.0f}")

        print(f"\n   Recommended Makefile settings:")
        print(f"   --parallel-workers {best['workers']} \\")
        print(f"   --batch-size-mcts 96 \\")
        print(f"   --mcts-simulations 100 \\")
        print(f"   --selfplay-games 128 \\")

if __name__ == '__main__':
    print("MPS Parallel Self-Play Test")
    print("=" * 70)
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    if not torch.backends.mps.is_available():
        print("\n❌ MPS not available on this system!")
        sys.exit(1)

    compare_configurations()
