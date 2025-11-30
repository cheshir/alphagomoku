#!/usr/bin/env python3
"""Test different parallel worker configurations with MPS"""

import torch
import time
import sys
from alphagomoku.model.network import GomokuNet
from alphagomoku.train.trainer import Trainer
from alphagomoku.selfplay.parallel import ParallelSelfPlay
from alphagomoku.selfplay.selfplay import SelfPlayWorker

def test_configuration(num_workers, num_games=8, simulations=100):
    """Test a specific worker configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: {num_workers} worker(s), {num_games} games, {simulations} sims")
    print(f"{'='*70}\n")

    # Create model and trainer (moves to MPS)
    model = GomokuNet(board_size=15, num_blocks=30, channels=192)
    trainer = Trainer(model, lr=1e-3)
    print(f"Main process device: {next(model.parameters()).device}")

    start = time.time()

    if num_workers == 1:
        # Single worker (no multiprocessing overhead)
        worker = SelfPlayWorker(
            model=model,
            mcts_simulations=simulations,
            batch_size=96,
            difficulty='medium'
        )
        data = worker.generate_batch(num_games)
    else:
        # Parallel workers
        parallel_worker = ParallelSelfPlay(
            model=model,
            mcts_simulations=simulations,
            adaptive_sims=False,
            batch_size=96,
            num_workers=num_workers,
            difficulty='medium'
        )
        data = parallel_worker.generate_batch(num_games, debug=False)

    elapsed = time.time() - start

    # Calculate metrics
    games_per_hour = num_games / (elapsed / 3600)
    avg_time_per_game = elapsed / num_games
    positions = len(data)

    # Estimate epoch time
    epoch_games = 128
    epoch_selfplay_minutes = (epoch_games / games_per_hour) * 60
    epoch_total_minutes = epoch_selfplay_minutes + 15  # +15 for training
    epochs_per_day = (24 * 60) / epoch_total_minutes

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Avg time/game: {avg_time_per_game:.1f}s")
    print(f"Games/hour: {games_per_hour:.0f}")
    print(f"Total positions: {positions}")
    print(f"\nProjected for 128 games/epoch:")
    print(f"  Self-play time: {epoch_selfplay_minutes:.0f} minutes")
    print(f"  Total epoch time: {epoch_total_minutes:.0f} minutes ({epoch_total_minutes/60:.1f} hours)")
    print(f"  Epochs per day: {epochs_per_day:.1f}")

    return {
        'workers': num_workers,
        'elapsed': elapsed,
        'games_per_hour': games_per_hour,
        'avg_time_per_game': avg_time_per_game,
        'positions': positions,
        'epoch_minutes': epoch_total_minutes,
        'epochs_per_day': epochs_per_day
    }

if __name__ == '__main__':
    print("="*70)
    print("PARALLEL WORKER MPS PERFORMANCE TEST")
    print("="*70)
    print(f"MPS Available: {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        print("\n❌ MPS not available!")
        sys.exit(1)

    # Test different configurations
    configs = [
        (1, "Single worker (MPS)"),
        (2, "Two workers (MPS each)"),
        (3, "Three workers (MPS each)"),
        (4, "Four workers (MPS each)"),
    ]

    results = []

    for num_workers, description in configs:
        try:
            result = test_configuration(num_workers, num_games=8, simulations=100)
            result['description'] = description
            result['status'] = 'success'
            results.append(result)

            # Give system a moment to cool down between tests
            print("\nWaiting 10 seconds before next test...")
            time.sleep(10)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'workers': num_workers,
                'description': description,
                'status': 'failed',
                'error': str(e)
            })

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Config':<25} {'Time/game':<12} {'Games/hr':<12} {'Epoch/day':<12} {'Speedup':<10}")
    print("-"*70)

    baseline_gph = None
    for r in results:
        if r['status'] == 'success':
            if baseline_gph is None:
                baseline_gph = r['games_per_hour']
                speedup = 1.0
            else:
                speedup = r['games_per_hour'] / baseline_gph

            print(f"{r['description']:<25} "
                  f"{r['avg_time_per_game']:<12.1f} "
                  f"{r['games_per_hour']:<12.0f} "
                  f"{r['epochs_per_day']:<12.1f} "
                  f"{speedup:<10.2f}x")
        else:
            print(f"{r['description']:<25} FAILED")

    # Find best configuration
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        best = max(successful, key=lambda x: x['games_per_hour'])

        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(f"\n✅ Best configuration: {best['description']}")
        print(f"   Workers: {best['workers']}")
        print(f"   Games/hour: {best['games_per_hour']:.0f}")
        print(f"   Speedup: {best['games_per_hour'] / baseline_gph:.2f}x vs single worker")
        print(f"   Epoch time: {best['epoch_minutes']:.0f} minutes ({best['epoch_minutes']/60:.1f} hours)")
        print(f"   Epochs/day: {best['epochs_per_day']:.1f}")

        print(f"\n📝 Recommended Makefile settings:")
        print(f"   --parallel-workers {best['workers']} \\")
        print(f"   --batch-size-mcts 96 \\")
        print(f"   --mcts-simulations 100 \\")
        print(f"   --selfplay-games 128 \\")
