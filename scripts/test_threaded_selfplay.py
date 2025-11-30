#!/usr/bin/env python3
"""Test threaded self-play with shared MPS model"""

import torch
import threading
import time
from queue import Queue
from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.train.trainer import Trainer

def threaded_worker(worker_id, model, num_games, results_queue):
    """Generate games in a thread using shared model"""
    try:
        worker = SelfPlayWorker(
            model=model,
            mcts_simulations=100,
            batch_size=64,
            difficulty='medium'
        )

        start = time.time()
        all_data = []
        for i in range(num_games):
            game_data = worker.generate_game()
            all_data.extend(game_data)

        elapsed = time.time() - start
        results_queue.put({
            'worker_id': worker_id,
            'status': 'success',
            'games': num_games,
            'positions': len(all_data),
            'time': elapsed,
            'games_per_hour': num_games / (elapsed / 3600)
        })
    except Exception as e:
        results_queue.put({
            'worker_id': worker_id,
            'status': 'error',
            'error': str(e)
        })

def test_threaded_selfplay(num_threads=1, games_per_thread=4):
    """Test self-play with multiple threads sharing MPS model"""
    # Create model and move to MPS (simulating train.py flow)
    model = GomokuNet(board_size=15, num_blocks=30, channels=192)
    trainer = Trainer(model, lr=1e-3)

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Running {num_threads} threads, {games_per_thread} games each")
    print("-" * 60)

    results_queue = Queue()
    threads = []

    overall_start = time.time()

    for i in range(num_threads):
        t = threading.Thread(
            target=threaded_worker,
            args=(i, model, games_per_thread, results_queue)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    overall_elapsed = time.time() - overall_start

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Print results
    total_games = 0
    total_positions = 0

    for r in results:
        if r['status'] == 'success':
            print(f"✅ Thread {r['worker_id']}: {r['games']} games in {r['time']:.1f}s "
                  f"({r['games_per_hour']:.0f} games/hour)")
            total_games += r['games']
            total_positions += r['positions']
        else:
            print(f"❌ Thread {r['worker_id']}: {r['error']}")

    overall_throughput = total_games / (overall_elapsed / 3600)

    print("-" * 60)
    print(f"Total: {total_games} games, {total_positions} positions in {overall_elapsed:.1f}s")
    print(f"Overall throughput: {overall_throughput:.0f} games/hour")
    print(f"Speedup vs single thread: {overall_throughput / (total_games/num_threads / (overall_elapsed/num_threads/3600)):.2f}x")

    return overall_throughput

if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: Single thread (baseline)")
    print("=" * 60)
    throughput_1 = test_threaded_selfplay(num_threads=1, games_per_thread=4)

    print("\n" + "=" * 60)
    print("Test 2: Two threads")
    print("=" * 60)
    throughput_2 = test_threaded_selfplay(num_threads=2, games_per_thread=4)

    print("\n" + "=" * 60)
    print("Test 3: Four threads")
    print("=" * 60)
    throughput_4 = test_threaded_selfplay(num_threads=4, games_per_thread=4)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1 thread:  {throughput_1:.0f} games/hour (baseline)")
    print(f"2 threads: {throughput_2:.0f} games/hour ({throughput_2/throughput_1:.2f}x speedup)")
    print(f"4 threads: {throughput_4:.0f} games/hour ({throughput_4/throughput_1:.2f}x speedup)")
    print("\nConclusion:")
    if throughput_2/throughput_1 > 1.3:
        print("✅ Threading provides significant speedup!")
        print(f"   Recommended: Use {2 if throughput_2/throughput_1 > throughput_4/throughput_2 else 4} threads")
    else:
        print("❌ Threading provides minimal benefit due to GIL")
        print("   Recommended: Stick with single worker")
