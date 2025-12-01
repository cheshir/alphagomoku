#!/usr/bin/env python3
"""Distributed self-play manager - manages multiple workers in a single process.

This script spawns multiple self-play workers as threads using a producer-consumer
pattern with version-based model updates.

Architecture:
- Worker threads: Generate individual games and push to local queue
- Accumulator thread: Pulls games from local queue, batches them, pushes to Redis
- Model updater thread: Checks Redis for new models, saves them with version numbers
- Workers check version number and update their models independently
"""

import argparse
import logging
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.queue import RedisQueue
from alphagomoku.utils.validation import (
    validate_redis_url,
    validate_selfplay_config,
    print_validation_errors
)


@dataclass
class WorkerStats:
    """Statistics for a single worker."""
    worker_id: str
    games_generated: int = 0
    positions_generated: int = 0
    errors: int = 0
    last_game_time: float = 0.0
    model_version: int = 0
    status: str = "initializing"  # initializing, generating, updating_model, idle, error, stopped


class SelfPlayManager:
    """Manages multiple self-play workers with unified statistics."""

    def __init__(
        self,
        num_workers: int,
        redis_url: str,
        model_preset: str,
        mcts_simulations: int,
        device: str,
        batch_size: int,
        model_update_frequency: int,
        batch_size_mcts: int,
        checkpoint_dir: str
    ):
        self.num_workers = num_workers
        self.redis_url = redis_url
        self.model_preset = model_preset
        self.mcts_simulations = mcts_simulations
        self.device = device
        self.batch_size = batch_size  # Games per Redis push
        self.model_update_frequency = model_update_frequency
        self.batch_size_mcts = batch_size_mcts
        self.checkpoint_dir = checkpoint_dir

        # Version-based model synchronization
        self.current_model_version = 0
        self.version_lock = threading.Lock()

        # Producer-consumer queue for games
        self.game_queue = queue.Queue()

        # Signal for model updater
        self.check_model_event = threading.Event()

        # Statistics
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.stats_lock = threading.Lock()
        self.batches_pushed = 0
        self.total_positions_pushed = 0
        self.start_time = None
        self.running = False

        # Redis queue
        self.queue = None

        # Setup logging to file only (no console spam during dashboard updates)
        log_file = os.path.join(checkpoint_dir, 'selfplay_manager.log')
        self.log_file = log_file

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False  # Don't propagate to root logger

    def setup(self):
        """Initialize shared resources."""
        self.logger.info("=" * 70)
        self.logger.info("Distributed Self-Play Manager Starting")
        self.logger.info("=" * 70)
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Number of workers: {self.num_workers}")
        self.logger.info(f"Model preset: {self.model_preset}")
        self.logger.info(f"MCTS simulations: {self.mcts_simulations}")
        self.logger.info(f"MCTS batch size: {self.batch_size_mcts}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size} games")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Connect to Redis
        try:
            self.queue = RedisQueue(redis_url=self.redis_url)
            self.logger.info("✓ Connected to Redis successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

        # Initialize model version 0 (random initialization or load from disk)
        self._initialize_model_v0()

        self.logger.info("=" * 70)

    def _initialize_model_v0(self):
        """Initialize version 0 model."""
        # Check if we have any checkpoints
        checkpoint_files = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith('model_v') and f.endswith('.pt')]
        )

        if checkpoint_files:
            # Load latest checkpoint
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                version = checkpoint.get('iteration', 0)

                with self.version_lock:
                    self.current_model_version = version

                self.logger.info(f"✓ Found existing checkpoint: {latest_checkpoint} (v{version})")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint {latest_checkpoint}: {e}")

        # No checkpoints found, try Redis
        self.logger.info("Checking for trained model in Redis queue...")
        model_data = self.queue.pull_model(timeout=0)

        if model_data:
            try:
                iteration = model_data['metadata'].get('iteration', 0)
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_v{iteration}.pt')

                torch.save({
                    'model_state': model_data['model_state'],
                    'iteration': iteration,
                    'metadata': model_data['metadata']
                }, checkpoint_path)

                with self.version_lock:
                    self.current_model_version = iteration

                self.logger.info(f"✓ Loaded model from Redis (v{iteration})")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load model from Redis: {e}")

        # Create initial random model (v0)
        self.logger.info("No trained model found. Creating v0 with random initialization.")
        model = GomokuNet.from_preset(self.model_preset, board_size=15, device=self.device)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model_v0.pt')

        torch.save({
            'model_state': model.state_dict(),
            'iteration': 0,
            'metadata': {'note': 'Random initialization'}
        }, checkpoint_path)

        with self.version_lock:
            self.current_model_version = 0

        self.logger.info(f"✓ Created model v0: {model.get_model_size():,} parameters")

    def worker_thread(self, worker_id: int):
        """Worker thread that generates games (producer)."""
        worker_name = f"worker-{worker_id}"

        # Initialize worker stats
        with self.stats_lock:
            self.worker_stats[worker_name] = WorkerStats(worker_id=worker_name)

        # Create worker's own model
        worker_model = GomokuNet.from_preset(self.model_preset, board_size=15, device=self.device)
        worker_model_version = -1  # Force initial load

        # Create self-play worker
        selfplay_worker = SelfPlayWorker(
            model=worker_model,
            mcts_simulations=self.mcts_simulations,
            adaptive_sims=False,
            batch_size=self.batch_size_mcts,
            difficulty='easy',
            disable_tqdm=True,  # Disable progress bars to avoid interfering with dashboard
        )

        try:
            while self.running:
                # Check for model updates (lock-free read)
                current_version = self.current_model_version

                if current_version > worker_model_version:
                    with self.stats_lock:
                        self.worker_stats[worker_name].status = "updating_model"

                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f'model_v{current_version}.pt'
                    )

                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        worker_model.load_state_dict(checkpoint['model_state'])
                        worker_model_version = current_version

                        with self.stats_lock:
                            self.worker_stats[worker_name].model_version = worker_model_version

                        self.logger.info(f"Worker {worker_id} updated to v{worker_model_version}")
                    except FileNotFoundError:
                        # File not ready yet, will retry next iteration
                        self.logger.debug(f"Worker {worker_id}: Model v{current_version} not found yet, retrying...")
                        time.sleep(0.1)
                        continue
                    except Exception as e:
                        self.logger.error(f"Worker {worker_id} failed to load model v{current_version}: {e}")
                        with self.stats_lock:
                            self.worker_stats[worker_name].errors += 1
                        time.sleep(1)
                        continue

                # Generate single game
                game_start = time.time()

                with self.stats_lock:
                    self.worker_stats[worker_name].status = "generating"

                game_positions = selfplay_worker.generate_batch(1)  # Returns list of positions from 1 game
                game_time = time.time() - game_start

                # Push all positions from this game to local queue (fast, no network)
                self.game_queue.put(game_positions)

                # Update statistics
                with self.stats_lock:
                    stats = self.worker_stats[worker_name]
                    stats.games_generated += 1
                    stats.positions_generated += len(game_positions)
                    stats.last_game_time = game_time
                    stats.status = "idle"

        except Exception as e:
            error_msg = f"Worker {worker_id} crashed: {e}"
            self.logger.error(error_msg, exc_info=True)
            with self.stats_lock:
                self.worker_stats[worker_name].status = "error"
                self.worker_stats[worker_name].errors += 1
            # Log to console as well
            print(f"\n❌ {error_msg}\nCheck log file: {self.log_file}\n", flush=True)

    def accumulator_thread(self):
        """Accumulator thread that batches games and pushes to Redis (consumer)."""
        batch = []
        games_in_batch = 0

        while self.running:
            try:
                # Pull from local queue with timeout
                game_positions = self.game_queue.get(timeout=1)
                batch.extend(game_positions)
                games_in_batch += 1

                # Push batch when we have enough games
                if games_in_batch >= self.batch_size:
                    self.queue.push_games(batch)

                    with self.stats_lock:
                        self.batches_pushed += 1
                        self.total_positions_pushed += len(batch)

                    self.logger.info(f"Pushed batch of {len(batch)} positions ({games_in_batch} games) to Redis (batch #{self.batches_pushed})")
                    batch = []
                    games_in_batch = 0

                    # Signal model updater every N batches
                    if self.batches_pushed % self.model_update_frequency == 0:
                        self.check_model_event.set()

            except queue.Empty:
                continue

        # Push remaining games on shutdown
        if batch:
            self.queue.push_games(batch)
            self.logger.info(f"Pushed final batch of {len(batch)} positions ({games_in_batch} games)")

    def model_updater_thread(self):
        """Model updater thread that checks Redis for new models."""
        while self.running:
            # Wait for signal from accumulator (with timeout)
            self.check_model_event.wait(timeout=60)
            self.check_model_event.clear()

            # Check Redis for new model
            model_data = self.queue.pull_model(timeout=0)

            if model_data:
                try:
                    iteration = model_data['metadata'].get('iteration', 0)
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f'model_v{iteration}.pt'
                    )

                    # Save checkpoint to disk
                    torch.save({
                        'model_state': model_data['model_state'],
                        'iteration': iteration,
                        'metadata': model_data['metadata']
                    }, checkpoint_path)

                    # Update version (atomic write with lock)
                    with self.version_lock:
                        self.current_model_version = iteration

                    self.logger.info(f"📥 Model updated to v{iteration}")

                except Exception as e:
                    self.logger.error(f"Failed to update model: {e}", exc_info=True)

    def print_stats(self):
        """Print combined statistics using cursor positioning (no flickering)."""
        # Build output in memory first, then print once
        lines = []

        lines.append("=" * 80)
        lines.append("  AlphaGomoku Distributed Self-Play - Live Statistics")
        lines.append("=" * 80)
        lines.append("")

        # Calculate totals
        total_games = 0
        total_positions = 0
        total_errors = 0

        with self.stats_lock:
            lines.append(f"{'Worker':<12} {'Status':<15} {'Ver':<5} {'Games':<8} {'Positions':<10} {'Last Game':<12} {'Errors':<6}")
            lines.append("-" * 80)

            for worker_name in sorted(self.worker_stats.keys()):
                stats = self.worker_stats[worker_name]
                total_games += stats.games_generated
                total_positions += stats.positions_generated
                total_errors += stats.errors

                # Status emoji
                status_emoji = {
                    'initializing': '🔄',
                    'generating': '⚙️ ',
                    'idle': '✓',
                    'updating_model': '📥',
                    'error': '❌',
                    'stopped': '⏹️ '
                }.get(stats.status, '?')

                lines.append(f"{worker_name:<12} {status_emoji} {stats.status:<13} "
                            f"v{stats.model_version:<4} {stats.games_generated:<8} {stats.positions_generated:<10} "
                            f"{stats.last_game_time:>6.1f}s      "
                            f"{stats.errors:<6}")

            lines.append("-" * 80)

            # Calculate rates
            elapsed = time.time() - self.start_time if self.start_time else 1
            games_per_hour = (total_games / elapsed) * 3600
            positions_per_hour = (total_positions / elapsed) * 3600

            lines.append("")
            lines.append(f"{'TOTAL':<12} {'':<15} {'':<5} {total_games:<8} {total_positions:<10}")
            lines.append("")
            lines.append(f"Runtime:          {elapsed / 3600:.2f} hours")
            lines.append(f"Games/hour:       {games_per_hour:.1f}")
            lines.append(f"Positions/hour:   {positions_per_hour:.1f}")
            lines.append(f"Batches pushed:   {self.batches_pushed}")
            lines.append(f"Current model:    v{self.current_model_version}")
            lines.append(f"Local queue size: {self.game_queue.qsize()} games")
            lines.append(f"Total errors:     {total_errors}")
            lines.append("")

            # Queue stats
            try:
                queue_stats = self.queue.get_stats()
                lines.append(f"Redis queue:      {queue_stats['queue_size']} batches")
                lines.append(f"Games pushed:     {queue_stats['games_pushed']}")
                lines.append(f"Games pulled:     {queue_stats['games_pulled']}")
            except:
                pass

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Log file: {self.log_file}")
        lines.append("Press Ctrl+C to stop all workers")
        lines.append("=" * 80)

        # Clear screen and print all at once (reduces flickering)
        # Use home position without clearing to preserve any error messages above
        output = "\n".join(lines)

        # Move cursor to home and clear from cursor to end of screen
        print("\033[H\033[J" + output, flush=True)

    def run(self):
        """Run the manager with all workers."""
        self.setup()

        # Print startup info to console (before dashboard takes over)
        print("\n" + "=" * 70)
        print("  AlphaGomoku Distributed Self-Play Manager")
        print("=" * 70)
        print(f"✓ Log file: {self.log_file}")
        print(f"✓ Workers: {self.num_workers}")
        print(f"✓ Device: {self.device}")
        print(f"✓ MCTS batch size: {self.batch_size_mcts}")
        print("\nStarting dashboard in 2 seconds...")
        print("(Logs will be written to file only)")
        print("=" * 70)
        time.sleep(2)

        # Clear screen before starting dashboard
        print("\033[2J\033[H", flush=True)

        # Start statistics printer thread
        def stats_printer():
            while self.running:
                self.print_stats()
                time.sleep(5)  # Update every 5 seconds

        stats_thread = threading.Thread(target=stats_printer, daemon=True)

        # Start accumulator thread
        accumulator = threading.Thread(target=self.accumulator_thread, daemon=True)

        # Start model updater thread
        updater = threading.Thread(target=self.model_updater_thread, daemon=True)

        # Start all worker threads
        worker_threads = []
        self.running = True
        self.start_time = time.time()

        self.logger.info(f"Starting {self.num_workers} worker threads...")

        for i in range(1, self.num_workers + 1):
            thread = threading.Thread(target=self.worker_thread, args=(i,), daemon=True)
            thread.start()
            worker_threads.append(thread)

        # Start supporting threads
        accumulator.start()
        updater.start()
        stats_thread.start()

        # Wait for interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("\n\nStopping all workers...")
            self.running = False

            # Wait for threads to finish (with timeout)
            for thread in worker_threads:
                thread.join(timeout=2)

            # Print final statistics
            self.print_stats()
            print("\n✓ All workers stopped")


def main():
    parser = argparse.ArgumentParser(description='Distributed self-play manager')
    parser.add_argument('--redis-url', type=str, required=True,
                        help='Redis connection URL (redis://:password@host:port/db)')
    parser.add_argument('--num-workers', type=int, default=6,
                        help='Number of parallel workers')
    parser.add_argument('--model-preset', type=str, choices=['small', 'medium', 'large'],
                        default='medium', help='Model preset to use')
    parser.add_argument('--mcts-simulations', type=int, default=50,
                        help='MCTS simulations per move')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda'],
                        default='cpu', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Games per batch to push to Redis')
    parser.add_argument('--model-update-frequency', type=int, default=10,
                        help='Check for new model every N batches pushed')
    parser.add_argument('--batch-size-mcts', type=int, default=32,
                        help='MCTS batch size for neural network evaluation')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save/load model checkpoints locally')

    args = parser.parse_args()

    # Validate arguments
    validation_errors = []

    # Validate Redis URL
    validation_errors.extend(validate_redis_url(args.redis_url))

    # Validate self-play configuration
    validation_errors.extend(validate_selfplay_config(
        args.mcts_simulations,
        args.batch_size_mcts,
        args.batch_size,
        args.model_update_frequency
    ))

    # Validate number of workers
    if args.num_workers < 1 or args.num_workers > 16:
        validation_errors.append(
            f"❌ Invalid number of workers: {args.num_workers}\n"
            f"   Must be between 1 and 16\n"
            f"   Recommended: 4-8 for CPU, 1 for MPS"
        )

    # Print validation errors and exit if any
    if validation_errors:
        logger = logging.getLogger(__name__)
        print_validation_errors(validation_errors, logger)
        return 1

    # Create and run manager
    manager = SelfPlayManager(
        num_workers=args.num_workers,
        redis_url=args.redis_url,
        model_preset=args.model_preset,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
        batch_size=args.batch_size,
        model_update_frequency=args.model_update_frequency,
        batch_size_mcts=args.batch_size_mcts,
        checkpoint_dir=args.checkpoint_dir
    )

    manager.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
