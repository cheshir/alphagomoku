#!/usr/bin/env python3
"""Distributed self-play manager - manages multiple workers as separate processes.

Uses multiprocessing (not threading) to bypass Python's GIL and achieve true
parallel execution across all CPU cores.

Architecture:
- Worker processes: Generate games independently (bypasses GIL!)
- Accumulator thread: Batches games and pushes to Redis
- Model updater thread: Downloads models from Redis
- Dashboard thread: Displays real-time statistics

Key design:
- Worker PROCESSES (not threads) -> each has own Python interpreter
- Shared memory using multiprocessing.Manager()
- True parallel execution: 8 workers = ~800% CPU (8 cores fully utilized)
"""

import argparse
import logging
import os
import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.queue import PositionQueue
from alphagomoku.utils.validation import (
    validate_redis_url,
    validate_selfplay_config,
    print_validation_errors
)


def worker_process(
    worker_id: int,
    model_preset: str,
    mcts_simulations: int,
    device: str,
    mcts_batch_size: int,  # RENAMED: NN inference batch size for MCTS
    checkpoint_dir: str,
    game_queue,  # Shared queue
    worker_stats_dict,  # Shared dict
    stats_lock,  # Shared lock
    current_version_value,  # Shared value
    running_value,  # Shared value
    log_file: str
):
    """Worker process that generates games (runs in separate process, bypasses GIL!).

    Args:
        mcts_batch_size: Neural network batch size for MCTS evaluations (not game count!)
    """
    # Setup logging
    logger = logging.getLogger(f"worker-{worker_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter(
        f'%(asctime)s - [Worker-{worker_id}] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.propagate = False

    worker_name = f"worker-{worker_id}"

    # Initialize stats
    with stats_lock:
        worker_stats_dict[worker_name] = {
            'worker_id': worker_name,
            'games_generated': 0,
            'positions_generated': 0,
            'total_time': 0.0,
            'status': 'initializing',
            'model_version': -1,
            'errors': 0
        }

    # Create model
    worker_model = GomokuNet.from_preset(model_preset, board_size=15, device=device)
    worker_model_version = -1

    # Create self-play worker
    selfplay_worker = SelfPlayWorker(
        model=worker_model,
        mcts_simulations=mcts_simulations,
        adaptive_sims=False,
        batch_size=mcts_batch_size,  # NN batch size for MCTS
        difficulty='easy',
        disable_tqdm=True,
    )

    logger.info("Worker initialized, waiting for model...")

    # Wait for initial model with timeout
    model_wait_start = time.time()
    model_loaded = False

    try:
        while running_value.value:
            # Check for model updates
            current_version = current_version_value.value

            if current_version > worker_model_version:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_v{current_version}.pt')

                # Update status based on whether this is initial load or update
                with stats_lock:
                    stats = dict(worker_stats_dict[worker_name])
                    if worker_model_version == -1:
                        stats['status'] = 'waiting_for_model'
                    else:
                        stats['status'] = 'updating_model'
                    worker_stats_dict[worker_name] = stats

                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    worker_model.load_state_dict(checkpoint['model_state'])
                    worker_model_version = current_version
                    model_loaded = True

                    with stats_lock:
                        stats = dict(worker_stats_dict[worker_name])
                        stats['model_version'] = worker_model_version
                        stats['status'] = 'ready'
                        worker_stats_dict[worker_name] = stats

                    logger.info(f"Updated to v{worker_model_version}")
                except FileNotFoundError:
                    # Check timeout for initial model load
                    if worker_model_version == -1:
                        wait_time = time.time() - model_wait_start
                        if wait_time > 60:  # 60 second timeout
                            raise TimeoutError(f"Model file not found after {wait_time:.1f}s: {checkpoint_path}")
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    with stats_lock:
                        stats = dict(worker_stats_dict[worker_name])
                        stats['errors'] += 1
                        stats['status'] = 'error'
                        worker_stats_dict[worker_name] = stats
                    time.sleep(1)
                    continue

            # Skip game generation if model not loaded yet
            if not model_loaded:
                time.sleep(0.1)
                continue

            # Generate game
            game_start = time.time()

            with stats_lock:
                stats = dict(worker_stats_dict[worker_name])
                stats['status'] = 'generating'
                worker_stats_dict[worker_name] = stats

            game_positions = selfplay_worker.generate_batch(1)
            game_time = time.time() - game_start

            # Push to queue
            game_queue.put(game_positions)

            # Update stats (must reassign entire dict for Manager.dict to sync)
            with stats_lock:
                stats = dict(worker_stats_dict[worker_name])  # Copy
                stats['games_generated'] += 1
                stats['positions_generated'] += len(game_positions)
                stats['total_time'] += game_time
                stats['status'] = 'idle'
                worker_stats_dict[worker_name] = stats  # Reassign

    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        with stats_lock:
            stats = dict(worker_stats_dict[worker_name])
            stats['status'] = 'error'
            stats['errors'] += 1
            worker_stats_dict[worker_name] = stats
        print(f"\n❌ Worker {worker_id} crashed: {e}\nCheck: {log_file}\n", flush=True)


class DistributedSelfPlayManager:
    """Manager that coordinates multiple self-play worker PROCESSES."""

    def __init__(
        self,
        num_workers: int,
        redis_url: str,
        model_preset: str,
        mcts_simulations: int,
        device: str,
        positions_per_push: int,  # How many positions to batch before pushing to Redis
        model_update_frequency: int,
        mcts_batch_size: int,  # NN inference batch size
        checkpoint_dir: str
    ):
        """Initialize distributed self-play manager.

        Args:
            positions_per_push: How many positions to accumulate before pushing to Redis
            mcts_batch_size: Neural network batch size for MCTS evaluations
        """
        self.num_workers = num_workers
        self.redis_url = redis_url
        self.model_preset = model_preset
        self.mcts_simulations = mcts_simulations
        self.device = device
        self.positions_per_push = positions_per_push
        self.model_update_frequency = model_update_frequency
        self.mcts_batch_size = mcts_batch_size
        self.checkpoint_dir = checkpoint_dir

        # Multiprocessing manager for shared state
        self.manager = mp.Manager()

        # Shared state (accessible across processes)
        self.current_model_version = self.manager.Value('i', 0)
        self.current_model_timestamp = self.manager.Value('c', b'')  # Redis timestamp string
        self.version_lock = self.manager.Lock()
        self.game_queue = self.manager.Queue()
        self.worker_stats_dict = self.manager.dict()
        self.stats_lock = self.manager.Lock()
        self.batches_pushed = self.manager.Value('i', 0)
        self.total_positions_pushed = self.manager.Value('i', 0)
        self.last_batch_time = self.manager.Value('d', 0.0)  # Timestamp of last Redis push
        self.accumulated_positions = self.manager.Value('i', 0)  # Positions in current batch
        self.running = self.manager.Value('b', False)

        # Thread-only state
        self.check_model_event = threading.Event()
        self.start_time = None

        # Redis
        self.queue = None

        # Logging - recreate log file on each start
        log_file = os.path.join(checkpoint_dir, 'selfplay_manager.log')
        self.log_file = log_file

        # Clear old log file
        if os.path.exists(log_file):
            os.remove(log_file)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def setup(self):
        """Initialize Redis and model.

        Strategy: Always check Redis first for latest model to ensure consistency
        across multiple workers on different machines.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Connect to Redis via PositionQueue abstraction
        self.queue = PositionQueue(self.redis_url)
        self.logger.info(f"✓ Connected to Redis")

        # PRIORITY 1: Check Redis for latest model (ensures consistency across workers)
        self.logger.info("Checking Redis for latest model...")
        model_data = self.queue.pull_model(timeout=0)

        if model_data:
            try:
                iteration = model_data['metadata'].get('iteration', 0)
                timestamp = model_data.get('timestamp', '')
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_v{iteration}.pt')
                torch.save({
                    'model_state': model_data['model_state'],
                    'iteration': iteration,
                    'metadata': model_data['metadata'],
                    'timestamp': timestamp
                }, checkpoint_path)
                with self.version_lock:
                    self.current_model_version.value = iteration
                    if timestamp:
                        self.current_model_timestamp.value = timestamp.encode('utf-8')
                self.logger.info(f"✓ Loaded model from Redis (v{iteration}, {timestamp})")
                return
            except Exception as e:
                self.logger.warning(f"Failed to save model from Redis: {e}")

        # PRIORITY 2: Fallback to local checkpoint if Redis unavailable
        import glob
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'model_v*.pt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
                version = checkpoint.get('iteration', 0)
                timestamp = checkpoint.get('timestamp', '')
                with self.version_lock:
                    self.current_model_version.value = version
                    if timestamp:
                        self.current_model_timestamp.value = timestamp.encode('utf-8')
                self.logger.info(f"✓ Loaded local checkpoint (v{version}, {timestamp}) - Redis was empty")
                self.logger.warning("⚠️  Using local checkpoint - may not be latest across workers!")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load local checkpoint: {e}")

        # PRIORITY 3: Create random model (cold start)
        self.logger.info("No trained model found, creating random initialization...")
        model = GomokuNet.from_preset(self.model_preset, board_size=15, device='cpu')
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model_v0.pt')
        torch.save({
            'model_state': model.state_dict(),
            'iteration': 0,
            'metadata': {'note': 'Random initialization'}
        }, checkpoint_path)
        with self.version_lock:
            self.current_model_version.value = 0
        self.logger.info(f"✓ Created model v0: {model.get_model_size():,} parameters")

    def accumulator_thread(self):
        """Accumulator thread (batches positions and pushes to Redis)."""
        batch = []

        while self.running.value:
            try:
                game_positions = self.game_queue.get(timeout=1)
                batch.extend(game_positions)
                self.accumulated_positions.value = len(batch)  # Update counter

                if len(batch) >= self.positions_per_push:
                    self.queue.push_positions(batch)
                    self.logger.info(f"Pushed batch: {len(batch)} positions")
                    self.batches_pushed.value += 1
                    self.total_positions_pushed.value += len(batch)
                    self.last_batch_time.value = time.time()
                    self.check_model_event.set()
                    batch = []
                    self.accumulated_positions.value = 0  # Reset counter

            except Exception:
                pass

        # Push remaining
        if batch:
            self.queue.push_positions(batch)
            self.logger.info(f"Pushed final batch: {len(batch)} positions")
            self.accumulated_positions.value = 0  # Reset counter

    def model_updater_thread(self):
        """Model updater thread (downloads models from Redis).

        Uses efficient timestamp-based polling:
        1. Periodically check lightweight timestamp (few bytes)
        2. Only download full model (~50MB) if timestamp changed
        """
        updates_since_check = 0
        last_timestamp = None  # Track last seen timestamp

        while self.running.value:
            self.check_model_event.wait(timeout=10)
            self.check_model_event.clear()

            updates_since_check += 1
            if updates_since_check < self.model_update_frequency:
                continue

            updates_since_check = 0

            try:
                # STEP 1: Check timestamp (lightweight, ~10 bytes)
                current_timestamp = self.queue.get_model_timestamp()
                if not current_timestamp:
                    continue  # No model in Redis yet

                # STEP 2: Compare timestamps
                if current_timestamp == last_timestamp:
                    continue  # Model hasn't changed, skip download

                # STEP 3: Download model only if timestamp changed
                model_data = self.queue.pull_model(timeout=0)
                if not model_data:
                    continue

                iteration = model_data['metadata'].get('iteration', 0)

                if iteration <= self.current_model_version.value:
                    # Timestamp changed but version didn't (shouldn't happen)
                    last_timestamp = current_timestamp
                    continue

                # Save model checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_v{iteration}.pt')
                torch.save({
                    'model_state': model_data['model_state'],
                    'iteration': iteration,
                    'metadata': model_data['metadata'],
                    'timestamp': current_timestamp
                }, checkpoint_path)

                with self.version_lock:
                    self.current_model_version.value = iteration
                    self.current_model_timestamp.value = current_timestamp.encode('utf-8')

                # Remember this timestamp
                last_timestamp = current_timestamp

                self.logger.info(f"✓ Updated to model v{iteration} (timestamp: {current_timestamp})")

            except Exception as e:
                self.logger.error(f"Model update failed: {e}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}m ago"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m ago"

    def _format_eta(self, minutes: int) -> str:
        """Format ETA in minutes to human-readable string."""
        if minutes < 1:
            return "< 1 min"
        elif minutes < 60:
            return f"~{minutes} min"
        else:
            hours = minutes // 60
            mins = minutes % 60
            return f"~{hours}h {mins}m"

    def _get_runtime_string(self) -> str:
        """Get formatted runtime string."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}", elapsed

    def _build_header(self) -> list:
        """Build dashboard header section."""
        lines = []
        lines.append("=" * 80)
        lines.append("  AlphaGomoku Distributed Self-Play (MULTIPROCESS)")
        lines.append("=" * 80)

        runtime_str, _ = self._get_runtime_string()

        # Get model version display
        model_timestamp = self.current_model_timestamp.value
        if model_timestamp:
            try:
                timestamp_str = model_timestamp.decode('utf-8')
                # Format: "2025-12-02T22:30:45.123456" -> "2025-12-02 22:30:45"
                if 'T' in timestamp_str:
                    date_part, time_part = timestamp_str.split('T')
                    time_clean = time_part.split('.')[0]  # Remove microseconds
                    model_display = f"{date_part} {time_clean}"
                else:
                    model_display = timestamp_str
            except:
                model_display = f"v{self.current_model_version.value}"
        else:
            model_display = f"v{self.current_model_version.value}"

        lines.append(f"\n⏱  Runtime: {runtime_str}  |  Model: {model_display}")
        return lines

    def _build_local_queue_stats(self) -> tuple:
        """Build local queue statistics section."""
        lines = []
        games_in_queue = self.game_queue.qsize() if hasattr(self.game_queue, 'qsize') else 0
        lines.append(f"\n📦 Local Queue: {games_in_queue} games (unbatched)")
        return lines, games_in_queue

    def _build_redis_stats(self) -> list:
        """Build Redis statistics section."""
        lines = []
        lines.append(f"\n📊 Redis Stats:")

        try:
            redis_stats = self.queue.get_stats()
            redis_queue_size = redis_stats['queue_size']
            positions_in_redis = redis_stats['positions_pushed'] - redis_stats['positions_pulled']

            lines.append(f"   Queue: {redis_queue_size} batches ({positions_in_redis:,} positions waiting)")
            lines.append(f"   Batches pushed: {self.batches_pushed.value}")
            lines.append(f"   Total positions pushed: {self.total_positions_pushed.value:,}")
            lines.append(f"   Positions pulled by trainer: {redis_stats['positions_pulled']:,}")
        except Exception:
            # Fallback if Redis query fails
            lines.append(f"   Batches pushed: {self.batches_pushed.value}")
            lines.append(f"   Total positions pushed: {self.total_positions_pushed.value:,}")

        return lines

    def _build_worker_stats(self):
        """Build worker statistics section."""
        lines = []
        lines.append(f"\n👷 Workers ({self.num_workers} processes):")

        total_games = 0
        total_positions = 0
        status_counts = {}

        with self.stats_lock:
            for worker_name in sorted(self.worker_stats_dict.keys()):
                stats = self.worker_stats_dict[worker_name]
                games = stats['games_generated']
                positions = stats['positions_generated']
                total_time = stats['total_time']
                status = stats['status']
                version = stats['model_version']
                errors = stats['errors']

                status_counts[status] = status_counts.get(status, 0) + 1
                total_games += games
                total_positions += positions

                avg_time = total_time / games if games > 0 else 0
                lines.append(
                    f"   {worker_name}: {games} games, {positions} pos, "
                    f"v{version}, {avg_time:.1f}s/game, {status}, errors:{errors}"
                )

        return lines, total_games, total_positions, status_counts

    def _build_summary_stats(self, total_games: int, total_positions: int, status_counts: dict) -> list:
        """Build summary statistics section."""
        lines = []
        lines.append(f"\n📈 Summary:")
        lines.append(f"   Total games: {total_games:,}")
        lines.append(f"   Total positions: {total_positions:,}")

        if total_games > 0:
            lines.append(f"   Avg positions/game: {total_positions / total_games:.1f}")
        else:
            lines.append("   Avg positions/game: N/A")

        status_str = ", ".join([f"{status}: {count}" for status, count in sorted(status_counts.items())])
        lines.append(f"   Worker status: {status_str}")

        return lines

    def _build_throughput_stats(self, elapsed: float, total_games: int, total_positions: int, games_in_queue: int) -> list:
        """Build throughput and prediction statistics section."""
        lines = []

        if elapsed <= 0 or total_games <= 0:
            return lines

        games_per_hour = (total_games / elapsed) * 3600
        positions_per_hour = (total_positions / elapsed) * 3600
        lines.append(f"   Throughput: {games_per_hour:.1f} games/hour, {positions_per_hour:,.0f} positions/hour")

        # Get actual accumulated positions from accumulator thread
        accumulated_positions = self.accumulated_positions.value
        positions_until_next_batch = max(0, self.positions_per_push - accumulated_positions)

        # Next batch ETA
        if positions_per_hour > 0:
            hours_until_batch = positions_until_next_batch / positions_per_hour
            minutes_until_batch = int(hours_until_batch * 60)
            eta_str = self._format_eta(minutes_until_batch)
            lines.append(f"   Next batch: {accumulated_positions}/{self.positions_per_push} positions ready, ETA {eta_str}")
        else:
            lines.append(f"   Next batch: {accumulated_positions}/{self.positions_per_push} positions ready")

        # Time since last batch
        if self.last_batch_time.value > 0:
            time_since_last = time.time() - self.last_batch_time.value
            time_str = self._format_duration(time_since_last)
            lines.append(f"   Last batch pushed: {time_str}")

        return lines

    def _build_footer(self) -> list:
        """Build dashboard footer section."""
        lines = []
        lines.append(f"\n💾 Log file: {self.log_file}")
        lines.append(f"⌨  Press Ctrl+C to stop")
        lines.append("=" * 80)
        return lines

    def print_stats(self):
        """Print dashboard (flicker-free)."""
        lines = []

        # Build all sections
        lines.extend(self._build_header())

        local_queue_lines, games_in_queue = self._build_local_queue_stats()
        lines.extend(local_queue_lines)

        lines.extend(self._build_redis_stats())

        worker_lines, total_games, total_positions, status_counts = self._build_worker_stats()
        lines.extend(worker_lines)

        lines.extend(self._build_summary_stats(total_games, total_positions, status_counts))

        _, elapsed = self._get_runtime_string()
        lines.extend(self._build_throughput_stats(elapsed, total_games, total_positions, games_in_queue))

        lines.extend(self._build_footer())

        # Single atomic print
        output = "\n".join(lines)
        print("\033[H\033[J" + output, flush=True)

    def run(self):
        """Run the manager."""
        self.setup()

        # Startup info
        print("\n" + "=" * 70)
        print("  AlphaGomoku Distributed Self-Play Manager (MULTIPROCESS)")
        print("=" * 70)
        print(f"✓ Log file: {self.log_file}")
        print(f"✓ Workers: {self.num_workers} PROCESSES (bypasses GIL!)")
        print(f"✓ Device: {self.device}")
        print(f"✓ NN batch size (MCTS inference): {self.mcts_batch_size}")
        print(f"✓ Expected CPU usage: ~{self.num_workers * 100}%")
        print("\nStarting dashboard in 2 seconds...")
        print("=" * 70)
        time.sleep(2)

        print("\033[2J\033[H", flush=True)

        # Start threads
        stats_thread = threading.Thread(
            target=lambda: (time.sleep(0), [self.print_stats() or time.sleep(5) for _ in iter(lambda: self.running.value, False)]),
            daemon=True
        )
        accumulator = threading.Thread(target=self.accumulator_thread, daemon=True)
        updater = threading.Thread(target=self.model_updater_thread, daemon=True)

        # Start worker PROCESSES
        worker_processes = []
        self.running.value = True
        self.start_time = time.time()

        self.logger.info(f"Starting {self.num_workers} worker PROCESSES...")

        for i in range(1, self.num_workers + 1):
            process = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.model_preset,
                    self.mcts_simulations,
                    self.device,
                    self.mcts_batch_size,
                    self.checkpoint_dir,
                    self.game_queue,
                    self.worker_stats_dict,
                    self.stats_lock,
                    self.current_model_version,
                    self.running,
                    self.log_file
                )
            )
            process.start()
            worker_processes.append(process)

        # Start threads
        stats_thread.start()
        accumulator.start()
        updater.start()

        self.logger.info("All workers and threads started")

        try:
            # Keep main thread alive
            for process in worker_processes:
                process.join()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
            self.running.value = False

            for process in worker_processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()

            print("✓ Stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed self-play manager (MULTIPROCESS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter Guide:
  --positions-per-push  How many positions to accumulate before pushing to Redis
                        (Default: 500 positions = ~10 games per Redis push)
                        This is what the GPU trainer will consume!

  --mcts-batch-size     Neural network batch size for MCTS inference
                        Higher = faster MCTS, more VRAM
                        (Default: 128 positions per NN forward pass)
"""
    )
    parser.add_argument('--redis-url', type=str, required=True, help='Redis connection URL')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--model-preset', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--mcts-simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--positions-per-push', type=int, default=500,
                        help='Positions to accumulate before pushing to Redis')
    parser.add_argument('--model-update-frequency', type=int, default=5,
                        help='Check for new model every N Redis pushes')
    parser.add_argument('--mcts-batch-size', type=int, default=128,
                        help='NN inference batch size for MCTS')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')

    args = parser.parse_args()

    # Set multiprocessing start method to 'spawn' (required for CUDA/MPS compatibility)
    mp.set_start_method('spawn', force=True)

    manager = DistributedSelfPlayManager(
        num_workers=args.num_workers,
        redis_url=args.redis_url,
        model_preset=args.model_preset,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
        positions_per_push=args.positions_per_push,
        model_update_frequency=args.model_update_frequency,
        mcts_batch_size=args.mcts_batch_size,
        checkpoint_dir=args.checkpoint_dir
    )

    manager.run()


if __name__ == "__main__":
    main()
