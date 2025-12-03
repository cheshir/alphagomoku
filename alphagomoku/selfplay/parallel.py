"""Parallel self-play workers for faster data generation"""

import multiprocessing as mp
import time
import os
import sys
import psutil
from dataclasses import dataclass
from typing import List, Optional

from tqdm.auto import tqdm

import torch

from ..model.network import GomokuNet
from ..utils.debug_logger import DebugLogger, NoOpLogger
from .selfplay import SelfPlayData, SelfPlayWorker


_GLOBAL_WORKER: Optional[SelfPlayWorker] = None
_WORKER_STATS: Optional[dict] = None  # Shared stats dict
_WORKER_DEVICE: Optional[str] = None  # Device this worker is using


@dataclass
class WorkerStats:
    """Statistics for a single worker."""
    worker_id: int
    status: str  # 'init', 'generating', 'idle', 'failed'
    games_completed: int
    positions_generated: int
    ram_mb: float
    gpu_mem_mb: float
    game_start_time: float
    total_time: float
    current_game_id: int


def _get_memory_stats(device=None):
    """Get current process memory usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024

    gpu_mem_mb = 0.0
    try:
        if torch.cuda.is_available() and device is not None:
            # Extract GPU ID from device string (e.g., 'cuda:0' -> 0)
            if isinstance(device, str) and device.startswith('cuda'):
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                gpu_mem_mb = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
            else:
                gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
    except:
        pass

    return ram_mb, gpu_mem_mb


def _worker_initializer(
    model_state_dict,
    board_size,
    mcts_simulations,
    adaptive_sims,
    batch_size,
    difficulty,
    model_config: Optional[dict] = None,
    stats_dict: Optional[dict] = None,
):
    """Initialise a process-local self-play worker reused across tasks."""

    global _GLOBAL_WORKER, _WORKER_STATS, _WORKER_DEVICE

    # Store shared stats dict
    _WORKER_STATS = stats_dict
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0

    # Update status to init
    if _WORKER_STATS is not None:
        ram_mb, gpu_mem_mb = _get_memory_stats()  # No device yet
        _WORKER_STATS[worker_id] = {
            'status': 'init',
            'games_completed': 0,
            'positions_generated': 0,
            'ram_mb': ram_mb,
            'gpu_mem_mb': gpu_mem_mb,
            'game_start_time': 0.0,
            'total_time': 0.0,
            'current_game_id': -1,
        }

    model_kwargs = dict(model_config or {})
    model_kwargs.setdefault("board_size", board_size)
    model = GomokuNet(**model_kwargs)

    cpu_state = {}
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        elif hasattr(value, "to") and callable(value.to):
            cpu_state[key] = value.to("cpu")
        elif hasattr(value, "detach") and callable(value.detach):
            detached = value.detach()
            cpu_state[key] = (
                detached.to("cpu") if hasattr(detached, "to") else detached
            )
        else:
            cpu_state[key] = value

    incompatible = model.load_state_dict(cpu_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Model state dict is incompatible with worker architecture."
        )
    model.eval()

    # Try to use GPU (CUDA or MPS) for hardware acceleration, fall back to CPU if unavailable
    device_used = 'cpu'
    try:
        if torch.cuda.is_available():
            # Distribute workers across available CUDA devices
            num_gpus = torch.cuda.device_count()
            # Use worker_id to assign GPU (round-robin distribution)
            gpu_id = (worker_id - 1) % num_gpus  # worker_id starts from 1
            device = f'cuda:{gpu_id}'
            model = model.to(device)
            device_used = device
            _WORKER_DEVICE = device  # Store device globally for memory stats
            print(f"[Worker-{worker_id}] Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)} ({num_gpus} GPUs available)", flush=True)
        elif torch.backends.mps.is_available():
            model = model.to('mps')
            device_used = 'mps'
            _WORKER_DEVICE = 'mps'
            print(f"[Worker-{worker_id}] Using MPS device for acceleration", flush=True)
        else:
            model = model.cpu()
            _WORKER_DEVICE = 'cpu'
            print(f"[Worker-{worker_id}] No GPU available, using CPU", flush=True)
    except Exception as e:
        print(f"[Worker-{worker_id}] GPU initialization failed ({e}), falling back to CPU", flush=True)
        model = model.cpu()
        _WORKER_DEVICE = 'cpu'

    _GLOBAL_WORKER = SelfPlayWorker(
        model=model,
        board_size=board_size,
        mcts_simulations=mcts_simulations,
        adaptive_sims=adaptive_sims,
        batch_size=batch_size,
        difficulty=difficulty,
    )

    # Log device confirmation
    actual_device = next(model.parameters()).device
    print(f"[Worker-{worker_id}] Model loaded on device: {actual_device}", flush=True)

    # Update status to idle
    if _WORKER_STATS is not None:
        ram_mb, gpu_mem_mb = _get_memory_stats(_WORKER_DEVICE)
        _WORKER_STATS[worker_id]['status'] = 'idle'
        _WORKER_STATS[worker_id]['ram_mb'] = ram_mb
        _WORKER_STATS[worker_id]['gpu_mem_mb'] = gpu_mem_mb


def _play_single_game(task_data) -> List[SelfPlayData]:
    """Generate a single self-play game via the process-local worker."""
    game_id, debug_enabled = task_data

    global _WORKER_STATS, _WORKER_DEVICE

    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialised before task execution")

    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0

    # Update status to generating
    start_time = time.time()
    if _WORKER_STATS is not None:
        ram_mb, gpu_mem_mb = _get_memory_stats(_WORKER_DEVICE)
        _WORKER_STATS[worker_id].update({
            'status': 'generating',
            'current_game_id': game_id,
            'game_start_time': start_time,
            'ram_mb': ram_mb,
            'gpu_mem_mb': gpu_mem_mb,
        })

    # Create logger instance for this specific task
    if debug_enabled:
        logger = DebugLogger(enabled=True)
    else:
        logger = NoOpLogger()

    logger.debug(f"Starting game {game_id}")

    try:
        result = _GLOBAL_WORKER.generate_game()
        game_time = time.time() - start_time

        logger.debug(f"Completed game {game_id}, {len(result)} positions")

        # Update stats after completion
        if _WORKER_STATS is not None:
            ram_mb, gpu_mem_mb = _get_memory_stats(_WORKER_DEVICE)
            _WORKER_STATS[worker_id].update({
                'status': 'idle',
                'games_completed': _WORKER_STATS[worker_id]['games_completed'] + 1,
                'positions_generated': _WORKER_STATS[worker_id]['positions_generated'] + len(result),
                'total_time': _WORKER_STATS[worker_id]['total_time'] + game_time,
                'current_game_id': -1,
                'ram_mb': ram_mb,
                'gpu_mem_mb': gpu_mem_mb,
            })

        return result

    except Exception as e:
        logger.debug(f"ERROR in game {game_id}: {e}")

        # Update status to failed
        if _WORKER_STATS is not None:
            _WORKER_STATS[worker_id]['status'] = 'failed'
            _WORKER_STATS[worker_id]['current_game_id'] = -1

        raise


class ParallelSelfPlay:
    """Parallel self-play data generation"""

    def __init__(
        self,
        model,
        board_size: int = 15,
        mcts_simulations: int = 800,
        adaptive_sims: bool = True,
        batch_size: int = 32,
        num_workers: int = None,
        difficulty: str = "medium",
    ):
        self.model = model
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.adaptive_sims = adaptive_sims
        self.difficulty = difficulty
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count()

    def _resolve_num_workers(self) -> int:
        workers = self.num_workers
        if not workers or workers <= 0:
            workers = mp.cpu_count()
        return max(1, workers)

    def _prepare_model_state_dict(self) -> dict:
        state_dict = {}
        for key, value in self.model.state_dict().items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach().to("cpu")
            elif hasattr(value, "to"):
                tensor = value.to("cpu")
            else:
                tensor = value
            state_dict[key] = tensor
        return state_dict

    def _extract_model_config(self) -> dict:
        config = {}
        if hasattr(self.model, "board_size"):
            config["board_size"] = getattr(self.model, "board_size")
        if hasattr(self.model, "channels"):
            config["channels"] = getattr(self.model, "channels")
        if hasattr(self.model, "blocks"):
            config["num_blocks"] = len(getattr(self.model, "blocks"))
        return config

    def _print_worker_stats(self, stats_dict, num_workers, batch_start_time):
        """Print worker statistics in a Jupyter-friendly format."""
        current_time = time.time()
        elapsed = current_time - batch_start_time

        print("\n" + "=" * 80, flush=True)
        print(f"⏱️  Elapsed: {elapsed:.1f}s | Workers: {num_workers}", flush=True)
        print("=" * 80, flush=True)

        for worker_id in range(1, num_workers + 1):
            if worker_id not in stats_dict:
                continue

            stats = stats_dict[worker_id]
            status = stats['status']
            games = stats['games_completed']
            positions = stats['positions_generated']
            ram = stats['ram_mb']
            gpu = stats['gpu_mem_mb']
            total_time = stats['total_time']
            current_game = stats['current_game_id']
            game_start = stats['game_start_time']

            # Calculate rates
            games_per_s = games / total_time if total_time > 0 else 0
            pos_per_s = positions / total_time if total_time > 0 else 0

            # Calculate ETA for current game
            eta_str = "N/A"
            if status == 'generating' and game_start > 0:
                game_elapsed = current_time - game_start
                # Estimate based on previous avg: ~30-40s per game
                avg_game_time = total_time / games if games > 0 else 35.0
                eta = max(0, avg_game_time - game_elapsed)
                eta_str = f"{eta:.0f}s"

            # Status emoji
            status_emoji = {
                'init': '🔵',
                'generating': '🟢',
                'idle': '⚪',
                'failed': '🔴',
            }.get(status, '❓')

            print(f"\n{status_emoji} Worker-{worker_id} | Status: {status.upper():<10} | Game: {current_game if current_game >= 0 else 'N/A'}", flush=True)
            print(f"   📊 Games: {games} | Positions: {positions}", flush=True)
            print(f"   💾 RAM: {ram:.0f}MB | GPU: {gpu:.0f}MB", flush=True)
            print(f"   ⚡ Speed: {games_per_s:.2f} games/s | {pos_per_s:.1f} pos/s", flush=True)
            if status == 'generating':
                print(f"   ⏳ ETA: {eta_str}", flush=True)

        print("=" * 80, flush=True)

    def generate_data(self, num_games: int, debug: bool = False) -> List[SelfPlayData]:
        """Generate self-play data using sequential or parallel workers."""

        if num_games <= 0:
            return []

        num_workers = self._resolve_num_workers()

        if num_workers == 1:
            worker = SelfPlayWorker(
                model=self.model,
                board_size=self.board_size,
                mcts_simulations=self.mcts_simulations,
                adaptive_sims=self.adaptive_sims,
                batch_size=self.batch_size,
                difficulty=self.difficulty,
            )
            return worker.generate_batch(num_games)

        model_state_dict = self._prepare_model_state_dict()
        model_config = self._extract_model_config()

        # Create shared manager for worker stats
        manager = mp.Manager()
        stats_dict = manager.dict()

        init_args = (
            model_state_dict,
            self.board_size,
            self.mcts_simulations,
            self.adaptive_sims,
            self.batch_size,
            self.difficulty,
            model_config,
            stats_dict,
        )

        positions_generated = 0
        all_data: List[SelfPlayData] = []
        batch_start_time = time.time()

        # Use spawn context for CUDA compatibility
        # CUDA cannot be initialized in forked processes
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(
            processes=num_workers, initializer=_worker_initializer, initargs=init_args
        )

        # Give workers time to initialize
        print(f"\n🚀 Starting {num_workers} workers...", flush=True)
        time.sleep(2)  # Wait for initialization

        try:
            # Initial stats print
            self._print_worker_stats(stats_dict, num_workers, batch_start_time)

            # Create tasks with debug flag
            tasks = [(game_id, debug) for game_id in range(num_games)]

            games_completed = 0
            last_stats_print = time.time()
            stats_interval = 10  # Print stats every 10 seconds

            # Use imap_unordered for real-time updates
            for game_data in pool.imap_unordered(_play_single_game, tasks):
                all_data.extend(game_data)
                positions_generated += len(game_data)
                games_completed += 1

                # Print stats periodically
                current_time = time.time()
                if current_time - last_stats_print >= stats_interval:
                    self._print_worker_stats(stats_dict, num_workers, batch_start_time)
                    last_stats_print = current_time

                # Also print on game completion
                print(f"✅ Game {games_completed}/{num_games} complete: {len(game_data)} positions", flush=True)

            # Final stats print
            self._print_worker_stats(stats_dict, num_workers, batch_start_time)

        finally:
            # CRITICAL: Force immediate cleanup of worker processes
            # This ensures MPS memory is released before training starts
            pool.close()
            pool.join()  # Wait for all workers to exit
            pool.terminate()  # Force kill any lingering processes
            pool.join()  # Wait for termination to complete

            # Give OS time to reclaim memory from terminated processes
            time.sleep(0.5)

            print(f"\n✅ All workers completed. Total: {positions_generated} positions\n", flush=True)

        return all_data

    def generate_batch(self, num_games: int, debug: bool = False) -> List[SelfPlayData]:
        """Backward-compatible alias for generate_data."""

        return self.generate_data(num_games, debug)
