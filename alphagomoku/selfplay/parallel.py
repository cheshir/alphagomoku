"""Parallel self-play workers for faster data generation"""

import multiprocessing as mp
from typing import List, Optional

from tqdm import tqdm

import torch

from ..model.network import GomokuNet
from ..utils.debug_logger import DebugLogger, NoOpLogger
from .selfplay import SelfPlayData, SelfPlayWorker


_GLOBAL_WORKER: Optional[SelfPlayWorker] = None


def _worker_initializer(
    model_state_dict,
    board_size,
    mcts_simulations,
    adaptive_sims,
    batch_size,
    difficulty,
    model_config: Optional[dict] = None,
):
    """Initialise a process-local self-play worker reused across tasks."""

    global _GLOBAL_WORKER

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

    # Force CPU-only inference in worker processes to avoid MPS/CUDA context issues
    model = model.cpu()
    for param in model.parameters():
        param.data = param.data.cpu()

    _GLOBAL_WORKER = SelfPlayWorker(
        model=model,
        board_size=board_size,
        mcts_simulations=mcts_simulations,
        adaptive_sims=adaptive_sims,
        batch_size=batch_size,
        difficulty=difficulty,
    )


def _play_single_game(task_data) -> List[SelfPlayData]:
    """Generate a single self-play game via the process-local worker."""
    game_id, debug_enabled = task_data

    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialised before task execution")

    # Create logger instance for this specific task
    if debug_enabled:
        logger = DebugLogger(enabled=True)
    else:
        logger = NoOpLogger()

    logger.debug(f"Starting game {game_id}")

    try:
        result = _GLOBAL_WORKER.generate_game()
        logger.debug(f"Completed game {game_id}, {len(result)} positions")
        return result
    except Exception as e:
        logger.debug(f"ERROR in game {game_id}: {e}")
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

        init_args = (
            model_state_dict,
            self.board_size,
            self.mcts_simulations,
            self.adaptive_sims,
            self.batch_size,
            self.difficulty,
            model_config,
        )

        positions_generated = 0
        all_data: List[SelfPlayData] = []
        
        with mp.Pool(
            processes=num_workers, initializer=_worker_initializer, initargs=init_args
        ) as pool:
            with tqdm(
                total=num_games,
                desc="Self-play",
                unit="game",
                leave=False,
                position=1,
            ) as game_pbar:
                # Create tasks with debug flag
                tasks = [(game_id, debug) for game_id in range(num_games)]

                # Use imap_unordered but without chunking for real-time updates
                for game_data in pool.imap_unordered(_play_single_game, tasks):
                    all_data.extend(game_data)
                    positions_generated += len(game_data)
                    game_pbar.update(1)
                    game_pbar.set_postfix({'positions': positions_generated})

        return all_data

    def generate_batch(self, num_games: int, debug: bool = False) -> List[SelfPlayData]:
        """Backward-compatible alias for generate_data."""

        return self.generate_data(num_games, debug)
