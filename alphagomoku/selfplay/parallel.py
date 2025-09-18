"""Parallel self-play workers for faster data generation"""

import multiprocessing as mp
from contextlib import nullcontext
from typing import List, Optional
from tqdm import tqdm

import torch

from ..model.network import GomokuNet
from .selfplay import SelfPlayData, SelfPlayWorker


def worker_process(
    model_state_dict,
    board_size,
    mcts_simulations,
    adaptive_sims,
    batch_size,
    num_games,
    worker_id,
    difficulty="medium",
    model_config: Optional[dict] = None,
):
    """Worker process for parallel self-play"""
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

    # Create worker with unified search support
    worker = SelfPlayWorker(
        model=model,
        board_size=board_size,
        mcts_simulations=mcts_simulations,
        adaptive_sims=adaptive_sims,
        batch_size=batch_size,
        difficulty=difficulty,
    )

    # Generate games
    all_data = []
    game_pbar = tqdm(range(num_games), desc=f"Worker {worker_id}", leave=False, unit="game")
    for i in game_pbar:
        game_data = worker.generate_game()
        all_data.extend(game_data)
        game_pbar.set_postfix({'positions': len(all_data)})
    game_pbar.close()
    return all_data


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

    def _build_worker_args(
        self, num_games: int, num_workers: int, model_state_dict: dict, model_config: dict
    ) -> List[tuple]:
        games_per_worker = num_games // num_workers
        remaining_games = num_games % num_workers

        worker_args = []
        for worker_id in range(num_workers):
            worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            if worker_games <= 0:
                continue
            worker_args.append(
                (
                    model_state_dict,
                    self.board_size,
                    self.mcts_simulations,
                    self.adaptive_sims,
                    self.batch_size,
                    worker_games,
                    worker_id,
                    self.difficulty,
                    model_config,
                )
            )
        return worker_args

    def generate_data(self, num_games: int) -> List[SelfPlayData]:
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
            data: List[SelfPlayData] = []
            for _ in range(num_games):
                data.extend(worker.generate_game())
            return data

        model_state_dict = self._prepare_model_state_dict()
        model_config = self._extract_model_config()
        worker_args = self._build_worker_args(
            num_games, num_workers, model_state_dict, model_config
        )

        if not worker_args:
            return []

        pool_obj = mp.Pool(processes=len(worker_args))
        try:
            with pool_obj as pool:
                results = pool.starmap(worker_process, worker_args)
        except AttributeError:
            with nullcontext(pool_obj) as pool:
                try:
                    results = pool.starmap(worker_process, worker_args)
                finally:
                    close = getattr(pool_obj, "close", None)
                    if callable(close):
                        close()
                    join = getattr(pool_obj, "join", None)
                    if callable(join):
                        join()

        all_data: List[SelfPlayData] = []
        for result in results:
            all_data.extend(result)

        return all_data

    def generate_batch(self, num_games: int) -> List[SelfPlayData]:
        """Backward-compatible alias for generate_data."""

        return self.generate_data(num_games)
