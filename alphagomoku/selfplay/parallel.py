"""Parallel self-play workers for faster data generation"""

import multiprocessing as mp
import numpy as np
import torch
from typing import List
from .selfplay import SelfPlayWorker, SelfPlayData


def worker_process(model_state_dict, board_size, mcts_simulations, adaptive_sims,
                  batch_size, num_games, worker_id, difficulty='medium'):
    """Worker process for parallel self-play"""
    # Recreate model in worker process
    from ..model.network import GomokuNet
    import torch

    model = GomokuNet(board_size=board_size)
    # Ensure loading on CPU to avoid MPS sharing issues
    model.load_state_dict({k: v.to('cpu') if isinstance(v, torch.Tensor) else v
                           for k, v in model_state_dict.items()})
    model.eval()
    
    # Create worker with unified search support
    worker = SelfPlayWorker(
        model=model,
        board_size=board_size,
        mcts_simulations=mcts_simulations,
        adaptive_sims=adaptive_sims,
        batch_size=batch_size,
        difficulty=difficulty
    )
    
    # Generate games
    all_data = []
    for i in range(num_games):
        print(f"Worker {worker_id}: Game {i+1}/{num_games}")
        game_data = worker.generate_game()
        all_data.extend(game_data)
    
    return all_data


class ParallelSelfPlay:
    """Parallel self-play data generation"""

    def __init__(self, model, board_size: int = 15, mcts_simulations: int = 800,
                 adaptive_sims: bool = True, batch_size: int = 32, num_workers: int = None,
                 difficulty: str = 'medium'):
        self.model = model
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.adaptive_sims = adaptive_sims
        self.difficulty = difficulty
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count()
    
    def generate_batch(self, num_games: int) -> List[SelfPlayData]:
        """Generate games in parallel across multiple processes"""
        
        # Distribute games across workers
        games_per_worker = num_games // self.num_workers
        remaining_games = num_games % self.num_workers
        
        # Prepare worker arguments
        worker_args = []
        for i in range(self.num_workers):
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            if worker_games > 0:
                worker_args.append((
                    self.model.state_dict(),
                    self.board_size,
                    self.mcts_simulations,
                    self.adaptive_sims,
                    self.batch_size,
                    worker_games,
                    i,
                    self.difficulty
                ))
        
        # Prepare CPU-only state dict to avoid sharing MPS/CUDA storages
        import torch
        cpu_state = {k: (v.detach().to('cpu') if isinstance(v, torch.Tensor) else v)
                     for k, v in self.model.state_dict().items()}

        # Run parallel workers
        with mp.Pool(len(worker_args)) as pool:
            # Inject cpu_state as first arg into each tuple
            worker_args_with_state = [
                (cpu_state,) + args[1:] if isinstance(args, tuple) else (cpu_state,)
                for args in worker_args
            ]
            # But our worker expects state first; rebuild tuples accordingly
            worker_args = []
            for args in worker_args_with_state:
                # args currently: (cpu_state, board_size, mcts_simulations, adaptive_sims, batch_size, worker_games, i, difficulty)
                worker_args.append(args)

            results = pool.starmap(worker_process, worker_args)
        
        # Combine results
        all_data = []
        for result in results:
            all_data.extend(result)
        
        print(f"Generated {len(all_data)} positions from {num_games} games")
        return all_data
