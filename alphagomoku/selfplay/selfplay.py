from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..env.gomoku_env import GomokuEnv
from ..mcts.adaptive import AdaptiveSimulator
from ..search import UnifiedSearch


@dataclass
class SelfPlayData:
    """Single training example from self-play"""

    state: np.ndarray  # Board state (5, 15, 15)
    policy: np.ndarray  # MCTS policy (225,)
    value: float  # Game outcome from this position


class SelfPlayWorker:
    """Generates self-play training data"""

    def __init__(
        self,
        model,
        board_size: int = 15,
        mcts_simulations: int = 800,
        adaptive_sims: bool = True,
        batch_size: int = 64,
        difficulty: str = "medium",
    ):
        self.model = model
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.adaptive_sims = adaptive_sims
        self.difficulty = difficulty
        self.env = GomokuEnv(board_size)

        # Use UnifiedSearch instead of plain MCTS for tactical training
        self.search = UnifiedSearch(model, self.env, difficulty=difficulty)
        # Keep reference to MCTS for adaptive simulations
        self.mcts = self.search.mcts
        self.adaptive_simulator = AdaptiveSimulator() if adaptive_sims else None
        # Ensure the initial simulation budget matches requested value
        # (UnifiedSearch picks defaults by difficulty; override here)
        self.mcts.num_simulations = self.mcts_simulations

    def generate_game(self, temperature_moves: int = 8, max_moves: int | None = None) -> List[SelfPlayData]:
        """Generate one self-play game and return training examples"""
        self.env.reset()
        self.mcts.root = None  # Reset tree
        game_data = []
        move_count = 0

        while not self.env.game_over:
            if max_moves is not None and move_count >= max_moves:
                break
            # Get current state for neural network
            state_tensor = self._get_state_tensor()

            # Adaptive simulation count
            if self.adaptive_sims and self.adaptive_simulator:
                sims = self.adaptive_simulator.get_simulations(
                    move_count, self.env.board
                )
                self.mcts.num_simulations = sims

            # Run unified search (MCTS + TSS + Endgame) to get policy
            temperature = 1.0 if move_count < temperature_moves else 0.0
            reuse_tree = move_count > 0  # Reuse tree after first move

            # Use unified search for enhanced tactical play
            search_result = self.search.search(self.env.board, temperature, reuse_tree)
            policy = search_result.action_probs
            # Store training example (value will be filled after game ends)
            game_data.append(
                SelfPlayData(
                    state=state_tensor.numpy(), policy=policy, value=0.0  # Placeholder
                )
            )

            # Select and make move
            if temperature > 0:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)

            # Reuse subtree for next search (unified search delegates to MCTS)
            self.search.reuse_subtree(action)
            self.env.step(action)
            move_count += 1

        # Fill in game outcome values
        game_result = self.env.winner  # 1, -1, or 0
        for i, data in enumerate(game_data):
            # Value from perspective of player who made the move
            player_at_move = 1 if i % 2 == 0 else -1
            data.value = game_result * player_at_move

        return game_data

    def _get_state_tensor(self) -> torch.Tensor:
        """Convert current environment state to neural network input"""
        board_size = self.env.board_size

        # Channel 0: Current player's stones
        own_stones = (self.env.board == self.env.current_player).astype(np.float32)

        # Channel 1: Opponent's stones
        opp_stones = (self.env.board == -self.env.current_player).astype(np.float32)

        # Channel 2: Last move
        last_move = np.zeros((board_size, board_size), dtype=np.float32)
        if self.env.last_move[0] >= 0:
            last_move[self.env.last_move[0], self.env.last_move[1]] = 1.0

        # Channel 3: Side to move
        side_to_move = np.ones((board_size, board_size), dtype=np.float32)

        # Channel 4: Pattern maps (placeholder)
        pattern_maps = np.zeros((board_size, board_size), dtype=np.float32)

        return torch.FloatTensor(
            np.stack([own_stones, opp_stones, last_move, side_to_move, pattern_maps])
        )

    def generate_batch(self, num_games: int) -> List[SelfPlayData]:
        """Generate multiple self-play games"""
        all_data = []
        game_pbar = tqdm(
            range(num_games), desc="Self-play", leave=False, unit="game", position=1
        )
        for i in game_pbar:
            game_data = self.generate_game()
            all_data.extend(game_data)
            game_pbar.set_postfix({'positions': len(all_data)})
        game_pbar.close()
        return all_data
