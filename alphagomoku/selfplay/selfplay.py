import numpy as np
import torch
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass
from ..env.gomoku_env import GomokuEnv
from ..mcts.mcts import MCTS
from ..mcts.adaptive import AdaptiveSimulator


@dataclass
class SelfPlayData:
    """Single training example from self-play"""
    state: np.ndarray  # Board state (5, 15, 15)
    policy: np.ndarray  # MCTS policy (225,)
    value: float  # Game outcome from this position


class SelfPlayWorker:
    """Generates self-play training data"""
    
    def __init__(self, model, board_size: int = 15, mcts_simulations: int = 800, 
                 adaptive_sims: bool = True, batch_size: int = 64):
        self.model = model
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.adaptive_sims = adaptive_sims
        self.env = GomokuEnv(board_size)
        self.mcts = MCTS(model, self.env, num_simulations=mcts_simulations, batch_size=batch_size)
        self.adaptive_simulator = AdaptiveSimulator() if adaptive_sims else None
    
    def generate_game(self, temperature_moves: int = 8) -> List[SelfPlayData]:
        """Generate one self-play game and return training examples"""
        self.env.reset()
        self.mcts.root = None  # Reset tree
        game_data = []
        move_count = 0
        
        while not self.env.game_over:
            # Get current state for neural network
            state_tensor = self._get_state_tensor()
            
            # Adaptive simulation count
            if self.adaptive_sims and self.adaptive_simulator:
                sims = self.adaptive_simulator.get_simulations(move_count, self.env.board)
                self.mcts.num_simulations = sims
            
            # Run MCTS to get policy
            temperature = 1.0 if move_count < temperature_moves else 0.0
            reuse_tree = move_count > 0  # Reuse tree after first move
            policy, _ = self.mcts.search(self.env.board, temperature, reuse_tree)
            
            # Store training example (value will be filled after game ends)
            game_data.append(SelfPlayData(
                state=state_tensor.numpy(),
                policy=policy,
                value=0.0  # Placeholder
            ))
            
            # Select and make move
            if temperature > 0:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)
            
            # Reuse subtree for next search
            self.mcts.reuse_subtree(action)
            
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
        
        return torch.FloatTensor(np.stack([own_stones, opp_stones, last_move, side_to_move, pattern_maps]))
    
    def generate_batch(self, num_games: int) -> List[SelfPlayData]:
        """Generate multiple self-play games"""
        all_data = []
        for i in range(num_games):
            print(f"\rGame {i+1}/{num_games}", end="", flush=True)
            game_data = self.generate_game()
            all_data.extend(game_data)
        print(f" - Generated {len(all_data)} positions")
        return all_data