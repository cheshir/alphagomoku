from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..env.gomoku_env import GomokuEnv
from ..mcts.adaptive import AdaptiveSimulator
from ..search import UnifiedSearch
from ..tss.tss_config import TSSConfig
from .temperature import TemperatureScheduler


@dataclass
class SelfPlayData:
    """Single training example from self-play."""

    state: np.ndarray  # Board state (5, board_size, board_size)
    policy: np.ndarray  # MCTS policy (board_size * board_size,)
    value: float  # Game outcome from this position
    current_player: int = 1  # Player perspective of this sample
    last_move: Optional[Tuple[int, int]] = None  # Last move coordinates if known
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional extra info


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
        epoch: int = 0,
        disable_tqdm: bool = False,
    ):
        self.model = model
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.adaptive_sims = adaptive_sims
        self.difficulty = difficulty
        self.epoch = epoch
        self.disable_tqdm = disable_tqdm
        self.env = GomokuEnv(board_size)

        # Get TSS config for current training epoch
        tss_config = TSSConfig.for_training_epoch(epoch)

        # Use UnifiedSearch instead of plain MCTS for tactical training
        self.search = UnifiedSearch(model, self.env, difficulty=difficulty,
                                    tss_config=tss_config)
        # Keep reference to MCTS for adaptive simulations
        self.mcts = self.search.mcts
        self.adaptive_simulator = AdaptiveSimulator() if adaptive_sims else None
        # Ensure the initial simulation budget matches requested value
        # (UnifiedSearch picks defaults by difficulty; override here)
        self.mcts.num_simulations = self.mcts_simulations

        # Phase 3: Initialize temperature scheduler
        self.temperature_scheduler = TemperatureScheduler(epoch=epoch)

    def generate_game(self, temperature_moves: int = 8, max_moves: int | None = None) -> List[SelfPlayData]:
        """Generate one self-play game and return training examples"""
        self.env.reset()
        self.mcts.root = None  # Reset tree
        game_data = []
        move_count = 0
        last_confidence = 0.0

        while not self.env.game_over:
            if max_moves is not None and move_count >= max_moves:
                # Force end game if max moves reached
                break
            # Get current state for neural network
            state_tensor = self._get_state_tensor()

            # Adaptive simulation count
            if self.adaptive_sims and self.adaptive_simulator:
                sims = self.adaptive_simulator.get_simulations(
                    move_count,
                    self.env.board,
                    confidence=last_confidence,
                )
                self.mcts.num_simulations = sims

            # Phase 3: Run unified search first to get policy
            reuse_tree = move_count > 0  # Reuse tree after first move
            search_result = self.search.search(self.env.board, temperature=1.0, reuse_tree=reuse_tree)
            policy = search_result.action_probs

            # Check if this is a critical/forced position (after getting search result)
            is_critical = search_result.is_forced if hasattr(search_result, 'is_forced') else False

            # Get temperature from scheduler
            temperature = self.temperature_scheduler.get_temperature(
                move_count,
                is_critical=is_critical
            )

            # Apply temperature to policy
            policy = self.temperature_scheduler.apply_temperature(policy, temperature)

            if self.adaptive_sims and self.adaptive_simulator:
                last_confidence = self.adaptive_simulator.get_confidence(policy)

            # Store training example (value will be filled after game ends)
            game_data.append(
                SelfPlayData(
                    state=state_tensor.numpy(),
                    policy=policy,
                    value=0.0,  # Placeholder until game finishes
                    current_player=self.env.current_player,
                    last_move=tuple(int(x) for x in self.env.last_move.tolist())
                    if self.env.last_move.size == 2
                    else None,
                    metadata={"move_index": move_count, "temperature": temperature},
                )
            )

            # Select and make move
            if self.temperature_scheduler.should_sample(temperature):
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

        # Channel 4: Pattern maps (NOW COMPUTED!)
        from ..utils.pattern_detector import get_pattern_features
        pattern_maps = get_pattern_features(self.env.board, self.env.current_player)

        return torch.FloatTensor(
            np.stack([own_stones, opp_stones, last_move, side_to_move, pattern_maps])
        )

    def generate_batch(self, num_games: int) -> List[SelfPlayData]:
        """Generate multiple self-play games"""
        all_data = []

        # Conditionally create progress bar based on disable_tqdm flag
        if not self.disable_tqdm:
            game_pbar = tqdm(
                range(num_games), desc="Self-play", leave=False, unit="game", position=1
            )
        else:
            game_pbar = range(num_games)

        for i in game_pbar:
            game_data = self.generate_game()
            all_data.extend(game_data)
            # Only update postfix if using tqdm
            if not self.disable_tqdm:
                game_pbar.set_postfix({'positions': len(all_data)})

        # Only close if using tqdm
        if not self.disable_tqdm:
            game_pbar.close()

        return all_data
