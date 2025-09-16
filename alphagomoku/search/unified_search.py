"""Unified search interface combining MCTS, TSS, and endgame solver."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..mcts.mcts import MCTS
from ..tss import tss_search, Position as TSSPosition
from ..endgame import endgame_search, EndgamePosition, should_use_endgame_solver
from ..env.gomoku_env import GomokuEnv


@dataclass
class SearchResult:
    """Result of unified search."""

    action_probs: np.ndarray  # Action probability distribution
    best_move: Optional[Tuple[int, int]]  # Best move (row, col)
    search_method: str  # 'endgame', 'tss', 'mcts'
    is_forced: bool  # Whether the move is forced
    evaluation: Optional[float]  # Position evaluation if available
    search_stats: Dict[str, Any]  # Search statistics


class UnifiedSearch:
    """
    Unified search interface that coordinates TSS, MCTS, and endgame solver.

    Search priority:
    1. Endgame solver (when few empty cells remain)
    2. TSS (for forced tactical sequences)
    3. MCTS (for general position evaluation)
    """

    def __init__(self, model, env: GomokuEnv, difficulty: str = 'medium'):
        self.model = model
        self.env = env
        self.difficulty = difficulty

        # Initialize search components based on difficulty
        config = self._get_difficulty_config(difficulty)

        self.mcts = MCTS(
            model=model,
            env=env,
            cpuct=config['cpuct'],
            num_simulations=config['mcts_sims'],
            batch_size=config.get('batch_size', 32)
        )

        self.tss_config = config['tss']
        self.endgame_config = config['endgame']

    def search(self, state: np.ndarray, temperature: float = 1.0,
               reuse_tree: bool = False) -> SearchResult:
        """
        Perform unified search on the given state.

        Args:
            state: Current board state
            temperature: Temperature for move selection
            reuse_tree: Whether to reuse MCTS tree

        Returns:
            SearchResult with action probabilities and metadata
        """
        # Convert state to different position formats
        stones = int(np.sum(state != 0))
        current_player = 1 if stones % 2 == 0 else -1
        last_move = self._get_last_move(state)

        # Create position objects for TSS and endgame
        tss_position = TSSPosition(
            board=state.copy(),
            current_player=current_player,
            last_move=last_move,
            board_size=self.env.board_size
        )

        endgame_position = EndgamePosition(
            board=state.copy(),
            current_player=current_player,
            last_move=last_move,
            board_size=self.env.board_size
        )

        # 1. Try endgame solver first
        if should_use_endgame_solver(endgame_position, self.difficulty):
            result = endgame_search(
                endgame_position,
                max_depth=self.endgame_config['max_depth'],
                time_limit=self.endgame_config['time_limit']
            )

            if result.best_move is not None:
                return self._create_endgame_result(result, state)

        # 2. Try TSS for tactical sequences
        if self.tss_config['enabled']:
            tss_result = tss_search(
                tss_position,
                depth=self.tss_config['depth'],
                time_cap_ms=self.tss_config['time_cap_ms']
            )

            if tss_result.forced_move is not None:
                return self._create_tss_result(tss_result, state)

        # 3. Fall back to MCTS
        action_probs, visits = self.mcts.search(state, temperature, reuse_tree)
        best_action = np.argmax(action_probs) if action_probs.sum() > 0 else None
        best_move = None
        if best_action is not None:
            best_move = divmod(best_action, self.env.board_size)

        return SearchResult(
            action_probs=action_probs,
            best_move=best_move,
            search_method='mcts',
            is_forced=False,
            evaluation=None,
            search_stats={
                'total_visits': int(visits.sum()) if len(visits) > 0 else 0,
                'method': 'mcts'
            }
        )

    def reuse_subtree(self, action: int):
        """Reuse MCTS subtree after making a move."""
        self.mcts.reuse_subtree(action)

    def _get_difficulty_config(self, difficulty: str) -> Dict[str, Any]:
        """Get configuration parameters for the given difficulty."""
        configs = {
            'easy': {
                'mcts_sims': 48,
                'cpuct': 1.6,
                'tss': {
                    'enabled': False,
                    'depth': 2,
                    'time_cap_ms': 30
                },
                'endgame': {
                    'enabled': False,
                    'max_depth': 10,
                    'time_limit': 0.1
                }
            },
            'medium': {
                'mcts_sims': 384,
                'cpuct': 1.8,
                'tss': {
                    'enabled': True,
                    'depth': 4,
                    'time_cap_ms': 100
                },
                'endgame': {
                    'enabled': True,
                    'max_depth': 16,
                    'time_limit': 0.3
                }
            },
            'strong': {
                'mcts_sims': 1600,
                'cpuct': 1.8,
                'tss': {
                    'enabled': True,
                    'depth': 6,
                    'time_cap_ms': 300
                },
                'endgame': {
                    'enabled': True,
                    'max_depth': 20,
                    'time_limit': 1.0
                }
            }
        }

        return configs.get(difficulty, configs['medium'])

    def _get_last_move(self, state: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Infer the last move from the board state.
        This is a simplified implementation - in practice you'd track this.
        """
        # For now, return None - in a real game this would be tracked
        return None

    def _create_endgame_result(self, endgame_result, state: np.ndarray) -> SearchResult:
        """Convert endgame solver result to SearchResult."""
        action_probs = np.zeros(self.env.board_size * self.env.board_size)

        if endgame_result.best_move:
            row, col = endgame_result.best_move
            action = row * self.env.board_size + col
            action_probs[action] = 1.0

        return SearchResult(
            action_probs=action_probs,
            best_move=endgame_result.best_move,
            search_method='endgame',
            is_forced=endgame_result.is_win or endgame_result.is_loss,
            evaluation=float(endgame_result.evaluation),
            search_stats={
                'nodes_searched': endgame_result.search_stats['nodes_searched'],
                'time_ms': endgame_result.search_stats['time_ms'],
                'depth_reached': endgame_result.search_stats['depth_reached'],
                'is_win': endgame_result.is_win,
                'is_loss': endgame_result.is_loss,
                'depth_to_mate': endgame_result.depth_to_mate,
                'method': 'endgame'
            }
        )

    def _create_tss_result(self, tss_result, state: np.ndarray) -> SearchResult:
        """Convert TSS result to SearchResult."""
        action_probs = np.zeros(self.env.board_size * self.env.board_size)

        if tss_result.forced_move:
            row, col = tss_result.forced_move
            action = row * self.env.board_size + col
            action_probs[action] = 1.0

        return SearchResult(
            action_probs=action_probs,
            best_move=tss_result.forced_move,
            search_method='tss',
            is_forced=True,
            evaluation=1.0 if tss_result.is_forced_win else -1.0 if tss_result.is_forced_defense else 0.0,
            search_stats={
                'nodes_visited': tss_result.search_stats.get('nodes_visited', 0),
                'time_ms': tss_result.search_stats.get('time_ms', 0),
                'reason': tss_result.search_stats.get('reason', 'tss'),
                'is_forced_win': tss_result.is_forced_win,
                'is_forced_defense': tss_result.is_forced_defense,
                'method': 'tss'
            }
        )