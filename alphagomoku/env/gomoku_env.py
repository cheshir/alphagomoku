from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GomokuEnv(gym.Env):
    """Gomoku environment following Gymnasium interface"""

    def __init__(self, board_size: int = 15, rule_profile: str = "classic"):
        super().__init__()
        if not isinstance(board_size, int) or board_size <= 0:
            raise ValueError("board_size must be a positive integer")
        self.board_size = board_size
        self.rule_profile = rule_profile

        # Action space: each cell on the board
        self.action_space = spaces.Discrete(board_size * board_size)

        # Observation space: board state + metadata
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-1, high=1, shape=(board_size, board_size), dtype=np.int8
                ),
                "current_player": spaces.Discrete(2),
                "last_move": spaces.Box(
                    low=-1, high=board_size - 1, shape=(2,), dtype=np.int8
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(board_size * board_size,), dtype=np.bool_
                ),
            }
        )

        self.reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1 for first player, -1 for second
        self.last_move = np.array([-1, -1], dtype=np.int8)  # No last move initially
        self.move_count = 0
        self.game_over = False
        self.winner = 0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        # If game already over, remain terminated and return current observation
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {"game_over": True}

        # Validate action bounds
        if not isinstance(action, (int, np.integer)) or not (
            0 <= int(action) < self.board_size * self.board_size
        ):
            self.game_over = True
            return self._get_observation(), -1.0, True, False, {"invalid_move": True}

        row, col = divmod(int(action), self.board_size)

        if self.board[row, col] != 0:
            # Invalid move - terminate game
            self.game_over = True
            return self._get_observation(), -1.0, True, False, {"invalid_move": True}

        # Make move
        self.board[row, col] = self.current_player
        self.last_move = np.array([row, col], dtype=np.int8)
        self.move_count += 1

        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0 if self.current_player == 1 else -1.0
            return self._get_observation(), reward, True, False, {"winner": self.winner}

        # Check for draw
        if self.move_count == self.board_size * self.board_size:
            self.game_over = True
            return self._get_observation(), 0.0, True, False, {"draw": True}

        # Switch player
        self.current_player *= -1

        return self._get_observation(), 0.0, False, False, {}

    def _get_observation(self) -> Dict:
        # Normalize board values and convert to current player's perspective
        normalized = np.clip(self.board, -1, 1)
        obs_board = normalized * self.current_player

        return {
            "board": obs_board,
            "current_player": 0 if self.current_player == 1 else 1,
            "last_move": self.last_move.copy(),
            "action_mask": self._get_action_mask(),
        }

    def _get_action_mask(self) -> np.ndarray:
        """Return mask of legal actions"""
        return self.board.reshape(-1) == 0

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win (5 in a row)"""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Count the placed stone

            # Check positive direction
            r, c = row + dr, col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == player
            ):
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == player
            ):
                count += 1
                r, c = r - dr, c - dc

            if count >= 5:
                return True

        return False

    def get_legal_actions(self) -> np.ndarray:
        """Get list of legal action indices"""
        return np.where(self._get_action_mask())[0]

    @property
    def terminated(self) -> bool:
        """Check if game is terminated (for test compatibility)"""
        return self.game_over

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the board state"""
        if mode == "human":
            symbols = {0: ".", 1: "X", -1: "O"}
            board_str = ""
            for row in self.board:
                board_str += " ".join(symbols[cell] for cell in row) + "\n"
            print(board_str)
        return None
