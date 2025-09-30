"""Game state management."""

import uuid
import time
from typing import Dict, Optional, Tuple
import numpy as np

from .models import GameState, Move


class Game:
    """Represents a single game session."""

    def __init__(self, difficulty: str, player_color: int, board_size: int = 15):
        self.game_id = str(uuid.uuid4())
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.board_size = board_size
        self.difficulty = difficulty
        self.player_color = player_color
        self.ai_color = -player_color
        self.current_player = 1  # Black always starts
        self.status = "in_progress"
        self.last_move: Optional[Tuple[int, int, int]] = None
        self.move_count = 0
        self.player_time = 0.0
        self.ai_time = 0.0
        self.player_move_start: Optional[float] = None
        self.ai_move_start: Optional[float] = None

    def start_player_timer(self):
        """Start timing a player move."""
        self.player_move_start = time.time()

    def stop_player_timer(self):
        """Stop timing a player move."""
        if self.player_move_start:
            self.player_time += time.time() - self.player_move_start
            self.player_move_start = None

    def start_ai_timer(self):
        """Start timing an AI move."""
        self.ai_move_start = time.time()

    def stop_ai_timer(self):
        """Stop timing an AI move."""
        if self.ai_move_start:
            self.ai_time += time.time() - self.ai_move_start
            self.ai_move_start = None

    def make_move(self, row: int, col: int, player: int) -> bool:
        """
        Make a move on the board.

        Returns:
            bool: True if move was valid and made, False otherwise.
        """
        if not self.is_valid_move(row, col):
            return False

        self.board[row, col] = player
        self.last_move = (row, col, player)
        self.current_player = -player
        self.move_count += 1

        # Check for win or draw
        if self.check_win(row, col, player):
            if player == self.player_color:
                self.status = "player_won"
            else:
                self.status = "ai_won"
        elif self.is_board_full():
            self.status = "draw"

        return True

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid."""
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        return self.board[row, col] == 0

    def check_win(self, row: int, col: int, player: int) -> bool:
        """Check if the last move resulted in a win."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Count the placed stone

            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                else:
                    break

            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                else:
                    break

            if count >= 5:
                return True

        return False

    def is_board_full(self) -> bool:
        """Check if the board is full."""
        return np.all(self.board != 0)

    def resign(self):
        """Player resigns."""
        self.status = "ai_won"

    def to_dict(self) -> dict:
        """Convert game state to dictionary."""
        return {
            "game_id": self.game_id,
            "board": self.board.tolist(),
            "board_size": self.board_size,
            "difficulty": self.difficulty,
            "player_color": self.player_color,
            "current_player": self.current_player,
            "status": self.status,
            "last_move": (
                Move(row=self.last_move[0], col=self.last_move[1], player=self.last_move[2])
                if self.last_move
                else None
            ),
            "move_count": self.move_count,
            "player_time": self.player_time,
            "ai_time": self.ai_time,
        }


class GameManager:
    """Manages all active game sessions."""

    def __init__(self):
        self.games: Dict[str, Game] = {}

    def create_game(self, difficulty: str, player_color: int) -> Game:
        """Create a new game."""
        game = Game(difficulty, player_color)
        self.games[game.game_id] = game
        return game

    def get_game(self, game_id: str) -> Optional[Game]:
        """Get a game by ID."""
        return self.games.get(game_id)

    def delete_game(self, game_id: str) -> bool:
        """Delete a game."""
        if game_id in self.games:
            del self.games[game_id]
            return True
        return False