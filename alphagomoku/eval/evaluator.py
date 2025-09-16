from typing import Dict, List, Tuple

import numpy as np

from ..env.gomoku_env import GomokuEnv
from ..mcts.mcts import MCTS


class Evaluator:
    """Evaluation framework for Gomoku models"""

    def __init__(self, model, board_size: int = 15):
        self.model = model
        self.board_size = board_size
        self.env = GomokuEnv(board_size)

    def play_game(self, player1_sims: int, player2_sims: int) -> Dict:
        """Play a single game between two MCTS configurations"""
        self.env.reset()

        mcts1 = MCTS(self.model, self.env, num_simulations=player1_sims)
        mcts2 = MCTS(self.model, self.env, num_simulations=player2_sims)

        move_count = 0
        while not self.env.game_over:
            current_mcts = mcts1 if self.env.current_player == 1 else mcts2

            # Get move from MCTS
            policy, _ = current_mcts.search(self.env.board, temperature=0.0)
            action = np.argmax(policy)

            # Make move
            self.env.step(action)
            move_count += 1

        return {
            "winner": self.env.winner,
            "moves": move_count,
            "player1_sims": player1_sims,
            "player2_sims": player2_sims,
        }

    def evaluate_strength(
        self, test_sims: int, baseline_sims: int, num_games: int = 100
    ) -> Dict:
        """Evaluate model strength against baseline"""
        results = []
        wins = losses = draws = 0

        for game_idx in range(num_games):
            # Alternate who plays first
            if game_idx % 2 == 0:
                result = self.play_game(test_sims, baseline_sims)
                if result["winner"] == 1:
                    wins += 1
                elif result["winner"] == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                result = self.play_game(baseline_sims, test_sims)
                if result["winner"] == -1:
                    wins += 1
                elif result["winner"] == 1:
                    losses += 1
                else:
                    draws += 1

            results.append(result)

        win_rate = wins / num_games
        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": num_games,
            "results": results,
        }

    def test_positions(self, positions: List[Tuple[np.ndarray, int]]) -> Dict:
        """Test model on specific positions (board, expected_move)"""
        correct = 0
        total = len(positions)

        mcts = MCTS(self.model, self.env, num_simulations=800)

        for board, expected_move in positions:
            policy, _ = mcts.search(board, temperature=0.0)
            predicted_move = np.argmax(policy)

            if predicted_move == expected_move:
                correct += 1

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }
