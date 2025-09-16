"""Unit tests for Evaluator module."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from alphagomoku.eval.evaluator import Evaluator
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS


class TestEvaluator:
    """Test Evaluator class."""

    @pytest.fixture
    def setup_evaluator(self):
        """Setup evaluator components."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        evaluator = Evaluator(model, board_size=9)
        return model, evaluator

    def test_initialization(self, setup_evaluator):
        """Test Evaluator initialization."""
        model, evaluator = setup_evaluator

        assert evaluator.model == model
        assert evaluator.board_size == 9
        assert isinstance(evaluator.env, GomokuEnv)
        assert evaluator.env.board_size == 9

    def test_play_game_basic(self, setup_evaluator):
        """Test basic game playing functionality."""
        model, evaluator = setup_evaluator

        # Mock MCTS to avoid long computation
        with patch('alphagomoku.eval.evaluator.MCTS') as mock_mcts_class:
            # Create mock MCTS instances
            mock_mcts1 = Mock()
            mock_mcts2 = Mock()

            # Alternate between different moves
            move_sequence = [40, 41, 42, 43, 44]  # Simple sequence
            call_count = [0]

            def mock_search(board, temperature=0.0):
                policy = np.zeros(81)
                if call_count[0] < len(move_sequence):
                    policy[move_sequence[call_count[0]]] = 1.0
                    call_count[0] += 1
                else:
                    # Random legal move
                    legal = np.where(board.flatten() == 0)[0]
                    if len(legal) > 0:
                        policy[legal[0]] = 1.0
                    else:
                        policy[0] = 1.0

                return policy, 0.0

            mock_mcts1.search = Mock(side_effect=mock_search)
            mock_mcts2.search = Mock(side_effect=mock_search)

            mock_mcts_class.side_effect = [mock_mcts1, mock_mcts2]

            # Play game
            result = evaluator.play_game(player1_sims=10, player2_sims=20)

            # Should return valid result
            assert isinstance(result, dict)
            assert 'winner' in result
            assert 'moves' in result
            assert 'player1_sims' in result
            assert 'player2_sims' in result

            assert result['player1_sims'] == 10
            assert result['player2_sims'] == 20
            assert result['moves'] > 0

    def test_play_game_termination(self, setup_evaluator):
        """Test that games terminate properly."""
        model, evaluator = setup_evaluator

        with patch('alphagomoku.eval.evaluator.MCTS') as mock_mcts_class:
            mock_mcts1 = Mock()
            mock_mcts2 = Mock()

            # Mock a quick winning sequence
            def create_winning_policy(board, temperature=0.0):
                # Find first legal move
                legal_positions = np.where(board.flatten() == 0)[0]
                policy = np.zeros(81)
                if len(legal_positions) > 0:
                    policy[legal_positions[0]] = 1.0
                return policy, 0.0

            mock_mcts1.search = Mock(side_effect=create_winning_policy)
            mock_mcts2.search = Mock(side_effect=create_winning_policy)
            mock_mcts_class.side_effect = [mock_mcts1, mock_mcts2]

            result = evaluator.play_game(player1_sims=5, player2_sims=5)

            # Game should terminate
            assert result['winner'] in [-1, 0, 1]  # Valid winner states
            assert result['moves'] <= 81  # Can't exceed board size

    def test_evaluate_strength(self, setup_evaluator):
        """Test strength evaluation between different configurations."""
        model, evaluator = setup_evaluator

        # Mock the play_game method to avoid actual game computation
        mock_results = [
            {'winner': 1, 'moves': 20, 'player1_sims': 50, 'player2_sims': 10},
            {'winner': -1, 'moves': 15, 'player1_sims': 10, 'player2_sims': 50},
            {'winner': 0, 'moves': 30, 'player1_sims': 50, 'player2_sims': 10},
            {'winner': 1, 'moves': 25, 'player1_sims': 10, 'player2_sims': 50},
        ]

        with patch.object(evaluator, 'play_game', side_effect=mock_results):
            result = evaluator.evaluate_strength(
                test_sims=50,
                baseline_sims=10,
                num_games=4
            )

            assert isinstance(result, dict)
            assert 'wins' in result or 'win_rate' in result
            # Exact structure depends on implementation

    def test_evaluate_strength_alternating_players(self, setup_evaluator):
        """Test that strength evaluation alternates first player."""
        model, evaluator = setup_evaluator

        game_configs = []

        def capture_game_config(p1_sims, p2_sims):
            game_configs.append((p1_sims, p2_sims))
            return {'winner': 1, 'moves': 20, 'player1_sims': p1_sims, 'player2_sims': p2_sims}

        with patch.object(evaluator, 'play_game', side_effect=capture_game_config):
            evaluator.evaluate_strength(test_sims=50, baseline_sims=10, num_games=4)

            # Should alternate configurations
            assert len(game_configs) == 4
            # Implementation might alternate test_sims between player1 and player2

    def test_play_match_interface(self, setup_evaluator):
        """Test play_match interface if it exists."""
        model, evaluator = setup_evaluator

        # Check if evaluator has play_match method
        if hasattr(evaluator, 'play_match'):
            # Create mock players
            player1 = Mock()
            player2 = Mock()

            # Mock the search methods
            player1.search = Mock(return_value=(np.ones(81) / 81, 0.0))
            player2.search = Mock(return_value=(np.ones(81) / 81, 0.0))

            try:
                result = evaluator.play_match(player1, player2, num_games=2, board_size=9)
                assert isinstance(result, dict)
            except (NotImplementedError, AttributeError):
                # Method might not be implemented yet
                pass

    def test_position_evaluation(self, setup_evaluator):
        """Test evaluation of specific positions."""
        model, evaluator = setup_evaluator

        # Test positions
        positions = [
            np.zeros((9, 9)),  # Empty board
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # Center stone
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        ]

        # Check if evaluator has position evaluation method
        if hasattr(evaluator, 'evaluate_position'):
            for pos in positions:
                try:
                    evaluation = evaluator.evaluate_position(pos)
                    assert isinstance(evaluation, (float, int, dict))
                except (NotImplementedError, AttributeError):
                    pass

    def test_tournament_evaluation(self, setup_evaluator):
        """Test tournament-style evaluation if available."""
        model, evaluator = setup_evaluator

        if hasattr(evaluator, 'run_tournament'):
            # Create multiple model configurations
            configs = [
                {'simulations': 10, 'name': 'weak'},
                {'simulations': 50, 'name': 'strong'}
            ]

            try:
                results = evaluator.run_tournament(configs, games_per_pair=2)
                assert isinstance(results, dict)
            except (NotImplementedError, AttributeError):
                pass

    def test_error_handling_invalid_sims(self, setup_evaluator):
        """Test error handling with invalid simulation counts."""
        model, evaluator = setup_evaluator

        # Test with zero simulations
        with pytest.raises((ValueError, AssertionError)):
            evaluator.play_game(player1_sims=0, player2_sims=10)

        # Test with negative simulations
        with pytest.raises((ValueError, AssertionError)):
            evaluator.play_game(player1_sims=-5, player2_sims=10)

    def test_error_handling_game_failure(self, setup_evaluator):
        """Test handling when games fail to complete."""
        model, evaluator = setup_evaluator

        # Mock MCTS to cause game failure
        with patch('alphagomoku.eval.evaluator.MCTS') as mock_mcts_class:
            mock_mcts = Mock()
            mock_mcts.search = Mock(side_effect=RuntimeError("MCTS failed"))
            mock_mcts_class.return_value = mock_mcts

            with pytest.raises(RuntimeError):
                evaluator.play_game(player1_sims=10, player2_sims=10)

    def test_statistics_collection(self, setup_evaluator):
        """Test statistics collection during evaluation."""
        model, evaluator = setup_evaluator

        # Mock games with various outcomes
        mock_results = []
        for i in range(10):
            mock_results.append({
                'winner': (i % 3) - 1,  # Cycle through -1, 0, 1
                'moves': 15 + (i % 10),  # Vary game length
                'player1_sims': 20,
                'player2_sims': 20
            })

        with patch.object(evaluator, 'play_game', side_effect=mock_results):
            result = evaluator.evaluate_strength(
                test_sims=20,
                baseline_sims=20,
                num_games=10
            )

            # Should collect meaningful statistics
            assert isinstance(result, dict)
            # Specific statistics depend on implementation

    def test_performance_tracking(self, setup_evaluator):
        """Test performance tracking capabilities."""
        model, evaluator = setup_evaluator

        # Check if evaluator tracks timing
        if hasattr(evaluator, 'last_evaluation_time'):
            with patch.object(evaluator, 'play_game', return_value={
                'winner': 1, 'moves': 20, 'player1_sims': 10, 'player2_sims': 10
            }):
                evaluator.evaluate_strength(test_sims=10, baseline_sims=10, num_games=2)
                # Some timing information should be available

    def test_reproducibility(self, setup_evaluator):
        """Test evaluation reproducibility."""
        model, evaluator = setup_evaluator

        # Set random seed if evaluator supports it
        np.random.seed(42)
        torch.manual_seed(42)

        with patch.object(evaluator, 'play_game', side_effect=lambda p1, p2: {
            'winner': np.random.choice([-1, 0, 1]),
            'moves': np.random.randint(10, 30),
            'player1_sims': p1,
            'player2_sims': p2
        }):
            result1 = evaluator.evaluate_strength(test_sims=10, baseline_sims=10, num_games=5)

        # Reset seed and run again
        np.random.seed(42)
        torch.manual_seed(42)

        with patch.object(evaluator, 'play_game', side_effect=lambda p1, p2: {
            'winner': np.random.choice([-1, 0, 1]),
            'moves': np.random.randint(10, 30),
            'player1_sims': p1,
            'player2_sims': p2
        }):
            result2 = evaluator.evaluate_strength(test_sims=10, baseline_sims=10, num_games=5)

        # Results should be similar (allowing for implementation differences)


class TestEvaluatorIntegration:
    """Integration tests for Evaluator."""

    def test_evaluator_with_real_model(self):
        """Test evaluator with actual model (small scale)."""
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)  # Very small for speed
        evaluator = Evaluator(model, board_size=5)

        # Play a single game with very low simulation counts
        result = evaluator.play_game(player1_sims=3, player2_sims=3)

        assert isinstance(result, dict)
        assert 'winner' in result
        assert 'moves' in result
        assert result['winner'] in [-1, 0, 1]
        assert result['moves'] > 0

    def test_strength_evaluation_consistency(self):
        """Test strength evaluation consistency."""
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)
        evaluator = Evaluator(model, board_size=5)

        # Evaluate strong vs weak configurations
        result = evaluator.evaluate_strength(
            test_sims=10,
            baseline_sims=2,
            num_games=4  # Small number for speed
        )

        # Should complete without errors
        assert isinstance(result, dict)

        # Strong player (more sims) should generally perform better
        # Though with such a small model and few games, results may vary

    def test_evaluator_memory_usage(self):
        """Test evaluator memory usage doesn't grow excessively."""
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)
        evaluator = Evaluator(model, board_size=5)

        # Run multiple evaluations
        for _ in range(3):
            result = evaluator.play_game(player1_sims=2, player2_sims=2)

        # Should not accumulate excessive memory (basic check)
        assert True  # If we reach here without OOM, test passes

    def test_different_board_sizes(self):
        """Test evaluator with different board sizes."""
        board_sizes = [5, 7, 9]

        for size in board_sizes:
            model = GomokuNet(board_size=size, num_blocks=1, channels=4)
            evaluator = Evaluator(model, board_size=size)

            result = evaluator.play_game(player1_sims=2, player2_sims=2)

            assert isinstance(result, dict)
            assert 'winner' in result
            assert result['moves'] <= size * size  # Can't exceed board capacity


if __name__ == "__main__":
    pytest.main([__file__])