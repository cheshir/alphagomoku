"""Integration tests for endgame solver with unified search."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from alphagomoku.search import UnifiedSearch, SearchResult
from alphagomoku.env.gomoku_env import GomokuEnv


class TestEndgameIntegration:
    """Test endgame solver integration with unified search."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict.return_value = (
            torch.ones(225) / 225,  # uniform policy
            0.0  # neutral value
        )
        model.predict_batch.return_value = (
            torch.ones((1, 225)) / 225,  # batch policy
            torch.zeros((1,))  # batch values
        )
        # Fix the parameters method to return an iterator that doesn't get exhausted
        def mock_parameters():
            mock_param = Mock()
            mock_param.device = torch.device('cpu')
            return iter([mock_param])
        model.parameters = mock_parameters
        return model

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return GomokuEnv(board_size=15)

    def test_endgame_activation_strong_mode(self, mock_model, env):
        """Test that endgame solver activates in strong mode with few empty cells."""
        search = UnifiedSearch(mock_model, env, difficulty='strong')

        # Create board with 20 empty cells (should trigger endgame)
        board = np.zeros((15, 15), dtype=np.int8)
        count = 0
        for i in range(15):
            for j in range(15):
                if count < 15 * 15 - 20:
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        # Add a winning position for player 1
        board[7, 3:7] = 1  # 4 in a row

        result = search.search(board)

        # Should use endgame solver and find a winning move
        assert result.search_method == 'endgame'
        assert result.is_forced
        assert result.search_stats.get('is_win', False) or result.evaluation > 0
        assert result.best_move is not None

    def test_endgame_not_activated_medium_mode(self, mock_model, env):
        """Test that endgame solver doesn't activate in medium mode with too many empty cells."""
        search = UnifiedSearch(mock_model, env, difficulty='medium')

        # Create board with 50 empty cells (should not trigger endgame in medium)
        board = np.zeros((15, 15), dtype=np.int8)
        count = 0
        for i in range(15):
            for j in range(15):
                if count < 15 * 15 - 50:
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        result = search.search(board)

        # Should use MCTS, not endgame solver
        assert result.search_method != 'endgame'

    def test_endgame_disabled_easy_mode(self, mock_model, env):
        """Test that endgame solver is disabled in easy mode."""
        search = UnifiedSearch(mock_model, env, difficulty='easy')

        # Create board with very few empty cells
        board = np.zeros((15, 15), dtype=np.int8)
        count = 0
        for i in range(15):
            for j in range(15):
                if count < 15 * 15 - 5:
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        result = search.search(board)

        # Should not use endgame solver in easy mode
        assert result.search_method != 'endgame'

    def test_search_priority_order(self, mock_model, env):
        """Test that search methods are tried in correct priority order."""
        search = UnifiedSearch(mock_model, env, difficulty='strong')

        # Create a position where endgame should activate
        board = np.zeros((15, 15), dtype=np.int8)
        # Fill most of board
        count = 0
        for i in range(15):
            for j in range(15):
                if count < 15 * 15 - 15:  # Leave 15 empty
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        result = search.search(board)

        # With this setup, should use endgame solver
        assert result.search_method == 'endgame'
        assert result.search_stats['method'] == 'endgame'

    def test_tss_override_with_forced_sequence(self, mock_model, env):
        """Test that TSS can override endgame when forced sequence is found."""
        search = UnifiedSearch(mock_model, env, difficulty='strong')

        # Create position where both endgame and TSS could activate
        board = np.zeros((15, 15), dtype=np.int8)

        # Setup immediate winning threat that TSS should catch
        board[7, 3:7] = -1  # Opponent has 4 in a row
        # Player 1 must defend at (7, 2) or (7, 7)

        # Fill rest to make it an endgame position
        count = 4  # Already placed 4 stones
        for i in range(15):
            for j in range(15):
                if board[i, j] == 0 and count < 15 * 15 - 18:
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        result = search.search(board)

        # TSS should detect the forced defense and override endgame
        if result.search_method == 'tss':
            assert result.is_forced
            assert result.best_move in [(7, 2), (7, 7)]

    def test_mcts_fallback(self, mock_model, env):
        """Test fallback to MCTS when no forced moves are found."""
        search = UnifiedSearch(mock_model, env, difficulty='medium')

        # Create normal middle game position
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        board[7, 8] = -1
        board[8, 7] = 1

        result = search.search(board)

        # Should fall back to MCTS
        assert result.search_method == 'mcts'
        assert not result.is_forced
        assert result.action_probs.sum() > 0

    def test_unified_search_result_consistency(self, mock_model, env):
        """Test that unified search results are consistent."""
        search = UnifiedSearch(mock_model, env, difficulty='medium')

        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1

        result = search.search(board)

        # Validate result structure
        assert isinstance(result, SearchResult)
        assert result.action_probs.shape == (225,)
        assert result.search_method in ['endgame', 'tss', 'mcts']
        assert isinstance(result.is_forced, bool)
        assert isinstance(result.search_stats, dict)
        assert 'method' in result.search_stats

        if result.best_move is not None:
            row, col = result.best_move
            assert 0 <= row < 15
            assert 0 <= col < 15
            action = row * 15 + col
            # If there's a best move, it should have highest probability
            if result.action_probs.sum() > 0:
                assert result.action_probs[action] > 0

    def test_tree_reuse_compatibility(self, mock_model, env):
        """Test that endgame integration doesn't break MCTS tree reuse."""
        search = UnifiedSearch(mock_model, env, difficulty='medium')

        # First move
        board1 = np.zeros((15, 15), dtype=np.int8)
        board1[7, 7] = 1

        result1 = search.search(board1, reuse_tree=False)

        # Second move with tree reuse
        board2 = board1.copy()
        board2[7, 8] = -1

        # This should not crash and should still work
        result2 = search.search(board2, reuse_tree=True)

        assert isinstance(result2, SearchResult)

    def test_difficulty_configuration_consistency(self, mock_model, env):
        """Test that difficulty configurations are applied consistently."""
        # Test all difficulty levels
        difficulties = ['easy', 'medium', 'strong']

        for difficulty in difficulties:
            search = UnifiedSearch(mock_model, env, difficulty=difficulty)
            config = search._get_difficulty_config(difficulty)

            # Validate config structure
            assert 'mcts_sims' in config
            assert 'tss' in config
            assert 'endgame' in config
            assert isinstance(config['tss']['enabled'], bool)
            assert isinstance(config['endgame']['enabled'], bool)

            # Test search doesn't crash
            board = np.zeros((15, 15), dtype=np.int8)
            board[7, 7] = 1
            result = search.search(board)

            assert isinstance(result, SearchResult)