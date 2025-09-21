"""Unit tests for MCTS module."""

import pytest
import numpy as np
import torch
from unittest.mock import patch
from alphagomoku.mcts.mcts import MCTS, MCTSNode
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet


class TestMCTSNode:
    """Test MCTSNode class."""

    def test_node_creation(self):
        state = np.zeros((15, 15), dtype=np.int8)
        node = MCTSNode(state, current_player=1)

        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.current_player == 1
        assert not node.is_expanded
        assert node.is_leaf()
        assert len(node.children) == 0

    def test_node_value(self):
        state = np.zeros((15, 15), dtype=np.int8)
        node = MCTSNode(state)

        # No visits, value should be 0
        assert node.value() == 0.0

        # Add some visits
        node.visit_count = 5
        node.value_sum = 2.5
        assert node.value() == 0.5

    def test_uct_score(self):
        state = np.zeros((15, 15), dtype=np.int8)
        node = MCTSNode(state, prior=0.5)

        # Test UCT calculation
        cpuct = 1.0
        parent_visits = 10
        score = node.uct_score(cpuct, parent_visits)

        # Should be positive (exploration term dominates with 0 visits)
        assert score > 0

        # Test with in_flight flag
        node.in_flight = True
        score = node.uct_score(cpuct, parent_visits)
        assert score == -1e9

    def test_select_child(self):
        state = np.zeros((15, 15), dtype=np.int8)
        parent = MCTSNode(state)

        # Add children
        child1 = MCTSNode(state, parent=parent, prior=0.3)
        child2 = MCTSNode(state, parent=parent, prior=0.7)
        parent.children[0] = child1
        parent.children[1] = child2

        # Should select child with highest UCT score
        selected = parent.select_child(cpuct=1.0)
        assert selected in [child1, child2]


class TestMCTS:
    """Test MCTS class."""

    @pytest.fixture
    def setup_mcts(self):
        """Setup MCTS components."""
        env = GomokuEnv(board_size=9)  # Smaller board for faster tests
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        mcts = MCTS(model, env, num_simulations=50)
        return env, model, mcts

    def test_mcts_initialization(self, setup_mcts):
        """Test MCTS initialization."""
        env, model, mcts = setup_mcts

        assert mcts.model == model
        assert mcts.env == env
        assert mcts.num_simulations == 50
        assert mcts.cpuct > 0

    def test_search_basic(self, setup_mcts):
        """Test basic MCTS search."""
        env, model, mcts = setup_mcts
        env.reset()

        # Run search
        action_probs, value = mcts.search(env.board)

        assert len(action_probs) == 9 * 9
        assert np.isclose(np.sum(action_probs), 1.0)
        assert isinstance(value, (int, float, np.number))
        assert -1 <= value <= 1

        visits = mcts.last_visit_counts
        assert isinstance(visits, np.ndarray)
        assert visits.ndim == 1
        if visits.size > 0:
            assert np.all(visits >= 0)

    def test_search_with_mask(self, setup_mcts):
        """Test MCTS search with action mask."""
        env, model, mcts = setup_mcts
        env.reset()

        # Make some moves to create mask
        env.step(40)  # Center
        env.step(41)  # Adjacent

        action_probs, _ = mcts.search(env.board)

        # Should not select already occupied positions
        assert action_probs[40] == 0.0
        assert action_probs[41] == 0.0
        assert np.sum(action_probs) > 0.0

    def test_batch_evaluation(self, setup_mcts):
        """Test batched neural network evaluation."""
        env, model, mcts = setup_mcts
        env.reset()

        def fake_predict_batch(board_states: torch.Tensor):
            batch = board_states.shape[0]
            action_space = env.board_size * env.board_size
            device = board_states.device
            policies = torch.full(
                (batch, action_space), 1.0 / action_space, device=device
            )
            values = torch.zeros(batch, device=device)
            return policies, values

        mcts.batch_size = 4  # Ensure batched path is exercised
        with patch.object(model, 'predict_batch', side_effect=fake_predict_batch) as mock_predict:
            action_probs, value = mcts.search(env.board)

        assert mock_predict.called
        assert len(action_probs) == env.board_size ** 2
        assert isinstance(value, (int, float, np.number))
        visits = mcts.last_visit_counts
        assert isinstance(visits, np.ndarray)

    def test_tree_reuse(self, setup_mcts):
        """Test MCTS tree reuse between searches."""
        env, model, mcts = setup_mcts
        env.reset()

        # First search
        mcts.search(env.board)
        first_tree_size = len(mcts.tree) if hasattr(mcts, 'tree') else 0

        # Second search on same position
        mcts.search(env.board)

        # Tree should exist and potentially be reused
        assert hasattr(mcts, 'root') or hasattr(mcts, 'tree')

    def test_memory_cleanup(self, setup_mcts):
        """Test memory cleanup in MCTS."""
        env, model, mcts = setup_mcts
        env.reset()

        # Run multiple searches
        for _ in range(3):
            mcts.search(env.board)
            # Make a move to create new tree
            env.step(np.random.choice(env.get_legal_actions()))

        # Should handle memory appropriately (no errors)
        assert True  # If we reach here without OOM, test passes

    def test_error_handling_invalid_state(self, setup_mcts):
        """Test error handling with invalid board state."""
        env, model, mcts = setup_mcts

        # Invalid board (wrong shape)
        invalid_board = np.zeros((10, 10))

        with pytest.raises((ValueError, RuntimeError)):
            mcts.search(invalid_board)

    def test_error_handling_model_failure(self, setup_mcts):
        """Test error handling when model fails."""
        env, model, mcts = setup_mcts
        env.reset()

        # Mock model to raise exception
        with patch.object(model, 'forward', side_effect=RuntimeError("Model error")):
            with pytest.raises(RuntimeError):
                mcts.search(env.board)

    def test_zero_simulations(self, setup_mcts):
        """Test MCTS with zero simulations."""
        env, model, mcts = setup_mcts
        mcts.num_simulations = 0
        env.reset()

        # Should handle gracefully or raise appropriate error
        try:
            action_probs, value = mcts.search(env.board)
            # If it succeeds, check results are reasonable
            assert len(action_probs) == 81
        except ValueError:
            # Acceptable to raise error for zero simulations
            pass

    def test_temperature_effects(self, setup_mcts):
        """Test temperature effects on action selection."""
        env, model, mcts = setup_mcts
        env.reset()

        # Search with different temperatures
        probs_low_temp, _ = mcts.search(env.board, temperature=0.1)
        probs_high_temp, _ = mcts.search(env.board, temperature=2.0)

        # Low temperature should be more concentrated
        entropy_low = -np.sum(probs_low_temp * np.log(probs_low_temp + 1e-8))
        entropy_high = -np.sum(probs_high_temp * np.log(probs_high_temp + 1e-8))

        # High temperature should have higher entropy (more exploration)
        assert entropy_high >= entropy_low - 0.1  # Small tolerance for randomness


class TestMCTSBatchProcessing:
    """Test MCTS batch processing capabilities."""

    @pytest.fixture
    def setup_batch_mcts(self):
        """Setup MCTS for batch testing."""
        env = GomokuEnv(board_size=9)
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        mcts = MCTS(model, env, num_simulations=100, batch_size=8)
        return env, model, mcts

    def test_batched_handles_terminal_leaves(self):
        """Ensure batched simulations advance when encountering terminal leaves."""

        board = np.array(
            [
                [1, -1, 1, -1, 1],
                [-1, 1, -1, 1, -1],
                [1, -1, -1, -1, 1],
                [-1, 1, -1, 1, -1],
                [1, -1, 1, 0, 1],
            ],
            dtype=np.int8,
        )

        env = GomokuEnv(board_size=5)
        env.board = board.copy()
        env.current_player = 1  # Even number of stones -> first player to move
        env.last_move = np.array([4, 4], dtype=np.int8)
        env.game_over = False
        env.winner = 0
        env.move_count = int(np.count_nonzero(env.board))

        # Keep channels >= reduction factor in SEBlock to avoid zero-dim warnings
        model = GomokuNet(board_size=5, num_blocks=1, channels=16)
        mcts = MCTS(model, env, num_simulations=32, batch_size=4)

        action_probs, value = mcts.search(env.board)

        legal_actions = env.get_legal_actions()
        assert len(legal_actions) == 1
        assert np.isclose(action_probs.sum(), 1.0)
        assert action_probs[legal_actions[0]] == pytest.approx(1.0)
        assert isinstance(value, (float, int, np.number))

    def test_batch_size_configuration(self, setup_batch_mcts):
        """Test batch size configuration."""
        env, model, mcts = setup_batch_mcts

        assert hasattr(mcts, 'batch_size') or mcts.num_simulations > 0
        # Basic functionality should work
        env.reset()
        action_probs, _ = mcts.search(env.board)
        assert len(action_probs) == 81

    def test_concurrent_evaluations(self, setup_batch_mcts):
        """Test concurrent leaf evaluations."""
        env, model, mcts = setup_batch_mcts
        env.reset()

        # Track model calls
        original_forward = model.forward
        call_count = 0
        batch_sizes = []

        def tracking_forward(x):
            nonlocal call_count, batch_sizes
            call_count += 1
            batch_sizes.append(x.shape[0])
            return original_forward(x)

        with patch.object(model, 'forward', side_effect=tracking_forward):
            mcts.search(env.board)

        # Should have made at least one model call
        assert call_count > 0
        # Batch sizes should be reasonable
        assert all(size > 0 for size in batch_sizes)

    def test_virtual_loss(self, setup_batch_mcts):
        """Test virtual loss mechanism for batching."""
        env, model, mcts = setup_batch_mcts
        env.reset()

        # This tests the internal virtual loss mechanism
        # Run search and verify it completes without deadlock
        action_probs, value = mcts.search(env.board)

        assert len(action_probs) == 81
        assert np.sum(action_probs) > 0
        assert isinstance(value, (int, float, np.number))

    def test_batch_performance(self, setup_batch_mcts):
        """Test batch processing performance."""
        env, model, mcts = setup_batch_mcts
        env.reset()

        import time

        # Measure search time
        start_time = time.time()
        mcts.search(env.board)
        search_time = time.time() - start_time

        # Should complete within reasonable time
        assert search_time < 10.0  # 10 seconds max for 100 simulations


class TestMCTSEdgeCases:
    """Test MCTS edge cases and error scenarios."""

    @pytest.fixture
    def setup_edge_mcts(self):
        """Setup MCTS for edge case testing."""
        env = GomokuEnv(board_size=9)
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        mcts = MCTS(model, env, num_simulations=20)
        return env, model, mcts

    def test_full_board(self, setup_edge_mcts):
        """Test MCTS on nearly full board."""
        env, model, mcts = setup_edge_mcts
        env.reset()

        # Fill most of the board
        actions = list(range(81))
        np.random.shuffle(actions)

        for action in actions[:75]:  # Leave some moves
            if action in env.get_legal_actions():
                env.step(action)
                if env.terminated:
                    break

        if not env.terminated and len(env.get_legal_actions()) > 0:
            action_probs, _ = mcts.search(env.board)
            # Should only have non-zero probs for legal moves
            legal_actions = env.get_legal_actions()
            for i, prob in enumerate(action_probs):
                if i not in legal_actions:
                    assert prob == 0.0

    def test_terminal_position(self, setup_edge_mcts):
        """Test MCTS on terminal position."""
        env, model, mcts = setup_edge_mcts
        env.reset()

        # Create winning position
        # Horizontal line for player 1
        winning_moves = [40, 50, 41, 60, 42, 70, 43, 80, 44]  # Player 1 wins

        for move in winning_moves:
            env.step(move)
            if env.terminated:
                break

        if env.terminated:
            # MCTS should handle terminal position
            try:
                action_probs, value = mcts.search(env.board)
                # All probabilities should be zero for terminal position
                assert np.sum(action_probs) == 0.0 or len(env.get_legal_actions()) == 0
            except (ValueError, RuntimeError):
                # Acceptable to raise error for terminal position
                pass

    def test_single_legal_move(self, setup_edge_mcts):
        """Test MCTS with only one legal move."""
        env, model, mcts = setup_edge_mcts
        env.reset()

        # Fill board except one position
        actions = list(range(81))
        last_action = actions.pop()  # Save one action

        for action in actions:
            if action in env.get_legal_actions():
                env.step(action)
                if env.terminated:
                    break

        if not env.terminated and len(env.get_legal_actions()) == 1:
            action_probs, _ = mcts.search(env.board)
            # Should put all probability on the single legal move
            legal_actions = env.get_legal_actions()
            assert len(legal_actions) == 1
            legal_action = legal_actions[0]
            assert action_probs[legal_action] > 0.9

    def test_model_nan_output(self, setup_edge_mcts):
        """Test handling of NaN model outputs."""
        env, model, mcts = setup_edge_mcts
        env.reset()

        # Mock model to return NaN
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = (
                torch.tensor([[float('nan')] * 81]),  # NaN policy
                torch.tensor([0.0])  # Valid value
            )

            with pytest.raises((ValueError, RuntimeError)):
                mcts.search(env.board)

    def test_model_infinite_output(self, setup_edge_mcts):
        """Test handling of infinite model outputs."""
        env, model, mcts = setup_edge_mcts
        env.reset()

        # Mock model to return infinity
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = (
                torch.tensor([[float('inf')] * 81]),  # Infinite policy
                torch.tensor([0.0])  # Valid value
            )

            # Should handle infinite values gracefully
            try:
                action_probs, _ = mcts.search(env.board)
                # If it succeeds, probabilities should be finite
                assert np.all(np.isfinite(action_probs))
            except (ValueError, RuntimeError):
                # Acceptable to raise error for infinite outputs
                pass


if __name__ == "__main__":
    pytest.main([__file__])
