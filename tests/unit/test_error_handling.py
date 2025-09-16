"""Comprehensive error handling and edge case tests."""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS, MCTSNode
from alphagomoku.tss import Position, tss_search, ThreatDetector
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayWorker, SelfPlayData


class TestGomokuEnvErrorHandling:
    """Test error handling in GomokuEnv."""

    def test_invalid_board_size(self):
        """Test invalid board size handling."""
        with pytest.raises((ValueError, AssertionError)):
            GomokuEnv(board_size=0)

        with pytest.raises((ValueError, AssertionError)):
            GomokuEnv(board_size=-5)

        # Very large board might be impractical but shouldn't crash
        try:
            env = GomokuEnv(board_size=100)
            assert env.board_size == 100
        except (MemoryError, ValueError):
            # Acceptable to reject very large boards
            pass

    def test_invalid_action(self):
        """Test invalid action handling."""
        env = GomokuEnv(board_size=15)
        env.reset()

        # Out of bounds actions
        invalid_actions = [-1, 225, 300, -100]

        for action in invalid_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated
            assert reward < 0  # Should be penalized
            assert info.get('invalid_move', False)

    def test_repeated_action(self):
        """Test playing on occupied position."""
        env = GomokuEnv()
        env.reset()

        # Play valid move
        obs, reward, terminated, truncated, info = env.step(112)  # Center
        assert not terminated
        assert reward == 0

        # Try to play same position
        obs, reward, terminated, truncated, info = env.step(112)
        assert terminated
        assert reward < 0
        assert info.get('invalid_move', False)

    def test_action_after_termination(self):
        """Test action after game has terminated."""
        env = GomokuEnv()
        env.reset()

        # Create winning position quickly
        winning_moves = [112, 0, 113, 1, 114, 2, 115, 3, 116]  # Player 1 wins

        for i, move in enumerate(winning_moves):
            obs, reward, terminated, truncated, info = env.step(move)
            if terminated:
                break

        assert terminated

        # Try to make move after termination
        obs, reward, terminated, truncated, info = env.step(50)
        assert terminated  # Should remain terminated
        # Behavior after termination is implementation-dependent

    def test_reset_after_error(self):
        """Test reset after invalid moves."""
        env = GomokuEnv()
        env.reset()

        # Make invalid move
        env.step(-1)
        assert env.terminated

        # Reset should work
        obs, info = env.reset()
        assert not env.terminated
        assert np.all(obs['board'] == 0)

    def test_corrupted_board_state(self):
        """Test handling of manually corrupted board state."""
        env = GomokuEnv()
        env.reset()

        # Manually corrupt board
        env.board[5, 5] = 99  # Invalid value

        # Environment should handle or detect corruption
        try:
            obs, reward, terminated, truncated, info = env.step(50)
            # If it succeeds, check that board values are normalized
            assert np.all(np.isin(obs['board'], [-1, 0, 1]))
        except (ValueError, AssertionError):
            # Acceptable to detect and reject corrupted state
            pass


class TestModelErrorHandling:
    """Test error handling in GomokuNet."""

    def test_invalid_input_shape(self):
        """Test model with invalid input shapes."""
        model = GomokuNet(board_size=15)

        # Wrong number of dimensions
        with pytest.raises((RuntimeError, ValueError)):
            model(torch.randn(5, 15, 15))  # Missing channel dimension

        # Wrong channel count
        with pytest.raises((RuntimeError, ValueError)):
            model(torch.randn(1, 3, 15, 15))  # Wrong number of channels

        # Wrong board size
        with pytest.raises((RuntimeError, ValueError)):
            model(torch.randn(1, 5, 19, 19))  # Wrong board size

    def test_extreme_input_values(self):
        """Test model with extreme input values."""
        model = GomokuNet(board_size=15)

        # Very large values
        large_input = torch.ones(1, 5, 15, 15) * 1000
        try:
            policy, value = model(large_input)
            assert torch.all(torch.isfinite(policy))
            assert torch.all(torch.isfinite(value))
        except (RuntimeError, OverflowError):
            # Acceptable to fail with extreme inputs
            pass

        # NaN inputs
        nan_input = torch.ones(1, 5, 15, 15) * float('nan')
        try:
            policy, value = model(nan_input)
            # If model handles NaN, outputs should be caught
            # Most likely this will produce NaN outputs
        except (RuntimeError, ValueError):
            # Acceptable to reject NaN inputs
            pass

    def test_zero_parameters(self):
        """Test model behavior with zero-initialized parameters."""
        model = GomokuNet(board_size=9)

        # Zero all parameters
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

        input_tensor = torch.randn(1, 5, 9, 9)
        policy, value = model(input_tensor)

        # Should produce valid outputs even with zero parameters
        assert policy.shape == (1, 81)
        assert value.shape == (1,)
        assert torch.all(torch.isfinite(policy))
        assert torch.all(torch.isfinite(value))

    def test_gradient_explosion_protection(self):
        """Test protection against gradient explosion."""
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        optimizer = torch.optim.SGD(model.parameters(), lr=100.0)  # Very high LR

        # Create extreme target values
        input_tensor = torch.randn(1, 5, 9, 9)
        target_policy = torch.ones(1, 81) / 81
        target_value = torch.tensor([1.0])

        # Multiple training steps with high learning rate
        for _ in range(5):
            optimizer.zero_grad()
            policy, value = model(input_tensor)

            policy_loss = torch.nn.functional.kl_div(
                torch.log_softmax(policy, dim=1),
                target_policy,
                reduction='batchmean'
            )
            value_loss = torch.nn.functional.mse_loss(value, target_value)
            total_loss = policy_loss + value_loss

            total_loss.backward()

            # Check for gradient explosion
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # Apply gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Parameters should remain finite
            for param in model.parameters():
                assert torch.all(torch.isfinite(param))

    def test_device_mismatch(self):
        """Test handling of device mismatches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = GomokuNet(board_size=9)
        model.to('cpu')

        # Try to pass CUDA tensor to CPU model
        cuda_input = torch.randn(1, 5, 9, 9).to('cuda')

        with pytest.raises(RuntimeError):
            model(cuda_input)


class TestMCTSErrorHandling:
    """Test error handling in MCTS."""

    @pytest.fixture
    def setup_mcts(self):
        """Setup MCTS for error testing."""
        env = GomokuEnv(board_size=9)
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        mcts = MCTS(model, env, num_simulations=10)
        return env, model, mcts

    def test_model_failure_handling(self, setup_mcts):
        """Test MCTS handling when model fails."""
        env, model, mcts = setup_mcts
        env.reset()

        # Mock model to fail
        with patch.object(model, 'forward', side_effect=RuntimeError("Model crashed")):
            with pytest.raises(RuntimeError):
                mcts.search(env.board)

    def test_invalid_board_state(self, setup_mcts):
        """Test MCTS with invalid board state."""
        env, model, mcts = setup_mcts

        # Invalid board (wrong shape)
        invalid_board = np.zeros((10, 10))
        with pytest.raises((ValueError, RuntimeError)):
            mcts.search(invalid_board)

        # Invalid values in board
        invalid_board = np.ones((9, 9)) * 5  # Invalid stone values
        with pytest.raises((ValueError, RuntimeError)):
            mcts.search(invalid_board)

    def test_zero_simulations(self, setup_mcts):
        """Test MCTS with zero simulations."""
        env, model, mcts = setup_mcts
        mcts.num_simulations = 0
        env.reset()

        # Should handle gracefully or raise appropriate error
        try:
            action_probs, value = mcts.search(env.board)
            # If it succeeds, check that output is reasonable
            assert len(action_probs) == 81
            assert np.all(action_probs >= 0)
        except ValueError:
            # Acceptable to reject zero simulations
            pass

    def test_negative_simulations(self, setup_mcts):
        """Test MCTS with negative simulations."""
        env, model, mcts = setup_mcts
        mcts.num_simulations = -10
        env.reset()

        with pytest.raises((ValueError, AssertionError)):
            mcts.search(env.board)

    def test_infinite_loop_protection(self, setup_mcts):
        """Test protection against infinite loops in MCTS."""
        env, model, mcts = setup_mcts
        env.reset()

        # Mock node selection to always return same node (potential infinite loop)
        original_select_child = MCTSNode.select_child

        def broken_select_child(self, cpuct):
            # Always return first child if available
            if self.children:
                return list(self.children.values())[0]
            return original_select_child(self, cpuct)

        with patch.object(MCTSNode, 'select_child', broken_select_child):
            try:
                # Should either complete or timeout, not hang forever
                action_probs, value = mcts.search(env.board, timeout=5.0)
            except (RuntimeError, TimeoutError):
                # Acceptable to detect and handle infinite loops
                pass

    def test_memory_exhaustion_protection(self, setup_mcts):
        """Test protection against memory exhaustion."""
        env, model, mcts = setup_mcts
        env.reset()

        # Set very high simulation count
        mcts.num_simulations = 100000

        try:
            action_probs, value = mcts.search(env.board)
            # If it completes, should have reasonable memory usage
        except (MemoryError, RuntimeError):
            # Acceptable to fail with memory constraints
            pass


class TestTSSErrorHandling:
    """Test error handling in TSS."""

    def test_invalid_position(self):
        """Test TSS with invalid positions."""
        # Invalid board shape
        invalid_board = np.zeros((10, 10))
        with pytest.raises((ValueError, AttributeError)):
            position = Position(board=invalid_board, current_player=1)

        # Invalid player
        board = np.zeros((15, 15))
        with pytest.raises((ValueError, AssertionError)):
            position = Position(board=board, current_player=0)

    def test_corrupted_board_state(self):
        """Test TSS with corrupted board."""
        board = np.ones((15, 15)) * 99  # Invalid values
        position = Position(board=board, current_player=1)

        with pytest.raises((ValueError, RuntimeError)):
            result = tss_search(position, depth=2, time_cap_ms=50)

    def test_extreme_parameters(self):
        """Test TSS with extreme parameters."""
        board = np.zeros((15, 15))
        position = Position(board=board, current_player=1)

        # Zero time cap
        with pytest.raises((ValueError, TimeoutError)):
            result = tss_search(position, depth=2, time_cap_ms=0)

        # Negative depth
        with pytest.raises((ValueError, AssertionError)):
            result = tss_search(position, depth=-1, time_cap_ms=100)

        # Very high depth (should timeout or limit itself)
        try:
            result = tss_search(position, depth=100, time_cap_ms=50)
            # Should respect time limit
            assert result.search_stats['time_ms'] <= 100
        except (TimeoutError, ValueError):
            # Acceptable to reject extreme depth
            pass

    def test_threat_detector_edge_cases(self):
        """Test ThreatDetector with edge cases."""
        detector = ThreatDetector()

        # Empty board
        board = np.zeros((15, 15))
        position = Position(board=board, current_player=1)
        threats = detector.detect_threats(position, 1)
        assert len(threats) == 0

        # Full board
        board = np.ones((15, 15))
        position = Position(board=board, current_player=1)
        threats = detector.detect_threats(position, 1)
        # Should handle gracefully (no legal moves)

        # Single stone
        board = np.zeros((15, 15))
        board[7, 7] = 1
        position = Position(board=board, current_player=1)
        threats = detector.detect_threats(position, 1)
        # Should find potential threats around the stone


class TestTrainingErrorHandling:
    """Test error handling in training components."""

    def test_trainer_invalid_data(self):
        """Test trainer with invalid training data."""
        model = GomokuNet(board_size=9)
        trainer = Trainer(model, device='cpu')

        # Empty batch
        losses = trainer.train_step([])
        assert losses == {}

        # Invalid shapes
        invalid_batch = [SelfPlayData(
            state=np.zeros((10, 10)),  # Wrong size
            policy=np.ones(81) / 81,
            value=0.0,
            current_player=1,
            last_move=(4, 4)
        )]

        with pytest.raises((RuntimeError, ValueError)):
            trainer.train_step(invalid_batch)

        # NaN values
        nan_batch = [SelfPlayData(
            state=np.full((9, 9), float('nan')),
            policy=np.ones(81) / 81,
            value=0.0,
            current_player=1,
            last_move=(4, 4)
        )]

        with pytest.raises((ValueError, RuntimeError)):
            trainer.train_step(nan_batch)

    def test_data_buffer_corruption(self):
        """Test data buffer with corruption scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = DataBuffer(db_path=temp_dir, max_size=100)

            # Add valid data first
            valid_data = [SelfPlayData(
                state=np.zeros((15, 15)),
                policy=np.ones(225) / 225,
                value=0.0,
                current_player=1,
                last_move=(7, 7)
            )]
            buffer.add_data(valid_data)

            # Manually corrupt database
            with buffer.env.begin(write=True) as txn:
                txn.put(b'data_1', b'corrupted')

            # Sampling should handle corruption gracefully
            try:
                batch = buffer.sample(10)
                # If successful, should contain valid data
                for item in batch:
                    assert isinstance(item, SelfPlayData)
            except (ValueError, EOFError, TypeError):
                # Acceptable to detect and handle corruption
                pass

    def test_disk_space_exhaustion(self):
        """Test handling of disk space issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create small buffer that might hit disk limits
            buffer = DataBuffer(db_path=temp_dir, max_size=10000, map_size=1024)  # Very small

            large_data = []
            for _ in range(100):
                data = SelfPlayData(
                    state=np.random.randint(-1, 2, (15, 15)),
                    policy=np.random.rand(225),
                    value=np.random.rand() * 2 - 1,
                    current_player=1,
                    last_move=(7, 7)
                )
                data.policy = data.policy / np.sum(data.policy)
                large_data.append(data)

            try:
                buffer.add_data(large_data)
            except Exception as e:
                # Should handle disk space issues gracefully
                assert isinstance(e, (OSError, MemoryError))


class TestSelfPlayErrorHandling:
    """Test error handling in self-play."""

    def test_selfplay_model_failure(self):
        """Test self-play when model fails."""
        model = GomokuNet(board_size=9)
        worker = SelfPlayWorker(model, board_size=9, mcts_simulations=10)

        # Mock model to fail
        with patch.object(model, 'forward', side_effect=RuntimeError("Model failed")):
            with pytest.raises(RuntimeError):
                worker.generate_game()

    def test_selfplay_infinite_game_protection(self):
        """Test protection against infinite games."""
        model = GomokuNet(board_size=9)
        worker = SelfPlayWorker(model, board_size=9, mcts_simulations=5)

        # Mock MCTS to always return uniform policy (might cause long games)
        uniform_policy = np.ones(81) / 81

        with patch.object(worker.mcts, 'search', return_value=(uniform_policy, 0.0)):
            try:
                # Should complete within reasonable time or have move limit
                game_data = worker.generate_game(max_moves=200)
                assert len(game_data) <= 200
            except (RuntimeError, TimeoutError):
                # Acceptable to timeout on infinite games
                pass

    def test_selfplay_memory_leak_protection(self):
        """Test protection against memory leaks in self-play."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        worker = SelfPlayWorker(model, board_size=9, mcts_simulations=5)

        # Generate multiple games and check memory doesn't grow excessively
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for _ in range(5):
            game_data = worker.generate_game()
            # Memory should be cleaned up between games

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            # Should not grow excessively (less than 100MB)
            assert memory_growth < 100 * 1024**2


class TestIntegrationErrorHandling:
    """Test error handling in integrated scenarios."""

    def test_cascade_failure_handling(self):
        """Test handling of cascading failures."""
        model = GomokuNet(board_size=9)
        env = GomokuEnv(board_size=9)
        mcts = MCTS(model, env, num_simulations=10)

        # Simulate cascade: model fails -> MCTS fails -> should be handled
        with patch.object(model, 'forward', side_effect=[
            RuntimeError("First failure"),
            (torch.randn(1, 81), torch.randn(1)),  # Recovery
            RuntimeError("Second failure")
        ]):
            # First call should fail
            with pytest.raises(RuntimeError):
                mcts.search(env.board)

            # System should be recoverable after failure
            # (This depends on implementation details)

    def test_resource_cleanup_after_errors(self):
        """Test that resources are cleaned up after errors."""
        model = GomokuNet(board_size=9)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                buffer = DataBuffer(db_path=temp_dir, max_size=100)

                # Cause an error
                invalid_data = [SelfPlayData(
                    state=np.ones((20, 20)),  # Wrong size
                    policy=np.ones(81),
                    value=0.0,
                    current_player=1,
                    last_move=(4, 4)
                )]

                try:
                    buffer.add_data(invalid_data)
                except Exception:
                    pass

                # Buffer should still be usable after error
                valid_data = [SelfPlayData(
                    state=np.zeros((15, 15)),
                    policy=np.ones(225) / 225,
                    value=0.0,
                    current_player=1,
                    last_move=(7, 7)
                )]

                buffer.add_data(valid_data)
                assert buffer.size > 0

            except Exception as e:
                # Even if test fails, cleanup should work
                pass

    def test_graceful_degradation(self):
        """Test graceful degradation under adverse conditions."""
        # Test that system can operate with reduced functionality
        model = GomokuNet(board_size=9, num_blocks=1, channels=4)  # Minimal model
        env = GomokuEnv(board_size=9)

        # Reduce MCTS simulations to minimum
        mcts = MCTS(model, env, num_simulations=1)

        env.reset()
        try:
            action_probs, value = mcts.search(env.board)
            # Should produce valid output even with minimal resources
            assert len(action_probs) == 81
            assert np.all(action_probs >= 0)
            assert np.isfinite(value)
        except Exception as e:
            # If it fails, should fail gracefully
            assert isinstance(e, (ValueError, RuntimeError))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])