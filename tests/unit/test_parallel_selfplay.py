"""Unit tests for parallel self-play components."""

import pytest
import numpy as np
import torch
import multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
from alphagomoku.selfplay.parallel import ParallelSelfPlay, _worker_initializer, _play_single_game
from alphagomoku.selfplay.selfplay import SelfPlayWorker, SelfPlayData
from alphagomoku.model.network import GomokuNet


class TestWorkerProcess:
    """Test worker process function."""

    def test_worker_process_setup(self):
        """Test worker process can be set up correctly."""
        # Skip this test as worker_process doesn't exist in current implementation
        # The ParallelSelfPlay class handles worker setup internally
        pytest.skip("worker_process function not exposed in current implementation")

    def test_worker_process_model_loading(self):
        """Test that worker process loads model correctly."""
        pytest.skip("worker_process function not exposed in current implementation")

    def test_worker_process_cpu_device_handling(self):
        """Test worker process handles CPU device properly."""
        pytest.skip("worker_process function not exposed in current implementation")


class TestParallelSelfPlay:
    """Test ParallelSelfPlay class."""

    @pytest.fixture
    def setup_parallel(self):
        """Setup parallel self-play components."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        parallel = ParallelSelfPlay(
            model=model,
            board_size=9,
            mcts_simulations=10,
            batch_size=2,
            num_workers=2
        )
        return model, parallel

    def test_initialization(self, setup_parallel):
        """Test ParallelSelfPlay initialization."""
        model, parallel = setup_parallel

        assert parallel.model == model
        assert parallel.board_size == 9
        assert parallel.mcts_simulations == 10
        assert parallel.batch_size == 2

    def test_generate_data_sequential(self, setup_parallel):
        """Test data generation in sequential mode."""
        model, parallel = setup_parallel

        # Mock the worker process to avoid actual multiprocessing
        mock_data = [
            SelfPlayData(
                state=np.zeros((9, 9)),
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(-1, -1)
            ),
            SelfPlayData(
                state=np.zeros((9, 9)),
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(-1, -1)
            )
        ]

        with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
            mock_worker = Mock()
            mock_worker.generate_batch.return_value = mock_data  # Fixed: use generate_batch
            mock_worker_class.return_value = mock_worker

            # Generate data with num_workers=1 (sequential)
            parallel.num_workers = 1
            data = parallel.generate_data(num_games=2)

            assert len(data) == 2
            mock_worker.generate_batch.assert_called_once_with(2)
            assert all(isinstance(d, SelfPlayData) for d in data)

    @pytest.mark.skipif(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="MPS has known issues with multiprocessing in PyTorch"
    )
    def test_generate_data_parallel(self, setup_parallel):
        """Test data generation in parallel mode."""
        model, parallel = setup_parallel

        mock_data = [
            SelfPlayData(
                state=np.zeros((9, 9)),
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(-1, -1)
            )
        ]

        # Mock multiprocessing pool
        with patch('alphagomoku.selfplay.parallel.mp.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            # imap_unordered returns an iterator that yields game_data for each game
            mock_pool.imap_unordered.return_value = iter([mock_data, mock_data, mock_data, mock_data])
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            data = parallel.generate_data(num_games=4)

            # Should have called pool.imap_unordered
            mock_pool.imap_unordered.assert_called_once()
            assert len(data) == 4  # 4 games, each returns 1 data point
            assert all(isinstance(d, SelfPlayData) for d in data)

    @pytest.mark.skipif(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="MPS has known issues with multiprocessing in PyTorch"
    )
    def test_worker_distribution(self, setup_parallel):
        """Test that Pool is created with correct number of workers."""
        model, parallel = setup_parallel
        parallel.num_workers = 3

        mock_data = [SelfPlayData(
            state=np.zeros((9, 9)),
            policy=np.ones(81) / 81,
            value=0.0,
            current_player=1,
            last_move=(-1, -1)
        )]

        with patch('alphagomoku.selfplay.parallel.mp.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            # Return 10 games worth of data
            mock_pool.imap_unordered.return_value = iter([mock_data] * 10)
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            data = parallel.generate_data(num_games=10)

            # Should create pool with 3 workers
            mock_pool_class.assert_called_once()
            call_kwargs = mock_pool_class.call_args[1]
            assert call_kwargs['processes'] == 3

            # Should have called imap_unordered with tasks for all 10 games
            mock_pool.imap_unordered.assert_called_once()
            call_args = mock_pool.imap_unordered.call_args[0]
            tasks = call_args[1]
            assert len(tasks) == 10

            # Should return all 10 game results
            assert len(data) == 10

    def test_model_state_dict_preparation(self, setup_parallel):
        """Test model state dict preparation for workers."""
        model, parallel = setup_parallel

        # Test the helper method directly
        state_dict = parallel._prepare_model_state_dict()

        # Should return a dict with model parameters
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # All tensors should be on CPU
        for key, value in state_dict.items():
            if hasattr(value, 'device'):
                assert value.device.type == 'cpu', f"Tensor {key} not on CPU"

        # Test config extraction
        config = parallel._extract_model_config()
        assert isinstance(config, dict)
        assert config["board_size"] == 9
        assert config["channels"] == 8
        assert config["num_blocks"] == 1

    @pytest.mark.skipif(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="MPS has known issues with multiprocessing in PyTorch"
    )
    def test_error_handling_worker_failure(self, setup_parallel):
        """Test error handling when worker process fails."""
        model, parallel = setup_parallel

        with patch('alphagomoku.selfplay.parallel.mp.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            # Make imap_unordered raise an error
            mock_pool.imap_unordered.side_effect = RuntimeError("Worker failed")
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            with pytest.raises(RuntimeError, match="Worker failed"):
                parallel.generate_data(num_games=1)

    @pytest.mark.skipif(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="MPS has known issues with multiprocessing in PyTorch"
    )
    def test_cpu_count_handling(self, setup_parallel):
        """Test handling of CPU count for worker determination."""
        model, parallel = setup_parallel

        # Test with None num_workers (should use CPU count)
        parallel.num_workers = None

        mock_data = [SelfPlayData(
            state=np.zeros((9, 9)),
            policy=np.ones(81) / 81,
            value=0.0,
            current_player=1,
            last_move=(-1, -1)
        )]

        with patch('alphagomoku.selfplay.parallel.mp.cpu_count', return_value=4):
            with patch('alphagomoku.selfplay.parallel.mp.Pool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool.imap_unordered.return_value = iter([mock_data] * 4)
                mock_pool.__enter__.return_value = mock_pool
                mock_pool.__exit__.return_value = None
                mock_pool_class.return_value = mock_pool

                data = parallel.generate_data(num_games=4)

                # Should create pool with CPU count workers
                call_kwargs = mock_pool_class.call_args[1]
                assert call_kwargs['processes'] == 4
                assert len(data) == 4

    def test_adaptive_simulations_parameter(self, setup_parallel):
        """Test adaptive simulations parameter is stored correctly."""
        model, parallel = setup_parallel

        # Test default value
        assert parallel.adaptive_sims == True  # From setup

        # Test changing it
        parallel.adaptive_sims = False
        assert parallel.adaptive_sims == False

        # Test it's used in sequential mode (easier to test)
        with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
            mock_worker = Mock()
            mock_worker.generate_batch.return_value = []
            mock_worker_class.return_value = mock_worker

            parallel.num_workers = 1
            parallel.generate_data(num_games=1)

            # Check worker was created with correct parameters
            call_kwargs = mock_worker_class.call_args[1]
            assert call_kwargs['adaptive_sims'] == False

    @pytest.mark.skipif(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="MPS has known issues with multiprocessing in PyTorch"
    )
    def test_memory_efficiency(self, setup_parallel):
        """Test memory efficiency considerations."""
        model, parallel = setup_parallel

        # Test with large number of games
        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            # Simulate memory-efficient batch processing
            mock_pool.starmap.return_value = [[] for _ in range(parallel.num_workers)]
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            # Should handle large numbers without issues
            data = parallel.generate_data(num_games=100)

            # Should complete without memory errors
            assert isinstance(data, list)


class TestParallelSelfPlayIntegration:
    """Integration tests for parallel self-play."""

    def test_end_to_end_small_scale(self):
        """Test end-to-end parallel self-play on small scale."""
        # Use very small configuration for fast testing
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)

        # Mock the actual game generation to avoid long running tests
        mock_game_data = [
            SelfPlayData(
                state=np.zeros((5, 5)),
                policy=np.ones(25) / 25,
                value=0.0,
                current_player=1,
                last_move=(-1, -1)
            )
        ]

        with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
            mock_worker = Mock()
            mock_worker.generate_batch.return_value = [mock_game_data[0], mock_game_data[0]]
            mock_worker_class.return_value = mock_worker

            parallel = ParallelSelfPlay(
                model=model,
                board_size=5,
                mcts_simulations=5,
                batch_size=1,
                num_workers=1  # Single worker for deterministic testing
            )

            data = parallel.generate_data(num_games=2)

            assert len(data) == 2
            assert all(d.state.shape == (5, 5) for d in data)
            assert all(len(d.policy) == 25 for d in data)

    def test_data_format_consistency(self):
        """Test that data format is consistent."""
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)

        mock_game_data = [
            SelfPlayData(
                state=np.random.rand(5, 5).astype(np.int8),
                policy=np.random.rand(25),
                value=np.random.rand(),
                current_player=1,
                last_move=(2, 2)
            )
        ]

        with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
            mock_worker = Mock()
            mock_worker.generate_batch.return_value = mock_game_data
            mock_worker_class.return_value = mock_worker

            parallel = ParallelSelfPlay(
                model=model,
                board_size=5,
                mcts_simulations=5,
                batch_size=1,
                num_workers=1
            )

            data = parallel.generate_data(num_games=1)

            # Check data format consistency
            assert len(data) == 1
            sample = data[0]
            assert hasattr(sample, 'state')
            assert hasattr(sample, 'policy')
            assert hasattr(sample, 'value')
            assert hasattr(sample, 'current_player')
            assert hasattr(sample, 'last_move')

    @pytest.mark.skipif(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        reason="MPS has known issues with multiprocessing in PyTorch"
    )
    def test_resource_cleanup(self):
        """Test proper resource cleanup after parallel execution."""
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)

        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool.starmap.return_value = [[]]
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            parallel = ParallelSelfPlay(
                model=model,
                board_size=5,
                num_workers=2
            )

            parallel.generate_data(num_games=2)

            # Pool should be properly entered and exited (context manager)
            mock_pool.__enter__.assert_called_once()
            mock_pool.__exit__.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
