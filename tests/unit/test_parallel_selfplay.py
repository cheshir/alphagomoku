"""Unit tests for parallel self-play components."""

import pytest
import numpy as np
import torch
import multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
from alphagomoku.selfplay.parallel import ParallelSelfPlay, worker_process
from alphagomoku.selfplay.selfplay import SelfPlayWorker, SelfPlayData
from alphagomoku.model.network import GomokuNet


class TestWorkerProcess:
    """Test worker process function."""

    def test_worker_process_setup(self):
        """Test worker process can be set up correctly."""
        # Create a small model for testing
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        model_state_dict = model.state_dict()

        # Mock the worker creation and execution
        with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
            mock_worker = Mock()
            mock_worker.generate_game.return_value = [
                SelfPlayData(
                    state=np.zeros((9, 9)),
                    policy=np.ones(81) / 81,
                    value=0.0,
                    current_player=1,
                    last_move=(-1, -1)
                )
            ]
            mock_worker_class.return_value = mock_worker

            # Test worker process
            result = worker_process(
                model_state_dict=model_state_dict,
                board_size=9,
                mcts_simulations=10,
                adaptive_sims=True,
                batch_size=2,
                num_games=2,
                worker_id=0
            )

            # Should return game data
            assert len(result) == 2  # 2 games, 1 data point each
            mock_worker_class.assert_called_once()
            assert mock_worker.generate_game.call_count == 2

    def test_worker_process_model_loading(self):
        """Test that worker process loads model correctly."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        model_state_dict = model.state_dict()

        with patch('alphagomoku.selfplay.parallel.GomokuNet') as mock_net_class:
            mock_model = Mock()
            mock_net_class.return_value = mock_model

            with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
                mock_worker = Mock()
                mock_worker.generate_game.return_value = []
                mock_worker_class.return_value = mock_worker

                worker_process(
                    model_state_dict=model_state_dict,
                    board_size=9,
                    mcts_simulations=10,
                    adaptive_sims=False,
                    batch_size=1,
                    num_games=1,
                    worker_id=0
                )

                # Model should be created and loaded
                mock_net_class.assert_called_once_with(board_size=9)
                mock_model.load_state_dict.assert_called_once()
                mock_model.eval.assert_called_once()

    def test_worker_process_cpu_device_handling(self):
        """Test worker process handles CPU device properly."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)

        # Create state dict with tensors on different device (simulated)
        model_state_dict = {}
        for k, v in model.state_dict().items():
            if isinstance(v, torch.Tensor):
                # Simulate tensor on non-CPU device
                mock_tensor = Mock()
                mock_tensor.to.return_value = v
                model_state_dict[k] = mock_tensor
            else:
                model_state_dict[k] = v

        with patch('alphagomoku.selfplay.parallel.GomokuNet') as mock_net_class:
            mock_model = Mock()
            mock_net_class.return_value = mock_model

            with patch('alphagomoku.selfplay.parallel.SelfPlayWorker') as mock_worker_class:
                mock_worker = Mock()
                mock_worker.generate_game.return_value = []
                mock_worker_class.return_value = mock_worker

                worker_process(
                    model_state_dict=model_state_dict,
                    board_size=9,
                    mcts_simulations=10,
                    adaptive_sims=False,
                    batch_size=1,
                    num_games=1,
                    worker_id=0
                )

                # CPU tensors should have .to('cpu') called
                for k, v in model_state_dict.items():
                    if hasattr(v, 'to'):
                        v.to.assert_called_with('cpu')


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
            )
        ]

        with patch('alphagomoku.selfplay.parallel.worker_process', return_value=mock_data):
            # Generate data with num_workers=1 (sequential)
            parallel.num_workers = 1
            data = parallel.generate_data(num_games=2)

            assert len(data) == 2
            assert all(isinstance(d, SelfPlayData) for d in data)

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
        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool.starmap.return_value = [mock_data, mock_data]  # 2 workers
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            data = parallel.generate_data(num_games=4)  # 2 games per worker

            # Should have called pool.starmap
            mock_pool.starmap.assert_called_once()
            args, _ = mock_pool.starmap.call_args
            assert len(args[1]) == 2  # 2 workers

    def test_worker_distribution(self, setup_parallel):
        """Test that games are distributed correctly among workers."""
        model, parallel = setup_parallel
        parallel.num_workers = 3

        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool.starmap.return_value = [[], [], []]  # 3 workers
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            parallel.generate_data(num_games=10)

            args, _ = mock_pool.starmap.call_args
            worker_args = args[1]

            # Check game distribution (should be roughly equal)
            total_games = sum(args[5] for args in worker_args)  # num_games is 6th argument
            assert total_games == 10

            # Each worker should get at least 3 games (10/3 = 3.33)
            for args in worker_args:
                assert args[5] >= 3

    def test_model_state_dict_preparation(self, setup_parallel):
        """Test model state dict preparation for workers."""
        model, parallel = setup_parallel

        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool.starmap.return_value = [[]]
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            parallel.generate_data(num_games=1)

            args, _ = mock_pool.starmap.call_args
            worker_args = args[1][0]  # First worker's args

            # Model state dict should be first argument
            model_state_dict = worker_args[0]
            assert isinstance(model_state_dict, dict)
            assert len(model_state_dict) > 0

    def test_error_handling_worker_failure(self, setup_parallel):
        """Test error handling when worker process fails."""
        model, parallel = setup_parallel

        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool.starmap.side_effect = RuntimeError("Worker failed")
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            with pytest.raises(RuntimeError):
                parallel.generate_data(num_games=1)

    def test_cpu_count_handling(self, setup_parallel):
        """Test handling of CPU count for worker determination."""
        model, parallel = setup_parallel

        # Test with None num_workers (should use CPU count)
        parallel.num_workers = None

        with patch('multiprocessing.cpu_count', return_value=4):
            with patch('multiprocessing.Pool') as mock_pool_class:
                mock_pool = Mock()
                mock_pool.starmap.return_value = [[] for _ in range(4)]
                mock_pool.__enter__.return_value = mock_pool
                mock_pool.__exit__.return_value = None
                mock_pool_class.return_value = mock_pool

                parallel.generate_data(num_games=4)

                # Should create pool with CPU count workers
                mock_pool_class.assert_called_once_with(processes=4)

    def test_adaptive_simulations_parameter(self, setup_parallel):
        """Test adaptive simulations parameter propagation."""
        model, parallel = setup_parallel
        parallel.adaptive_sims = True

        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool.starmap.return_value = [[]]
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.__exit__.return_value = None
            mock_pool_class.return_value = mock_pool

            parallel.generate_data(num_games=1)

            args, _ = mock_pool.starmap.call_args
            worker_args = args[1][0]

            # adaptive_sims should be 4th argument (index 3)
            assert worker_args[3] == True

    def test_memory_efficiency(self, setup_parallel):
        """Test memory efficiency considerations."""
        model, parallel = setup_parallel

        # Test with large number of games
        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
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
            mock_worker.generate_game.return_value = mock_game_data
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
        """Test that parallel and sequential generation produce consistent data format."""
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
            mock_worker.generate_game.return_value = mock_game_data
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

    def test_resource_cleanup(self):
        """Test proper resource cleanup after parallel execution."""
        model = GomokuNet(board_size=5, num_blocks=1, channels=4)

        with patch('multiprocessing.Pool') as mock_pool_class:
            mock_pool = Mock()
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