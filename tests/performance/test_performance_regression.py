"""Performance regression tests for AlphaGomoku."""

import pytest
import time
import numpy as np
import torch
import psutil
import threading
from contextlib import contextmanager
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.tss import Position, tss_search
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayData
import tempfile
import shutil


@contextmanager
def performance_timer():
    """Context manager to measure execution time."""
    start_time = time.time()
    yield lambda: time.time() - start_time


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    yield lambda: process.memory_info().rss - initial_memory


class TestModelPerformance:
    """Test neural network model performance."""

    @pytest.fixture
    def setup_models(self):
        """Setup models of different sizes."""
        small_model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        medium_model = GomokuNet(board_size=15, num_blocks=6, channels=64)
        large_model = GomokuNet(board_size=15, num_blocks=12, channels=128)

        return {
            'small': small_model,
            'medium': medium_model,
            'large': large_model
        }

    def test_model_inference_speed(self, setup_models):
        """Test model inference speed benchmarks."""
        models = setup_models

        # Benchmark parameters
        batch_sizes = [1, 4, 16, 32]
        expected_times = {
            'small': {'1': 0.01, '4': 0.02, '16': 0.05, '32': 0.1},
            'medium': {'1': 0.05, '4': 0.1, '16': 0.3, '32': 0.6},
            'large': {'1': 0.1, '4': 0.2, '16': 0.6, '32': 1.2}
        }

        for model_name, model in models.items():
            model.eval()
            board_size = 9 if model_name == 'small' else 15

            for batch_size in batch_sizes:
                # Prepare input
                x = torch.randn(batch_size, 5, board_size, board_size)

                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        model(x)

                # Benchmark
                with performance_timer() as get_time:
                    with torch.no_grad():
                        for _ in range(10):
                            policy, value = model(x)

                avg_time = get_time() / 10
                expected_time = expected_times[model_name][str(batch_size)]

                # Allow 3x variance for different hardware
                assert avg_time < expected_time * 3, \
                    f"{model_name} model batch_size {batch_size}: {avg_time:.3f}s > {expected_time * 3:.3f}s"

    def test_model_memory_usage(self, setup_models):
        """Test model memory usage."""
        models = setup_models

        expected_memory = {
            'small': 10 * 1024**2,    # 10MB
            'medium': 50 * 1024**2,   # 50MB
            'large': 200 * 1024**2    # 200MB
        }

        for model_name, model in models.items():
            with memory_monitor() as get_memory_delta:
                # Load model and run inference
                model.eval()
                board_size = 9 if model_name == 'small' else 15
                x = torch.randn(16, 5, board_size, board_size)

                with torch.no_grad():
                    for _ in range(5):
                        policy, value = model(x)

            memory_used = get_memory_delta()
            expected_mem = expected_memory[model_name]

            # Allow 2x variance
            assert memory_used < expected_mem * 2, \
                f"{model_name} model used {memory_used / 1024**2:.1f}MB > {expected_mem * 2 / 1024**2:.1f}MB"

    def test_batch_processing_efficiency(self, setup_models):
        """Test that batch processing is more efficient than sequential."""
        model = setup_models['medium']
        model.eval()

        # Sequential processing
        with performance_timer() as get_sequential_time:
            with torch.no_grad():
                for _ in range(16):
                    x = torch.randn(1, 5, 15, 15)
                    policy, value = model(x)

        sequential_time = get_sequential_time()

        # Batch processing
        with performance_timer() as get_batch_time:
            with torch.no_grad():
                x = torch.randn(16, 5, 15, 15)
                policy, value = model(x)

        batch_time = get_batch_time()

        # Batch should be significantly faster
        efficiency_ratio = sequential_time / batch_time
        assert efficiency_ratio > 2.0, \
            f"Batch processing not efficient enough: {efficiency_ratio:.2f}x speedup"


class TestMCTSPerformance:
    """Test MCTS performance."""

    @pytest.fixture
    def setup_mcts(self):
        """Setup MCTS with different configurations."""
        env = GomokuEnv(board_size=15)
        model = GomokuNet(board_size=15, num_blocks=4, channels=32)

        configs = {
            'fast': MCTS(model, env, num_simulations=100, cpuct=1.0),
            'medium': MCTS(model, env, num_simulations=400, cpuct=1.5),
            'slow': MCTS(model, env, num_simulations=800, cpuct=2.0)
        }

        return env, configs

    def test_mcts_search_speed(self, setup_mcts):
        """Test MCTS search speed benchmarks."""
        env, mcts_configs = setup_mcts

        # Expected times (seconds) for different configurations
        expected_times = {
            'fast': 0.5,
            'medium': 2.0,
            'slow': 4.0
        }

        for config_name, mcts in mcts_configs.items():
            env.reset()

            # Warm up
            mcts.search(env.board)

            # Benchmark
            with performance_timer() as get_time:
                for _ in range(3):
                    action_probs, value = mcts.search(env.board)

            avg_time = get_time() / 3
            expected_time = expected_times[config_name]

            # Allow 2x variance
            assert avg_time < expected_time * 2, \
                f"MCTS {config_name}: {avg_time:.2f}s > {expected_time * 2:.2f}s"

    def test_mcts_memory_scaling(self, setup_mcts):
        """Test MCTS memory usage scales reasonably with simulations."""
        env, mcts_configs = setup_mcts
        env.reset()

        memory_usage = {}

        for config_name, mcts in mcts_configs.items():
            with memory_monitor() as get_memory_delta:
                # Run multiple searches to build tree
                for _ in range(3):
                    action_probs, value = mcts.search(env.board)
                    # Make a move to expand tree
                    legal_actions = env.get_legal_actions()
                    if legal_actions:
                        env.step(legal_actions[0])

            memory_usage[config_name] = get_memory_delta()

        # Memory should scale reasonably with simulations
        # slow (800 sims) should use more memory than fast (100 sims)
        assert memory_usage['slow'] > memory_usage['fast'], \
            "Memory usage should increase with simulation count"

        # But not excessively (less than 10x)
        ratio = memory_usage['slow'] / max(memory_usage['fast'], 1)
        assert ratio < 10, f"Memory scaling too high: {ratio:.1f}x"

    def test_mcts_tree_reuse_efficiency(self, setup_mcts):
        """Test MCTS tree reuse provides performance benefit."""
        env, mcts_configs = setup_mcts
        mcts = mcts_configs['medium']
        env.reset()

        # First search (no tree reuse)
        with performance_timer() as get_first_time:
            action_probs1, value1 = mcts.search(env.board)
        first_time = get_first_time()

        # Second search on same position (potential tree reuse)
        with performance_timer() as get_second_time:
            action_probs2, value2 = mcts.search(env.board)
        second_time = get_second_time()

        # Tree reuse should provide some benefit (at least not much slower)
        assert second_time <= first_time * 1.5, \
            f"Tree reuse inefficient: {second_time:.3f}s vs {first_time:.3f}s"


class TestTSSPerformance:
    """Test TSS performance."""

    def test_tss_search_speed(self):
        """Test TSS search speed for different depths."""
        # Create tactical position
        board = np.zeros((15, 15), dtype=np.int8)

        # Create open-four threat
        for i in range(4):
            board[7, 5 + i] = -1

        position = Position(board=board, current_player=1)

        depths_and_times = [
            (2, 0.01),   # 10ms
            (4, 0.05),   # 50ms
            (6, 0.20),   # 200ms
        ]

        for depth, expected_time in depths_and_times:
            with performance_timer() as get_time:
                for _ in range(5):
                    result = tss_search(position, depth=depth, time_cap_ms=1000)

            avg_time = get_time() / 5

            # Allow 3x variance
            assert avg_time < expected_time * 3, \
                f"TSS depth {depth}: {avg_time:.3f}s > {expected_time * 3:.3f}s"

    def test_tss_timeout_compliance(self):
        """Test that TSS respects timeout limits."""
        # Create complex position for longer search
        board = np.random.randint(-1, 2, (15, 15)).astype(np.int8)
        position = Position(board=board, current_player=1)

        time_caps = [10, 50, 100]  # milliseconds

        for time_cap in time_caps:
            with performance_timer() as get_time:
                result = tss_search(position, depth=8, time_cap_ms=time_cap)

            actual_time_ms = get_time() * 1000

            # Should respect timeout (allow 50% tolerance for overhead)
            assert actual_time_ms < time_cap * 1.5, \
                f"TSS timeout violated: {actual_time_ms:.1f}ms > {time_cap * 1.5:.1f}ms"


class TestTrainingPerformance:
    """Test training pipeline performance."""

    def test_training_step_speed(self):
        """Test training step performance."""
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        trainer = Trainer(model, device='cpu')

        # Create training batch
        batch_sizes = [8, 16, 32]

        for batch_size in batch_sizes:
            batch = []
            for _ in range(batch_size):
                data = SelfPlayData(
                    state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                    policy=np.random.rand(81),
                    value=np.random.rand() * 2 - 1,
                    current_player=1,
                    last_move=(4, 4)
                )
                data.policy = data.policy / np.sum(data.policy)
                batch.append(data)

            # Benchmark training step
            with performance_timer() as get_time:
                for _ in range(5):
                    losses = trainer.train_step(batch)

            avg_time = get_time() / 5
            expected_time = 0.05 * (batch_size / 8)  # Scale with batch size

            # Allow 3x variance
            assert avg_time < expected_time * 3, \
                f"Training step batch_size {batch_size}: {avg_time:.3f}s > {expected_time * 3:.3f}s"

    def test_data_buffer_performance(self):
        """Test data buffer performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = DataBuffer(db_path=temp_dir, max_size=1000)

            # Test add performance
            data_sizes = [10, 50, 100]

            for data_size in data_sizes:
                data_batch = []
                for _ in range(data_size):
                    data = SelfPlayData(
                        state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                        policy=np.random.rand(225),
                        value=np.random.rand() * 2 - 1,
                        current_player=1,
                        last_move=(7, 7)
                    )
                    data.policy = data.policy / np.sum(data.policy)
                    data_batch.append(data)

                # Benchmark add operation
                with performance_timer() as get_add_time:
                    buffer.add_data(data_batch)

                add_time = get_add_time()
                expected_time = 0.01 * data_size  # 10ms per example

                assert add_time < expected_time * 3, \
                    f"Buffer add {data_size} examples: {add_time:.3f}s > {expected_time * 3:.3f}s"

                # Test sample performance
                with performance_timer() as get_sample_time:
                    for _ in range(10):
                        batch = buffer.sample(32)

                avg_sample_time = get_sample_time() / 10
                expected_sample_time = 0.01  # 10ms

                assert avg_sample_time < expected_sample_time * 3, \
                    f"Buffer sample: {avg_sample_time:.3f}s > {expected_sample_time * 3:.3f}s"


class TestSelfPlayPerformance:
    """Test self-play performance."""

    def test_selfplay_game_generation_speed(self):
        """Test self-play game generation performance."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)

        worker = SelfPlayWorker(
            model=model,
            board_size=9,
            mcts_simulations=50,  # Reduced for performance test
            adaptive_sims=False,
            batch_size=4
        )

        # Benchmark game generation
        with performance_timer() as get_time:
            game_data = worker.generate_game()

        game_time = get_time()
        expected_time = 2.0  # 2 seconds for 9x9 board with 50 simulations

        # Allow 3x variance
        assert game_time < expected_time * 3, \
            f"Self-play game: {game_time:.2f}s > {expected_time * 3:.2f}s"

        # Should generate reasonable amount of data
        assert len(game_data) > 5, "Should generate multiple data points per game"

    def test_selfplay_memory_efficiency(self):
        """Test self-play memory efficiency."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)

        worker = SelfPlayWorker(
            model=model,
            board_size=9,
            mcts_simulations=30,
            adaptive_sims=False,
            batch_size=4
        )

        with memory_monitor() as get_memory_delta:
            # Generate multiple games
            for _ in range(3):
                game_data = worker.generate_game()

        memory_used = get_memory_delta()
        expected_memory = 100 * 1024**2  # 100MB

        # Should not use excessive memory
        assert memory_used < expected_memory, \
            f"Self-play memory usage: {memory_used / 1024**2:.1f}MB > {expected_memory / 1024**2:.1f}MB"


class TestIntegrationPerformance:
    """Test integrated performance scenarios."""

    def test_full_training_iteration_performance(self):
        """Test performance of full training iteration."""
        # Small-scale integration test
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        trainer = Trainer(model, device='cpu')

        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = DataBuffer(db_path=temp_dir, max_size=100)

            # Simulate training iteration
            with performance_timer() as get_time:
                # 1. Generate some self-play data (mocked for speed)
                mock_data = []
                for _ in range(10):
                    data = SelfPlayData(
                        state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                        policy=np.random.rand(81),
                        value=np.random.rand() * 2 - 1,
                        current_player=1,
                        last_move=(4, 4)
                    )
                    data.policy = data.policy / np.sum(data.policy)
                    mock_data.append(data)

                # 2. Add to buffer
                buffer.add_data(mock_data)

                # 3. Sample and train
                for _ in range(5):
                    batch = buffer.sample(8)
                    if batch:
                        losses = trainer.train_step(batch)

            iteration_time = get_time()
            expected_time = 1.0  # 1 second for small iteration

            assert iteration_time < expected_time * 3, \
                f"Training iteration: {iteration_time:.2f}s > {expected_time * 3:.2f}s"

    def test_concurrent_performance(self):
        """Test performance under concurrent operations."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        env = GomokuEnv(board_size=9)
        mcts = MCTS(model, env, num_simulations=50)

        def search_task():
            env_local = GomokuEnv(board_size=9)
            env_local.reset()
            for _ in range(3):
                action_probs, value = mcts.search(env_local.board)

        # Run concurrent searches
        with performance_timer() as get_time:
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=search_task)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        concurrent_time = get_time()
        expected_time = 2.0  # Should complete within reasonable time

        assert concurrent_time < expected_time, \
            f"Concurrent performance: {concurrent_time:.2f}s > {expected_time:.2f}s"

    def test_memory_leak_detection(self):
        """Test for potential memory leaks."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        env = GomokuEnv(board_size=9)
        mcts = MCTS(model, env, num_simulations=20)

        initial_memory = psutil.Process().memory_info().rss

        # Run many iterations
        for _ in range(50):
            env.reset()
            action_probs, value = mcts.search(env.board)

            # Make random moves
            for _ in range(5):
                legal_actions = env.get_legal_actions()
                if legal_actions:
                    env.step(np.random.choice(legal_actions))
                    if env.terminated:
                        break

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        max_acceptable_increase = 50 * 1024**2
        assert memory_increase < max_acceptable_increase, \
            f"Potential memory leak: {memory_increase / 1024**2:.1f}MB increase"


if __name__ == "__main__":
    # Run with custom markers for performance tests
    pytest.main([__file__, "-v", "-s"])