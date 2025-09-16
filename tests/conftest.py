"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer


@pytest.fixture(scope="session")
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
    # Reset after session if needed


@pytest.fixture
def small_model():
    """Small model for fast testing."""
    return GomokuNet(board_size=5, num_blocks=1, channels=4)


@pytest.fixture
def medium_model():
    """Medium model for realistic testing."""
    return GomokuNet(board_size=9, num_blocks=2, channels=8)


@pytest.fixture
def small_env():
    """Small environment for fast testing."""
    return GomokuEnv(board_size=5)


@pytest.fixture
def medium_env():
    """Medium environment for realistic testing."""
    return GomokuEnv(board_size=9)


@pytest.fixture
def small_mcts(small_model, small_env):
    """Small MCTS configuration for fast testing."""
    return MCTS(small_model, small_env, num_simulations=5)


@pytest.fixture
def medium_mcts(medium_model, medium_env):
    """Medium MCTS configuration for realistic testing."""
    return MCTS(medium_model, medium_env, num_simulations=20)


@pytest.fixture
def trainer_small(small_model):
    """Trainer with small model."""
    return Trainer(small_model, device='cpu', lr=0.01)


@pytest.fixture
def trainer_medium(medium_model):
    """Trainer with medium model."""
    return Trainer(medium_model, device='cpu', lr=0.001)


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_buffer_small(temp_data_dir):
    """Small data buffer for testing."""
    return DataBuffer(db_path=temp_data_dir, max_size=50, map_size=1024**2)


@pytest.fixture
def sample_training_data():
    """Sample training data for tests."""
    from alphagomoku.selfplay.selfplay import SelfPlayData

    data = []
    for i in range(10):
        state = np.random.randint(-1, 2, (9, 9)).astype(np.int8)
        policy = np.random.rand(81)
        policy = policy / np.sum(policy)

        data.append(SelfPlayData(
            state=state,
            policy=policy,
            value=np.random.rand() * 2 - 1,
            current_player=1 if i % 2 == 0 else -1,
            last_move=(i % 9, (i + 1) % 9)
        ))

    return data


@pytest.fixture
def tactical_position():
    """Gomoku position with tactical elements for TSS testing."""
    board = np.zeros((15, 15), dtype=np.int8)

    # Create open three
    board[7, 6] = 1
    board[7, 7] = 1
    board[7, 8] = 1

    # Add some opponent stones
    board[8, 7] = -1
    board[6, 7] = -1

    return board


@pytest.fixture
def winning_position():
    """Gomoku position close to winning."""
    board = np.zeros((15, 15), dtype=np.int8)

    # Four in a row (one move from win)
    for i in range(4):
        board[7, 5 + i] = 1

    # Some opponent stones
    board[6, 6] = -1
    board[6, 7] = -1
    board[8, 6] = -1

    return board


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit test")
    config.addinivalue_line("markers", "integration: Integration test")
    config.addinivalue_line("markers", "performance: Performance test")
    config.addinivalue_line("markers", "slow: Slow test")
    config.addinivalue_line("markers", "gpu: Requires GPU")
    config.addinivalue_line("markers", "parallel: Uses multiprocessing")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add markers based on file path
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add specific component markers
        if "mcts" in item.name.lower():
            item.add_marker(pytest.mark.mcts)

        if "tss" in item.name.lower():
            item.add_marker(pytest.mark.tss)

        if "parallel" in item.name.lower():
            item.add_marker(pytest.mark.parallel)


@pytest.fixture(autouse=True)
def reset_torch_threads():
    """Reset PyTorch thread count for consistent testing."""
    torch.set_num_threads(1)  # Consistent threading for tests
    yield
    torch.set_num_threads(torch.get_num_threads())  # Reset to default


@pytest.fixture
def mock_model_forward():
    """Mock model forward pass for faster testing."""
    def mock_forward(batch_size, action_size):
        policy = torch.rand(batch_size, action_size)
        value = torch.rand(batch_size)
        return policy, value

    return mock_forward


# Performance fixtures
@pytest.fixture
def performance_monitor():
    """Monitor for tracking test performance."""
    import time
    import psutil

    class PerfMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss

        def stop(self):
            if self.start_time is None:
                return {}

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            return {
                'duration': end_time - self.start_time,
                'memory_delta': end_memory - self.start_memory,
                'peak_memory': end_memory
            }

    return PerfMonitor()