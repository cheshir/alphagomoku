"""Unit tests for DataBuffer module."""

import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import patch
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayData


class TestDataBuffer:
    """Test DataBuffer class."""

    def test_initialization(self, temp_data_dir):
        """Test DataBuffer initialization."""
        buffer = DataBuffer(db_path=temp_data_dir, max_size=1000)

        assert buffer.db_path == temp_data_dir
        assert buffer.max_size == 1000
        assert buffer.size == 0
        assert buffer.write_idx == 0
        assert buffer.env is not None

    def test_add_single_data(self, data_buffer_small):
        """Test adding single training example."""
        buffer = data_buffer_small

        data = [SelfPlayData(
            state=np.zeros((5, 15, 15), dtype=np.float32),
            policy=np.ones(225) / 225,
            value=0.5
        )]

        buffer.add_data(data)
        assert buffer.size == 1  # Lazy augmentation stores 1 original

    def test_sample_data(self, data_buffer_small):
        """Test sampling data from buffer."""
        buffer = data_buffer_small

        # Add some data
        data_list = []
        for i in range(3):
            data = SelfPlayData(
                state=np.random.rand(5, 15, 15).astype(np.float32),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1
            )
            data.policy = data.policy / np.sum(data.policy)
            data_list.append(data)

        buffer.add_data(data_list)

        # Sample data
        batch_size = 10
        sampled = buffer.sample(batch_size)

        assert len(sampled) == batch_size
        assert all(isinstance(item, SelfPlayData) for item in sampled)

    def test_empty_buffer_sampling(self, data_buffer_small):
        """Test sampling from empty buffer."""
        buffer = data_buffer_small

        # Try to sample from empty buffer
        sampled = buffer.sample(10)

        # Should return empty list
        assert len(sampled) == 0

    def test_data_persistence(self, temp_data_dir):
        """Test that data persists across buffer instances."""
        # Create buffer and add data
        buffer1 = DataBuffer(db_path=temp_data_dir, max_size=100)
        data = [SelfPlayData(
            state=np.ones((5, 15, 15), dtype=np.float32),
            policy=np.ones(225) / 225,
            value=0.5
        )]
        buffer1.add_data(data)
        size_after_add = buffer1.size

        # Close and reopen
        buffer1.env.close()
        buffer2 = DataBuffer(db_path=temp_data_dir, max_size=100)

        # Size should be restored
        assert buffer2.size == size_after_add

    def test_cleanup(self, data_buffer_small):
        """Test proper cleanup of resources."""
        buffer = data_buffer_small

        # Test that environment can be closed
        buffer.env.close()

        # After closing, should not be able to add data
        data = [SelfPlayData(
            state=np.zeros((5, 15, 15), dtype=np.float32),
            policy=np.ones(225) / 225,
            value=0.0
        )]

        with pytest.raises(Exception):  # Should raise error with closed env
            buffer.add_data(data)


class TestDataBufferIntegration:
    """Integration tests for DataBuffer."""

    def test_integration_with_selfplay_data(self, temp_data_dir):
        """Test integration with realistic SelfPlay data."""
        buffer = DataBuffer(db_path=temp_data_dir, max_size=500)

        # Create realistic game data sequence
        game_data = []
        for move_num in range(10):
            # Create policy with some concentration
            policy = np.random.rand(225) * 0.1
            policy[move_num * 20 % 225] = 0.5  # Concentrated probability
            policy = policy / np.sum(policy)

            game_data.append(SelfPlayData(
                state=np.random.rand(5, 15, 15).astype(np.float32),
                policy=policy,
                value=(move_num - 5) * 0.1  # Simulate changing evaluation
            ))

        buffer.add_data(game_data)

        # Test sampling
        batch = buffer.sample(32)
        assert len(batch) == 32
        assert all(isinstance(sample, SelfPlayData) for sample in batch)

        # Test data consistency
        for sample in batch[:5]:  # Check first few samples
            assert sample.state.shape == (5, 15, 15)
            assert len(sample.policy) == 225
            assert np.isclose(np.sum(sample.policy), 1.0)
            assert -1 <= sample.value <= 1

    def test_large_scale_operations(self, temp_data_dir):
        """Test buffer with larger scale operations."""
        buffer = DataBuffer(db_path=temp_data_dir, max_size=1000)

        # Add data in multiple batches
        for batch_num in range(20):
            batch_data = []
            for i in range(5):
                policy = np.random.rand(225)
                policy = policy / np.sum(policy)

                batch_data.append(SelfPlayData(
                    state=np.random.rand(5, 15, 15).astype(np.float32),
                    policy=policy,
                    value=np.random.rand() * 2 - 1
                ))

            buffer.add_data(batch_data)

        # Buffer should handle large amounts of data
        assert buffer.size > 0
        assert buffer.size <= buffer.max_size  # Should not exceed capacity

        # Sampling should work efficiently
        large_batch = buffer.sample(100)
        assert len(large_batch) == 100

    def test_buffer_reuse_across_training(self, temp_data_dir):
        """Test buffer reuse across multiple training sessions."""
        # Session 1: Add initial data
        buffer1 = DataBuffer(db_path=temp_data_dir, max_size=200)

        session1_data = []
        for i in range(10):
            data = SelfPlayData(
                state=np.random.rand(5, 15, 15).astype(np.float32),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1
            )
            data.policy = data.policy / np.sum(data.policy)
            session1_data.append(data)

        buffer1.add_data(session1_data)
        size_after_session1 = buffer1.size
        buffer1.env.close()

        # Session 2: Reuse buffer and add more data
        buffer2 = DataBuffer(db_path=temp_data_dir, max_size=200)
        assert buffer2.size == size_after_session1

        session2_data = []
        for i in range(5):
            data = SelfPlayData(
                state=np.random.rand(5, 15, 15).astype(np.float32),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1
            )
            data.policy = data.policy / np.sum(data.policy)
            session2_data.append(data)

        buffer2.add_data(session2_data)

        # Should have accumulated data from both sessions
        assert buffer2.size > size_after_session1


if __name__ == "__main__":
    pytest.main([__file__])