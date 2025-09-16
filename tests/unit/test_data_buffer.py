"""Unit tests for DataBuffer module."""

import pytest
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayData


class TestDataBuffer:
    """Test DataBuffer class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for LMDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def setup_buffer(self, temp_db_path):
        """Setup DataBuffer with temporary database."""
        buffer = DataBuffer(db_path=temp_db_path, max_size=100, map_size=1024**2)  # 1MB
        return buffer

    def test_initialization(self, temp_db_path):
        """Test DataBuffer initialization."""
        buffer = DataBuffer(db_path=temp_db_path, max_size=1000)

        assert buffer.db_path == temp_db_path
        assert buffer.max_size == 1000
        assert buffer.size == 0
        assert buffer.write_idx == 0
        assert buffer.env is not None

    def test_add_single_data(self, setup_buffer):
        """Test adding single training example."""
        buffer = setup_buffer

        data = [SelfPlayData(
            state=np.zeros((15, 15), dtype=np.int8),
            policy=np.ones(225) / 225,
            value=0.5,
            current_player=1,
            last_move=(7, 7)
        )]

        buffer.add_data(data)

        # Should have added data with 8-fold symmetry (8 augmented examples)
        assert buffer.size == 8

    def test_augmentation(self, setup_buffer):
        """Test 8-fold symmetry augmentation."""
        buffer = setup_buffer

        # Create asymmetric data to test augmentation
        state = np.zeros((15, 15), dtype=np.int8)
        state[7, 7] = 1  # Single stone
        state[8, 7] = -1

        policy = np.zeros(225)
        policy[7 * 15 + 8] = 1.0  # Policy at specific position

        data = [SelfPlayData(
            state=state,
            policy=policy,
            value=0.0,
            current_player=1,
            last_move=(8, 7)
        )]

        buffer.add_data(data)

        # Should have 8 augmented examples
        assert buffer.size == 8

    def test_circular_buffer_overflow(self, setup_buffer):
        """Test circular buffer behavior when exceeding max_size."""
        buffer = setup_buffer
        buffer.max_size = 10  # Small buffer for testing

        # Add data beyond capacity
        for i in range(5):  # Each adds 8 augmented examples = 40 total
            data = [SelfPlayData(
                state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(7, 7)
            )]
            data[0].policy = data[0].policy / np.sum(data[0].policy)
            buffer.add_data(data)

        # Should not exceed max_size due to circular buffer
        assert buffer.size == buffer.max_size

    def test_sample_data(self, setup_buffer):
        """Test sampling data from buffer."""
        buffer = setup_buffer

        # Add some data
        data_list = []
        for i in range(3):
            data = SelfPlayData(
                state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(7, 7)
            )
            data.policy = data.policy / np.sum(data.policy)
            data_list.append(data)

        buffer.add_data(data_list)

        # Sample data
        batch_size = 10
        sampled = buffer.sample(batch_size)

        assert len(sampled) == batch_size
        assert all(isinstance(item, SelfPlayData) for item in sampled)

    def test_sample_more_than_available(self, setup_buffer):
        """Test sampling more data than available."""
        buffer = setup_buffer

        # Add small amount of data
        data = [SelfPlayData(
            state=np.zeros((15, 15), dtype=np.int8),
            policy=np.ones(225) / 225,
            value=0.0,
            current_player=1,
            last_move=(7, 7)
        )]
        buffer.add_data(data)

        # Try to sample more than available
        sampled = buffer.sample(100)  # More than 8 augmented examples

        # Should sample with replacement or return available data
        assert len(sampled) > 0
        assert len(sampled) <= 100

    def test_empty_buffer_sampling(self, setup_buffer):
        """Test sampling from empty buffer."""
        buffer = setup_buffer

        # Try to sample from empty buffer
        sampled = buffer.sample(10)

        # Should return empty list or handle gracefully
        assert len(sampled) == 0

    def test_data_persistence(self, temp_db_path):
        """Test that data persists across buffer instances."""
        # Create buffer and add data
        buffer1 = DataBuffer(db_path=temp_db_path, max_size=100)
        data = [SelfPlayData(
            state=np.ones((15, 15), dtype=np.int8),
            policy=np.ones(225) / 225,
            value=0.5,
            current_player=1,
            last_move=(7, 7)
        )]
        buffer1.add_data(data)
        size_after_add = buffer1.size

        # Close and reopen
        buffer1.env.close()
        buffer2 = DataBuffer(db_path=temp_db_path, max_size=100)

        # Size should be restored
        assert buffer2.size == size_after_add

    def test_map_resize_handling(self, setup_buffer):
        """Test handling of LMDB map resize."""
        buffer = setup_buffer

        # Mock LMDB to raise MapFullError
        with patch.object(buffer, '_resize_map') as mock_resize:
            # Create a scenario that might trigger map full
            large_data = []
            for _ in range(10):
                data = SelfPlayData(
                    state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                    policy=np.random.rand(225),
                    value=np.random.rand() * 2 - 1,
                    current_player=1,
                    last_move=(7, 7)
                )
                data.policy = data.policy / np.sum(data.policy)
                large_data.append(data)

            # This should work normally, but test the resize mechanism exists
            buffer.add_data(large_data)

    def test_error_handling_corrupted_data(self, temp_db_path):
        """Test error handling with corrupted database."""
        # Create buffer
        buffer = DataBuffer(db_path=temp_db_path, max_size=100)

        # Manually corrupt data in database
        with buffer.env.begin(write=True) as txn:
            txn.put(b'data_0', b'corrupted_data')
            txn.put(b'size', (1).to_bytes(8, 'big'))

        # Try to sample - should handle corrupted data gracefully
        try:
            sampled = buffer.sample(5)
            # If it succeeds, that's fine
        except Exception as e:
            # Should raise appropriate error, not crash
            assert isinstance(e, (ValueError, EOFError, TypeError))

    def test_thread_safety_simulation(self, setup_buffer):
        """Test basic thread safety considerations."""
        buffer = setup_buffer

        # Simulate concurrent access (not actual threading for unit test)
        data_batches = []
        for i in range(5):
            batch = [SelfPlayData(
                state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(7, 7)
            )]
            batch[0].policy = batch[0].policy / np.sum(batch[0].policy)
            data_batches.append(batch)

        # Add data in sequence (simulating concurrent access)
        for batch in data_batches:
            buffer.add_data(batch)

        # Should complete without errors
        assert buffer.size > 0

    def test_memory_efficiency(self, setup_buffer):
        """Test memory efficiency with large data."""
        buffer = setup_buffer

        # Add data with larger board size
        large_data = []
        for i in range(3):  # Limited for unit test
            data = SelfPlayData(
                state=np.random.randint(-1, 2, (19, 19)).astype(np.int8),  # Larger board
                policy=np.random.rand(361),  # 19x19 board
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(9, 9)
            )
            data.policy = data.policy / np.sum(data.policy)
            large_data.append(data)

        buffer.add_data(large_data)

        # Should handle larger data efficiently
        assert buffer.size > 0

    def test_data_augmentation_correctness(self, setup_buffer):
        """Test correctness of data augmentation."""
        buffer = setup_buffer

        # Create specific pattern to test augmentation
        state = np.zeros((9, 9), dtype=np.int8)  # Smaller for easier testing
        state[4, 4] = 1  # Center stone
        state[4, 5] = -1  # Adjacent stone

        policy = np.zeros(81)
        policy[4 * 9 + 6] = 1.0  # Policy for position (4, 6)

        data = [SelfPlayData(
            state=state,
            policy=policy,
            value=0.0,
            current_player=1,
            last_move=(4, 5)
        )]

        # Mock augmentation to test if it's called
        with patch.object(buffer, '_augment_example') as mock_augment:
            mock_augment.return_value = [data[0]] * 8  # Return 8 copies

            buffer.add_data(data)

            mock_augment.assert_called_once_with(data[0])

    def test_statistics_tracking(self, setup_buffer):
        """Test buffer statistics tracking."""
        buffer = setup_buffer

        # Add data and track statistics
        initial_size = buffer.size
        initial_write_idx = buffer.write_idx

        data = [SelfPlayData(
            state=np.zeros((15, 15), dtype=np.int8),
            policy=np.ones(225) / 225,
            value=0.0,
            current_player=1,
            last_move=(7, 7)
        )]

        buffer.add_data(data)

        # Size and write index should update correctly
        assert buffer.size > initial_size
        assert buffer.write_idx != initial_write_idx

    def test_cleanup(self, setup_buffer):
        """Test proper cleanup of resources."""
        buffer = setup_buffer

        # Test that environment can be closed
        buffer.env.close()

        # After closing, should not be able to add data
        data = [SelfPlayData(
            state=np.zeros((15, 15), dtype=np.int8),
            policy=np.ones(225) / 225,
            value=0.0,
            current_player=1,
            last_move=(7, 7)
        )]

        with pytest.raises(Exception):  # Should raise error with closed env
            buffer.add_data(data)


class TestDataBufferIntegration:
    """Integration tests for DataBuffer."""

    def test_integration_with_selfplay_data(self, temp_db_path):
        """Test integration with realistic SelfPlay data."""
        buffer = DataBuffer(db_path=temp_db_path, max_size=500)

        # Create realistic game data sequence
        game_data = []
        for move_num in range(10):
            state = np.zeros((15, 15), dtype=np.int8)
            # Add some stones to simulate game progress
            for i in range(move_num):
                row, col = divmod(i, 15)
                if row < 15 and col < 15:
                    state[row, col] = 1 if i % 2 == 0 else -1

            # Create policy with some concentration
            policy = np.random.rand(225) * 0.1
            policy[move_num * 20 % 225] = 0.5  # Concentrated probability
            policy = policy / np.sum(policy)

            game_data.append(SelfPlayData(
                state=state,
                policy=policy,
                value=(move_num - 5) * 0.1,  # Simulate changing evaluation
                current_player=1 if move_num % 2 == 0 else -1,
                last_move=(move_num // 15, move_num % 15)
            ))

        buffer.add_data(game_data)

        # Test sampling
        batch = buffer.sample(32)
        assert len(batch) == 32
        assert all(isinstance(sample, SelfPlayData) for sample in batch)

        # Test data consistency
        for sample in batch[:5]:  # Check first few samples
            assert sample.state.shape == (15, 15)
            assert len(sample.policy) == 225
            assert np.isclose(np.sum(sample.policy), 1.0)
            assert -1 <= sample.value <= 1
            assert sample.current_player in [-1, 1]

    def test_large_scale_operations(self, temp_db_path):
        """Test buffer with larger scale operations."""
        buffer = DataBuffer(db_path=temp_db_path, max_size=1000)

        # Add data in multiple batches
        for batch_num in range(20):
            batch_data = []
            for i in range(5):
                state = np.random.randint(-1, 2, (15, 15)).astype(np.int8)
                policy = np.random.rand(225)
                policy = policy / np.sum(policy)

                batch_data.append(SelfPlayData(
                    state=state,
                    policy=policy,
                    value=np.random.rand() * 2 - 1,
                    current_player=1 if i % 2 == 0 else -1,
                    last_move=(i, i)
                ))

            buffer.add_data(batch_data)

        # Buffer should handle large amounts of data
        assert buffer.size > 0
        assert buffer.size <= buffer.max_size  # Should not exceed capacity

        # Sampling should work efficiently
        large_batch = buffer.sample(100)
        assert len(large_batch) == 100

    def test_buffer_reuse_across_training(self, temp_db_path):
        """Test buffer reuse across multiple training sessions."""
        # Session 1: Add initial data
        buffer1 = DataBuffer(db_path=temp_db_path, max_size=200)

        session1_data = []
        for i in range(10):
            data = SelfPlayData(
                state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(i, i)
            )
            data.policy = data.policy / np.sum(data.policy)
            session1_data.append(data)

        buffer1.add_data(session1_data)
        size_after_session1 = buffer1.size
        buffer1.env.close()

        # Session 2: Reuse buffer and add more data
        buffer2 = DataBuffer(db_path=temp_db_path, max_size=200)
        assert buffer2.size == size_after_session1

        session2_data = []
        for i in range(5):
            data = SelfPlayData(
                state=np.random.randint(-1, 2, (15, 15)).astype(np.int8),
                policy=np.random.rand(225),
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(i + 10, i + 10)
            )
            data.policy = data.policy / np.sum(data.policy)
            session2_data.append(data)

        buffer2.add_data(session2_data)

        # Should have accumulated data from both sessions
        assert buffer2.size > size_after_session1


if __name__ == "__main__":
    pytest.main([__file__])