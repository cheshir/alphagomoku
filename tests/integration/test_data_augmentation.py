"""Integration tests for data augmentation in the training pipeline."""

import tempfile
import shutil
import numpy as np
import pytest
from alphagomoku.selfplay.selfplay import SelfPlayData
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.utils.symmetry import BoardSymmetry


class TestDataBufferAugmentation:
    """Test that DataBuffer correctly applies symmetry augmentations."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_lazy_augmentation_sampling(self, temp_db_path):
        """Test that lazy augmentation applies random symmetries correctly."""
        N = 5

        # Create a recognizable position: single stone at (1, 2)
        state = np.zeros((5, N, N), dtype=np.float32)
        state[0, 1, 2] = 1.0  # Current player stone at (1, 2)
        state[2, 1, 2] = 1.0  # Last move marker at (1, 2)

        # Policy concentrated at (1, 2)
        policy = np.zeros(N * N, dtype=np.float32)
        policy[1 * N + 2] = 1.0

        example = SelfPlayData(
            state=state,
            policy=policy,
            value=1.0,
            current_player=1,
            last_move=(1, 2),
            metadata={"move": 0}
        )

        # Create buffer with lazy augmentation
        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=True)
        buffer.add_data([example])

        # Sample multiple times and verify different symmetries are applied
        symmetries_seen = set()
        for _ in range(100):
            batch = buffer.sample_batch(1)
            assert len(batch) == 1

            sample = batch[0]

            # Find where the stone ended up
            stone_pos = np.where(sample.state[0] == 1.0)
            assert len(stone_pos[0]) == 1
            r, c = stone_pos[0][0], stone_pos[1][0]

            # Find where policy is concentrated
            policy_grid = sample.policy.reshape(N, N)
            policy_pos = np.unravel_index(np.argmax(policy_grid), policy_grid.shape)

            # Stone and policy should be at same position
            assert (r, c) == policy_pos, "State and policy not aligned after augmentation"

            # Last move should also match
            if sample.last_move is not None:
                assert sample.last_move == (r, c), "Last move not aligned with state/policy"

            # Track which symmetry this likely represents
            symmetries_seen.add((r, c))

        # We should see at least 4 different orientations out of 8 possible
        # (statistical test - with 100 samples we should see good variety)
        # Note: some symmetries may map to the same position, so we expect >= 4
        assert len(symmetries_seen) >= 4, \
            f"Only saw {len(symmetries_seen)} different orientations, expected variety"

    def test_eager_augmentation_generates_all_8(self, temp_db_path):
        """Test that eager augmentation generates all 8 symmetries."""
        N = 5

        # Create a position
        state = np.zeros((5, N, N), dtype=np.float32)
        state[0, 2, 3] = 1.0
        state[2, 2, 3] = 1.0  # Last move marker

        policy = np.zeros(N * N, dtype=np.float32)
        policy[2 * N + 3] = 1.0

        example = SelfPlayData(
            state=state,
            policy=policy,
            value=0.5,
            current_player=1,
            last_move=(2, 3),
            metadata={}
        )

        # Create buffer with eager augmentation
        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=False)
        buffer.add_data([example])

        # Should have stored 8 examples (all symmetries)
        assert buffer.size == 8

        # Sample all and verify they're all different but valid
        all_positions = set()
        for _ in range(8):
            batch = buffer.sample_batch(1)
            sample = batch[0]

            # Find stone position
            stone_pos = np.where(sample.state[0] == 1.0)
            r, c = stone_pos[0][0], stone_pos[1][0]
            all_positions.add((r, c))

            # Verify consistency
            policy_grid = sample.policy.reshape(N, N)
            assert policy_grid[r, c] == 1.0

        # Should have seen 8 different positions
        # (some might be the same due to symmetries, but we should see several unique ones)
        assert len(all_positions) >= 4  # At least half should be unique

    def test_augmentation_preserves_value(self, temp_db_path):
        """Test that value target is unchanged by augmentation."""
        N = 5
        state = np.random.rand(5, N, N).astype(np.float32)
        policy = np.random.rand(N * N).astype(np.float32)
        policy /= policy.sum()

        original_value = 0.75

        example = SelfPlayData(
            state=state,
            policy=policy,
            value=original_value,
            current_player=-1,
            last_move=None,
            metadata={}
        )

        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=True)
        buffer.add_data([example])

        # Sample many times and verify value never changes
        for _ in range(50):
            batch = buffer.sample_batch(1)
            assert batch[0].value == original_value

    def test_augmentation_preserves_probability_mass(self, temp_db_path):
        """Test that policy probability mass is preserved."""
        N = 5
        state = np.random.rand(5, N, N).astype(np.float32)
        policy = np.random.rand(N * N).astype(np.float32)
        policy /= policy.sum()

        example = SelfPlayData(
            state=state,
            policy=policy,
            value=0.0,
            current_player=1,
            last_move=None,
            metadata={}
        )

        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=True)
        buffer.add_data([example])

        # Sample and verify probability mass
        for _ in range(20):
            batch = buffer.sample_batch(1)
            sample_policy = batch[0].policy
            assert np.isclose(sample_policy.sum(), 1.0, atol=1e-6)

    def test_multiple_examples_independent_augmentation(self, temp_db_path):
        """Test that different examples get independent random augmentations."""
        N = 5

        examples = []
        for i in range(5):
            state = np.zeros((5, N, N), dtype=np.float32)
            state[0, i, i] = 1.0  # Different position for each

            policy = np.zeros(N * N, dtype=np.float32)
            policy[i * N + i] = 1.0

            examples.append(SelfPlayData(
                state=state,
                policy=policy,
                value=float(i),
                current_player=1,
                last_move=(i, i),
                metadata={}
            ))

        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=True)
        buffer.add_data(examples)

        # Sample a batch
        batch = buffer.sample_batch(5)
        assert len(batch) == 5

        # Each sample should have undergone independent augmentation
        # Check that they have different patterns (not all the same symmetry)
        stone_positions = []
        for sample in batch:
            stone_pos = np.where(sample.state[0] > 0)
            if len(stone_pos[0]) > 0:
                stone_positions.append((stone_pos[0][0], stone_pos[1][0]))

        # Should have variety in positions
        unique_positions = len(set(stone_positions))
        assert unique_positions >= 3, "Samples don't show enough variety"


class TestSymmetryConsistency:
    """Test consistency between state, policy, and last_move after augmentation."""

    def test_last_move_state_policy_alignment(self):
        """Test that last_move, state last_move channel, and policy stay aligned."""
        N = 7
        last_move_orig = (3, 4)

        # Create state with last_move channel
        state = np.zeros((5, N, N), dtype=np.float32)
        state[2, 3, 4] = 1.0  # Channel 2 = last move

        # Policy concentrated at last_move
        policy = np.zeros(N * N, dtype=np.float32)
        policy[3 * N + 4] = 0.8
        policy /= policy.sum()

        # Test all 8 symmetries
        for sym_id in range(8):
            state_sym, policy_sym, _, last_move_sym, _, _ = BoardSymmetry.apply_symmetry(
                state, policy, 0.0, sym_id, last_move_orig
            )

            r_exp, c_exp = last_move_sym

            # Check last_move channel
            assert state_sym[2, r_exp, c_exp] == 1.0, \
                f"Symmetry {sym_id}: last_move channel not at expected position"

            # Check policy has high prob at that position
            policy_grid = policy_sym.reshape(N, N)
            assert policy_grid[r_exp, c_exp] >= 0.7, \
                f"Symmetry {sym_id}: policy not concentrated at expected position"


class TestBatchGeneration:
    """Test that batches are generated correctly with augmentation."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_batch_shapes_consistent(self, temp_db_path):
        """Test that all samples in a batch have consistent shapes."""
        N = 5

        examples = []
        for _ in range(10):
            state = np.random.rand(5, N, N).astype(np.float32)
            policy = np.random.rand(N * N).astype(np.float32)
            policy /= policy.sum()

            examples.append(SelfPlayData(
                state=state,
                policy=policy,
                value=np.random.randn(),
                current_player=1,
                last_move=None,
                metadata={}
            ))

        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=True)
        buffer.add_data(examples)

        # Sample batch
        batch = buffer.sample_batch(8)

        # All states should have same shape
        state_shapes = [sample.state.shape for sample in batch]
        assert all(shape == (5, N, N) for shape in state_shapes)

        # All policies should have same shape
        policy_shapes = [sample.policy.shape for sample in batch]
        assert all(shape == (N * N,) for shape in policy_shapes)

    def test_no_nans_or_infs_in_batch(self, temp_db_path):
        """Test that augmentation never introduces NaNs or Infs."""
        N = 5

        examples = []
        for _ in range(20):
            state = np.random.rand(5, N, N).astype(np.float32)
            policy = np.random.rand(N * N).astype(np.float32)
            policy /= policy.sum()

            examples.append(SelfPlayData(
                state=state,
                policy=policy,
                value=np.random.randn(),
                current_player=1,
                last_move=None,
                metadata={}
            ))

        buffer = DataBuffer(temp_db_path, max_size=100, lazy_augmentation=True)
        buffer.add_data(examples)

        # Sample many batches
        for _ in range(50):
            batch = buffer.sample_batch(10)

            for sample in batch:
                assert not np.any(np.isnan(sample.state))
                assert not np.any(np.isinf(sample.state))
                assert not np.any(np.isnan(sample.policy))
                assert not np.any(np.isinf(sample.policy))
                assert not np.isnan(sample.value)
                assert not np.isinf(sample.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
