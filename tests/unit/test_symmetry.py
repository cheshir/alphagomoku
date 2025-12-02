"""Unit tests for board symmetry transformations."""

import numpy as np
import pytest
from alphagomoku.utils.symmetry import BoardSymmetry


class TestCoordinateTransformations:
    """Test coordinate mapping correctness for all 8 symmetries."""

    def test_identity_transformation(self):
        """Test identity leaves coordinates unchanged."""
        N = 5
        for r in range(N):
            for c in range(N):
                r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=0)
                assert (r_new, c_new) == (r, c)

    def test_rotation_90_cw(self):
        """Test 90° clockwise rotation: (r, c) -> (c, N-1-r)."""
        N = 5
        test_cases = [
            ((0, 0), (0, 4)),  # Top-left -> Top-right
            ((0, 4), (4, 4)),  # Top-right -> Bottom-right
            ((4, 4), (4, 0)),  # Bottom-right -> Bottom-left
            ((4, 0), (0, 0)),  # Bottom-left -> Top-left
            ((2, 2), (2, 2)),  # Center -> Center
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=1)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_rotation_180(self):
        """Test 180° rotation: (r, c) -> (N-1-r, N-1-c)."""
        N = 5
        test_cases = [
            ((0, 0), (4, 4)),  # Top-left -> Bottom-right
            ((0, 4), (4, 0)),  # Top-right -> Bottom-left
            ((2, 2), (2, 2)),  # Center -> Center
            ((1, 3), (3, 1)),  # Arbitrary point
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=2)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_rotation_270_cw(self):
        """Test 270° clockwise rotation: (r, c) -> (N-1-c, r)."""
        N = 5
        test_cases = [
            ((0, 0), (4, 0)),  # Top-left -> Bottom-left
            ((0, 4), (0, 0)),  # Top-right -> Top-left
            ((4, 4), (0, 4)),  # Bottom-right -> Top-right
            ((4, 0), (4, 4)),  # Bottom-left -> Bottom-right
            ((2, 2), (2, 2)),  # Center -> Center
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=3)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_flip_vertical(self):
        """Test vertical flip (left-right): (r, c) -> (r, N-1-c)."""
        N = 5
        test_cases = [
            ((0, 0), (0, 4)),  # Top-left -> Top-right
            ((0, 4), (0, 0)),  # Top-right -> Top-left
            ((2, 2), (2, 2)),  # Center -> Center
            ((3, 1), (3, 3)),  # Arbitrary point
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=4)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_flip_horizontal(self):
        """Test horizontal flip (top-bottom): (r, c) -> (N-1-r, c)."""
        N = 5
        test_cases = [
            ((0, 0), (4, 0)),  # Top-left -> Bottom-left
            ((4, 0), (0, 0)),  # Bottom-left -> Top-left
            ((2, 2), (2, 2)),  # Center -> Center
            ((1, 3), (3, 3)),  # Arbitrary point
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=5)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_flip_main_diagonal(self):
        """Test main diagonal flip (transpose): (r, c) -> (c, r)."""
        N = 5
        test_cases = [
            ((0, 0), (0, 0)),  # On diagonal
            ((4, 4), (4, 4)),  # On diagonal
            ((0, 4), (4, 0)),  # Off diagonal
            ((1, 3), (3, 1)),  # Arbitrary point
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=6)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_flip_anti_diagonal(self):
        """Test anti-diagonal flip: (r, c) -> (N-1-c, N-1-r)."""
        N = 5
        test_cases = [
            ((0, 4), (0, 4)),  # On anti-diagonal
            ((4, 0), (4, 0)),  # On anti-diagonal
            ((0, 0), (4, 4)),  # Off anti-diagonal
            ((1, 3), (1, 3)),  # On anti-diagonal
        ]
        for (r, c), (r_exp, c_exp) in test_cases:
            r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id=7)
            assert (r_new, c_new) == (r_exp, c_exp), f"Failed for ({r}, {c})"

    def test_invalid_symmetry_id(self):
        """Test that invalid symmetry IDs raise ValueError."""
        with pytest.raises(ValueError):
            BoardSymmetry.transform_coordinates(0, 0, 5, sym_id=8)
        with pytest.raises(ValueError):
            BoardSymmetry.transform_coordinates(0, 0, 5, sym_id=-1)


class TestBoardTensorTransformations:
    """Test transformations on actual board tensors."""

    def test_unique_index_tensor_transformations(self):
        """Test that each cell transforms correctly using unique indices."""
        N = 5
        # Create tensor where each cell has unique value = r * N + c
        board = np.arange(N * N, dtype=np.float32).reshape(N, N)

        for sym_id in range(8):
            board_sym = BoardSymmetry.transform_board_tensor(board, sym_id)

            # Verify each cell ended up in the right place
            for r in range(N):
                for c in range(N):
                    original_value = r * N + c
                    r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id)
                    assert board_sym[r_new, c_new] == original_value, \
                        f"Symmetry {sym_id}: cell ({r},{c}) with value {original_value} " \
                        f"should be at ({r_new},{c_new}) but found {board_sym[r_new, c_new]}"

    def test_3d_tensor_transformations(self):
        """Test transformations on 3D tensors [C, H, W]."""
        N = 5
        C = 3
        # Create 3D tensor where each channel has unique pattern
        board_3d = np.zeros((C, N, N), dtype=np.float32)
        for ch in range(C):
            board_3d[ch] = np.arange(N * N).reshape(N, N) + ch * 100

        for sym_id in range(8):
            board_sym = BoardSymmetry.transform_board_tensor(board_3d, sym_id)

            # Verify each channel transforms independently but identically
            for ch in range(C):
                for r in range(N):
                    for c in range(N):
                        original_value = r * N + c + ch * 100
                        r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id)
                        assert board_sym[ch, r_new, c_new] == original_value, \
                            f"Symmetry {sym_id}, channel {ch}: " \
                            f"cell ({r},{c}) should be at ({r_new},{c_new})"


class TestPolicyTransformations:
    """Test policy transformations in both flat and grid formats."""

    def test_policy_grid_transformation(self):
        """Test policy transformation in grid format [N, N]."""
        N = 5
        # Create policy with probability only at (1, 3)
        policy_grid = np.zeros((N, N), dtype=np.float32)
        policy_grid[1, 3] = 1.0

        for sym_id in range(8):
            policy_sym = BoardSymmetry.transform_policy_grid(policy_grid, sym_id)

            # Find where the probability ended up
            r_new, c_new = BoardSymmetry.transform_coordinates(1, 3, N, sym_id)

            # Verify probability is at the correct location
            assert policy_sym[r_new, c_new] == 1.0, \
                f"Symmetry {sym_id}: probability should be at ({r_new},{c_new})"
            assert policy_sym.sum() == 1.0, \
                f"Symmetry {sym_id}: total probability should be 1.0"

    def test_policy_flat_transformation(self):
        """Test policy transformation in flat format [N*N]."""
        N = 5
        # Create policy with probability only at position (1, 3)
        policy_flat = np.zeros(N * N, dtype=np.float32)
        idx_orig = 1 * N + 3  # Row-major: index = r * N + c
        policy_flat[idx_orig] = 1.0

        for sym_id in range(8):
            policy_sym = BoardSymmetry.transform_policy_flat(policy_flat, N, sym_id)

            # Find where the probability should be
            r_new, c_new = BoardSymmetry.transform_coordinates(1, 3, N, sym_id)
            idx_new = r_new * N + c_new

            # Verify probability is at the correct index
            assert policy_sym[idx_new] == 1.0, \
                f"Symmetry {sym_id}: probability should be at index {idx_new} " \
                f"(position ({r_new},{c_new}))"
            assert policy_sym.sum() == 1.0, \
                f"Symmetry {sym_id}: total probability should be 1.0"

    def test_policy_mass_preservation(self):
        """Test that policy probability mass is preserved."""
        N = 5
        # Create random policy
        policy_grid = np.random.rand(N, N).astype(np.float32)
        policy_grid /= policy_grid.sum()  # Normalize

        for sym_id in range(8):
            policy_sym = BoardSymmetry.transform_policy_grid(policy_grid, sym_id)
            assert np.isclose(policy_sym.sum(), 1.0), \
                f"Symmetry {sym_id}: probability mass not preserved"


class TestRoundtripProperties:
    """Test that applying a symmetry and its inverse returns the original."""

    def test_inverse_symmetries(self):
        """Test that applying symmetry and its inverse gives identity."""
        N = 5
        # Create random board
        board = np.random.rand(N, N).astype(np.float32)

        for sym_id in range(8):
            inverse_id = BoardSymmetry.INVERSE_SYMMETRIES[sym_id]

            # Apply symmetry then inverse
            board_sym = BoardSymmetry.transform_board_tensor(board, sym_id)
            board_restored = BoardSymmetry.transform_board_tensor(board_sym, inverse_id)

            assert np.allclose(board, board_restored), \
                f"Symmetry {sym_id} -> {inverse_id} did not restore original"

    def test_policy_state_consistency(self):
        """Test that state and policy transform consistently."""
        N = 5
        # Create state with last_move at (2, 3)
        state = np.zeros((5, N, N), dtype=np.float32)
        state[2, 2, 3] = 1.0  # Channel 2 = last_move

        # Create policy with probability at (2, 3)
        policy = np.zeros((N, N), dtype=np.float32)
        policy[2, 3] = 1.0

        last_move = (2, 3)

        for sym_id in range(8):
            state_sym, policy_sym, _, last_move_sym, _, _ = BoardSymmetry.apply_symmetry(
                state, policy, value=0.0, sym_id=sym_id, last_move=last_move
            )

            # Verify last_move transformed correctly
            r_new, c_new = BoardSymmetry.transform_coordinates(2, 3, N, sym_id)
            assert last_move_sym == (r_new, c_new), \
                f"Symmetry {sym_id}: last_move should be ({r_new}, {c_new})"

            # Verify last_move channel and policy align
            assert state_sym[2, r_new, c_new] == 1.0, \
                f"Symmetry {sym_id}: last_move channel should have 1.0 at ({r_new}, {c_new})"
            assert policy_sym[r_new, c_new] == 1.0, \
                f"Symmetry {sym_id}: policy should have 1.0 at ({r_new}, {c_new})"


class TestFullAugmentation:
    """Test the complete augmentation pipeline."""

    def test_apply_symmetry_complete(self):
        """Test apply_symmetry with all components."""
        N = 5
        # Create complete example
        state = np.random.rand(5, N, N).astype(np.float32)
        policy = np.random.rand(N * N).astype(np.float32)
        policy /= policy.sum()
        value = 0.5
        last_move = (2, 3)
        current_player = 1
        metadata = {"move_index": 10}

        for sym_id in range(8):
            state_sym, policy_sym, value_sym, last_move_sym, player_sym, meta_sym = \
                BoardSymmetry.apply_symmetry(
                    state, policy, value, sym_id, last_move, current_player, metadata
                )

            # Verify shapes
            assert state_sym.shape == state.shape
            assert policy_sym.shape == policy.shape

            # Verify invariants
            assert value_sym == value
            assert player_sym == current_player
            assert meta_sym == metadata

            # Verify probability mass
            assert np.isclose(policy_sym.sum(), policy.sum())

            # Verify no NaNs or Infs
            assert not np.any(np.isnan(state_sym))
            assert not np.any(np.isnan(policy_sym))

    def test_verify_consistency_function(self):
        """Test the consistency verification function."""
        N = 5
        state = np.random.rand(5, N, N).astype(np.float32)
        policy = np.random.rand(N, N).astype(np.float32)
        policy /= policy.sum()

        # All symmetries should pass consistency check
        for sym_id in range(8):
            assert BoardSymmetry.verify_symmetry_consistency(state, policy, sym_id)

    def test_random_symmetry_generation(self):
        """Test random symmetry generation."""
        # Generate 1000 random symmetries and verify they're all valid
        for _ in range(1000):
            sym_id = BoardSymmetry.get_random_symmetry()
            assert 0 <= sym_id < 8


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_board_size_1(self):
        """Test transformations on 1x1 board."""
        N = 1
        board = np.array([[5.0]])

        for sym_id in range(8):
            board_sym = BoardSymmetry.transform_board_tensor(board, sym_id)
            assert board_sym[0, 0] == 5.0, f"Symmetry {sym_id} failed for 1x1 board"

    def test_large_board_size(self):
        """Test transformations on larger boards."""
        N = 19  # Go board size
        board = np.arange(N * N, dtype=np.float32).reshape(N, N)

        for sym_id in range(8):
            board_sym = BoardSymmetry.transform_board_tensor(board, sym_id)

            # Spot check a few positions
            test_positions = [(0, 0), (0, N-1), (N-1, 0), (N-1, N-1), (N//2, N//2)]
            for r, c in test_positions:
                original_value = r * N + c
                r_new, c_new = BoardSymmetry.transform_coordinates(r, c, N, sym_id)
                assert board_sym[r_new, c_new] == original_value

    def test_none_last_move(self):
        """Test that None last_move is handled correctly."""
        N = 5
        state = np.random.rand(5, N, N).astype(np.float32)
        policy = np.random.rand(N, N).astype(np.float32)

        for sym_id in range(8):
            _, _, _, last_move_sym, _, _ = BoardSymmetry.apply_symmetry(
                state, policy, 0.0, sym_id, last_move=None
            )
            assert last_move_sym is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
