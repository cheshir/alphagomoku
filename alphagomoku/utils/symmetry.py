"""Board symmetry transformations for Gomoku.

This module implements the 8 symmetries of a square board (dihedral group D4):
- Identity
- 3 rotations (90°, 180°, 270° clockwise)
- 4 reflections (vertical, horizontal, main diagonal, anti-diagonal)

These are used for data augmentation during training to improve generalization,
without modifying self-play or MCTS logic.
"""

from typing import Optional, Tuple
import numpy as np


class BoardSymmetry:
    """Handles board symmetry transformations for Gomoku.

    Symmetry IDs (0-7) correspond to the 8 elements of the dihedral group D4:
        0: Identity               (r, c) -> (r, c)
        1: Rotate 90° CW         (r, c) -> (c, N-1-r)
        2: Rotate 180°           (r, c) -> (N-1-r, N-1-c)
        3: Rotate 270° CW        (r, c) -> (N-1-c, r)
        4: Flip vertical         (r, c) -> (r, N-1-c)
        5: Flip horizontal       (r, c) -> (N-1-r, c)
        6: Flip main diagonal    (r, c) -> (c, r)
        7: Flip anti-diagonal    (r, c) -> (N-1-c, N-1-r)

    All transformations preserve:
    - Value target (win/loss/draw is rotation-invariant)
    - Current player
    - Legal move structure
    """

    # Symmetry names for debugging/logging
    SYMMETRY_NAMES = [
        "identity",
        "rot90_cw",
        "rot180",
        "rot270_cw",
        "flip_vertical",
        "flip_horizontal",
        "flip_main_diag",
        "flip_anti_diag",
    ]

    # Inverse symmetries: applying symmetry s then inverse[s] gives identity
    INVERSE_SYMMETRIES = [
        0,  # identity -> identity
        3,  # rot90_cw -> rot270_cw
        2,  # rot180 -> rot180
        1,  # rot270_cw -> rot90_cw
        4,  # flip_vertical -> flip_vertical
        5,  # flip_horizontal -> flip_horizontal
        6,  # flip_main_diag -> flip_main_diag
        7,  # flip_anti_diag -> flip_anti_diag
    ]

    @staticmethod
    def transform_coordinates(r: int, c: int, board_size: int, sym_id: int) -> Tuple[int, int]:
        """Transform board coordinates according to symmetry.

        Args:
            r: Row coordinate (0-based)
            c: Column coordinate (0-based)
            board_size: Size of the square board (N)
            sym_id: Symmetry ID (0-7)

        Returns:
            (r', c'): Transformed coordinates
        """
        N = board_size

        if sym_id == 0:  # Identity
            return r, c
        elif sym_id == 1:  # Rotate 90° CW
            return c, N - 1 - r
        elif sym_id == 2:  # Rotate 180°
            return N - 1 - r, N - 1 - c
        elif sym_id == 3:  # Rotate 270° CW
            return N - 1 - c, r
        elif sym_id == 4:  # Flip vertical (left-right)
            return r, N - 1 - c
        elif sym_id == 5:  # Flip horizontal (top-bottom)
            return N - 1 - r, c
        elif sym_id == 6:  # Flip main diagonal
            return c, r
        elif sym_id == 7:  # Flip anti-diagonal
            return N - 1 - c, N - 1 - r
        else:
            raise ValueError(f"Invalid symmetry ID: {sym_id}. Must be 0-7.")

    @staticmethod
    def transform_board_tensor(board: np.ndarray, sym_id: int) -> np.ndarray:
        """Transform a board tensor according to symmetry.

        Works for both 2D [H, W] and 3D [C, H, W] tensors.
        All spatial dimensions are transformed identically.

        Args:
            board: Board tensor of shape [H, W] or [C, H, W]
            sym_id: Symmetry ID (0-7)

        Returns:
            Transformed board tensor of the same shape
        """
        if sym_id == 0:  # Identity
            return board.copy()

        # Handle both 2D and 3D cases
        is_3d = board.ndim == 3
        spatial_axes = (1, 2) if is_3d else (0, 1)

        if sym_id == 1:  # Rotate 90° CW
            # numpy rot90 rotates counter-clockwise by default, so k=-1 for CW
            result = np.rot90(board, k=-1, axes=spatial_axes)
        elif sym_id == 2:  # Rotate 180°
            result = np.rot90(board, k=2, axes=spatial_axes)
        elif sym_id == 3:  # Rotate 270° CW (= 90° CCW)
            result = np.rot90(board, k=1, axes=spatial_axes)
        elif sym_id == 4:  # Flip vertical (left-right)
            axis = 2 if is_3d else 1
            result = np.flip(board, axis=axis)
        elif sym_id == 5:  # Flip horizontal (top-bottom)
            axis = 1 if is_3d else 0
            result = np.flip(board, axis=axis)
        elif sym_id == 6:  # Flip main diagonal (transpose)
            result = np.swapaxes(board, *spatial_axes)
        elif sym_id == 7:  # Flip anti-diagonal
            # Anti-diagonal = transpose + 180° rotation
            result = np.swapaxes(board, *spatial_axes)
            result = np.rot90(result, k=2, axes=spatial_axes)
        else:
            raise ValueError(f"Invalid symmetry ID: {sym_id}. Must be 0-7.")

        return result.copy()

    @staticmethod
    def transform_policy_grid(policy_grid: np.ndarray, sym_id: int) -> np.ndarray:
        """Transform policy in grid format [N, N].

        Args:
            policy_grid: Policy as grid of shape [N, N]
            sym_id: Symmetry ID (0-7)

        Returns:
            Transformed policy grid [N, N]
        """
        return BoardSymmetry.transform_board_tensor(policy_grid, sym_id)

    @staticmethod
    def transform_policy_flat(policy_flat: np.ndarray, board_size: int, sym_id: int) -> np.ndarray:
        """Transform policy in flat format [N*N].

        The flat policy is in row-major order: index k = r * N + c

        Args:
            policy_flat: Policy as flat vector of length N*N
            board_size: Board size N
            sym_id: Symmetry ID (0-7)

        Returns:
            Transformed policy flat [N*N]
        """
        N = board_size

        if sym_id == 0:  # Identity
            return policy_flat.copy()

        # Convert to grid, transform, flatten back
        policy_grid = policy_flat.reshape(N, N)
        policy_grid_transformed = BoardSymmetry.transform_policy_grid(policy_grid, sym_id)
        return policy_grid_transformed.flatten()

    @staticmethod
    def transform_last_move(
        last_move: Optional[Tuple[int, int]],
        board_size: int,
        sym_id: int
    ) -> Optional[Tuple[int, int]]:
        """Transform last move coordinates according to symmetry.

        Args:
            last_move: Last move as (row, col) or None
            board_size: Board size N
            sym_id: Symmetry ID (0-7)

        Returns:
            Transformed last move coordinates or None
        """
        if last_move is None:
            return None

        r, c = last_move
        r_new, c_new = BoardSymmetry.transform_coordinates(r, c, board_size, sym_id)
        return (r_new, c_new)

    @staticmethod
    def apply_symmetry(
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
        sym_id: int,
        last_move: Optional[Tuple[int, int]] = None,
        current_player: int = 1,
        metadata: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[Tuple[int, int]], int, dict]:
        """Apply symmetry to a complete training example.

        This is the main entry point for data augmentation.

        Args:
            state: State tensor [C, H, W] where H=W=board_size
            policy: Policy either as flat [N*N] or grid [N, N]
            value: Value target (unchanged by symmetry)
            sym_id: Symmetry ID (0-7)
            last_move: Optional last move coordinates
            current_player: Current player (unchanged by symmetry)
            metadata: Optional metadata dict (unchanged by symmetry)

        Returns:
            Tuple of (state_sym, policy_sym, value, last_move_sym, current_player, metadata)
        """
        # Transform state
        state_sym = BoardSymmetry.transform_board_tensor(state, sym_id)

        # Transform policy (handle both flat and grid formats)
        if policy.ndim == 1:
            # Flat format
            board_size = state.shape[-1]
            policy_sym = BoardSymmetry.transform_policy_flat(policy, board_size, sym_id)
        elif policy.ndim == 2:
            # Grid format
            policy_sym = BoardSymmetry.transform_policy_grid(policy, sym_id)
        else:
            raise ValueError(f"Policy must be 1D or 2D, got shape {policy.shape}")

        # Transform last move
        board_size = state.shape[-1]
        last_move_sym = BoardSymmetry.transform_last_move(last_move, board_size, sym_id)

        # Value, current_player, metadata are unchanged
        return (
            state_sym,
            policy_sym,
            value,
            last_move_sym,
            current_player,
            metadata.copy() if metadata else {}
        )

    @staticmethod
    def get_random_symmetry() -> int:
        """Get a random symmetry ID for data augmentation.

        Returns:
            Random symmetry ID in [0, 7]
        """
        return np.random.randint(0, 8)

    @staticmethod
    def verify_symmetry_consistency(
        state: np.ndarray,
        policy: np.ndarray,
        sym_id: int,
        last_move: Optional[Tuple[int, int]] = None,
        tolerance: float = 1e-6
    ) -> bool:
        """Verify that state and policy transformations are consistent.

        Checks that:
        1. If last_move is set, the corresponding cells in state and policy align
        2. Policy probability mass is preserved
        3. No NaNs or Infs introduced

        Args:
            state: State tensor [C, H, W]
            policy: Policy (flat or grid)
            sym_id: Symmetry ID to test
            last_move: Optional last move to verify consistency
            tolerance: Numerical tolerance for checks

        Returns:
            True if transformation is consistent, False otherwise
        """
        board_size = state.shape[-1]

        # Apply transformation
        state_sym, policy_sym, _, last_move_sym, _, _ = BoardSymmetry.apply_symmetry(
            state, policy, 0.0, sym_id, last_move
        )

        # Check 1: No NaNs or Infs
        if np.any(np.isnan(state_sym)) or np.any(np.isinf(state_sym)):
            return False
        if np.any(np.isnan(policy_sym)) or np.any(np.isinf(policy_sym)):
            return False

        # Check 2: Policy probability mass preserved
        policy_flat = policy.flatten() if policy.ndim == 2 else policy
        policy_sym_flat = policy_sym.flatten() if policy_sym.ndim == 2 else policy_sym

        if abs(policy_flat.sum() - policy_sym_flat.sum()) > tolerance:
            return False

        # Check 3: Last move consistency (if last_move channel exists)
        if last_move is not None and last_move_sym is not None:
            # Assuming channel 2 is last_move in state [C, H, W]
            if state.shape[0] >= 3:
                last_move_channel = state[2]
                last_move_channel_sym = state_sym[2]

                # Original position should have value at last_move
                r_orig, c_orig = last_move
                if last_move_channel[r_orig, c_orig] > 0:
                    # Transformed position should have value at last_move_sym
                    r_sym, c_sym = last_move_sym
                    if last_move_channel_sym[r_sym, c_sym] <= 0:
                        return False

        return True
