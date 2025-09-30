"""Opening move logic for Gomoku."""

from typing import List, Tuple

import numpy as np


def get_first_move_near_opponent(
    state: np.ndarray,
    board_size: int = 15,
    distance: int = 2,
) -> List[int]:
    """Get good first move positions near opponent's opening stone.

    When the AI responds to the player's first move, it should play nearby
    to create immediate tactical interaction rather than playing far away.

    Args:
        state: Board state (board_size x board_size)
        board_size: Size of the board (default 15)
        distance: How many squares away to consider (default 2)
            - 1: Adjacent (knight's move distance)
            - 2: Two squares away (standard opening response)
            - 3: Wider opening

    Returns:
        List of action indices (flattened) that are good first moves,
        sorted by priority (closest first)
    """
    # Check if this is truly the first move (only 1 stone on board)
    stones = np.where(state != 0)
    if len(stones[0]) != 1:
        return []  # Not a first-move situation

    # Get opponent's stone position
    opp_row, opp_col = stones[0][0], stones[1][0]

    # Generate candidate positions at various distances
    candidates = []

    # Distance 1: Adjacent + diagonal (8 positions)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = opp_row + dr, opp_col + dc
            if 0 <= r < board_size and 0 <= c < board_size and state[r, c] == 0:
                action = r * board_size + c
                candidates.append((1, action))  # Priority 1 (highest)

    # Distance 2: Two squares away (knight moves and beyond)
    if distance >= 2:
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                if dr == 0 and dc == 0:
                    continue
                # Skip if already in distance 1
                if abs(dr) <= 1 and abs(dc) <= 1:
                    continue

                r, c = opp_row + dr, opp_col + dc
                if 0 <= r < board_size and 0 <= c < board_size and state[r, c] == 0:
                    action = r * board_size + c
                    candidates.append((2, action))  # Priority 2

    # Distance 3: Three squares away
    if distance >= 3:
        for dr in [-3, -2, -1, 0, 1, 2, 3]:
            for dc in [-3, -2, -1, 0, 1, 2, 3]:
                if dr == 0 and dc == 0:
                    continue
                # Skip if already covered
                if abs(dr) <= 2 and abs(dc) <= 2:
                    continue

                r, c = opp_row + dr, opp_col + dc
                if 0 <= r < board_size and 0 <= c < board_size and state[r, c] == 0:
                    action = r * board_size + c
                    candidates.append((3, action))  # Priority 3

    # Sort by priority (lower number = higher priority)
    candidates.sort(key=lambda x: x[0])

    # Return just the actions
    return [action for _, action in candidates]


def boost_opening_moves(
    policy: np.ndarray,
    state: np.ndarray,
    board_size: int = 15,
    boost_factor: float = 10.0,
    distance: int = 2,
) -> np.ndarray:
    """Boost policy probabilities for good opening moves.

    Args:
        policy: Neural network policy output (flattened, board_size^2)
        state: Current board state
        board_size: Size of the board
        boost_factor: How much to boost good opening moves (default 10.0)
        distance: Distance parameter for get_first_move_near_opponent

    Returns:
        Modified policy with boosted opening moves
    """
    # Check if this is truly the first move (only 1 stone on board)
    stones = np.where(state != 0)
    if len(stones[0]) != 1:
        return policy  # Not a first-move situation

    opp_row, opp_col = stones[0][0], stones[1][0]

    # Create boosted policy
    boosted_policy = policy.copy()

    # Distance 1: Immediately adjacent (8 positions) - HIGHEST priority
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = opp_row + dr, opp_col + dc
            if 0 <= r < board_size and 0 <= c < board_size and state[r, c] == 0:
                action = r * board_size + c
                # Very strong boost for adjacent moves
                boosted_policy[action] *= boost_factor * 10.0

    # Distance 2: Two squares away - lower priority
    if distance >= 2:
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                if dr == 0 and dc == 0:
                    continue
                # Skip if already in distance 1
                if abs(dr) <= 1 and abs(dc) <= 1:
                    continue

                r, c = opp_row + dr, opp_col + dc
                if 0 <= r < board_size and 0 <= c < board_size and state[r, c] == 0:
                    action = r * board_size + c
                    # Moderate boost for distance-2 moves
                    boosted_policy[action] *= boost_factor

    # Renormalize (but keep only legal moves)
    legal_mask = (state.reshape(-1) == 0)
    boosted_policy = boosted_policy * legal_mask

    total = boosted_policy.sum()
    if total > 0:
        boosted_policy /= total

    return boosted_policy


def is_center_opening(state: np.ndarray, board_size: int = 15) -> bool:
    """Check if opponent opened in center region.

    Center openings are more common and might want different response.

    Args:
        state: Board state
        board_size: Size of board

    Returns:
        True if opponent's first move is in center 5x5 region
    """
    stones = np.where(state != 0)
    if len(stones[0]) != 1:
        return False

    opp_row, opp_col = stones[0][0], stones[1][0]
    center = board_size // 2

    # Check if within 2 squares of center
    return abs(opp_row - center) <= 2 and abs(opp_col - center) <= 2


def get_opening_response(
    state: np.ndarray,
    board_size: int = 15,
) -> Tuple[int, int]:
    """Get a good opening response position (row, col).

    Simple heuristic for first move response:
    - If opponent plays center, play diagonally adjacent
    - If opponent plays corner/edge, play nearby

    Args:
        state: Board state
        board_size: Size of board

    Returns:
        (row, col) position for AI's first move
    """
    stones = np.where(state != 0)
    if len(stones[0]) != 1:
        raise ValueError("Not a first-move situation")

    opp_row, opp_col = stones[0][0], stones[1][0]
    center = board_size // 2

    # If opponent played center or near-center, respond diagonally
    if abs(opp_row - center) <= 1 and abs(opp_col - center) <= 1:
        # Play diagonal to center
        candidates = [
            (opp_row + 1, opp_col + 1),
            (opp_row + 1, opp_col - 1),
            (opp_row - 1, opp_col + 1),
            (opp_row - 1, opp_col - 1),
        ]
    else:
        # Play one square away in a good direction
        # Prefer diagonal, then adjacent
        candidates = [
            (opp_row + 1, opp_col + 1),
            (opp_row + 1, opp_col),
            (opp_row, opp_col + 1),
            (opp_row - 1, opp_col - 1),
            (opp_row - 1, opp_col),
            (opp_row, opp_col - 1),
            (opp_row + 1, opp_col - 1),
            (opp_row - 1, opp_col + 1),
        ]

    # Return first valid candidate
    for r, c in candidates:
        if 0 <= r < board_size and 0 <= c < board_size and state[r, c] == 0:
            return (r, c)

    # Fallback: shouldn't happen
    return (center, center)
