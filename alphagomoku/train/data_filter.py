"""Minimal data filtering for training - trust the learning process.

PHILOSOPHY: The network + MCTS should learn what moves are good/bad.
We only filter truly pathological cases that waste compute.

REMOVED from previous version:
- Aggressive edge move filtering
- Tactical pattern checking (let network learn tactics!)
- Distance-based filtering (network should learn proximity matters)

KEPT (essential only):
- Very low confidence moves (MCTS is completely uncertain)
- Moves far outside the play area (obvious waste)
"""

import numpy as np
from typing import List
from ..selfplay.selfplay import SelfPlayData


def filter_minimal(data: List[SelfPlayData], board_size: int = 15) -> List[SelfPlayData]:
    """Minimal filtering - only remove pathological cases.

    Removes only:
    1. Moves very far (>4 cells) from any stone after move 5
    2. Very low confidence moves (MCTS nearly uniform)

    Args:
        data: List of self-play data
        board_size: Board size

    Returns:
        Filtered list
    """
    filtered = []
    removed_far = 0
    removed_low_conf = 0

    for example in data:
        # Extract move from policy (argmax = move played)
        move_action = np.argmax(example.policy)
        move_r, move_c = divmod(move_action, board_size)

        # Get board state
        board = example.state[0] - example.state[1]  # Reconstruct board
        num_stones = np.sum(board != 0)

        # Rule 1: Only filter moves VERY far from play (>4 cells away)
        # This catches obvious mistakes but doesn't constrain learning
        if num_stones >= 5:
            min_dist = _compute_min_distance(board, move_r, move_c)
            if min_dist > 4:  # Very loose threshold
                removed_far += 1
                continue

        # Rule 2: Filter very low confidence (MCTS nearly uniform)
        max_prob = example.policy.max()
        if max_prob < 0.1:  # MCTS has almost no preference
            removed_low_conf += 1
            continue

        # Keep this example
        filtered.append(example)

    # Log filtering stats
    total = len(data)
    kept = len(filtered)
    removed = total - kept

    if removed > 0:
        print(f"🔍 Minimal filtering: Kept {kept}/{total} ({kept/total*100:.1f}%)")
        if removed_far > 0:
            print(f"   ❌ Removed {removed_far} moves very far from play (>4 cells)")
        if removed_low_conf > 0:
            print(f"   ❌ Removed {removed_low_conf} very low confidence moves (<10%)")

    return filtered


def _compute_min_distance(board: np.ndarray, r: int, c: int) -> int:
    """Compute minimum Chebyshev distance from (r,c) to any stone on board."""
    occupied = np.argwhere(board != 0)
    if len(occupied) == 0:
        return 999

    min_dist = 999
    for stone_r, stone_c in occupied:
        dist = max(abs(r - stone_r), abs(c - stone_c))
        min_dist = min(min_dist, dist)

    return min_dist


def apply_all_filters(data: List[SelfPlayData], board_size: int = 15) -> List[SelfPlayData]:
    """Apply minimal data filtering.

    PHILOSOPHY: Trust the learning process. Only filter obvious pathologies.

    Args:
        data: Raw self-play data
        board_size: Board size

    Returns:
        Filtered data (most examples kept)
    """
    if not data:
        return data

    return filter_minimal(data, board_size)


# Backward compatibility aliases
filter_stupid_moves = filter_minimal
filter_low_confidence_moves = lambda data, threshold=0.1: filter_minimal(data)
