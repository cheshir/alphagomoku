"""Filter training data to exclude stupid/nonsensical moves.

This is CRITICAL for preventing the model from learning bad habits.
Even with good self-play, MCTS explores random moves that should not
be in the training data.
"""

import numpy as np
from typing import List
from ..selfplay.selfplay import SelfPlayData


def filter_stupid_moves(data: List[SelfPlayData], board_size: int = 15) -> List[SelfPlayData]:
    """Filter out training examples with stupid moves.

    Removes examples where:
    1. Move is on board edge in early game (moves 0-10)
    2. Move is >3 cells from nearest stone (after move 5)
    3. Move ignores obvious tactical threats

    Args:
        data: List of self-play data
        board_size: Board size

    Returns:
        Filtered list (stupid moves removed)
    """
    filtered = []
    removed_edge = 0
    removed_far = 0
    removed_tactical = 0

    for example in data:
        # Extract move from policy (argmax = move played)
        move_action = np.argmax(example.policy)
        move_r, move_c = divmod(move_action, board_size)

        # Get board state (channel 0 = own stones, channel 1 = opponent stones)
        board = example.state[0] - example.state[1]  # Reconstruct board
        num_stones = np.sum(board != 0)

        # Get pattern channel (used for tactical checks)
        pattern_channel = example.state[4]  # Channel 4 = patterns
        move_pattern_value = pattern_channel[move_r, move_c]
        max_pattern_value = pattern_channel.max()

        # CRITICAL: Never filter forced/winning moves!
        # If this move has very high pattern value (>0.9), it's likely a forced win/defense
        is_forced_move = move_pattern_value > 0.9

        # Rule 1: No edge moves in early game (UNLESS it's a forced move)
        if not is_forced_move and num_stones < 10:
            is_edge = (move_r == 0 or move_r == board_size-1 or
                      move_c == 0 or move_c == board_size-1)
            if is_edge:
                removed_edge += 1
                continue

        # Rule 2: No moves far from existing stones (UNLESS it's a forced move)
        if not is_forced_move and num_stones >= 5:
            min_dist = _compute_min_distance(board, move_r, move_c)
            if min_dist > 3:
                removed_far += 1
                continue

        # Rule 3: Check for tactical blunders (UNLESS it's a forced move)
        # If pattern channel shows high values elsewhere, this move is bad
        if not is_forced_move:
            if max_pattern_value > 0.8 and move_pattern_value < 0.3:
                # This is likely a tactical blunder
                removed_tactical += 1
                continue

        # This example is OK
        filtered.append(example)

    # Log filtering stats
    total = len(data)
    kept = len(filtered)
    removed = total - kept

    if removed > 0:
        print(f"📊 Data filtering: Kept {kept}/{total} ({kept/total*100:.1f}%)")
        if removed_edge > 0:
            print(f"   ❌ Removed {removed_edge} edge moves in early game")
        if removed_far > 0:
            print(f"   ❌ Removed {removed_far} distant moves (>3 cells away)")
        if removed_tactical > 0:
            print(f"   ❌ Removed {removed_tactical} tactical blunders")

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


def filter_low_confidence_moves(data: List[SelfPlayData], threshold: float = 0.1) -> List[SelfPlayData]:
    """Filter moves where the policy is too flat (low confidence).

    If MCTS is very uncertain (policy is nearly uniform), the training
    signal is weak. Better to exclude these examples.

    Args:
        data: List of self-play data
        threshold: Minimum max policy value to keep

    Returns:
        Filtered list
    """
    filtered = []
    removed = 0

    for example in data:
        max_prob = example.policy.max()

        # If policy is too flat (nearly uniform), skip it
        if max_prob < threshold:
            removed += 1
            continue

        filtered.append(example)

    if removed > 0:
        kept = len(filtered)
        total = len(data) + removed
        print(f"📊 Low-confidence filtering: Kept {kept}/{total} ({kept/total*100:.1f}%)")
        print(f"   ❌ Removed {removed} low-confidence examples (max_prob < {threshold})")

    return filtered


def apply_all_filters(data: List[SelfPlayData], board_size: int = 15) -> List[SelfPlayData]:
    """Apply all data quality filters.

    Args:
        data: Raw self-play data
        board_size: Board size

    Returns:
        High-quality filtered data
    """
    print(f"\n🔍 Filtering training data (starting with {len(data)} examples)...")

    # Filter 1: Remove stupid moves
    data = filter_stupid_moves(data, board_size)

    # Filter 2: Remove low-confidence moves
    data = filter_low_confidence_moves(data, threshold=0.15)

    print(f"✅ Final dataset: {len(data)} high-quality examples\n")

    return data
