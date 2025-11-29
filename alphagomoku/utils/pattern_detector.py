"""Tactical pattern detection for Gomoku neural network input."""

import numpy as np
from typing import Tuple


def detect_patterns(board: np.ndarray, player: int) -> np.ndarray:
    """Detect tactical patterns on the board for a specific player.

    Creates a heatmap where cells with higher values indicate
    more important tactical positions for the given player.

    Args:
        board: Board state (H, W) with values in {-1, 0, 1}
        player: Player to detect patterns for (1 or -1)

    Returns:
        Pattern heatmap (H, W) with float values [0, 1]
    """
    h, w = board.shape
    pattern_map = np.zeros((h, w), dtype=np.float32)

    # Check all positions
    for r in range(h):
        for c in range(w):
            if board[r, c] != 0:
                continue

            # Evaluate this empty position for tactical value
            score = 0.0

            # Check all 4 directions
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # Check sequences in this direction
                own_score = _evaluate_direction(board, r, c, dr, dc, player)
                opp_score = _evaluate_direction(board, r, c, dr, dc, -player)

                score += own_score
                score += opp_score * 1.2  # Defense slightly more important

            pattern_map[r, c] = min(1.0, score)

    return pattern_map


def _evaluate_direction(
    board: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    player: int
) -> float:
    """Evaluate tactical value of placing a stone at (r,c) in one direction.

    Returns a score indicating the importance of this position.
    """
    h, w = board.shape

    # Count consecutive stones and open ends in both directions
    # Forward direction
    forward_count = 0
    forward_blocked = False
    for i in range(1, 5):
        nr, nc = r + dr * i, c + dc * i
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            forward_blocked = True
            break
        if board[nr, nc] == -player:
            forward_blocked = True
            break
        if board[nr, nc] == player:
            forward_count += 1
        else:
            break

    # Backward direction
    backward_count = 0
    backward_blocked = False
    for i in range(1, 5):
        nr, nc = r - dr * i, c - dc * i
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            backward_blocked = True
            break
        if board[nr, nc] == -player:
            backward_blocked = True
            break
        if board[nr, nc] == player:
            backward_count += 1
        else:
            break

    total_count = forward_count + backward_count
    open_ends = 2 - int(forward_blocked) - int(backward_blocked)

    # Score based on pattern type
    if total_count >= 4:
        # Immediate win
        return 1.0
    elif total_count == 3:
        if open_ends == 2:
            # Open four (very strong)
            return 0.95
        elif open_ends == 1:
            # Broken four (strong defense/attack)
            return 0.7
    elif total_count == 2:
        if open_ends == 2:
            # Open three (important)
            return 0.5
        elif open_ends == 1:
            # Broken three
            return 0.25
    elif total_count == 1:
        if open_ends == 2:
            # Open two
            return 0.15

    return 0.0


def compute_proximity_mask(board: np.ndarray, max_distance: int = 2) -> np.ndarray:
    """Compute proximity mask penalizing moves far from existing stones.

    Phase 2: Proximity Penalty Implementation
    This is a key Gomoku heuristic: almost all good moves are adjacent to existing stones.

    Args:
        board: Board state (H, W) with values in {-1, 0, 1}
        max_distance: Maximum distance to consider (default 2)

    Returns:
        Proximity mask (H, W) with values [0.05, 1.0]
        - 1.0 for moves adjacent to stones (distance 1)
        - 0.5 for moves 2 cells away
        - 0.2 for moves 3 cells away
        - 0.05 for moves 4+ cells away
        - 1.0 for first 5 moves (opening exception)
    """
    h, w = board.shape
    proximity_mask = np.ones((h, w), dtype=np.float32) * 0.05  # Default: far away

    # Count stones on board
    num_stones = np.sum(board != 0)

    # Opening exception: first move only (num_stones == 0)
    if num_stones == 0:
        # Empty board: prefer center area
        center_r, center_c = h // 2, w // 2
        for r in range(h):
            for c in range(w):
                dist_to_center = max(abs(r - center_r), abs(c - center_c))
                if dist_to_center <= 2:
                    proximity_mask[r, c] = 1.0
                elif dist_to_center <= 4:
                    proximity_mask[r, c] = 0.7
                else:
                    proximity_mask[r, c] = 0.3
        return proximity_mask

    # Find all occupied cells
    occupied = np.argwhere(board != 0)

    if len(occupied) == 0:
        # Empty board should not happen here, but handle gracefully
        return np.ones((h, w), dtype=np.float32)

    # For each empty cell, compute distance to nearest stone
    for r in range(h):
        for c in range(w):
            if board[r, c] != 0:
                proximity_mask[r, c] = 0.0  # Occupied cells
                continue

            # Compute minimum Chebyshev distance to any stone
            # Chebyshev distance = max(|dr|, |dc|) - matches Gomoku threat distance
            min_dist = float('inf')
            for stone_r, stone_c in occupied:
                dist = max(abs(r - stone_r), abs(c - stone_c))
                min_dist = min(min_dist, dist)

            # Assign penalty based on distance
            if min_dist == 1:
                proximity_mask[r, c] = 1.0      # Adjacent: full weight
            elif min_dist == 2:
                proximity_mask[r, c] = 0.5      # 2 cells away: half weight
            elif min_dist == 3:
                proximity_mask[r, c] = 0.2      # 3 cells away: weak
            else:
                proximity_mask[r, c] = 0.05     # 4+ cells away: very weak

    return proximity_mask


def get_pattern_features(board: np.ndarray, current_player: int) -> np.ndarray:
    """Get pattern features for neural network input.

    Combines patterns for both players, emphasizing current player's perspective.

    Phase 2 Enhancement: Now includes proximity penalty to discourage distant moves.

    Args:
        board: Board state (H, W)
        current_player: Current player to move (1 or -1)

    Returns:
        Pattern heatmap (H, W) for NN input
    """
    own_patterns = detect_patterns(board, current_player)
    opp_patterns = detect_patterns(board, -current_player)

    # Combine: own patterns + opponent patterns (defense)
    # Weight defense slightly higher
    combined = own_patterns + opp_patterns * 1.2

    # Phase 2: Apply proximity penalty
    proximity_mask = compute_proximity_mask(board, max_distance=2)
    combined = combined * proximity_mask

    # If no patterns detected, proximity mask becomes the primary feature
    # This guides the model to play near existing stones even without threats
    if combined.max() < 1e-6:
        combined = proximity_mask * 0.3  # Give some baseline signal

    # Normalize to [0, 1]
    max_val = combined.max()
    if max_val > 0:
        combined = combined / max_val

    return combined.astype(np.float32)


def detect_immediate_threats(board: np.ndarray, player: int) -> Tuple[np.ndarray, bool]:
    """Detect immediate winning moves or forced defenses.

    Args:
        board: Board state (H, W)
        player: Player to check for (1 or -1)

    Returns:
        (threat_mask, has_immediate_threat) where:
        - threat_mask: Boolean array (H, W) marking critical positions
        - has_immediate_threat: True if immediate win/loss threat exists
    """
    h, w = board.shape
    threat_mask = np.zeros((h, w), dtype=bool)
    has_threat = False

    # Check for immediate wins (open fours or four in a row with gap)
    for r in range(h):
        for c in range(w):
            if board[r, c] != 0:
                continue

            # Try placing stone here
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                # Check if this completes 5 in a row for player
                if _completes_five(board, r, c, dr, dc, player):
                    threat_mask[r, c] = True
                    has_threat = True
                # Check if this blocks opponent's five
                elif _completes_five(board, r, c, dr, dc, -player):
                    threat_mask[r, c] = True
                    has_threat = True

    return threat_mask, has_threat


def _completes_five(board: np.ndarray, r: int, c: int, dr: int, dc: int, player: int) -> bool:
    """Check if placing a stone at (r,c) completes 5 in a row."""
    h, w = board.shape

    # Count in forward direction
    count = 1  # Count the placed stone
    for i in range(1, 5):
        nr, nc = r + dr * i, c + dc * i
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if board[nr, nc] == player:
            count += 1
        else:
            break

    # Count in backward direction
    for i in range(1, 5):
        nr, nc = r - dr * i, c - dc * i
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if board[nr, nc] == player:
            count += 1
        else:
            break

    return count >= 5
