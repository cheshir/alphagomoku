#!/usr/bin/env python3
"""Test that model correctly handles forced winning moves.

Verifies:
1. TSS detects immediate wins
2. TSS prioritizes wins over defense
3. Data filtering doesn't remove forced wins
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.tss.tss_search import TSSSearcher
from alphagomoku.tss.position import Position
from alphagomoku.train.data_filter import filter_stupid_moves
from alphagomoku.selfplay.selfplay import SelfPlayData
from alphagomoku.utils.pattern_detector import get_pattern_features


def test_tss_immediate_win():
    """Test that TSS detects and returns immediate winning moves."""
    print("=" * 60)
    print("TEST 1: TSS Immediate Win Detection")
    print("=" * 60)

    board_size = 15
    searcher = TSSSearcher(board_size=board_size)

    # Create position with immediate win available
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[7, 5:9] = 1  # Four in a row: _ X X X X _

    position = Position(
        board=board,
        current_player=1,
        board_size=board_size
    )

    # Search should find immediate win
    result = searcher.search(position, depth=4, time_cap_ms=1000)

    assert result.forced_move is not None, "TSS should find immediate win"
    assert result.is_forced_win, "Move should be marked as forced win"
    assert result.forced_move in [(7, 4), (7, 9)], f"Win move should complete the row, got {result.forced_move}"

    print(f"✓ TSS found immediate win: {result.forced_move}")
    print(f"✓ Marked as forced win: {result.is_forced_win}")
    print(f"✓ Search reason: {result.search_stats.get('reason', 'unknown')}")
    print()


def test_tss_win_over_defense():
    """Test that TSS prioritizes our win over opponent's threat."""
    print("=" * 60)
    print("TEST 2: TSS Win Priority Over Defense")
    print("=" * 60)

    board_size = 15
    searcher = TSSSearcher(board_size=board_size)

    # Create position where BOTH players have immediate wins
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[7, 5:9] = 1   # Our four: _ X X X X _
    board[8, 5:9] = -1  # Opponent's four: _ O O O O _

    position = Position(
        board=board,
        current_player=1,
        board_size=board_size
    )

    # We move first, so we should win (not defend)
    result = searcher.search(position, depth=4, time_cap_ms=1000)

    assert result.forced_move is not None, "TSS should find a move"
    assert result.is_forced_win, "Should prioritize our win"
    assert result.forced_move[0] == 7, f"Should complete our row (row 7), not defend row 8, got {result.forced_move}"

    print(f"✓ TSS prioritized our win: {result.forced_move}")
    print(f"✓ Marked as forced win (not defense): {result.is_forced_win}")
    print(f"✓ Search reason: {result.search_stats.get('reason', 'unknown')}")
    print()


def test_data_filtering_preserves_wins():
    """Test that data filtering does NOT remove forced winning moves."""
    print("=" * 60)
    print("TEST 3: Data Filtering Preserves Forced Wins")
    print("=" * 60)

    board_size = 15

    # Create example with winning move on EDGE (should NOT be filtered!)
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[0, 1:5] = 1  # Four in a row on top edge: _ X X X X _

    # Build state tensor
    own = (board == 1).astype(np.float32)
    opp = (board == -1).astype(np.float32)
    last = np.zeros_like(board, dtype=np.float32)
    side = np.ones_like(board, dtype=np.float32)
    pattern = get_pattern_features(board, 1)

    # Verify pattern gives high value to winning move
    winning_move = (0, 5)  # Complete on right
    pattern_value = pattern[winning_move]
    print(f"Pattern value at winning move {winning_move}: {pattern_value:.3f}")

    assert pattern_value > 0.9, f"Winning move should have high pattern value, got {pattern_value}"

    state = np.stack([own, opp, last, side, pattern])

    # Policy points to winning move
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[0 * board_size + 5] = 1.0  # (0, 5) = complete on right

    example = SelfPlayData(
        state=state,
        policy=policy,
        value=1.0,
        current_player=1
    )

    # Filter - should KEEP this example (it's a forced win)
    filtered = filter_stupid_moves([example], board_size=board_size)

    assert len(filtered) == 1, f"Forced winning move should NOT be filtered, got {len(filtered)} examples"

    print(f"✓ Winning edge move preserved: {winning_move}")
    print(f"✓ Pattern value: {pattern_value:.3f} (>0.9 threshold)")
    print(f"✓ Example kept after filtering")
    print()


def test_data_filtering_removes_stupid():
    """Test that data filtering DOES remove stupid edge moves."""
    print("=" * 60)
    print("TEST 4: Data Filtering Removes Stupid Edge Moves")
    print("=" * 60)

    board_size = 15

    # Create example with stupid edge move (NOT a winning move)
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[7, 7] = 1  # One stone in center

    # Build state tensor
    own = (board == 1).astype(np.float32)
    opp = (board == -1).astype(np.float32)
    last = np.zeros_like(board, dtype=np.float32)
    side = np.ones_like(board, dtype=np.float32)
    pattern = get_pattern_features(board, 1)

    # Verify pattern gives LOW value to edge move
    stupid_move = (0, 0)  # Top-left corner (stupid!)
    pattern_value = pattern[stupid_move]
    print(f"Pattern value at stupid edge move {stupid_move}: {pattern_value:.3f}")

    assert pattern_value < 0.2, f"Stupid edge move should have low pattern value, got {pattern_value}"

    state = np.stack([own, opp, last, side, pattern])

    # Policy points to stupid edge move
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[0] = 1.0  # (0, 0) = top-left corner (BAD!)

    example = SelfPlayData(
        state=state,
        policy=policy,
        value=0.0,
        current_player=1
    )

    # Filter - should REMOVE this example (stupid edge move)
    filtered = filter_stupid_moves([example], board_size=board_size)

    assert len(filtered) == 0, f"Stupid edge move SHOULD be filtered, got {len(filtered)} examples"

    print(f"✓ Stupid edge move removed: {stupid_move}")
    print(f"✓ Pattern value: {pattern_value:.3f} (<0.9 threshold)")
    print(f"✓ Example filtered out correctly")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Forced Win Handling Tests")
    print("=" * 60)
    print()

    try:
        test_tss_immediate_win()
        test_tss_win_over_defense()
        test_data_filtering_preserves_wins()
        test_data_filtering_removes_stupid()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("✓ TSS correctly detects immediate wins")
        print("✓ TSS prioritizes wins over defense")
        print("✓ Data filtering preserves forced wins")
        print("✓ Data filtering removes stupid moves")
        print()
        print("🎯 Model will learn to complete winning sequences!")
        return 0

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
