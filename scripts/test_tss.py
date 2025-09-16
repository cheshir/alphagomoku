#!/usr/bin/env python3
"""Test script for TSS functionality."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.tss import Position, ThreatDetector, ThreatType, tss_search


def print_board(board, last_move=None):
    """Print board with coordinates."""
    print("   " + " ".join(f"{i:2d}" for i in range(15)))
    for r in range(15):
        row_str = f"{r:2d} "
        for c in range(15):
            if last_move and (r, c) == last_move:
                if board[r, c] == 1:
                    row_str += " ●"  # Last move by player 1
                elif board[r, c] == -1:
                    row_str += " ○"  # Last move by player -1
                else:
                    row_str += " +"  # Last move position (empty)
            elif board[r, c] == 1:
                row_str += " X"
            elif board[r, c] == -1:
                row_str += " O"
            else:
                row_str += " ."
        print(row_str)
    print()


def test_threat_detection():
    """Test basic threat detection."""
    print("=== Testing Threat Detection ===")
    
    detector = ThreatDetector()
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Create open three pattern: .XXX.
    board[7, 6] = 1
    board[7, 7] = 1
    board[7, 8] = 1
    
    position = Position(board=board, current_player=1)
    print_board(board)
    
    threats = detector.detect_threats(position, 1)
    print(f"Detected {len(threats)} threats for player 1:")
    for r, c, threat_type in threats:
        print(f"  ({r}, {c}): {threat_type.value}")
    
    print()


def test_forced_defense():
    """Test forced defense detection."""
    print("=== Testing Forced Defense ===")
    
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Create opponent's open four threat: .XXXX.
    for i in range(4):
        board[7, 5 + i] = -1
    
    position = Position(board=board, current_player=1, last_move=(7, 8))
    print("Opponent has open four threat:")
    print_board(board, last_move=(7, 8))
    
    result = tss_search(position, depth=2, time_cap_ms=100)
    
    print(f"TSS Result:")
    print(f"  Forced defense: {result.is_forced_defense}")
    print(f"  Forced move: {result.forced_move}")
    print(f"  Stats: {result.search_stats}")
    
    if result.forced_move:
        r, c = result.forced_move
        print(f"  Must play at ({r}, {c}) to defend")
    
    print()


def test_forced_win():
    """Test forced win detection."""
    print("=== Testing Forced Win ===")
    
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Create winning setup: open three with space for double threat
    board[7, 6] = 1
    board[7, 7] = 1
    board[7, 8] = 1
    
    # Add some supporting stones
    board[6, 7] = 1
    board[8, 7] = 1
    
    position = Position(board=board, current_player=1, last_move=(8, 7))
    print("Position with potential forced win:")
    print_board(board, last_move=(8, 7))
    
    result = tss_search(position, depth=4, time_cap_ms=200)
    
    print(f"TSS Result:")
    print(f"  Forced win: {result.is_forced_win}")
    print(f"  Forced move: {result.forced_move}")
    print(f"  Stats: {result.search_stats}")
    
    if result.forced_move:
        r, c = result.forced_move
        print(f"  Winning move at ({r}, {c})")
    
    print()


def test_performance():
    """Test TSS performance."""
    print("=== Testing Performance ===")
    
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Create complex tactical position
    moves = [
        (7, 7), (7, 8), (8, 7), (8, 8),  # Center cluster
        (6, 6), (9, 9), (6, 8), (8, 6),  # Diagonal threats
        (5, 7), (10, 7)  # Vertical extension
    ]
    
    for i, (r, c) in enumerate(moves):
        board[r, c] = 1 if i % 2 == 0 else -1
    
    position = Position(board=board, current_player=1, last_move=(10, 7))
    print("Complex tactical position:")
    print_board(board, last_move=(10, 7))
    
    # Test different depths and time limits
    for depth in [2, 4, 6]:
        for time_cap in [50, 100, 200]:
            result = tss_search(position, depth=depth, time_cap_ms=time_cap)
            stats = result.search_stats
            print(f"  Depth {depth}, Time {time_cap}ms: "
                  f"{stats['nodes_visited']} nodes, {stats['time_ms']:.1f}ms")
    
    print()


def main():
    """Run all TSS tests."""
    print("AlphaGomoku TSS Test Suite")
    print("=" * 40)
    
    try:
        test_threat_detection()
        test_forced_defense()
        test_forced_win()
        test_performance()
        
        print("✅ All TSS tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())