#!/usr/bin/env python3
"""Test model's tactical awareness on critical positions."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.utils.pattern_detector import get_pattern_features


def create_test_position(pattern_name: str) -> tuple:
    """Create a test position to evaluate tactical awareness.

    Returns:
        (board, expected_move, description)
    """
    board_size = 15

    if pattern_name == "complete_five":
        # Four in a row horizontally, should complete to five
        board = np.zeros((board_size, board_size), dtype=np.int8)
        board[7, 5:9] = 1  # Four stones
        expected = (7, 9)  # Complete on the right
        desc = "Complete open-four to win (horizontal)"

    elif pattern_name == "block_five":
        # Opponent has four, MUST block
        board = np.zeros((board_size, board_size), dtype=np.int8)
        board[7, 5:9] = -1
        expected = [(7, 4), (7, 9)]  # Either end
        desc = "Block opponent's open-four (CRITICAL)"

    elif pattern_name == "block_broken_four":
        # Opponent has X X X _ X, must block gap
        board = np.zeros((board_size, board_size), dtype=np.int8)
        board[7, [5, 6, 7, 9]] = -1  # Gap at 8
        expected = (7, 8)
        desc = "Block opponent's broken-four"

    elif pattern_name == "center_start":
        # Empty board or just center stone - should NOT play on edge
        board = np.zeros((board_size, board_size), dtype=np.int8)
        board[7, 7] = -1  # Opponent plays center
        expected = "not_edge"  # Should play near center, NOT on edges
        desc = "Respond to center opening (should NOT play on edge)"

    elif pattern_name == "edge_avoidance":
        # Early game with few stones - avoid edges
        board = np.zeros((board_size, board_size), dtype=np.int8)
        board[7, 7] = -1
        board[8, 8] = 1
        expected = "not_edge"
        desc = "Early game - avoid board edges"

    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    return board, expected, desc


def board_to_input(board: np.ndarray, current_player: int = 1) -> torch.Tensor:
    """Convert board to neural network input."""
    h, w = board.shape

    own = (board == current_player).astype(np.float32)
    opp = (board == -current_player).astype(np.float32)
    last = np.zeros((h, w), dtype=np.float32)
    side = np.ones((h, w), dtype=np.float32)
    pattern = get_pattern_features(board, current_player)

    state = np.stack([own, opp, last, side, pattern])
    return torch.from_numpy(state).float().unsqueeze(0)


def test_position(model: GomokuNet, pattern_name: str) -> bool:
    """Test model on a specific tactical position.

    Returns:
        True if model passes the test
    """
    board, expected, desc = create_test_position(pattern_name)
    board_size = board.shape[0]

    # Get model prediction
    state = board_to_input(board, current_player=1)
    with torch.no_grad():
        policy_logits, value = model(state)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()

    # Get top predicted move
    top_action = np.argmax(policy)
    top_move = (top_action // board_size, top_action % board_size)
    top_prob = policy[top_action]

    # Get top 5 moves
    top5_indices = np.argsort(policy)[-5:][::-1]
    top5_moves = [(idx // board_size, idx % board_size) for idx in top5_indices]
    top5_probs = [policy[idx] for idx in top5_indices]

    print(f"\n{'='*60}")
    print(f"Test: {desc}")
    print(f"Pattern: {pattern_name}")
    print(f"{'='*60}")

    # Visualize board
    print("\nBoard (X=player, O=opponent):")
    for r in range(board_size):
        row_str = ""
        for c in range(board_size):
            if board[r, c] == 1:
                row_str += "X "
            elif board[r, c] == -1:
                row_str += "O "
            else:
                row_str += ". "
        print(row_str)

    print(f"\nTop 5 predicted moves:")
    for i, (move, prob) in enumerate(zip(top5_moves, top5_probs), 1):
        print(f"  {i}. {move} - {prob:.4f}")

    print(f"\nValue estimate: {value.item():.3f}")

    # Check if correct
    if isinstance(expected, tuple):
        # Single expected move
        passed = top_move == expected
        print(f"\nExpected: {expected}")
        print(f"Predicted: {top_move}")
        print(f"Result: {'✓ PASS' if passed else '✗ FAIL'}")

    elif isinstance(expected, list):
        # Multiple acceptable moves
        passed = top_move in expected
        print(f"\nExpected one of: {expected}")
        print(f"Predicted: {top_move}")
        print(f"Result: {'✓ PASS' if passed else '✗ FAIL'}")

    elif expected == "not_edge":
        # Should not play on edges
        r, c = top_move
        is_edge = (r == 0 or r == board_size-1 or c == 0 or c == board_size-1)
        passed = not is_edge
        print(f"\nConstraint: Should NOT play on edge")
        print(f"Predicted: {top_move} (edge={is_edge})")
        print(f"Result: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test model tactical awareness')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/mps/cuda)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = GomokuNet(board_size=15, num_blocks=12, channels=64)

    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(args.device)
    print(f"Model loaded on {args.device}")

    # Run tests
    test_cases = [
        "complete_five",
        "block_five",
        "block_broken_four",
        "center_start",
        "edge_avoidance",
    ]

    results = {}
    for test_name in test_cases:
        results[test_name] = test_position(model, test_name)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
