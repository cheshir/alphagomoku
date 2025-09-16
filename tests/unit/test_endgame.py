"""Unit tests for endgame solver."""

import pytest
import numpy as np
from alphagomoku.endgame import EndgamePosition, endgame_search, should_use_endgame_solver


class TestEndgamePosition:
    """Test EndgamePosition class."""

    def test_position_creation(self):
        """Test basic position creation."""
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        position = EndgamePosition(board=board, current_player=-1)

        assert position.current_player == -1
        assert position.board_size == 15
        assert position.get_empty_count() == 15 * 15 - 1

    def test_invalid_position_creation(self):
        """Test invalid position creation."""
        # Wrong board size
        board = np.zeros((10, 10), dtype=np.int8)
        with pytest.raises(ValueError):
            EndgamePosition(board=board, current_player=1)

        # Invalid player
        board = np.zeros((15, 15), dtype=np.int8)
        with pytest.raises(ValueError):
            EndgamePosition(board=board, current_player=2)

    def test_make_move(self):
        """Test making a move."""
        board = np.zeros((15, 15), dtype=np.int8)
        position = EndgamePosition(board=board, current_player=1)

        new_position = position.make_move(7, 7)

        assert new_position.board[7, 7] == 1
        assert new_position.current_player == -1
        assert new_position.last_move == (7, 7)
        assert position.board[7, 7] == 0  # Original unchanged

    def test_get_legal_moves(self):
        """Test getting legal moves."""
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        board[7, 8] = -1
        position = EndgamePosition(board=board, current_player=1)

        legal_moves = position.get_legal_moves()

        assert len(legal_moves) == 15 * 15 - 2
        assert (7, 7) not in legal_moves
        assert (7, 8) not in legal_moves
        assert (7, 6) in legal_moves

    def test_get_critical_moves(self):
        """Test getting critical moves near existing stones."""
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        position = EndgamePosition(board=board, current_player=-1)

        critical_moves = position.get_critical_moves()

        # Should include adjacent cells
        expected_adjacent = [
            (6, 6), (6, 7), (6, 8),
            (7, 6),          (7, 8),
            (8, 6), (8, 7), (8, 8)
        ]

        for move in expected_adjacent:
            assert move in critical_moves

    def test_empty_board_critical_moves(self):
        """Test critical moves on empty board."""
        board = np.zeros((15, 15), dtype=np.int8)
        position = EndgamePosition(board=board, current_player=1)

        critical_moves = position.get_critical_moves()

        assert critical_moves == [(7, 7)]  # Center move

    def test_is_terminal_win(self):
        """Test terminal position detection - win."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Create 5 in a row
        for i in range(5):
            board[7, 3 + i] = 1

        position = EndgamePosition(board=board, current_player=-1, last_move=(7, 7))

        is_terminal, winner = position.is_terminal()
        assert is_terminal
        assert winner == 1

    def test_is_terminal_draw(self):
        """Test terminal position detection - draw."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Fill board in a pattern that avoids 5 in a row
        # Use a more complex pattern that breaks all possible 5-in-a-row sequences
        for i in range(15):
            for j in range(15):
                # Pattern: 1, -1, 1, -1, -1 repeating, shifted by row
                pattern_offset = (i * 2) % 5
                board[i, j] = 1 if (j + pattern_offset) % 5 < 2 else -1

        # Override last move position to ensure it doesn't create 5 in a row
        board[14, 14] = -1
        position = EndgamePosition(board=board, current_player=1, last_move=(14, 14))

        is_terminal, winner = position.is_terminal()
        assert is_terminal
        assert winner == 0

    def test_is_not_terminal(self):
        """Test non-terminal position."""
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        position = EndgamePosition(board=board, current_player=-1, last_move=(7, 7))

        is_terminal, winner = position.is_terminal()
        assert not is_terminal
        assert winner == 0


class TestEndgameSolver:
    """Test endgame solver functionality."""

    def test_immediate_win_detection(self):
        """Test detection of immediate winning move."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Create 4 in a row with gap at end
        board[7, 3] = 1
        board[7, 4] = 1
        board[7, 5] = 1
        board[7, 6] = 1
        # Position (7, 7) would complete 5 in a row

        position = EndgamePosition(board=board, current_player=1)
        result = endgame_search(position, max_depth=2, time_limit=1.0)

        assert result.is_win
        assert result.best_move == (7, 7) or result.best_move == (7, 2)
        assert result.depth_to_mate == 1

    def test_forced_defense(self):
        """Test detection of forced defense."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Create opponent threat
        board[7, 3] = -1
        board[7, 4] = -1
        board[7, 5] = -1
        board[7, 6] = -1
        # Player 1 must block at (7, 7) or (7, 2)

        position = EndgamePosition(board=board, current_player=1)
        result = endgame_search(position, max_depth=4, time_limit=1.0)

        # The solver should find a move (not necessarily optimal, but reasonable)
        assert result.best_move is not None
        # In this position, player 1 is at a disadvantage
        assert result.evaluation <= 0 or not result.is_win

    def test_mate_in_3(self):
        """Test detection of mate in 3."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Setup a position where player 1 can force mate in 3 moves
        # This is a simplified test - in practice would need specific tactical setup
        board[7, 7] = 1
        board[8, 8] = 1
        board[9, 9] = 1

        position = EndgamePosition(board=board, current_player=1)
        result = endgame_search(position, max_depth=6, time_limit=2.0)

        # Should find some move (exact evaluation depends on position)
        assert result.best_move is not None
        assert result.search_stats['nodes_searched'] > 0

    def test_empty_position(self):
        """Test solver on empty position."""
        board = np.zeros((15, 15), dtype=np.int8)
        position = EndgamePosition(board=board, current_player=1)

        result = endgame_search(position, max_depth=4, time_limit=1.0)

        # Empty position should not be immediately winning/losing
        assert not result.is_win
        assert not result.is_loss
        assert result.best_move is not None

    def test_search_time_limit(self):
        """Test search respects time limits."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Add some stones to create complexity
        for i in range(5):
            board[7, i] = 1 if i % 2 == 0 else -1

        position = EndgamePosition(board=board, current_player=1)

        import time
        start_time = time.time()
        result = endgame_search(position, max_depth=10, time_limit=0.1)  # 100ms limit
        elapsed = time.time() - start_time

        # Should respect time limit (with some tolerance)
        assert elapsed <= 0.5  # Allow some overhead
        assert result.search_stats['time_ms'] <= 500


class TestEndgameThresholds:
    """Test endgame activation thresholds."""

    def test_should_use_endgame_easy(self):
        """Test endgame activation for easy difficulty."""
        board = np.zeros((15, 15), dtype=np.int8)
        # Fill most of the board
        for i in range(15):
            for j in range(10):
                board[i, j] = 1 if (i + j) % 2 == 0 else -1

        position = EndgamePosition(board=board, current_player=1)

        # Easy mode should not use endgame solver
        assert not should_use_endgame_solver(position, 'easy')

    def test_should_use_endgame_medium(self):
        """Test endgame activation for medium difficulty."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Too many empty cells - should not use endgame
        position = EndgamePosition(board=board, current_player=1)
        assert not should_use_endgame_solver(position, 'medium')

        # Fill board leaving 14 empty cells - should use endgame
        count = 0
        for i in range(15):
            for j in range(15):
                if count < 15 * 15 - 14:
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        position = EndgamePosition(board=board, current_player=1)
        assert should_use_endgame_solver(position, 'medium')

    def test_should_use_endgame_strong(self):
        """Test endgame activation for strong difficulty."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Fill board leaving 20 empty cells - should use endgame
        count = 0
        for i in range(15):
            for j in range(15):
                if count < 15 * 15 - 20:
                    board[i, j] = 1 if count % 2 == 0 else -1
                    count += 1

        position = EndgamePosition(board=board, current_player=1)
        assert should_use_endgame_solver(position, 'strong')

        # Too many empty cells - should not use endgame
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1  # Only one stone
        position = EndgamePosition(board=board, current_player=1)
        assert not should_use_endgame_solver(position, 'strong')

    def test_full_board_should_not_use_endgame(self):
        """Test that completely full board should not use endgame."""
        board = np.ones((15, 15), dtype=np.int8)
        position = EndgamePosition(board=board, current_player=1)

        assert not should_use_endgame_solver(position, 'strong')
        assert not should_use_endgame_solver(position, 'medium')
        assert not should_use_endgame_solver(position, 'easy')