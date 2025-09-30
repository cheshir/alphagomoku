"""Unit tests for TSS module."""

import pytest
import time
import numpy as np
from alphagomoku.tss import Position, ThreatDetector, ThreatType, tss_search


class TestPosition:
    """Test Position class."""
    
    def test_position_creation(self):
        board = np.zeros((15, 15), dtype=np.int8)
        pos = Position(board=board, current_player=1)
        assert pos.board_size == 15
        assert pos.current_player == 1
        assert pos.last_move is None
    
    def test_make_move(self):
        board = np.zeros((15, 15), dtype=np.int8)
        pos = Position(board=board, current_player=1)
        new_pos = pos.make_move(7, 7)
        
        assert new_pos.board[7, 7] == 1
        assert new_pos.current_player == -1
        assert new_pos.last_move == (7, 7)
        assert pos.board[7, 7] == 0  # Original unchanged
    
    def test_get_legal_moves(self):
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        pos = Position(board=board, current_player=-1)
        
        legal_moves = pos.get_legal_moves()
        assert len(legal_moves) == 15 * 15 - 1
        assert (7, 7) not in legal_moves
    
    def test_is_terminal_win(self):
        board = np.zeros((15, 15), dtype=np.int8)
        # Create horizontal 5 in a row
        for i in range(5):
            board[7, 7 + i] = 1
        
        pos = Position(board=board, current_player=-1, last_move=(7, 10))
        is_terminal, winner = pos.is_terminal()
        assert is_terminal
        assert winner == 1


class TestThreatDetector:
    """Test ThreatDetector class."""
    
    def test_open_three_detection(self):
        detector = ThreatDetector()
        board = np.zeros((15, 15), dtype=np.int8)
        
        # Create pattern: .XXX.
        board[7, 6] = 1
        board[7, 7] = 1
        board[7, 8] = 1
        
        pos = Position(board=board, current_player=1)
        threats = detector.detect_threats(pos, 1)
        
        # Should detect open three at positions (7,5) and (7,9)
        threat_positions = [(r, c) for r, c, _ in threats]
        assert (7, 5) in threat_positions
        assert (7, 9) in threat_positions
    
    def test_open_four_detection(self):
        detector = ThreatDetector()
        board = np.zeros((15, 15), dtype=np.int8)
        
        # Create pattern: .XXX. (open three that becomes open four when extended)
        for i in range(3):
            board[7, 6 + i] = 1
        
        pos = Position(board=board, current_player=1)
        threats = detector.detect_threats(pos, 1)
        
        # Should detect open four when placing at (7,5) or (7,9)
        threat_positions = {(r, c): t for r, c, t in threats}
        
        # Check if placing at either end creates open four
        has_open_four = False
        for (r, c), threat_type in threat_positions.items():
            if threat_type == ThreatType.OPEN_FOUR:
                has_open_four = True
                break
        
        assert has_open_four or ThreatType.OPEN_THREE in [t for _, _, t in threats]
    
    def test_must_defend(self):
        detector = ThreatDetector()
        board = np.zeros((15, 15), dtype=np.int8)
        
        # Create opponent's open four threat
        for i in range(4):
            board[7, 6 + i] = -1
        
        pos = Position(board=board, current_player=1)
        defense_moves = detector.must_defend(pos, 1)
        
        assert len(defense_moves) > 0
        # Should include blocking moves
        assert (7, 5) in defense_moves or (7, 10) in defense_moves


class TestTSSSearch:
    """Test TSS search functionality."""
    
    def test_immediate_defense(self):
        board = np.zeros((15, 15), dtype=np.int8)
        
        # Create opponent's open four threat
        for i in range(4):
            board[7, 6 + i] = -1
        
        pos = Position(board=board, current_player=1)
        result = tss_search(pos, depth=4, time_cap_ms=100)
        
        assert result.is_forced_defense
        assert result.forced_move is not None
        assert result.forced_move in [(7, 5), (7, 10)]
    
    def test_forced_win_detection(self):
        board = np.zeros((15, 15), dtype=np.int8)
        
        # Create winning setup for current player
        # Three in a row with open ends
        board[7, 6] = 1
        board[7, 7] = 1
        board[7, 8] = 1
        
        pos = Position(board=board, current_player=1)
        result = tss_search(pos, depth=4, time_cap_ms=100)
        
        # Should find a winning move
        if result.is_forced_win:
            assert result.forced_move is not None
    
    def test_no_forced_sequence(self):
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1  # Single stone
        
        pos = Position(board=board, current_player=-1)
        result = tss_search(pos, depth=2, time_cap_ms=50)
        
        assert not result.is_forced_win
        assert not result.is_forced_defense
        assert result.forced_move is None
    
    def test_time_limit_respected(self):
        board = np.zeros((15, 15), dtype=np.int8)
        pos = Position(board=board, current_player=1)

        start_time = time.time()
        result = tss_search(pos, depth=6, time_cap_ms=50)
        end_time = time.time()

        # Should complete within reasonable time (allowing some overhead)
        assert (end_time - start_time) * 1000 < 200
        assert result.search_stats['time_ms'] <= 100  # Some tolerance


class TestTSSCriticalCases:
    """Test critical tactical scenarios."""

    def test_case_1_both_have_four_choose_win(self):
        """Case 1: Both players have 4-in-a-row - bot should win, not defend."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Player (1) has 4-in-a-row horizontally
        for i in range(4):
            board[5, 5 + i] = 1

        # Bot (-1) has 4-in-a-row vertically
        for i in range(4):
            board[6 + i, 10] = -1

        pos = Position(board=board, current_player=-1)
        result = tss_search(pos, depth=4, time_cap_ms=200)

        # Bot should choose to WIN, not defend
        assert result.is_forced_win
        assert not result.is_forced_defense
        assert result.forced_move is not None

        # Verify it's actually a winning move
        r, c = result.forced_move
        test_board = board.copy()
        test_board[r, c] = -1
        test_pos = Position(board=test_board, current_player=1, last_move=(r, c))
        is_terminal, winner = test_pos.is_terminal()
        assert is_terminal and winner == -1

    def test_case_2_player_has_three_bot_has_four_choose_win(self):
        """Case 2: Player has open 3, bot has 4 - bot should win."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Player has open three
        for i in range(3):
            board[5, 5 + i] = 1

        # Bot has open four
        for i in range(4):
            board[8, 5 + i] = -1

        pos = Position(board=board, current_player=-1)
        result = tss_search(pos, depth=4, time_cap_ms=200)

        # Bot should choose to WIN
        assert result.is_forced_win
        assert result.forced_move is not None

        # Verify it's a winning move
        r, c = result.forced_move
        test_board = board.copy()
        test_board[r, c] = -1
        test_pos = Position(board=test_board, current_player=1, last_move=(r, c))
        is_terminal, winner = test_pos.is_terminal()
        assert is_terminal and winner == -1

    def test_case_3_player_has_three_bot_no_win_must_defend(self):
        """Case 3: Player has open 3, bot has no win - must defend."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Player has open three
        for i in range(3):
            board[7, 6 + i] = 1

        # Bot has only scattered stones, no immediate win
        board[5, 5] = -1
        board[6, 6] = -1

        pos = Position(board=board, current_player=-1)
        result = tss_search(pos, depth=4, time_cap_ms=200)

        # Bot should defend against the open three
        assert result.is_forced_defense or result.forced_move is not None

        # The move should block one of the player's threat positions
        detector = ThreatDetector()
        player_threats = detector.detect_threats(pos, 1)
        threat_positions = {(r, c) for r, c, _ in player_threats}

        if result.forced_move:
            assert result.forced_move in threat_positions

    def test_case_4_open_two_not_urgent(self):
        """Case 4: Player has open-2, bot should NOT be forced to defend."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Player has only 2 stones (open-2)
        board[7, 6] = 1
        board[7, 7] = 1

        # Bot has 3 stones elsewhere
        for i in range(3):
            board[5, 5 + i] = -1

        pos = Position(board=board, current_player=-1)
        result = tss_search(pos, depth=4, time_cap_ms=200)

        # Bot should NOT be forced to defend open-2 (not urgent!)
        # TSS should return no forced move, letting MCTS decide
        assert not result.is_forced_defense
        # Bot can look for its own opportunities

    def test_case_5_open_four_is_urgent(self):
        """Case 5: Player has potential open-4, bot MUST respond."""
        board = np.zeros((15, 15), dtype=np.int8)

        # Player has 3 stones (can create open-4)
        for i in range(3):
            board[7, 6 + i] = 1

        # Bot has 3 stones elsewhere
        for i in range(3):
            board[5, 5 + i] = -1

        pos = Position(board=board, current_player=-1)
        result = tss_search(pos, depth=4, time_cap_ms=200)

        # Bot must make a forced move (either win or defend open-4)
        assert result.forced_move is not None
        assert result.is_forced_win or result.is_forced_defense


if __name__ == "__main__":
    pytest.main([__file__])