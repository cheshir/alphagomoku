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


if __name__ == "__main__":
    pytest.main([__file__])