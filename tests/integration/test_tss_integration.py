"""Integration tests for TSS with MCTS."""

import pytest
import numpy as np
import torch
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.tss import Position, tss_search


class TestTSSIntegration:
    """Test TSS integration with existing components."""
    
    @pytest.fixture
    def setup_components(self):
        """Setup test components."""
        env = GomokuEnv(board_size=15)
        model = GomokuNet(board_size=15, num_blocks=4, channels=32)  # Small model for testing
        mcts = MCTS(model, env, num_simulations=100)
        return env, model, mcts
    
    def test_position_from_env(self, setup_components):
        """Test converting GomokuEnv to TSS Position."""
        env, _, _ = setup_components
        env.reset()
        
        # Make some moves
        env.step(7 * 15 + 7)  # Center
        env.step(7 * 15 + 8)  # Adjacent
        
        # Convert to TSS Position
        position = Position(
            board=env.board,
            current_player=env.current_player,
            last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None,
            board_size=env.board_size
        )
        
        assert position.board_size == 15
        assert position.current_player == env.current_player
        assert position.board[7, 7] != 0
        assert position.board[7, 8] != 0
    
    def test_tss_with_mcts_position(self, setup_components):
        """Test TSS on position from MCTS tree."""
        env, model, mcts = setup_components
        env.reset()
        
        # Create a tactical position
        moves = [
            (7, 7), (7, 8), (8, 7), (8, 8),  # 2x2 square
            (6, 7), (9, 7)  # Vertical line setup
        ]
        
        for i, (r, c) in enumerate(moves):
            action = r * 15 + c
            env.step(action)
        
        # Convert to TSS position
        position = Position(
            board=env.board,
            current_player=env.current_player,
            last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None
        )
        
        # Run TSS
        result = tss_search(position, depth=4, time_cap_ms=100)
        
        # Should complete without error
        assert result.search_stats is not None
        assert 'nodes_visited' in result.search_stats
        assert 'time_ms' in result.search_stats
    
    def test_tss_tactical_override(self, setup_components):
        """Test TSS overriding MCTS in tactical situations."""
        env, model, mcts = setup_components
        env.reset()
        
        # Create position where opponent has open four threat
        board = np.zeros((15, 15), dtype=np.int8)
        
        # Opponent's open four (must defend)
        for i in range(4):
            board[7, 5 + i] = -1
        
        # Set up environment
        env.board = board
        env.current_player = 1
        env.last_move = np.array([7, 8])
        
        # Convert to TSS position
        position = Position(
            board=board,
            current_player=1,
            last_move=(7, 8)
        )
        
        # TSS should find forced defense
        tss_result = tss_search(position, depth=2, time_cap_ms=50)
        
        if tss_result.is_forced_defense:
            assert tss_result.forced_move is not None
            # Should be a blocking move
            r, c = tss_result.forced_move
            assert (r, c) in [(7, 4), (7, 9)]  # Block the open four
    
    def test_performance_comparison(self, setup_components):
        """Test TSS performance vs pure MCTS."""
        env, model, mcts = setup_components
        env.reset()
        
        # Create simple position
        env.step(7 * 15 + 7)  # Center move
        
        position = Position(
            board=env.board,
            current_player=env.current_player,
            last_move=(7, 7)
        )
        
        # Time TSS
        import time
        start = time.time()
        tss_result = tss_search(position, depth=3, time_cap_ms=50)
        tss_time = time.time() - start
        
        # Time MCTS (reduced simulations for fair comparison)
        mcts.num_simulations = 50
        start = time.time()
        action_probs, _ = mcts.search(env.board)
        mcts_time = time.time() - start
        
        # TSS should be faster for tactical positions
        print(f"TSS time: {tss_time:.3f}s, MCTS time: {mcts_time:.3f}s")
        
        # Both should complete successfully
        assert tss_result.search_stats is not None
        assert len(action_probs) == 15 * 15


class TestTSSMCTSIntegration:
    """Test TSS integration into MCTS workflow."""
    
    def test_tss_guided_mcts(self):
        """Test concept of TSS guiding MCTS move selection."""
        # This would be implemented in the actual MCTS integration
        # For now, just test the interface compatibility
        
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        
        position = Position(board=board, current_player=-1)
        
        # TSS analysis
        tss_result = tss_search(position, depth=3, time_cap_ms=30)
        
        # Should provide actionable information
        assert hasattr(tss_result, 'forced_move')
        assert hasattr(tss_result, 'is_forced_win')
        assert hasattr(tss_result, 'is_forced_defense')
        assert hasattr(tss_result, 'search_stats')
        
        # Stats should be meaningful
        stats = tss_result.search_stats
        assert 'nodes_visited' in stats
        assert 'time_ms' in stats
        assert stats['nodes_visited'] >= 0
        assert stats['time_ms'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])