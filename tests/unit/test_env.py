import pytest
import numpy as np
from alphagomoku.env.gomoku_env import GomokuEnv


class TestGomokuEnv:
    
    def test_init(self):
        env = GomokuEnv(board_size=15)
        assert env.board_size == 15
        assert env.action_space.n == 225
    
    def test_reset(self):
        env = GomokuEnv()
        obs, info = env.reset()
        
        assert obs['board'].shape == (15, 15)
        assert np.all(obs['board'] == 0)
        assert obs['current_player'] == 0
        assert np.all(obs['last_move'] == -1)
        assert np.sum(obs['action_mask']) == 225
    
    def test_valid_move(self):
        env = GomokuEnv()
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(112)  # Center
        
        assert obs['board'][7, 7] == -1  # Current player perspective (switched to player 2)
        assert reward == 0.0
        assert not terminated
        assert obs['current_player'] == 1  # Switched to second player
    
    def test_invalid_move(self):
        env = GomokuEnv()
        env.reset()
        env.step(112)  # Center
        
        # Try to play same position
        obs, reward, terminated, truncated, info = env.step(112)
        
        assert reward == -1.0
        assert terminated
        assert info.get('invalid_move', False)
    
    def test_win_detection(self):
        env = GomokuEnv()
        env.reset()
        
        # Create horizontal win for player 1
        positions = [112, 113, 114, 115, 116]  # Row 7, cols 7-11
        dummy_positions = [0, 1, 2, 3]  # For player 2
        
        for i, pos in enumerate(positions[:-1]):
            env.step(pos)  # Player 1
            if i < len(dummy_positions):
                env.step(dummy_positions[i])  # Player 2
        
        # Winning move
        obs, reward, terminated, truncated, info = env.step(positions[-1])
        
        assert terminated
        assert reward == 1.0
        assert info.get('winner') == 1
    
    def test_legal_actions(self):
        env = GomokuEnv()
        env.reset()
        
        legal_actions = env.get_legal_actions()
        assert len(legal_actions) == 225
        
        env.step(112)  # Make a move
        legal_actions = env.get_legal_actions()
        assert len(legal_actions) == 224
        assert 112 not in legal_actions