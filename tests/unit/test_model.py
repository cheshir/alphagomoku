import pytest
import torch
from alphagomoku.model.network import GomokuNet


class TestGomokuNet:
    
    def test_init(self):
        model = GomokuNet(board_size=15, num_blocks=12, channels=64)
        assert model.board_size == 15
        assert model.channels == 64
        assert len(model.blocks) == 12
    
    def test_forward_shape(self):
        model = GomokuNet(board_size=15)
        batch_size = 4
        
        # Input: (batch, 5 channels, 15, 15)
        x = torch.randn(batch_size, 5, 15, 15)
        
        policy, value = model(x)
        
        assert policy.shape == (batch_size, 225)  # 15*15 actions
        assert value.shape == (batch_size,)
    
    def test_predict(self):
        model = GomokuNet(board_size=15)
        
        # Single state input
        state = torch.randn(5, 15, 15)
        
        policy, value = model.predict(state)
        
        assert policy.shape == (225,)
        assert isinstance(value, float)
        assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_model_size(self):
        model = GomokuNet(board_size=15, num_blocks=12, channels=64)
        size = model.get_model_size()
        
        assert isinstance(size, int)
        assert size > 0
        # Should be around 2-3M parameters for this configuration
        assert 2_000_000 < size < 4_000_000