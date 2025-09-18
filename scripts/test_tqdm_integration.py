#!/usr/bin/env python3
"""Test script to verify tqdm integration works correctly"""

import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
import tempfile
import os


def test_tqdm_integration():
    """Test that tqdm progress bars work correctly"""
    tqdm.write("🧪 Testing tqdm integration...")
    
    # Initialize small model for testing
    model = GomokuNet(board_size=15, num_blocks=2, channels=16)
    tqdm.write(f"Model parameters: {model.get_model_size():,}")
    
    # Test self-play with tqdm
    tqdm.write("\n1. Testing self-play progress bars...")
    selfplay_worker = SelfPlayWorker(model, mcts_simulations=10)
    
    # Generate a small batch to test progress bars
    game_data = selfplay_worker.generate_batch(3)  # 3 games
    tqdm.write(f"✓ Generated {len(game_data)} positions")
    
    # Test trainer with tqdm
    tqdm.write("\n2. Testing trainer progress bars...")
    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = os.path.join(temp_dir, 'test_buffer')
        data_buffer = DataBuffer(buffer_path, max_size=1000)
        data_buffer.add_data(game_data)
        
        trainer = Trainer(model, lr=0.01)
        metrics = trainer.train_epoch(data_buffer, batch_size=16, steps_per_epoch=5)
        
        if metrics:
            tqdm.write(f"✓ Training metrics: loss={metrics['total_loss']:.4f}")
    
    tqdm.write("\n✅ tqdm integration test completed successfully!")
    tqdm.write("Console output should now be much more compact during training.")


if __name__ == '__main__':
    test_tqdm_integration()