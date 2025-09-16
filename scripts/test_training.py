#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
import tempfile
import os


def test_training_pipeline():
    """Test the complete training pipeline"""
    print("Testing AlphaGomoku training pipeline...")
    
    # Initialize model
    model = GomokuNet(board_size=15, num_blocks=4, channels=32)  # Smaller for testing
    print(f"Model parameters: {model.get_model_size():,}")
    
    # Test self-play
    print("\n1. Testing self-play...")
    selfplay_worker = SelfPlayWorker(model, num_simulations=50)  # Fewer sims for testing
    game_data = selfplay_worker.generate_game()
    print(f"Generated {len(game_data)} training examples from one game")
    
    # Test data buffer
    print("\n2. Testing data buffer...")
    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = os.path.join(temp_dir, 'test_buffer')
        data_buffer = DataBuffer(buffer_path, max_size=1000)
        
        data_buffer.add_data(game_data)
        print(f"Buffer size after adding data: {len(data_buffer)}")
        
        batch = data_buffer.sample_batch(32)
        print(f"Sampled batch size: {len(batch)}")
    
    # Test trainer
    print("\n3. Testing trainer...")
    trainer = Trainer(model, lr=0.01)
    
    if batch:
        metrics = trainer.train_step(batch)
        print(f"Training metrics: {metrics}")
    
    print("\n✅ Training pipeline test completed successfully!")


if __name__ == '__main__':
    test_training_pipeline()