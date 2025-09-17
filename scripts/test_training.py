#!/usr/bin/env python3

import sys
from pathlib import Path
from tqdm import tqdm

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
    tqdm.write("Testing AlphaGomoku training pipeline...")
    
    # Initialize model
    model = GomokuNet(board_size=15, num_blocks=4, channels=32)  # Smaller for testing
    tqdm.write(f"Model parameters: {model.get_model_size():,}")
    
    # Test self-play
    tqdm.write("\n1. Testing self-play...")
    selfplay_worker = SelfPlayWorker(model, mcts_simulations=50)  # Fewer sims for testing
    game_data = selfplay_worker.generate_game()
    tqdm.write(f"Generated {len(game_data)} training examples from one game")
    
    # Test data buffer
    tqdm.write("\n2. Testing data buffer...")
    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = os.path.join(temp_dir, 'test_buffer')
        data_buffer = DataBuffer(buffer_path, max_size=1000)
        
        data_buffer.add_data(game_data)
        tqdm.write(f"Buffer size after adding data: {len(data_buffer)}")
        
        batch = data_buffer.sample_batch(32)
        tqdm.write(f"Sampled batch size: {len(batch)}")
    
    # Test trainer
    tqdm.write("\n3. Testing trainer...")
    trainer = Trainer(model, lr=0.01)
    
    if batch:
        metrics = trainer.train_step(batch)
        tqdm.write(f"Training metrics: {metrics}")
    
    tqdm.write("\n✅ Training pipeline test completed successfully!")


if __name__ == '__main__':
    test_training_pipeline()