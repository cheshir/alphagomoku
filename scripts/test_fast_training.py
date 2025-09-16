#!/usr/bin/env python3

import sys
from pathlib import Path
import time
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
import tempfile


def test_fast_training():
    """Test training pipeline with minimal computation"""
    print("Testing fast training pipeline...")
    
    # Small model for speed
    model = GomokuNet(board_size=9, num_blocks=2, channels=16)
    print(f"✓ Model: {model.get_model_size():,} parameters")
    
    # Fast self-play worker
    selfplay_worker = SelfPlayWorker(
        model,
        board_size=9,
        mcts_simulations=5,  # Very fast
        adaptive_sims=False,
        difficulty="easy"
    )
    
    # Initialize trainer
    trainer = Trainer(model, lr=0.01)
    print(f"✓ Trainer device: {trainer.device}")
    
    # Initialize data buffer
    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = os.path.join(temp_dir, 'fast_buffer')
        data_buffer = DataBuffer(buffer_path, max_size=1000, map_size=1024**2)
        
        # Mini training loop
        for epoch in range(2):
            print(f"\\nEpoch {epoch+1}/2")
            
            # Generate minimal self-play data
            print("  Generating data...")
            start_time = time.time()
            selfplay_data = selfplay_worker.generate_batch(2)  # Just 2 games
            selfplay_time = time.time() - start_time
            print(f"  ✓ Generated {len(selfplay_data)} positions in {selfplay_time:.1f}s")
            
            # Add to buffer
            data_buffer.add_data(selfplay_data)
            print(f"  ✓ Buffer size: {len(data_buffer)}")
            
            # Train
            print("  Training...")
            start_time = time.time()
            metrics = trainer.train_epoch(data_buffer, batch_size=16)
            train_time = time.time() - start_time
            
            if metrics:
                print(f"  ✓ Training completed in {train_time:.1f}s")
                print(f"    Loss: {metrics['total_loss']:.3f}, Acc: {metrics['policy_accuracy']:.3f}")
    
    print("\\n✅ Fast training test completed successfully!")


if __name__ == '__main__':
    test_fast_training()