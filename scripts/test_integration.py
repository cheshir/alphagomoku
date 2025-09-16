#!/usr/bin/env python3

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.eval.evaluator import Evaluator
import tempfile
import os


def test_integration():
    """Test integration without heavy computation"""
    print("Testing AlphaGomoku integration...")
    
    # Initialize small model for fast testing
    model = GomokuNet(board_size=9, num_blocks=2, channels=16)
    print(f"✓ Model initialized: {model.get_model_size():,} parameters")
    
    # Test self-play with minimal simulations
    print("\n1. Testing self-play...")
    selfplay_worker = SelfPlayWorker(
        model, 
        board_size=9,
        mcts_simulations=5,  # Very low for speed
        adaptive_sims=False,
        difficulty="easy"
    )
    
    start_time = time.time()
    game_data = selfplay_worker.generate_game()
    selfplay_time = time.time() - start_time
    print(f"✓ Generated {len(game_data)} positions in {selfplay_time:.2f}s")
    
    # Test data buffer
    print("\n2. Testing data buffer...")
    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = os.path.join(temp_dir, 'test_buffer')
        data_buffer = DataBuffer(buffer_path, max_size=1000, map_size=1024**2)
        
        data_buffer.add_data(game_data)
        print(f"✓ Buffer size after adding: {len(data_buffer)}")
        
        batch = data_buffer.sample_batch(8)
        print(f"✓ Sampled batch size: {len(batch)}")
    
    # Test trainer
    print("\n3. Testing trainer...")
    trainer = Trainer(model, lr=0.01)
    print(f"✓ Trainer device: {trainer.device}")
    
    if batch:
        start_time = time.time()
        metrics = trainer.train_step(batch)
        train_time = time.time() - start_time
        print(f"✓ Training step completed in {train_time:.2f}s")
        print(f"  Metrics: loss={metrics['total_loss']:.3f}, acc={metrics['policy_accuracy']:.3f}")
    
    # Test evaluator
    print("\n4. Testing evaluator...")
    evaluator = Evaluator(model, board_size=9)
    
    # Quick game with minimal simulations
    start_time = time.time()
    result = evaluator.play_game(player1_sims=3, player2_sims=3)
    eval_time = time.time() - start_time
    print(f"✓ Evaluation game completed in {eval_time:.2f}s")
    print(f"  Result: winner={result['winner']}, moves={result['moves']}")
    
    print("\n✅ All integration tests passed!")
    print(f"Total test time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    test_integration()