#!/usr/bin/env python3

import os
import sys
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# MPS optimization settings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'  # Fail if ops fallback to CPU
os.environ['OMP_NUM_THREADS'] = '1'  # Avoid OMP conflicts
os.environ['MKL_NUM_THREADS'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.selfplay.parallel import ParallelSelfPlay


def _plot_training_progress(history, current_epoch):
    """Plot training progress during training"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['policy_acc'])
    plt.title('Policy Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['value_mae'])
    plt.title('Value MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_progress_epoch_{current_epoch}.png', dpi=100, bbox_inches='tight')
    plt.close()


def _generate_final_report(history, args, model_path):
    """Generate comprehensive training report"""
    plt.figure(figsize=(15, 10))
    
    # Training metrics
    plt.subplot(2, 3, 1)
    plt.plot(history['loss'], 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['policy_acc'], 'g-', linewidth=2)
    plt.title('Policy Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['value_mae'], 'r-', linewidth=2)
    plt.title('Value MAE', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    
    # Training time per epoch
    plt.subplot(2, 3, 4)
    plt.plot(history['epoch_times'], 'purple', linewidth=2)
    plt.title('Epoch Duration', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 3, 5)
    plt.axis('off')
    if history['loss']:
        stats_text = f"""Training Summary:
        
• Total Epochs: {len(history['loss'])}
• Final Loss: {history['loss'][-1]:.4f}
• Final Policy Acc: {history['policy_acc'][-1]:.3f}
• Final Value MAE: {history['value_mae'][-1]:.3f}
• Avg Epoch Time: {np.mean(history['epoch_times']):.1f}s
• Total Training Time: {sum(history['epoch_times'])/3600:.1f}h
        
Model: {model_path}
Games per Epoch: {args.selfplay_games}
Batch Size: {args.batch_size}
Learning Rate: {args.lr}"""
        plt.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Training configuration
    plt.subplot(2, 3, 6)
    plt.axis('off')
    config_text = f"""Configuration:
    
• Board Size: 15x15
• Model Blocks: 12
• Model Channels: 64
• MCTS Simulations: {args.mcts_simulations}
• Adaptive Sims: {args.adaptive_sims}
• MCTS Batch Size: {args.batch_size_mcts}
• Search Difficulty: {args.difficulty} (MCTS + TSS + Endgame)
• Parallel Workers: {args.parallel_workers}
• Buffer Max Size: {args.buffer_max_size:,}
• Map Size: {args.map_size_gb}GB
• Temperature Moves: 8
• Optimizer: AdamW
• Weight Decay: 1e-4
• LR Schedule: StepLR
• Device: MPS/CPU"""
    plt.text(0.1, 0.9, config_text, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('AlphaGomoku Training Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Training report saved as 'training_report.png'")
    if history['loss']:
        print(f"📊 Final metrics: Loss={history['loss'][-1]:.4f}, "
              f"Acc={history['policy_acc'][-1]:.3f}, MAE={history['value_mae'][-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train AlphaGomoku model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--selfplay-games', type=int, default=100, help='Self-play games per iteration')
    parser.add_argument('--mcts-simulations', type=int, default=800, help='MCTS simulations per move')
    parser.add_argument('--map-size-gb', type=int, default=8, help='LMDB map size in GB')
    parser.add_argument('--buffer-max-size', type=int, default=2_000_000, help='Maximum buffer size (positions)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint (use "auto" for latest)')
    parser.add_argument('--adaptive-sims', action='store_true', help='Use adaptive simulation scheduling')
    parser.add_argument('--parallel-workers', type=int, default=1, help='Number of parallel selfplay workers')
    parser.add_argument('--batch-size-mcts', type=int, default=32, help='MCTS batch size for neural network evaluation')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'strong'], default='medium',
                        help='Training difficulty (affects TSS/endgame solver usage)')

    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model = GomokuNet(board_size=15, num_blocks=12, channels=64)
    print(f"Model parameters: {model.get_model_size():,}")
    
    # Initialize trainer
    trainer = Trainer(model, lr=args.lr)
    print(f"Trainer device: {trainer.device}")
    
    # Optimize for MPS
    if torch.backends.mps.is_available():
        torch.set_num_threads(1)
        print("✓ MPS optimizations enabled")
    
    # Initialize data buffer
    buffer_path = os.path.join(args.data_dir, 'replay_buffer')
    map_size = args.map_size_gb * 1024**3  # Convert GB to bytes
    data_buffer = DataBuffer(buffer_path, max_size=args.buffer_max_size, map_size=map_size)
    print(f"Data buffer: max_size={args.buffer_max_size:,}, map_size={args.map_size_gb}GB")
    
    # Initialize self-play worker with unified search
    if args.parallel_workers > 1:
        selfplay_worker = ParallelSelfPlay(
            model=model,
            mcts_simulations=args.mcts_simulations,
            adaptive_sims=args.adaptive_sims,
            batch_size=args.batch_size_mcts,
            num_workers=args.parallel_workers,
            difficulty=args.difficulty
        )
        print(f"Parallel selfplay: {args.parallel_workers} workers")
    else:
        selfplay_worker = SelfPlayWorker(
            model=model,
            mcts_simulations=args.mcts_simulations,
            adaptive_sims=args.adaptive_sims,
            batch_size=args.batch_size_mcts,
            difficulty=args.difficulty
        )
    
    print(f"MCTS simulations: {args.mcts_simulations} (adaptive: {args.adaptive_sims})")
    print(f"MCTS batch size: {args.batch_size_mcts}")
    print(f"Unified search difficulty: {args.difficulty} (includes MCTS + TSS + Endgame solver)")
    
    start_epoch = 0
    training_history = {'loss': [], 'policy_acc': [], 'value_mae': [], 'epoch_times': []}
    
    if args.resume:
        if args.resume == 'auto':
            # Find latest checkpoint
            checkpoints = list(Path(args.checkpoint_dir).glob('model_epoch_*.pt'))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                args.resume = str(latest)
                print(f"Auto-resume: found {args.resume}")
            else:
                print("No checkpoints found for auto-resume")
                args.resume = None
        
        if args.resume:
            checkpoint = trainer.load_checkpoint(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Generate self-play data
        print("🎮 Generating self-play data...")
        selfplay_start = time.time()
        selfplay_data = selfplay_worker.generate_batch(args.selfplay_games)
        
        try:
            data_buffer.add_data(selfplay_data)
            selfplay_time = time.time() - selfplay_start
            print(f"   ✓ Generated {len(selfplay_data)} positions in {selfplay_time:.1f}s")
            print(f"   📊 Buffer size: {len(data_buffer):,} positions")
        except Exception as e:
            print(f"   ❌ Error adding data to buffer: {e}")
            print(f"   💡 Try increasing --map-size-gb (current: {args.map_size_gb}GB)")
            print(f"   💡 Or reducing --buffer-max-size (current: {args.buffer_max_size:,})")
            raise
        
        # Train
        print("\n🧠 Training neural network...")
        train_start = time.time()
        metrics = trainer.train_epoch(data_buffer, args.batch_size)
        train_time = time.time() - train_start
        
        if metrics:
            print(f"\n📈 Training Results:")
            print(f"   Loss: {metrics['total_loss']:.4f} | "
                  f"Policy Acc: {metrics['policy_accuracy']:.3f} | "
                  f"Value MAE: {metrics['value_mae']:.3f}")
            print(f"   LR: {metrics['lr']:.6f} | Time: {train_time:.1f}s")
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pt')
            trainer.save_checkpoint(checkpoint_path, epoch, metrics)
            print(f"\n💾 Saved checkpoint: {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start
        print(f"\n⏱️  Epoch completed in {epoch_time:.1f}s")
        
        # Track history
        if metrics:
            training_history['loss'].append(metrics['total_loss'])
            training_history['policy_acc'].append(metrics['policy_accuracy'])
            training_history['value_mae'].append(metrics['value_mae'])
        training_history['epoch_times'].append(epoch_time)
        
        # Show progress chart every 5 epochs
        if (epoch + 1) % 5 == 0 and len(training_history['loss']) > 1:
            _plot_training_progress(training_history, epoch + 1)
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'model_final.pt')
    trainer.save_checkpoint(final_path, args.epochs - 1, metrics)
    
    # Generate final report
    _generate_final_report(training_history, args, final_path)
    print(f"\n🎉 Training complete! Final model: {final_path}")


if __name__ == '__main__':
    main()