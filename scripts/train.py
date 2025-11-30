#!/usr/bin/env python3

import os
import sys
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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
from alphagomoku.tss import TSSConfig, set_default_config
from alphagomoku.train.tactical_augmentation import augment_with_tactical_data
from alphagomoku.train.data_filter import apply_all_filters


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


def _generate_final_report(history, args, model_path, effective_lr_schedule: str, model=None):
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
    # Derive friendly labels from effective schedule
    schedule_label = 'Cosine+Warmup' if effective_lr_schedule == 'cosine' else 'StepLR'
    warmup_display = args.warmup_epochs if effective_lr_schedule == 'cosine' else 0

    # Get model info if available
    model_info = ""
    if model:
        model_size = model.get_model_size()
        model_info = f"• Model: {getattr(args, 'model_preset', 'custom')} ({model_size:,} params)\n"
        model_info += f"• Blocks: {model.num_blocks}, Channels: {model.num_channels}\n"
    else:
        model_info = "• Model Blocks: 30 (5M params)\n• Model Channels: 192\n"

    config_text = f"""Configuration:

• Board Size: 15x15
{model_info}• MCTS Simulations: {args.mcts_simulations}
• Adaptive Sims: {args.adaptive_sims}
• MCTS Batch Size: {args.batch_size_mcts}
• Search Difficulty: {args.difficulty} (MCTS + TSS + Endgame)
• Parallel Workers: {args.parallel_workers}
• Buffer Max Size: {args.buffer_max_size:,}
• Map Size: {args.map_size_gb}GB
• Temperature Moves: 8
• Optimizer: AdamW
• Weight Decay: 1e-4
• LR Schedule: {schedule_label}
• Warmup Epochs: {warmup_display}"""
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


def _append_metrics_csv(csv_path: str, epoch: int, history: dict, extra: dict):
    """Append training metrics to a CSV file, creating header on first write."""
    import csv
    import os

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    headers = [
        'epoch', 'loss', 'policy_acc', 'value_mae', 'lr',
        'epoch_time', 'selfplay_time', 'train_time', 'buffer_size', 'positions'
    ]
    # Current values (may be None if metrics not computed)
    loss = history['loss'][-1] if history['loss'] else ''
    acc = history['policy_acc'][-1] if history['policy_acc'] else ''
    mae = history['value_mae'][-1] if history['value_mae'] else ''
    lr = extra.get('lr', '')
    row = [
        epoch,
        loss,
        acc,
        mae,
        lr,
        extra.get('epoch_time', ''),
        extra.get('selfplay_time', ''),
        extra.get('train_time', ''),
        extra.get('buffer_size', ''),
        extra.get('positions', ''),
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


def _get_hardware_config(device: str) -> dict:
    """Get optimal training configuration based on hardware

    Model size is FIXED at 18 blocks × 192 channels (5.2M params) for all devices.
    Balanced choice: strong enough for competitive play, trains in reasonable time.
    Only batch size and checkpointing are adjusted per hardware.
    """
    # FIXED model architecture for all devices
    # 18 blocks × 192 channels = 5.2M parameters
    # Optimal balance for Gomoku 15×15: strength + efficiency + training time
    config = {
        'batch_size': None,
        'use_checkpoint': False,
        'num_blocks': 18,
        'channels': 192,
        'description': ''
    }

    if device == 'cuda':
        # Check GPU memory for batch size optimization
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)

            # Memory requirements for 5.2M model (18 blocks × 192 channels):
            # - Model: ~20 MB
            # - Optimizer: ~40 MB
            # - Batch 2048: ~18 GB (without checkpointing)
            # - Batch 1024: ~9 GB (without checkpointing)
            # - Batch 512: ~5 GB (without checkpointing)

            if gpu_memory_gb >= 32:  # A100, A6000, RTX 6000, etc.
                config['batch_size'] = 2048
                config['use_checkpoint'] = False
                config['description'] = f'{gpu_name} ({gpu_memory_gb:.0f}GB) - Large batch'
            elif gpu_memory_gb >= 20:  # RTX 4090, RTX 3090 Ti, A5000
                config['batch_size'] = 1024
                config['use_checkpoint'] = False
                config['description'] = f'{gpu_name} ({gpu_memory_gb:.0f}GB) - Medium batch'
            elif gpu_memory_gb >= 12:  # RTX 4080, RTX 3080, RTX 3090
                config['batch_size'] = 512
                config['use_checkpoint'] = False
                config['description'] = f'{gpu_name} ({gpu_memory_gb:.0f}GB) - Standard batch'
            elif gpu_memory_gb >= 8:  # RTX 4060 Ti, RTX 3070
                config['batch_size'] = 512
                config['use_checkpoint'] = True
                config['description'] = f'{gpu_name} ({gpu_memory_gb:.0f}GB) - Checkpointing enabled'
            else:  # < 8 GB (RTX 3060, etc.)
                config['batch_size'] = 256
                config['use_checkpoint'] = True
                config['description'] = f'{gpu_name} ({gpu_memory_gb:.0f}GB) - Small batch + checkpointing'

    elif device == 'mps':
        # MPS has ~18 GB limit, check system RAM to avoid swapping
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / 1024**3

            if ram_gb >= 32:  # 32+ GB RAM
                config['batch_size'] = 512
                config['use_checkpoint'] = False
                config['description'] = f'Apple Silicon ({ram_gb:.0f}GB RAM) - No swapping'
            else:  # 16 GB RAM (will swap without checkpointing)
                config['batch_size'] = 256
                config['use_checkpoint'] = True
                config['description'] = f'Apple Silicon ({ram_gb:.0f}GB RAM) - Checkpointing to reduce memory'
        except ImportError:
            # Fallback if psutil not available
            config['batch_size'] = 512
            config['use_checkpoint'] = False
            config['description'] = 'Apple Silicon - Default config'

    elif device == 'cpu':
        config['batch_size'] = 128
        config['use_checkpoint'] = False
        config['description'] = 'CPU - Small batch for acceptable speed'

    return config


def _select_device(device_arg: str) -> str:
    """Select training device based on argument and availability"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"🚀 Auto-detected device: CUDA (GPU: {torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("🚀 Auto-detected device: MPS (Apple Silicon)")
        else:
            device = 'cpu'
            print("⚠️  Auto-detected device: CPU (no GPU available)")
    else:
        device = device_arg
        # Validate the selected device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        print(f"✓ Using device: {device.upper()}")

    return device


def _log_memory_stats(label: str, device: str):
    """Log detailed memory statistics for debugging"""
    if device == 'mps' and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        driver = torch.mps.driver_allocated_memory() / 1024**3
        tqdm.write(f"   [{label}] MPS allocated: {allocated:.2f} GB, driver: {driver:.2f} GB")
    elif device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        tqdm.write(f"   [{label}] CUDA allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

    # Also log system memory if psutil available
    try:
        import psutil
        process = psutil.Process()
        rss_gb = process.memory_info().rss / 1024**3
        tqdm.write(f"   [{label}] Process RSS: {rss_gb:.2f} GB")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description='Train AlphaGomoku model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--model-preset', type=str, choices=['small', 'medium', 'large'], default=None,
                        help='Model preset: small (1.2M), medium (3M), large (5M)')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-schedule', type=str, choices=['step', 'cosine', 'auto'], default='step',
                        help='Learning rate schedule ("auto" uses cosine if warmup>0, else step)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Warmup epochs for cosine schedule')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='Minimum LR for cosine schedule')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--selfplay-games', type=int, default=100, help='Self-play games per iteration')
    parser.add_argument('--mcts-simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--map-size-gb', type=int, default=8, help='LMDB map size in GB')
    parser.add_argument('--buffer-max-size', type=int, default=2_000_000, help='Maximum buffer size (positions)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint (use "auto" for latest)')
    parser.add_argument('--adaptive-sims', action='store_true', help='Use adaptive simulation scheduling')
    parser.add_argument('--parallel-workers', type=int, default=1, help='Number of parallel selfplay workers')
    parser.add_argument('--batch-size-mcts', type=int, default=64, help='MCTS batch size for neural network evaluation')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'strong'], default='medium',
                        help='Training difficulty (affects TSS/endgame solver usage)')
    parser.add_argument('--debug-memory', action='store_true', help='Enable detailed memory logging')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                        help='Device to use for training (auto=detect best available)')
    parser.add_argument('--eval-frequency', type=int, default=0,
                        help='Evaluate model every N epochs (0=disable)')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Number of games for evaluation')

    args = parser.parse_args()

    # Select device
    device = _select_device(args.device)

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize model
    if args.model_preset:
        # Use model preset (new approach)
        model = GomokuNet.from_preset(args.model_preset, board_size=15, device=device)
        print(f"\n🧠 Model Configuration:")
        print(f"   Preset: {args.model_preset}")
        print(f"   Parameters: {model.get_model_size():,}")

        # Get gradient checkpointing from model
        use_checkpoint = model.use_checkpoint
        print(f"   Gradient checkpointing: {'✓ Enabled' if use_checkpoint else '✗ Disabled'}")
    else:
        # Use hardware-optimized configuration (legacy)
        hw_config = _get_hardware_config(device)
        print(f"\n⚙️  Hardware Configuration: {hw_config['description']}")

        model = GomokuNet(
            board_size=15,
            num_blocks=hw_config['num_blocks'],
            channels=hw_config['channels'],
            use_checkpoint=hw_config['use_checkpoint']
        )
        print(f"\n🧠 Model Configuration:")
        print(f"   Parameters: {model.get_model_size():,}")
        print(f"   Blocks: {hw_config['num_blocks']}, Channels: {hw_config['channels']}")
        print(f"   Gradient checkpointing: {'✓ Enabled' if hw_config['use_checkpoint'] else '✗ Disabled'}")

    # Auto-configure batch size if not specified by user
    if args.batch_size == 512:  # Default value
        # Get optimal batch size based on device and model size
        if device == 'cuda' and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            model_size = model.get_model_size()

            if model_size < 2_000_000:  # Small model
                args.batch_size = 1024 if gpu_memory_gb >= 16 else 512
            elif model_size < 4_000_000:  # Medium model
                args.batch_size = 512 if gpu_memory_gb >= 16 else 256
            else:  # Large model
                args.batch_size = 512 if gpu_memory_gb >= 24 else 256
        elif device == 'mps':
            model_size = model.get_model_size()
            args.batch_size = 512 if model_size < 2_000_000 else 256

        print(f"   Auto-configured batch size: {args.batch_size}")
    else:
        print(f"   Using user-specified batch size: {args.batch_size}")

    # Determine effective LR schedule
    effective_lr_schedule = args.lr_schedule
    if effective_lr_schedule == 'auto':
        effective_lr_schedule = 'cosine' if args.warmup_epochs > 0 else 'step'

    # Initialize trainer with selected device
    trainer = Trainer(
        model,
        lr=args.lr,
        lr_schedule=effective_lr_schedule,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr=args.min_lr,
        device=device,
    )
    print(f"Trainer device: {trainer.device}")
    print(f"LR schedule: {effective_lr_schedule} (warmup_epochs={args.warmup_epochs}, min_lr={args.min_lr})")
    
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

    # Describe what each difficulty level actually includes
    difficulty_descriptions = {
        'easy': 'MCTS only (pure AlphaZero, no TSS/endgame - fast, GPU-accelerated)',
        'medium': 'MCTS + TSS + Endgame solver (tactical, slower)',
        'strong': 'MCTS + TSS + Endgame solver (maximum settings, slowest)'
    }
    difficulty_desc = difficulty_descriptions.get(args.difficulty, 'unknown')
    print(f"Search configuration: {args.difficulty} ({difficulty_desc})")
    print(f"   ℹ️  Training with 'easy' is recommended (4-6x faster, AlphaZero methodology)")
    print(f"   ℹ️  Use 'medium' or 'strong' for inference/evaluation, not training")
    
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
    epoch_pbar = tqdm(
        range(start_epoch, args.epochs), desc="Epochs", unit="epoch", position=0
    )
    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Memory logging at epoch start
        if args.debug_memory:
            tqdm.write(f"\n🔍 Epoch {epoch} Memory Tracking:")
            _log_memory_stats("Epoch start", device)

        # Update TSS config based on current epoch (progressive learning)
        # Only configure TSS if using medium/strong difficulty (TSS-enhanced training)
        if args.difficulty in ('medium', 'strong'):
            tss_config = TSSConfig.for_training_epoch(epoch)
            set_default_config(tss_config)

            # Log TSS config changes at key epochs
            if epoch % 25 == 0:
                tqdm.write(f"\n📊 Epoch {epoch} TSS Configuration:")
                tqdm.write(f"   - defend_immediate_five: {tss_config.defend_immediate_five}")
                tqdm.write(f"   - defend_open_four: {tss_config.defend_open_four}")
                tqdm.write(f"   - defend_broken_four: {tss_config.defend_broken_four}")
                tqdm.write(f"   - defend_open_three: {tss_config.defend_open_three}")
        elif epoch % 25 == 0:
            tqdm.write(f"\n📊 Epoch {epoch}: Training with pure MCTS (AlphaZero style, no TSS)")

        # Generate self-play data
        selfplay_start = time.time()
        if args.debug_memory:
            _log_memory_stats("Before selfplay", device)

        selfplay_data = selfplay_worker.generate_batch(args.selfplay_games)

        if args.debug_memory:
            _log_memory_stats("After selfplay", device)

        # CRITICAL: Aggressive cleanup to prevent memory fragmentation
        # Different strategies for different devices
        if device == 'mps' and torch.backends.mps.is_available():
            # Log memory before cleanup (always show this, not just debug mode)
            allocated_gb = torch.mps.current_allocated_memory() / 1024**3
            driver_gb = torch.mps.driver_allocated_memory() / 1024**3
            tqdm.write(f"   💾 MPS before cleanup: allocated={allocated_gb:.2f} GB, driver={driver_gb:.2f} GB")

            # Force garbage collection
            import gc
            gc.collect()

            # Clear MPS cache multiple times (sometimes needed)
            torch.mps.empty_cache()
            torch.mps.synchronize()
            time.sleep(0.5)  # Give OS time to reclaim
            torch.mps.empty_cache()

            # Log memory after cleanup (always show this)
            allocated_gb = torch.mps.current_allocated_memory() / 1024**3
            driver_gb = torch.mps.driver_allocated_memory() / 1024**3
            tqdm.write(f"   ✨ MPS after cleanup: allocated={allocated_gb:.2f} GB, driver={driver_gb:.2f} GB")
            tqdm.write(f"   📉 Freed: {allocated_gb:.2f} GB")

        elif device == 'cuda' and torch.cuda.is_available():
            # CUDA cleanup
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            tqdm.write(f"   💾 CUDA before cleanup: {allocated_before:.2f} GB")

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            allocated_after = torch.cuda.memory_allocated() / 1024**3
            tqdm.write(f"   ✨ CUDA after cleanup: {allocated_after:.2f} GB")
            tqdm.write(f"   📉 Freed: {allocated_before - allocated_after:.2f} GB")

        # Phase 4: AUGMENT with tactical training examples (30% of data for stronger tactics)
        # Higher ratio early in training, gradually reduce
        augmentation_ratio = 0.3 if epoch < 100 else 0.2
        selfplay_data = augment_with_tactical_data(selfplay_data, board_size=15, augmentation_ratio=augmentation_ratio)

        # Phase 5: FILTER stupid moves from training data
        # This is CRITICAL - prevents model from learning bad habits
        selfplay_data = apply_all_filters(selfplay_data, board_size=15)

        try:
            data_buffer.add_data(selfplay_data)
            selfplay_time = time.time() - selfplay_start
            epoch_pbar.set_postfix({
                'positions': len(selfplay_data),
                'buffer': f"{len(data_buffer):,}",
                'selfplay_time': f"{selfplay_time:.1f}s"
            })
        except Exception as e:
            tqdm.write(f"❌ Error adding data to buffer: {e}")
            tqdm.write(f"💡 Try increasing --map-size-gb (current: {args.map_size_gb}GB)")
            tqdm.write(f"💡 Or reducing --buffer-max-size (current: {args.buffer_max_size:,})")
            raise
        
        # Train
        train_start = time.time()
        if args.debug_memory:
            _log_memory_stats("Before training", device)

        metrics = trainer.train_epoch(data_buffer, args.batch_size)
        train_time = time.time() - train_start

        if args.debug_memory:
            _log_memory_stats("After training", device)

        if metrics:
            epoch_pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'acc': f"{metrics['policy_accuracy']:.3f}",
                'mae': f"{metrics['value_mae']:.3f}",
                'lr': f"{metrics['lr']:.1e}",
                'train_time': f"{train_time:.1f}s"
            })
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pt')
        trainer.save_checkpoint(checkpoint_path, epoch, metrics)
        
        epoch_time = time.time() - epoch_start
        
        # Track history
        if metrics:
            training_history['loss'].append(metrics['total_loss'])
            training_history['policy_acc'].append(metrics['policy_accuracy'])
            training_history['value_mae'].append(metrics['value_mae'])
        training_history['epoch_times'].append(epoch_time)
        
        # Persist metrics to CSV and refresh report each epoch
        csv_path = os.path.join(args.checkpoint_dir, 'training_metrics.csv')
        _append_metrics_csv(
            csv_path,
            epoch,
            training_history,
            {
                'epoch_time': round(epoch_time, 3),
                'selfplay_time': round(selfplay_time, 3),
                'train_time': round(train_time, 3),
                'buffer_size': len(data_buffer),
                'positions': len(selfplay_data),
                'lr': metrics['lr'] if metrics else ''
            },
        )

        # Always update the comprehensive report after each epoch
        try:
            _generate_final_report(training_history, args, checkpoint_path, effective_lr_schedule, model)
        except Exception as e:
            tqdm.write(f"⚠️ Failed to update report: {e}")
    
    epoch_pbar.close()
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'model_final.pt')
    trainer.save_checkpoint(final_path, args.epochs - 1, metrics)

    # Generate final report
    _generate_final_report(training_history, args, final_path, effective_lr_schedule, model)
    tqdm.write(f"🎉 Training complete! Final model: {final_path}")


if __name__ == '__main__':
    main()
