#!/usr/bin/env python3
"""Distributed training worker for AlphaGomoku.

This worker pulls games from a Redis queue, trains the model, and publishes
updated models back to the queue.

Example:
    python scripts/distributed_training_worker.py \
        --redis-url redis://:password@REDIS_DOMAIN:6379/0 \
        --model-preset medium \
        --batch-size 1024 \
        --device cuda
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.model.network import GomokuNet
from alphagomoku.train.trainer import Trainer
from alphagomoku.queue import RedisQueue
from alphagomoku.utils.validation import (
    validate_redis_url,
    validate_training_config,
    print_validation_errors
)


def setup_logging(worker_id: str = "training") -> logging.Logger:
    """Setup logging for worker."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{worker_id}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def _log_training_metrics(
    checkpoint_dir: str,
    iteration: int,
    metrics: dict,
    train_time: float,
    buffer_size: int,
    positions_pulled: int
):
    """Log training metrics to CSV file (persistent across restarts)."""
    from alphagomoku.utils import append_metrics_to_csv

    csv_path = os.path.join(checkpoint_dir, 'training_metrics.csv')

    headers = [
        'iteration', 'total_loss', 'policy_loss', 'value_loss',
        'policy_accuracy', 'value_mae', 'lr', 'train_time',
        'buffer_size', 'positions_pulled'
    ]

    values = {
        'iteration': iteration,
        'total_loss': metrics.get('total_loss', ''),
        'policy_loss': metrics.get('policy_loss', ''),
        'value_loss': metrics.get('value_loss', ''),
        'policy_accuracy': metrics.get('policy_accuracy', ''),
        'value_mae': metrics.get('value_mae', ''),
        'lr': metrics.get('lr', ''),
        'train_time': train_time,
        'buffer_size': buffer_size,
        'positions_pulled': positions_pulled,
    }

    formatters = {
        'total_loss': '.6f',
        'policy_loss': '.6f',
        'value_loss': '.6f',
        'policy_accuracy': '.6f',
        'value_mae': '.6f',
        'lr': '.8f',
        'train_time': '.3f',
    }

    append_metrics_to_csv(csv_path, headers, values, formatters)


def main():
    parser = argparse.ArgumentParser(description='Distributed training worker')
    parser.add_argument('--redis-url', type=str, required=True,
                        help='Redis connection URL (redis://:password@host:port/db)')
    parser.add_argument('--model-preset', type=str, choices=['small', 'medium', 'large'],
                        default='medium', help='Model preset to use')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda'],
                        default='cuda', help='Device to use')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine schedule')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs for LR schedule')
    parser.add_argument('--publish-frequency', type=int, default=5,
                        help='Publish model every N training batches')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_distributed',
                        help='Directory to save checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data_distributed',
                        help='Directory for replay buffer (persistent LMDB storage)')
    parser.add_argument('--replay-buffer-size', type=int, default=500000,
                        help='Maximum positions in replay buffer')
    parser.add_argument('--min-position-batches-for-training', type=int, default=50,
                        help='Minimum position batches in queue before training (each batch ~1000 positions)')
    parser.add_argument('--position-batches-per-training-pull', type=int, default=50,
                        help='Position batches to pull per training iteration (each batch ~1000 positions)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("training-worker")

    logger.info("=" * 60)
    logger.info("Distributed Training Worker Starting")
    logger.info("=" * 60)

    # Validate arguments
    validation_errors = []

    # Validate Redis URL
    validation_errors.extend(validate_redis_url(args.redis_url))

    # Validate training configuration
    validation_errors.extend(validate_training_config(
        args.batch_size,
        args.lr,
        args.min_lr,
        args.publish_frequency,
        args.min_position_batches_for_training,
        args.position_batches_per_training_pull
    ))

    # Print validation errors and exit if any
    if validation_errors:
        print_validation_errors(validation_errors, logger)
        return 1

    # Log configuration
    logger.info(f"Redis URL: {args.redis_url.split('@')[0]}@***")  # Hide password
    logger.info(f"Model preset: {args.model_preset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr} -> {args.min_lr}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Publish frequency: every {args.publish_frequency} batches")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info("✓ Configuration validated successfully")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Connect to Redis
    try:
        queue = RedisQueue(redis_url=args.redis_url)
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return 1

    # Initialize model
    logger.info(f"Initializing {args.model_preset} model...")
    model = GomokuNet.from_preset(args.model_preset, board_size=15, device=args.device)
    logger.info(f"Model initialized: {model.get_model_size():,} parameters")

    # Initialize trainer
    trainer = Trainer(
        model,
        lr=args.lr,
        lr_schedule='cosine',
        warmup_epochs=args.warmup_epochs,
        max_epochs=1000,  # Will train indefinitely
        min_lr=args.min_lr,
        device=args.device,
    )
    logger.info(f"Trainer initialized on {args.device}")

    # Try to resume from latest checkpoint
    training_iteration = 0
    total_positions_trained = 0
    start_time = time.time()

    import glob
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'model_iteration_*.pt')))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        try:
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            checkpoint = trainer.load_checkpoint(latest_checkpoint)
            training_iteration = checkpoint.get('iteration', 0)
            total_positions_trained = checkpoint.get('total_positions', 0)
            logger.info(f"✓ Resumed from iteration {training_iteration}")
            logger.info(f"✓ Total positions trained so far: {total_positions_trained}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting from scratch")
    else:
        logger.info("No checkpoint found, starting from scratch")

    # Create persistent replay buffer
    from alphagomoku.train.data_buffer import DataBuffer
    buffer = DataBuffer(args.data_dir, max_size=args.replay_buffer_size)
    logger.info(f"✓ Replay buffer initialized: {args.data_dir} (max {args.replay_buffer_size:,} positions)")
    logger.info(f"  Current buffer size: {len(buffer):,} positions")

    logger.info("Starting training loop...")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        while True:
            # Register worker heartbeat
            queue.register_worker('training', 'training-worker')

            # Check queue status
            stats = queue.get_stats()
            queue_size = stats['queue_size']

            if queue_size < args.min_position_batches_for_training:
                logger.info(
                    f"Queue has {queue_size} position batches "
                    f"(need {args.min_position_batches_for_training}). Waiting..."
                )
                time.sleep(10)
                continue

            # Pull position batches from queue
            logger.info(f"Pulling {args.position_batches_per_training_pull} position batches from queue...")
            pull_start = time.time()
            games = queue.pull_games(batch_size=args.position_batches_per_training_pull, timeout=30)
            pull_time = time.time() - pull_start

            if not games:
                logger.warning("No games available after timeout. Retrying...")
                continue

            logger.info(
                f"Pulled {len(games)} positions in {pull_time:.1f}s "
                f"({len(games)/pull_time:.1f} positions/s)"
            )

            # Add positions to persistent replay buffer
            buffer.add_data(games)
            logger.info(f"Replay buffer size: {len(buffer):,} / {args.replay_buffer_size:,} positions")

            # Train on the replay buffer
            logger.info(f"Training on replay buffer (batch size: {args.batch_size})...")
            train_start = time.time()
            metrics = trainer.train_epoch(buffer, args.batch_size)
            train_time = time.time() - train_start

            training_iteration += 1
            total_positions_trained += len(games)

            # Log metrics to console
            if metrics:
                logger.info(
                    f"Iteration {training_iteration} complete in {train_time:.1f}s: "
                    f"loss={metrics['total_loss']:.4f}, "
                    f"policy_acc={metrics['policy_accuracy']:.3f}, "
                    f"value_mae={metrics['value_mae']:.3f}, "
                    f"lr={metrics['lr']:.1e}"
                )

                # Log metrics to CSV (persistent across restarts)
                _log_training_metrics(
                    args.checkpoint_dir,
                    training_iteration,
                    metrics,
                    train_time,
                    len(buffer),
                    len(games)
                )
            else:
                logger.warning(f"Training iteration {training_iteration} produced no metrics")

            # Total stats
            elapsed_time = time.time() - start_time
            positions_per_hour = (total_positions_trained / elapsed_time) * 3600
            logger.info(
                f"Total trained: {total_positions_trained} positions "
                f"({positions_per_hour:.0f} positions/hour)"
            )

            # Publish model periodically
            if training_iteration % args.publish_frequency == 0:
                logger.info("Publishing updated model to queue...")
                try:
                    model_state = model.state_dict()
                    # Move to CPU for serialization (smaller, device-agnostic)
                    model_state = {k: v.cpu() for k, v in model_state.items()}

                    metadata = {
                        'iteration': training_iteration,
                        'total_positions': total_positions_trained,
                        'metrics': metrics if metrics else {},
                    }

                    queue.push_model(model_state, metadata)
                    logger.info(f"Model published (iteration {training_iteration})")

                    # Save checkpoint locally with full training state
                    checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f'model_iteration_{training_iteration}.pt'
                    )
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "scheduler_state_dict": trainer.scheduler.state_dict(),
                        "iteration": training_iteration,
                        "total_positions": total_positions_trained,
                        "step": trainer.step,
                        "metrics": metrics if metrics else {},
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                except Exception as e:
                    logger.error(f"Failed to publish model: {e}", exc_info=True)

            # Check queue stats
            stats = queue.get_stats()
            workers = queue.get_active_workers()
            logger.info(
                f"Queue status: {stats['queue_size']} batches remaining, "
                f"{workers['selfplay']} self-play workers, "
                f"{workers['training']} training workers"
            )
            logger.info("-" * 60)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Worker stopped by user")
        logger.info(f"Training iterations: {training_iteration}")
        logger.info(f"Total positions trained: {total_positions_trained}")
        logger.info(f"Runtime: {(time.time() - start_time) / 3600:.2f} hours")

        # Save final model with full state
        logger.info("Saving final model...")
        checkpoint_path = os.path.join(args.checkpoint_dir, 'model_final.pt')
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "iteration": training_iteration,
            "total_positions": total_positions_trained,
            "step": trainer.step,
            "metrics": metrics if 'metrics' in locals() else {},
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Final checkpoint saved: {checkpoint_path}")

        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
