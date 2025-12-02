#!/usr/bin/env python3
"""Distributed self-play worker for AlphaGomoku.

This worker generates self-play games continuously and pushes them to a Redis queue.
It periodically fetches new models from the queue to improve game quality.

Example:
    python scripts/distributed_selfplay_worker.py \
        --redis-url redis://:password@REDIS_DOMAIN:6379/0 \
        --model-preset small \
        --mcts-simulations 50 \
        --device cpu \
        --worker-id mac-worker-1
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
from alphagomoku.selfplay.selfplay import SelfPlayWorker
from alphagomoku.queue import RedisQueue
from alphagomoku.utils.validation import (
    validate_redis_url,
    validate_selfplay_config,
    print_validation_errors
)


def setup_logging(worker_id: str) -> logging.Logger:
    """Setup logging for worker."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{worker_id}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Distributed self-play worker')
    parser.add_argument('--redis-url', type=str, required=True,
                        help='Redis connection URL (redis://:password@host:port/db)')
    parser.add_argument('--model-preset', type=str, choices=['small', 'medium', 'large'],
                        default='small', help='Model preset to use')
    parser.add_argument('--mcts-simulations', type=int, default=50,
                        help='MCTS simulations per move')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda'],
                        default='cpu', help='Device to use')
    parser.add_argument('--worker-id', type=str, required=True,
                        help='Unique worker identifier')
    parser.add_argument('--games-per-batch', type=int, default=10,
                        help='Games to generate before pushing to queue')
    parser.add_argument('--model-update-frequency', type=int, default=10,
                        help='Fetch new model every N batches')
    parser.add_argument('--batch-size-mcts', type=int, default=32,
                        help='MCTS batch size for neural network evaluation')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save/load model checkpoints locally')
    parser.add_argument('--enable-cache', action='store_true',
                        help='Enable local caching of generated positions for testing/replay')
    parser.add_argument('--cache-dir', type=str, default='./position_cache',
                        help='Directory to cache positions locally (only used if --enable-cache)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.worker_id)

    logger.info("=" * 60)
    logger.info("Distributed Self-Play Worker Starting")
    logger.info("=" * 60)

    # Validate arguments
    validation_errors = []

    # Validate Redis URL
    validation_errors.extend(validate_redis_url(args.redis_url))

    # Validate self-play configuration
    validation_errors.extend(validate_selfplay_config(
        args.mcts_simulations,
        args.batch_size_mcts,
        args.games_per_batch,
        args.model_update_frequency
    ))

    # Print validation errors and exit if any
    if validation_errors:
        print_validation_errors(validation_errors, logger)
        return 1

    # Log configuration
    logger.info(f"Worker ID: {args.worker_id}")
    logger.info(f"Redis URL: {args.redis_url.split('@')[0]}@***")  # Hide password
    logger.info(f"Model preset: {args.model_preset}")
    logger.info(f"MCTS simulations: {args.mcts_simulations}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Games per batch: {args.games_per_batch}")
    logger.info(f"Model update frequency: every {args.model_update_frequency} batches")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.enable_cache:
        logger.info(f"Position caching: ENABLED")
        logger.info(f"Cache directory: {args.cache_dir}")
    else:
        logger.info(f"Position caching: DISABLED")
    logger.info("✓ Configuration validated successfully")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup position cache if enabled
    position_cache = None
    if args.enable_cache:
        from alphagomoku.utils import PositionCache
        position_cache = PositionCache(args.cache_dir)
        logger.info(f"✓ Position cache initialized: {args.cache_dir}")

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

    # Try to load model in order: 1) local checkpoint, 2) Redis queue
    local_checkpoint = os.path.join(args.checkpoint_dir, 'latest_model.pt')
    model_loaded = False

    # Try local checkpoint first (faster, no network)
    if os.path.exists(local_checkpoint):
        try:
            logger.info(f"Loading model from local checkpoint: {local_checkpoint}")
            checkpoint = torch.load(local_checkpoint, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            iteration = checkpoint.get('iteration', 'unknown')
            logger.info(f"✓ Loaded model from local checkpoint (iteration: {iteration})")
            model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load local checkpoint: {e}")

    # If no local checkpoint, try Redis
    if not model_loaded:
        logger.info("Checking for trained model in Redis queue...")
        model_data = queue.pull_model(timeout=0)
        if model_data:
            try:
                model.load_state_dict(model_data['model_state'])
                iteration = model_data['metadata'].get('iteration', 'unknown')
                logger.info(f"✓ Loaded model from Redis queue (iteration: {iteration})")

                # Save to local checkpoint for future use
                logger.info(f"Saving model to local checkpoint: {local_checkpoint}")
                torch.save({
                    'model_state': model_data['model_state'],
                    'iteration': iteration,
                    'metadata': model_data['metadata']
                }, local_checkpoint)
                logger.info("✓ Model saved locally")
                model_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load model from queue: {e}")

    if not model_loaded:
        logger.info("No trained model found. Using random initialization.")

    # Initialize self-play worker
    selfplay_worker = SelfPlayWorker(
        model=model,
        mcts_simulations=args.mcts_simulations,
        adaptive_sims=False,  # Keep it simple for distributed training
        batch_size=args.batch_size_mcts,
        difficulty='easy',  # Always use easy for training
    )

    logger.info("Starting self-play loop...")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    batch_count = 0
    total_games = 0
    total_positions = 0
    start_time = time.time()

    try:
        while True:
            batch_start = time.time()

            # Register worker heartbeat
            queue.register_worker('selfplay', args.worker_id)

            # Generate games
            logger.info(f"Generating {args.games_per_batch} games (batch {batch_count + 1})...")
            games = selfplay_worker.generate_batch(args.games_per_batch)
            positions_count = len(games)

            # Save to local cache if enabled
            if position_cache is not None:
                cache_file = position_cache.save_batch(
                    games,
                    metadata={
                        'worker_id': args.worker_id,
                        'batch': batch_count + 1,
                        'games': args.games_per_batch,
                    }
                )
                logger.info(f"Cached to: {os.path.basename(cache_file)}")

            # Push to queue
            queue.push_games(games)
            batch_time = time.time() - batch_start

            # Update stats
            batch_count += 1
            total_games += args.games_per_batch
            total_positions += positions_count

            # Log progress
            elapsed_time = time.time() - start_time
            games_per_hour = (total_games / elapsed_time) * 3600
            logger.info(
                f"Batch {batch_count} complete: "
                f"{positions_count} positions in {batch_time:.1f}s "
                f"({batch_time/args.games_per_batch:.1f}s/game)"
            )
            logger.info(
                f"Total: {total_games} games, {total_positions} positions "
                f"({games_per_hour:.1f} games/hour)"
            )

            # Check queue stats
            stats = queue.get_stats()
            logger.info(
                f"Queue: {stats['queue_size']} batches, "
                f"{stats['games_pushed']} positions pushed, "
                f"{stats['games_pulled']} positions pulled"
            )

            # Update model periodically
            if batch_count % args.model_update_frequency == 0:
                logger.info("Checking for updated model...")
                model_data = queue.pull_model(timeout=0)
                if model_data:
                    try:
                        model.load_state_dict(model_data['model_state'])
                        iteration = model_data['metadata'].get('iteration', 'unknown')
                        logger.info(f"✓ Updated to model from iteration {iteration}")

                        # Save updated model to local checkpoint
                        torch.save({
                            'model_state': model_data['model_state'],
                            'iteration': iteration,
                            'metadata': model_data['metadata']
                        }, local_checkpoint)
                        logger.info(f"✓ Model saved to {local_checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to load updated model: {e}")
                else:
                    logger.info("No new model available")

            logger.info("-" * 60)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Worker stopped by user")
        logger.info(f"Total games generated: {total_games}")
        logger.info(f"Total positions: {total_positions}")
        logger.info(f"Runtime: {(time.time() - start_time) / 3600:.2f} hours")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
