#!/usr/bin/env python3
"""Replay cached positions to Redis queue.

This script allows you to replay previously cached self-play positions to Redis,
useful for testing distributed training without running self-play workers.

Example:
    # Replay all cached positions:
    python scripts/replay_cache_to_redis.py \
        --cache-dir ./position_cache \
        --redis-url redis://:password@localhost:6379/0

    # Replay only first 10 batches:
    python scripts/replay_cache_to_redis.py \
        --cache-dir ./position_cache \
        --redis-url redis://:password@localhost:6379/0 \
        --batch-limit 10

    # Replay with delay to simulate real-time generation:
    python scripts/replay_cache_to_redis.py \
        --cache-dir ./position_cache \
        --redis-url redis://:password@localhost:6379/0 \
        --delay 2.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.utils import PositionCache, replay_cache_to_redis
from alphagomoku.utils.validation import validate_redis_url, print_validation_errors


def main():
    parser = argparse.ArgumentParser(
        description='Replay cached positions to Redis queue',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Directory containing cached position batches')
    parser.add_argument('--redis-url', type=str, required=True,
                        help='Redis connection URL (redis://:password@host:port/db)')
    parser.add_argument('--batch-limit', type=int, default=None,
                        help='Maximum number of batches to replay (default: all)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between batches in seconds (default: 0)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show cache statistics without replaying')

    args = parser.parse_args()

    print("=" * 60)
    print("Replay Cached Positions to Redis")
    print("=" * 60)

    # Validate arguments
    validation_errors = validate_redis_url(args.redis_url)

    if not Path(args.cache_dir).exists():
        validation_errors.append(f"Cache directory does not exist: {args.cache_dir}")

    if validation_errors:
        print_validation_errors(validation_errors)
        return 1

    # Show cache statistics
    print(f"\nCache directory: {args.cache_dir}")
    cache = PositionCache(args.cache_dir)
    stats = cache.get_cache_stats()

    if stats['total_batches'] == 0:
        print("❌ No cached batches found!")
        return 1

    print(f"\nCache Statistics:")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Total positions: {stats['total_positions']:,}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Oldest batch: {stats['oldest_batch']} ({stats['oldest_time']})")
    print(f"  Newest batch: {stats['newest_batch']} ({stats['newest_time']})")

    if args.stats_only:
        print("\n✓ Stats only mode - not replaying to Redis")
        return 0

    # Confirm replay
    batches_to_replay = min(args.batch_limit, stats['total_batches']) if args.batch_limit else stats['total_batches']
    print(f"\nReplay Configuration:")
    print(f"  Redis URL: {args.redis_url.split('@')[0]}@***")
    print(f"  Batches to replay: {batches_to_replay}")
    print(f"  Delay between batches: {args.delay}s")

    try:
        input("\nPress Enter to start replay (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 0

    # Replay to Redis
    print()
    try:
        replay_stats = replay_cache_to_redis(
            cache_dir=args.cache_dir,
            redis_url=args.redis_url,
            batch_limit=args.batch_limit,
            delay_seconds=args.delay
        )
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Replay interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during replay: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
