#!/usr/bin/env python3
"""Monitor distributed training queue status.

Simple CLI tool to monitor Redis queue status, worker activity, and statistics.

Example:
    python scripts/monitor_queue.py --redis-url redis://:password@REDIS_DOMAIN:6379/0
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.queue import RedisQueue


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def format_time_elapsed(seconds: float) -> str:
    """Format elapsed time in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_status(queue: RedisQueue, start_time: float, prev_stats: dict) -> dict:
    """Print queue status and return current stats."""
    # Clear screen (ANSI escape code)
    print("\033[2J\033[H", end="")

    # Get current stats
    stats = queue.get_stats()
    workers = queue.get_active_workers()
    health = queue.health_check()

    # Calculate rates
    elapsed = time.time() - start_time
    positions_pushed_rate = 0
    positions_pulled_rate = 0

    if prev_stats:
        time_delta = 5.0  # Refresh interval
        positions_pushed_delta = stats['games_pushed'] - prev_stats['games_pushed']
        positions_pulled_delta = stats['games_pulled'] - prev_stats['games_pulled']
        positions_pushed_rate = (positions_pushed_delta / time_delta) * 3600  # per hour
        positions_pulled_rate = (positions_pulled_delta / time_delta) * 3600  # per hour

    # Print header
    print("=" * 70)
    print("  AlphaGomoku Distributed Training Queue Monitor")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
          f"Uptime: {format_time_elapsed(elapsed)}")
    print()

    # Connection status
    status_symbol = "✓" if health else "✗"
    status_color = "\033[92m" if health else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    print(f"  Connection: {status_color}{status_symbol} {'Healthy' if health else 'Disconnected'}{reset_color}")
    print()

    # Queue status
    print("─" * 70)
    print("  QUEUE STATUS")
    print("─" * 70)
    queue_size = stats['queue_size']
    queue_bar = "█" * min(queue_size, 50)
    print(f"  Game Batches:     {format_number(queue_size)} {queue_bar}")
    print(f"  Models Available: {stats['models_available']}")
    print()

    # Worker status
    print("─" * 70)
    print("  ACTIVE WORKERS")
    print("─" * 70)
    selfplay_workers = workers['selfplay']
    training_workers = workers['training']
    print(f"  Self-Play Workers: {selfplay_workers}")
    print(f"  Training Workers:  {training_workers}")
    print()

    # Statistics
    print("─" * 70)
    print("  STATISTICS")
    print("─" * 70)
    print(f"  Positions Pushed:  {format_number(stats['games_pushed'])} "
          f"({positions_pushed_rate:.1f}/hour)")
    print(f"  Positions Pulled:  {format_number(stats['games_pulled'])} "
          f"({positions_pulled_rate:.1f}/hour)")
    print(f"  Models Pushed: {format_number(stats['models_pushed'])}")
    print(f"  Models Pulled: {format_number(stats['models_pulled'])}")
    print()

    # Training progress estimate
    positions_in_queue = stats['games_pushed'] - stats['games_pulled']
    if stats['games_pulled'] > 0:
        training_efficiency = (stats['games_pulled'] / stats['games_pushed']) * 100
        print("─" * 70)
        print("  TRAINING PROGRESS")
        print("─" * 70)
        print(f"  Positions Processed:  {format_number(stats['games_pulled'])} / "
              f"{format_number(stats['games_pushed'])} ({training_efficiency:.1f}%)")
        print(f"  Positions Pending:    {format_number(positions_in_queue)}")
        print()

    # Footer
    print("─" * 70)
    print("  Press Ctrl+C to exit | Refreshing every 5 seconds")
    print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Monitor distributed training queue')
    parser.add_argument('--redis-url', type=str, required=True,
                        help='Redis connection URL (redis://:password@host:port/db)')
    parser.add_argument('--refresh', type=int, default=5,
                        help='Refresh interval in seconds (default: 5)')

    args = parser.parse_args()

    # Connect to Redis
    try:
        queue = RedisQueue(redis_url=args.redis_url)
        print("Connected to Redis successfully")
        time.sleep(1)
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        return 1

    start_time = time.time()
    prev_stats = None

    try:
        while True:
            prev_stats = print_status(queue, start_time, prev_stats)
            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        return 0

    except Exception as e:
        print(f"\n\nError: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
