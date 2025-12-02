"""Local cache for self-play positions with replay capability.

This module provides utilities to save generated positions locally and replay them
to Redis later. Useful for testing and debugging distributed training.
"""

import os
import pickle
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class PositionCache:
    """Local cache for self-play positions.

    Saves positions to disk in batches, allowing later replay to Redis queue.
    Useful for testing distributed training without running self-play workers.

    Example:
        >>> cache = PositionCache('./cache')
        >>> cache.save_batch(positions, metadata={'worker': 'test-1'})
        >>>
        >>> # Later, replay to Redis:
        >>> batches = cache.list_batches()
        >>> for batch_file in batches[:10]:
        ...     positions = cache.load_batch(batch_file)
        ...     queue.push_games(positions)
    """

    def __init__(self, cache_dir: str = './position_cache'):
        """Initialize position cache.

        Args:
            cache_dir: Directory to store cached position batches
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_batch(
        self,
        positions: List,
        metadata: Optional[dict] = None
    ) -> str:
        """Save a batch of positions to cache.

        Args:
            positions: List of SelfPlayData objects
            metadata: Optional metadata (worker_id, timestamp, etc.)

        Returns:
            Path to saved batch file
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'batch_{timestamp}_{len(positions)}pos.pkl'
        filepath = self.cache_dir / filename

        data = {
            'positions': positions,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat(),
            'count': len(positions),
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return str(filepath)

    def load_batch(self, batch_file: str) -> List:
        """Load a batch of positions from cache.

        Args:
            batch_file: Path to batch file (or just filename)

        Returns:
            List of SelfPlayData objects
        """
        filepath = Path(batch_file)
        if not filepath.is_absolute():
            filepath = self.cache_dir / filepath

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return data['positions']

    def load_batch_with_metadata(self, batch_file: str) -> dict:
        """Load a batch with full metadata.

        Args:
            batch_file: Path to batch file

        Returns:
            Dictionary with 'positions', 'metadata', 'timestamp', 'count'
        """
        filepath = Path(batch_file)
        if not filepath.is_absolute():
            filepath = self.cache_dir / filepath

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def list_batches(self, sort_by_time: bool = True) -> List[str]:
        """List all cached batch files.

        Args:
            sort_by_time: Sort by modification time (oldest first)

        Returns:
            List of batch file paths
        """
        batches = list(self.cache_dir.glob('batch_*.pkl'))

        if sort_by_time:
            batches.sort(key=lambda p: p.stat().st_mtime)

        return [str(b) for b in batches]

    def get_cache_stats(self) -> dict:
        """Get statistics about cached positions.

        Returns:
            Dictionary with cache statistics
        """
        batches = self.list_batches()

        if not batches:
            return {
                'total_batches': 0,
                'total_positions': 0,
                'total_size_mb': 0.0,
                'oldest_batch': None,
                'newest_batch': None,
            }

        total_positions = 0
        total_size = 0

        for batch_file in batches:
            filepath = Path(batch_file)
            total_size += filepath.stat().st_size

            # Extract position count from filename
            try:
                filename = filepath.name
                count_str = filename.split('_')[-1].replace('pos.pkl', '')
                total_positions += int(count_str)
            except (ValueError, IndexError):
                # Fallback: load file to count
                data = self.load_batch_with_metadata(batch_file)
                total_positions += data['count']

        oldest = Path(batches[0])
        newest = Path(batches[-1])

        return {
            'total_batches': len(batches),
            'total_positions': total_positions,
            'total_size_mb': total_size / 1024**2,
            'oldest_batch': oldest.name,
            'newest_batch': newest.name,
            'oldest_time': datetime.fromtimestamp(oldest.stat().st_mtime).isoformat(),
            'newest_time': datetime.fromtimestamp(newest.stat().st_mtime).isoformat(),
        }

    def clear_cache(self, keep_last_n: Optional[int] = None) -> int:
        """Clear cached batches.

        Args:
            keep_last_n: If specified, keep the N most recent batches

        Returns:
            Number of batches deleted
        """
        batches = self.list_batches(sort_by_time=True)

        if keep_last_n is not None:
            batches_to_delete = batches[:-keep_last_n] if keep_last_n > 0 else batches
        else:
            batches_to_delete = batches

        for batch_file in batches_to_delete:
            os.remove(batch_file)

        return len(batches_to_delete)


def replay_cache_to_redis(
    cache_dir: str,
    redis_url: str,
    batch_limit: Optional[int] = None,
    delay_seconds: float = 0.0
) -> dict:
    """Replay cached positions to Redis queue.

    Useful for testing distributed training without running self-play workers.

    Args:
        cache_dir: Directory containing cached batches
        redis_url: Redis connection URL
        batch_limit: Maximum number of batches to replay (None = all)
        delay_seconds: Delay between batches (simulate real-time generation)

    Returns:
        Statistics about replay operation

    Example:
        >>> stats = replay_cache_to_redis(
        ...     './cache',
        ...     'redis://:password@localhost:6379/0',
        ...     batch_limit=10,
        ...     delay_seconds=1.0
        ... )
        >>> print(f"Replayed {stats['positions_pushed']} positions")
    """
    from ..queue import RedisQueue

    cache = PositionCache(cache_dir)
    queue = RedisQueue(redis_url)

    batches = cache.list_batches(sort_by_time=True)

    if batch_limit is not None:
        batches = batches[:batch_limit]

    total_positions = 0
    total_batches = 0
    start_time = time.time()

    print(f"Replaying {len(batches)} batches from cache to Redis...")

    for i, batch_file in enumerate(batches, 1):
        positions = cache.load_batch(batch_file)
        queue.push_games(positions)

        total_positions += len(positions)
        total_batches += 1

        print(f"  [{i}/{len(batches)}] Pushed {len(positions)} positions "
              f"(total: {total_positions})")

        if delay_seconds > 0 and i < len(batches):
            time.sleep(delay_seconds)

    elapsed = time.time() - start_time

    stats = {
        'batches_replayed': total_batches,
        'positions_pushed': total_positions,
        'elapsed_seconds': elapsed,
        'positions_per_second': total_positions / elapsed if elapsed > 0 else 0,
    }

    print(f"\n✓ Replay complete:")
    print(f"  Batches: {stats['batches_replayed']}")
    print(f"  Positions: {stats['positions_pushed']}")
    print(f"  Time: {stats['elapsed_seconds']:.1f}s")
    print(f"  Rate: {stats['positions_per_second']:.1f} positions/s")

    return stats
