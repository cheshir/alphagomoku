"""High-level position queue abstraction for distributed training.

This module provides a clean interface for pushing/pulling positions (training data)
between self-play workers and training workers, hiding Redis implementation details.
"""

from typing import List, Optional
from ..selfplay.selfplay import SelfPlayData
from .redis_queue import RedisQueue


class PositionQueue:
    """High-level interface for position queue operations.

    This abstraction makes it clear that we're working with positions (training examples),
    not games. It wraps the lower-level RedisQueue and provides consistent terminology.
    """

    def __init__(self, redis_url: str):
        """Initialize position queue.

        Args:
            redis_url: Redis connection URL
        """
        self._redis_queue = RedisQueue(redis_url)

    def push_positions(self, positions: List[SelfPlayData]) -> int:
        """Push positions to the queue for training.

        Args:
            positions: List of training positions (SelfPlayData objects)

        Returns:
            Number of positions pushed
        """
        return self._redis_queue.push_games(positions)

    def pull_positions(self, max_batches: int = 50, timeout: int = 0) -> List[SelfPlayData]:
        """Pull positions from the queue for training.

        Args:
            max_batches: Maximum number of batches to pull from Redis
            timeout: Seconds to wait if queue is empty (0 = don't wait)

        Returns:
            List of positions ready for training
        """
        return self._redis_queue.pull_games(batch_size=max_batches, timeout=timeout)

    def get_model_timestamp(self) -> Optional[str]:
        """Get the timestamp of the latest model (lightweight check).

        This is used by workers to efficiently poll for model updates without
        downloading the full model (~50MB).

        Returns:
            ISO format timestamp string or None if no model exists
        """
        return self._redis_queue.get_model_timestamp()

    def push_model(self, model_state: dict, metadata: Optional[dict] = None) -> None:
        """Push a trained model to Redis (replaces previous model).

        Uses efficient two-key strategy:
        - latest_model: The actual model data (~50MB)
        - latest_model_timestamp: ISO timestamp for polling

        Args:
            model_state: Model state dict (from model.state_dict())
            metadata: Optional metadata (iteration, metrics, etc.)
        """
        self._redis_queue.push_model(model_state, metadata)

    def pull_model(self, timeout: int = 0) -> Optional[dict]:
        """Pull latest model from Redis.

        Efficient pattern:
        1. Call get_model_timestamp() periodically (cheap)
        2. Only call pull_model() if timestamp changed (expensive)

        Args:
            timeout: Seconds to wait if no model available (0 = don't wait)

        Returns:
            Dictionary with 'model_state', 'metadata', 'timestamp' or None
        """
        return self._redis_queue.pull_model(timeout)

    def get_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics:
            - queue_size: Number of position batches in queue
            - positions_pushed: Total positions pushed (note: Redis calls this 'games_pushed')
            - positions_pulled: Total positions pulled (note: Redis calls this 'games_pulled')
            - models_pushed: Total models pushed
            - models_pulled: Total models pulled
        """
        stats = self._redis_queue.get_stats()
        # Rename for clarity (Redis internally calls these 'games' but they're positions)
        return {
            'queue_size': stats['queue_size'],
            'positions_pushed': stats['games_pushed'],
            'positions_pulled': stats['games_pulled'],
            'models_available': stats['models_available'],
            'models_pushed': stats['models_pushed'],
            'models_pulled': stats['models_pulled'],
        }

    def health_check(self) -> bool:
        """Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        return self._redis_queue.health_check()

    def clear_positions(self) -> None:
        """Clear all positions from queue (USE WITH CAUTION!)."""
        self._redis_queue.clear_queue(self._redis_queue.GAMES_QUEUE)

    def reset_stats(self) -> None:
        """Reset all statistics (USE WITH CAUTION!)."""
        self._redis_queue.reset_stats()
