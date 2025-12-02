"""Simple Redis queue wrapper for distributed AlphaGomoku training.

This module provides a simple interface for pushing/pulling games and models
between self-play workers and training workers.
"""

import os
import pickle
import time
from typing import Any, List, Optional
import redis


class RedisQueue:
    """Simple Redis queue wrapper for distributed training.

    Queue structure:
        - games:queue: List of serialized game data
        - latest_model: Current model state (single key, not a queue)
        - latest_model_timestamp: ISO timestamp of last model update (for polling)
        - stats:games_pushed: Counter for total games pushed
        - stats:games_pulled: Counter for total games pulled
        - stats:models_pushed: Counter for total models pushed
        - stats:models_pulled: Counter for total models pulled
        - workers:selfplay:{worker_id}: Heartbeat timestamp
        - workers:training:{worker_id}: Heartbeat timestamp
    """

    GAMES_QUEUE = "games:queue"
    LATEST_MODEL = "latest_model"
    LATEST_MODEL_TIMESTAMP = "latest_model_timestamp"
    STATS_GAMES_PUSHED = "stats:games_pushed"
    STATS_GAMES_PULLED = "stats:games_pulled"
    STATS_MODELS_PUSHED = "stats:models_pushed"
    STATS_MODELS_PULLED = "stats:models_pulled"

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection.

        Args:
            redis_url: Redis connection URL (default: from REDIS_URL env var)
                      Format: redis://:password@host:port/db
        """
        if redis_url is None:
            redis_url = os.environ.get('REDIS_URL')
            if not redis_url:
                raise ValueError(
                    "REDIS_URL not provided and not found in environment. "
                    "Set REDIS_URL environment variable or pass redis_url parameter."
                )

        try:
            self.redis = redis.from_url(
                redis_url,
                decode_responses=False,  # We need binary for pickle
                socket_connect_timeout=10,
                socket_timeout=30,  # 30s timeout for large model uploads (~50MB)
            )
            # Test connection
            self.redis.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis at {redis_url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Redis initialization error: {e}")

    def push_games(self, games: List[Any]) -> int:
        """Push self-play games to the queue.

        Args:
            games: List of SelfPlayData objects or game dictionaries

        Returns:
            Number of games pushed
        """
        if not games:
            return 0

        # Serialize games
        serialized = pickle.dumps(games)

        # Push to queue (right side)
        self.redis.rpush(self.GAMES_QUEUE, serialized)

        # Update stats
        self.redis.incrby(self.STATS_GAMES_PUSHED, len(games))

        return len(games)

    def pull_games(self, batch_size: int = 50, timeout: int = 0) -> List[Any]:
        """Pull games from the queue.

        Args:
            batch_size: Maximum number of game batches to pull
            timeout: Seconds to wait if queue is empty (0 = don't wait)

        Returns:
            List of games (may be empty if queue is empty and timeout=0)
        """
        games = []

        for _ in range(batch_size):
            if timeout > 0 and len(games) == 0:
                # Use blocking pop for first item if timeout specified
                result = self.redis.blpop(self.GAMES_QUEUE, timeout=timeout)
                if result is None:
                    break
                _, serialized = result
            else:
                # Non-blocking pop
                serialized = self.redis.lpop(self.GAMES_QUEUE)
                if serialized is None:
                    break

            # Deserialize
            batch = pickle.loads(serialized)
            games.extend(batch)

        # Update stats
        if games:
            self.redis.incrby(self.STATS_GAMES_PULLED, len(games))

        return games

    def push_model(self, model_state: dict, metadata: Optional[dict] = None) -> None:
        """Push a trained model to Redis (replaces previous model).

        Uses two keys:
        - latest_model: The actual model data (~50MB)
        - latest_model_timestamp: ISO timestamp string for efficient polling

        Args:
            model_state: Model state dict (from model.state_dict())
            metadata: Optional metadata (iteration, metrics, etc.)
        """
        from datetime import datetime

        timestamp = datetime.utcnow().isoformat()

        data = {
            'model_state': model_state,
            'metadata': metadata or {},
            'timestamp': timestamp,
        }

        # Serialize model data
        serialized = pickle.dumps(data)

        # Atomic update: set both keys together
        pipeline = self.redis.pipeline()
        pipeline.set(self.LATEST_MODEL, serialized)
        pipeline.set(self.LATEST_MODEL_TIMESTAMP, timestamp)
        pipeline.incr(self.STATS_MODELS_PUSHED)
        pipeline.execute()

        # Note: Old model is immediately replaced (memory efficient)

    def get_model_timestamp(self) -> Optional[str]:
        """Get the timestamp of the latest model (lightweight check).

        This is used by workers to efficiently poll for model updates without
        downloading the full model (~50MB).

        Returns:
            ISO format timestamp string or None if no model exists
        """
        timestamp = self.redis.get(self.LATEST_MODEL_TIMESTAMP)
        if timestamp is None:
            return None
        return timestamp.decode('utf-8') if isinstance(timestamp, bytes) else timestamp

    def pull_model(self, timeout: int = 0) -> Optional[dict]:
        """Pull latest model from Redis.

        Workers should:
        1. Call get_model_timestamp() periodically (cheap)
        2. Only call pull_model() if timestamp changed (expensive)

        Args:
            timeout: Seconds to wait if no model available (0 = don't wait)

        Returns:
            Dictionary with 'model_state', 'metadata', 'timestamp' or None
        """
        # Get latest model directly from key
        serialized = self.redis.get(self.LATEST_MODEL)

        if serialized is None:
            if timeout > 0:
                # TODO: Implement blocking wait using pub/sub for efficiency
                # For now, just return None (workers will poll)
                pass
            return None

        # Deserialize
        data = pickle.loads(serialized)

        # Update stats
        self.redis.incr(self.STATS_MODELS_PULLED)

        return data

    def get_queue_size(self) -> int:
        """Get number of game batches in queue."""
        return self.redis.llen(self.GAMES_QUEUE)

    def get_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        # Check if model exists (1 or 0, not a list length)
        model_exists = 1 if self.redis.exists(self.LATEST_MODEL) else 0

        return {
            'queue_size': self.get_queue_size(),
            'models_available': model_exists,  # Now 0 or 1, not list length
            'games_pushed': int(self.redis.get(self.STATS_GAMES_PUSHED) or 0),
            'games_pulled': int(self.redis.get(self.STATS_GAMES_PULLED) or 0),
            'models_pushed': int(self.redis.get(self.STATS_MODELS_PUSHED) or 0),
            'models_pulled': int(self.redis.get(self.STATS_MODELS_PULLED) or 0),
        }

    def register_worker(self, worker_type: str, worker_id: str) -> None:
        """Register a worker with heartbeat.

        Args:
            worker_type: 'selfplay' or 'training'
            worker_id: Unique worker identifier
        """
        key = f"workers:{worker_type}:{worker_id}"
        self.redis.setex(key, 60, time.time())  # Expire after 60 seconds

    def get_active_workers(self) -> dict:
        """Get count of active workers by type.

        Returns:
            Dictionary with worker counts: {'selfplay': N, 'training': M}
        """
        selfplay_keys = self.redis.keys("workers:selfplay:*")
        training_keys = self.redis.keys("workers:training:*")

        return {
            'selfplay': len(selfplay_keys),
            'training': len(training_keys),
        }

    def health_check(self) -> bool:
        """Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self.redis.ping()
            return True
        except Exception:
            return False

    def clear_queue(self, queue_name: Optional[str] = None) -> None:
        """Clear a queue (USE WITH CAUTION!).

        Args:
            queue_name: Name of queue to clear (None = clear all)
        """
        if queue_name is None:
            self.redis.delete(self.GAMES_QUEUE)
            self.redis.delete(self.MODELS_QUEUE)
        else:
            self.redis.delete(queue_name)

    def reset_stats(self) -> None:
        """Reset all statistics (USE WITH CAUTION!)."""
        self.redis.delete(self.STATS_GAMES_PUSHED)
        self.redis.delete(self.STATS_GAMES_PULLED)
        self.redis.delete(self.STATS_MODELS_PUSHED)
        self.redis.delete(self.STATS_MODELS_PULLED)
