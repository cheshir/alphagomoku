"""Configuration validation utilities for distributed training."""

from typing import List, Optional


def validate_redis_url(url: Optional[str]) -> List[str]:
    """Validate Redis URL format.

    Args:
        url: Redis URL to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not url:
        errors.append(
            "❌ ERROR: REDIS_URL is not set\n"
            "\n"
            "Please set REDIS_URL in one of these ways:\n"
            "  1. Create .env file with: REDIS_URL=redis://:password@host:port/db\n"
            "  2. Export in shell: export REDIS_URL='redis://:password@host:port/db'\n"
            "\n"
            "Example .env file:\n"
            "  REDIS_URL=redis://default:password@queue.cheshir.me:6379/0"
        )
    elif not url.startswith('redis://'):
        errors.append(
            f"❌ Invalid Redis URL format: {url}\n"
            f"   Expected format: redis://:password@host:port/db\n"
            f"   Example: redis://default:password@queue.cheshir.me:6379/0"
        )

    return errors


def validate_selfplay_config(
    mcts_simulations: int,
    batch_size_mcts: int,
    games_per_batch: int,
    model_update_frequency: int
) -> List[str]:
    """Validate self-play worker configuration.

    Args:
        mcts_simulations: Number of MCTS simulations per move
        batch_size_mcts: Batch size for neural network evaluation
        games_per_batch: Games to generate per batch
        model_update_frequency: How often to fetch new model

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate MCTS simulations
    if mcts_simulations < 10 or mcts_simulations > 2000:
        errors.append(
            f"❌ Invalid MCTS simulations: {mcts_simulations}\n"
            f"   Must be between 10 and 2000\n"
            f"   Recommended: 50-100 for CPU workers, 100-200 for MPS"
        )

    # Validate batch size
    if batch_size_mcts < 1 or batch_size_mcts > 512:
        errors.append(
            f"❌ Invalid MCTS batch size: {batch_size_mcts}\n"
            f"   Must be between 1 and 512\n"
            f"   Recommended: 32 for CPU, 64-96 for MPS"
        )

    # Validate games per batch
    if games_per_batch < 1 or games_per_batch > 100:
        errors.append(
            f"❌ Invalid games per batch: {games_per_batch}\n"
            f"   Must be between 1 and 100\n"
            f"   Recommended: 10"
        )

    # Validate model update frequency
    if model_update_frequency < 1 or model_update_frequency > 100:
        errors.append(
            f"❌ Invalid model update frequency: {model_update_frequency}\n"
            f"   Must be between 1 and 100\n"
            f"   Recommended: 10 (pull model every 10 game batches)"
        )

    return errors


def validate_training_config(
    batch_size: int,
    lr: float,
    min_lr: float,
    publish_frequency: int,
    min_position_batches_for_training: int,
    position_batches_per_training_pull: int
) -> List[str]:
    """Validate training worker configuration.

    Args:
        batch_size: Training batch size
        lr: Learning rate
        min_lr: Minimum learning rate
        publish_frequency: How often to publish model
        min_position_batches_for_training: Minimum position batches before training
        position_batches_per_training_pull: Position batches to pull per training iteration

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate batch size
    if batch_size < 32 or batch_size > 4096:
        errors.append(
            f"❌ Invalid batch size: {batch_size}\n"
            f"   Must be between 32 and 4096\n"
            f"   Recommended: 512-1024 for T4, 1024-2048 for A100"
        )

    # Validate learning rates
    if lr <= 0 or lr > 1:
        errors.append(
            f"❌ Invalid learning rate: {lr}\n"
            f"   Must be between 0 and 1\n"
            f"   Recommended: 1e-3"
        )

    if min_lr <= 0 or min_lr >= lr:
        errors.append(
            f"❌ Invalid min learning rate: {min_lr}\n"
            f"   Must be between 0 and {lr}\n"
            f"   Recommended: 1e-6"
        )

    # Validate publish frequency
    if publish_frequency < 1 or publish_frequency > 100:
        errors.append(
            f"❌ Invalid publish frequency: {publish_frequency}\n"
            f"   Must be between 1 and 100\n"
            f"   Recommended: 5 (publish model every 5 training iterations)"
        )

    # Validate min position batches for training
    if min_position_batches_for_training < 1 or min_position_batches_for_training > 1000:
        errors.append(
            f"❌ Invalid min position batches for training: {min_position_batches_for_training}\n"
            f"   Must be between 1 and 1000\n"
            f"   Recommended: 50 (wait for 50+ position batches, ~50,000 positions total)"
        )

    # Validate position batches per training pull
    if position_batches_per_training_pull < 1 or position_batches_per_training_pull > 500:
        errors.append(
            f"❌ Invalid position batches per training pull: {position_batches_per_training_pull}\n"
            f"   Must be between 1 and 500\n"
            f"   Recommended: 50 (pull 50 position batches, ~50,000 positions per training iteration)"
        )

    return errors


def print_validation_errors(errors: List[str], logger) -> None:
    """Print validation errors and exit.

    Args:
        errors: List of error messages
        logger: Logger instance
    """
    if errors:
        logger.error("Configuration validation failed:")
        logger.error("")
        for error in errors:
            logger.error(error)
        logger.error("")
        logger.error("Please fix the configuration and try again.")
