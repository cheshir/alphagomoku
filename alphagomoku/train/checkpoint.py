"""Checkpoint management for training.

Provides unified checkpoint save/load interface for both single-node
and distributed training modes.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Checkpoint:
    """Training checkpoint with all state needed to resume training.

    A checkpoint represents a snapshot of training state that can be saved
    and restored. It contains the model weights, optimizer state, scheduler
    state, and training progress metadata.

    Attributes:
        model_state: Model state_dict
        iteration: Training iteration/epoch number
        total_positions: Cumulative positions trained on
        metrics: Training metrics (loss, accuracy, etc.)
        optimizer_state: Optional optimizer state_dict
        scheduler_state: Optional scheduler state_dict
        step: Optional optimizer step count
        extra: Additional arbitrary fields
    """

    model_state: Dict[str, Any]
    iteration: int
    total_positions: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    step: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for saving.

        Returns:
            Dictionary with all checkpoint fields in standard format.
            Includes both 'iteration' and 'epoch' keys for backward compatibility.
        """
        checkpoint_dict = {
            "model_state_dict": self.model_state,
            "iteration": self.iteration,
            "epoch": self.iteration,  # Legacy compatibility
            "total_positions": self.total_positions,
            "metrics": self.metrics,
        }

        if self.optimizer_state is not None:
            checkpoint_dict["optimizer_state_dict"] = self.optimizer_state
        if self.scheduler_state is not None:
            checkpoint_dict["scheduler_state_dict"] = self.scheduler_state
        if self.step is not None:
            checkpoint_dict["step"] = self.step

        # Add any extra fields
        checkpoint_dict.update(self.extra)

        return checkpoint_dict

    @classmethod
    def from_dict(cls, checkpoint_dict: Dict[str, Any]) -> 'Checkpoint':
        """Create checkpoint from dictionary loaded from file.

        Args:
            checkpoint_dict: Dictionary loaded from checkpoint file

        Returns:
            Checkpoint object with all fields populated

        Note:
            Handles both old format (epoch) and new format (iteration)
            for backward compatibility.
        """
        # Get iteration (handles both 'iteration' and legacy 'epoch' keys)
        iteration = checkpoint_dict.get('iteration', checkpoint_dict.get('epoch', 0))

        # Extract standard fields
        model_state = checkpoint_dict['model_state_dict']
        total_positions = checkpoint_dict.get('total_positions', 0)
        metrics = checkpoint_dict.get('metrics', {})
        optimizer_state = checkpoint_dict.get('optimizer_state_dict')
        scheduler_state = checkpoint_dict.get('scheduler_state_dict')
        step = checkpoint_dict.get('step')

        # Collect extra fields (not part of standard schema)
        standard_keys = {
            'model_state_dict', 'iteration', 'epoch', 'total_positions',
            'metrics', 'optimizer_state_dict', 'scheduler_state_dict', 'step'
        }
        extra = {k: v for k, v in checkpoint_dict.items() if k not in standard_keys}

        return cls(
            model_state=model_state,
            iteration=iteration,
            total_positions=total_positions,
            metrics=metrics,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            step=step,
            extra=extra
        )

    def save(self, path: str) -> None:
        """Save checkpoint to file.

        Args:
            path: Path to save checkpoint file
        """
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'Checkpoint':
        """Load checkpoint from file.

        Args:
            path: Path to checkpoint file
            device: Device to map tensors to (e.g., 'cpu', 'cuda', 'mps')

        Returns:
            Checkpoint object loaded from file
        """
        checkpoint_dict = torch.load(
            path,
            map_location=device,
            weights_only=False
        )
        return cls.from_dict(checkpoint_dict)

    def restore_model(self, model: nn.Module, strict: bool = True) -> None:
        """Restore model state from checkpoint.

        Args:
            model: Model to restore state into
            strict: Whether to strictly enforce state_dict keys match
        """
        model.load_state_dict(self.model_state, strict=strict)

    def restore_optimizer(self, optimizer: optim.Optimizer) -> None:
        """Restore optimizer state from checkpoint.

        Args:
            optimizer: Optimizer to restore state into

        Raises:
            ValueError: If checkpoint has no optimizer state
        """
        if self.optimizer_state is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(self.optimizer_state)

    def restore_scheduler(self, scheduler: Any) -> None:
        """Restore scheduler state from checkpoint.

        Args:
            scheduler: Scheduler to restore state into

        Raises:
            ValueError: If checkpoint has no scheduler state
        """
        if self.scheduler_state is None:
            raise ValueError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(self.scheduler_state)

    def restore_all(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> None:
        """Restore all available states from checkpoint.

        Args:
            model: Model to restore
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            strict: Whether to strictly enforce model state_dict keys match
        """
        self.restore_model(model, strict=strict)

        if optimizer is not None and self.optimizer_state is not None:
            self.restore_optimizer(optimizer)

        if scheduler is not None and self.scheduler_state is not None:
            self.restore_scheduler(scheduler)

    @classmethod
    def from_training_state(
        cls,
        model: nn.Module,
        iteration: int,
        total_positions: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: Optional[int] = None,
        **extra_fields
    ) -> 'Checkpoint':
        """Create checkpoint from current training state.

        Args:
            model: Model to save
            iteration: Current training iteration/epoch
            total_positions: Cumulative positions trained
            metrics: Training metrics
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            step: Optional optimizer step count
            **extra_fields: Additional fields to include

        Returns:
            Checkpoint object ready to save
        """
        return cls(
            model_state=model.state_dict(),
            iteration=iteration,
            total_positions=total_positions,
            metrics=metrics if metrics is not None else {},
            optimizer_state=optimizer.state_dict() if optimizer is not None else None,
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            step=step,
            extra=extra_fields
        )

    def __repr__(self) -> str:
        """String representation of checkpoint."""
        parts = [
            f"Checkpoint(iteration={self.iteration}",
            f"total_positions={self.total_positions:,}",
        ]
        if self.metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in list(self.metrics.items())[:3])
            parts.append(f"metrics=[{metric_str}]")
        if self.optimizer_state is not None:
            parts.append("has_optimizer=True")
        if self.scheduler_state is not None:
            parts.append("has_scheduler=True")
        if self.step is not None:
            parts.append(f"step={self.step}")

        return ", ".join(parts) + ")"
