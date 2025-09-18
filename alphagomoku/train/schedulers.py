"""Custom learning rate schedulers."""

import math
from typing import Dict, Any


class WarmupCosineScheduler:
    """Cosine decay with linear warmup over epochs.

    - Warms up linearly from 0 to base_lr over `warmup_epochs`.
    - Then decays with cosine from base_lr to min_lr over the remaining epochs
      up to `max_epochs`.
    - Call `step()` once per epoch.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float = 0.0,
        start_epoch: int = 0,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.max_epochs = max(int(max_epochs), 1)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.epoch = int(start_epoch)

        # Initialize LR for start_epoch
        self._update_lr()

    def _lr_at_epoch(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # Linear warmup from 0 to base_lr
            return self.base_lr * float(epoch + 1) / float(self.warmup_epochs)

        # Cosine phase
        cosine_span = max(self.max_epochs - self.warmup_epochs, 1)
        # Progress in [0, 1]
        progress = min(
            max(epoch - self.warmup_epochs, 0) / float(cosine_span), 1.0
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def _update_lr(self):
        lr = self._lr_at_epoch(self.epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self.epoch += 1
        self._update_lr()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.warmup_epochs = int(state.get("warmup_epochs", self.warmup_epochs))
        self.max_epochs = int(state.get("max_epochs", self.max_epochs))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.epoch = int(state.get("epoch", self.epoch))
        self._update_lr()

