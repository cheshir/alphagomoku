from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..model.network import GomokuNet
from ..selfplay.selfplay import SelfPlayData
from .data_buffer import DataBuffer


class Trainer:
    """Neural network trainer for AlphaGomoku"""

    def __init__(
        self,
        model: GomokuNet,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = None,
    ):
        self.model = model
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )

        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()

        self.writer = SummaryWriter()
        self.step = 0

    def train_step(self, batch: List[SelfPlayData]) -> Dict[str, float]:
        """Single training step with optimized tensor conversion"""
        if not batch:
            return {}

        # Optimize tensor creation by batching numpy arrays first
        states_np = np.stack([data.state for data in batch])
        policies_np = np.stack([data.policy for data in batch])
        values_np = np.array([data.value for data in batch])

        # Single tensor conversion and device transfer
        states = torch.from_numpy(states_np).float().to(self.device)
        policies = torch.from_numpy(policies_np).float().to(self.device)
        values = torch.from_numpy(values_np).float().to(self.device)
        # Forward pass
        self.optimizer.zero_grad()
        pred_policies, pred_values = self.model(states)

        # Compute losses
        log_probs = torch.log_softmax(pred_policies, dim=1)
        policy_loss = self.policy_loss_fn(log_probs, policies)
        value_loss = self.value_loss_fn(pred_values, values)
        total_loss = policy_loss + value_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            policy_acc = (
                (pred_policies.argmax(dim=1) == policies.argmax(dim=1)).float().mean()
            )
            value_mae = torch.abs(pred_values - values).mean()

        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "policy_accuracy": policy_acc.item(),
            "value_mae": value_mae.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.step)

        self.step += 1
        return metrics

    def train_epoch(
        self,
        data_buffer: DataBuffer,
        batch_size: int = 512,
        steps_per_epoch: int = 1000,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = []

        step_pbar = tqdm(range(steps_per_epoch), desc="Training", leave=False, unit="step")
        for i in step_pbar:
            batch = data_buffer.sample_batch(batch_size)
            if batch:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                if metrics:
                    step_pbar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'acc': f"{metrics['policy_accuracy']:.3f}"
                    })

        step_pbar.close()
        self.scheduler.step()

        # Average metrics
        if epoch_metrics:
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            return avg_metrics

        return {}

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "step": self.step,
            "metrics": metrics,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        return checkpoint
