from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    """DW-ResNet block with SE attention"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        return self.relu(out)


class GomokuNet(nn.Module):
    """DW-ResNet-SE network for Gomoku with policy and value heads"""

    def __init__(self, board_size: int = 15, num_blocks: int = 12, channels: int = 64):
        super().__init__()
        self.board_size = board_size
        self.channels = channels

        # Input processing - 5 channels: own stones, opponent stones, last move, side-to-move, pattern maps
        self.input_conv = nn.Sequential(
            nn.Conv2d(5, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(
            32 * board_size * board_size, board_size * board_size
        )

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(16 * board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = self.value_fc(value)

        return policy, value.squeeze(-1)

    def predict(self, board_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Single prediction for MCTS with optimized tensor handling"""
        self.eval()
        with torch.inference_mode():
            # Assume board_state is already on the correct device (handled by caller)
            # Add batch dimension only if needed
            if board_state.dim() == 3:
                board_state = board_state.unsqueeze(0)

            policy_logits, value = self.forward(board_state)
            policy = F.softmax(policy_logits, dim=1).squeeze(0)
            return policy, value.item()

    def predict_batch(
        self, board_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch prediction for MCTS with optimized tensor handling"""
        self.eval()
        with torch.inference_mode():
            # Assume board_states is already on the correct device (handled by caller)
            policy_logits, values = self.forward(board_states)
            policies = F.softmax(policy_logits, dim=1)
            return policies, values

    def get_model_size(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
