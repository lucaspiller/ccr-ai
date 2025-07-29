"""Value head for state value estimation in PPO training."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ValueConfig:
    """Configuration for value head architecture."""

    input_dim: int = 128
    hidden_dim: int = 64
    output_dim: int = 1
    use_bias: bool = True
    weight_init: str = "orthogonal"
    weight_gain: float = 0.01


class ValueHead(nn.Module):
    """Value head for PPO critic.

    Estimates state values for advantage calculation in PPO training.
    Uses the same 128-d fused embedding as the policy head.
    """

    def __init__(self, config: Optional[ValueConfig] = None):
        """Initialize value head.

        Args:
            config: Value head configuration (uses defaults if None)
        """
        super().__init__()

        if config is None:
            config = ValueConfig()

        self.config = config

        # 128 → ReLU → 64 → scalar architecture (matches existing PPO implementation)
        self.value_head = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim, bias=config.use_bias),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim, bias=config.use_bias),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for stable training."""
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                if self.config.weight_init == "orthogonal":
                    nn.init.orthogonal_(layer.weight, gain=self.config.weight_gain)
                elif self.config.weight_init == "xavier":
                    nn.init.xavier_uniform_(layer.weight, gain=self.config.weight_gain)
                elif self.config.weight_init == "normal":
                    nn.init.normal_(layer.weight, std=self.config.weight_gain)
                else:
                    raise ValueError(f"Unknown weight_init: {self.config.weight_init}")

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value head.

        Args:
            x: State embeddings [batch_size, input_dim]

        Returns:
            Value predictions [batch_size, 1]
        """
        return self.value_head(x)
