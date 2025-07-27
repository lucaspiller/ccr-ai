"""
Data types and configuration for the state-fusion layer.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class FusionConfig:
    """
    Configuration for the state-fusion layer.

    Attributes:
        hidden_dims: List of hidden layer sizes for the fusion MLP
        dropout_rate: Dropout rate during training (0.0 disables dropout)
        use_layer_norm: Whether to apply LayerNorm to output
        activation: Activation function name ('relu', 'gelu', 'tanh')
        gradient_clipping: Maximum gradient norm (None disables clipping)
        weight_init: Weight initialization strategy ('xavier', 'kaiming', 'normal')
    """

    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    activation: str = "relu"
    gradient_clipping: Optional[float] = 1.0
    weight_init: str = "xavier"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]

        # Validation
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if self.activation not in ["relu", "gelu", "tanh"]:
            raise ValueError("activation must be one of: relu, gelu, tanh")

        if self.weight_init not in ["xavier", "kaiming", "normal"]:
            raise ValueError("weight_init must be one of: xavier, kaiming, normal")

        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")


@dataclass
class FusedStateOutput:
    """
    Output from the state-fusion layer.

    Attributes:
        fused_embedding: 128-dimensional tensor representing the fused state
        source_step: Step number from original game state
        source_tick: Tick number from original game state
        fusion_time_ms: Time taken for fusion in milliseconds (if profiling enabled)
    """

    fused_embedding: torch.Tensor
    source_step: Optional[int] = None
    source_tick: Optional[int] = None
    fusion_time_ms: Optional[float] = None

    def __post_init__(self):
        """Validate the fused embedding shape."""
        if self.fused_embedding.dim() != 1:
            raise ValueError(
                f"fused_embedding must be 1D, got {self.fused_embedding.dim()}D"
            )

        if self.fused_embedding.size(0) != 128:
            raise ValueError(
                f"fused_embedding must have 128 dimensions, got {self.fused_embedding.size(0)}"
            )

    def to_device(self, device: torch.device) -> "FusedStateOutput":
        """Move the output to a different device."""
        return FusedStateOutput(
            fused_embedding=self.fused_embedding.to(device),
            source_step=self.source_step,
            source_tick=self.source_tick,
            fusion_time_ms=self.fusion_time_ms,
        )

    def detach(self) -> "FusedStateOutput":
        """Detach tensors from computation graph."""
        return FusedStateOutput(
            fused_embedding=self.fused_embedding.detach(),
            source_step=self.source_step,
            source_tick=self.source_tick,
            fusion_time_ms=self.fusion_time_ms,
        )


# Constants for input dimensions (matching perception layer output)
GRID_TENSOR_CHANNELS = 28
BOARD_HEIGHT = 10
BOARD_WIDTH = 14
GLOBAL_FEATURES_DIM = 16
CAT_EMBEDDING_DIM = 32

# Calculated input dimension for fusion MLP
FUSION_INPUT_DIM = (
    (GRID_TENSOR_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH)
    + GLOBAL_FEATURES_DIM
    + CAT_EMBEDDING_DIM
)
FUSION_OUTPUT_DIM = 128

# Expected dimension: 28 * 10 * 14 + 16 + 32 = 3920 + 48 = 3968
assert FUSION_INPUT_DIM == 3968, f"Expected 3968 input dims, got {FUSION_INPUT_DIM}"
