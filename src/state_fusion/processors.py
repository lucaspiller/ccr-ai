"""
State-fusion processor for combining perception outputs into decision-ready embeddings.
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..perception.data_types import PerceptionOutput
from .data_types import (BOARD_HEIGHT, BOARD_WIDTH, CAT_EMBEDDING_DIM,
                         FUSION_INPUT_DIM, FUSION_OUTPUT_DIM,
                         GLOBAL_FEATURES_DIM, GRID_TENSOR_CHANNELS,
                         FusedStateOutput, FusionConfig)


class FusionMLP(nn.Module):
    """
    Multi-layer perceptron for fusing perception outputs.

    Takes concatenated perception features and compresses them to 128-d embedding.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        # Build MLP layers
        layers = []
        input_dim = FUSION_INPUT_DIM

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Add activation
            if config.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())

            # Add dropout if enabled
            if config.dropout_rate > 0.0:
                layers.append(nn.Dropout(config.dropout_rate))

            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, FUSION_OUTPUT_DIM))

        # Optional layer normalization
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(FUSION_OUTPUT_DIM))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights according to config."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(
                        module.weight, nonlinearity=self.config.activation
                    )
                elif self.config.weight_init == "normal":
                    nn.init.normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion MLP.

        Args:
            x: Input tensor of shape (batch_size, FUSION_INPUT_DIM) or (FUSION_INPUT_DIM,)

        Returns:
            Fused embedding of shape (batch_size, 128) or (128,)
        """
        return self.mlp(x)


class StateFusionProcessor:
    """
    Main processor for fusing perception outputs into decision-ready embeddings.

    Combines grid tensor, global features, and cat embedding into a single
    128-dimensional representation for the policy and value heads.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize the state fusion processor.

        Args:
            config: Configuration for fusion architecture
        """
        self.config = config or FusionConfig()
        self.fusion_mlp = FusionMLP(self.config)
        self._profiling_enabled = False

        # Performance tracking
        self._fusion_times = []
        self._error_counts = {"shape": 0, "general": 0}

    def fuse(self, perception_output: PerceptionOutput) -> FusedStateOutput:
        """
        Fuse perception outputs into a single embedding.

        Args:
            perception_output: Output from perception layer

        Returns:
            FusedStateOutput containing 128-d embedding

        Raises:
            ValueError: If input shapes are incompatible
        """
        start_time = time.time() if self._profiling_enabled else None

        try:
            # Validate input shapes
            self._validate_perception_output(perception_output)

            # Flatten grid tensor
            grid_flat = perception_output.grid_tensor.flatten()

            # Concatenate all features
            fused_input = torch.cat(
                [
                    grid_flat,
                    perception_output.global_features,
                    perception_output.cat_embedding,
                ]
            )

            # Pass through fusion MLP
            fused_embedding = self.fusion_mlp(fused_input)

            # Calculate fusion time
            fusion_time_ms = None
            if self._profiling_enabled and start_time is not None:
                fusion_time_ms = (time.time() - start_time) * 1000
                self._fusion_times.append(fusion_time_ms)

            # Create output
            return FusedStateOutput(
                fused_embedding=fused_embedding,
                source_step=perception_output.source_step,
                source_tick=perception_output.source_tick,
                fusion_time_ms=fusion_time_ms,
            )

        except Exception as e:
            self._error_counts["general"] += 1
            if isinstance(e, ValueError):
                self._error_counts["shape"] += 1
            raise

    def fuse_batch(
        self, perception_outputs: list[PerceptionOutput]
    ) -> list[FusedStateOutput]:
        """
        Fuse a batch of perception outputs.

        Args:
            perception_outputs: List of perception outputs

        Returns:
            List of fused state outputs
        """
        return [self.fuse(output) for output in perception_outputs]

    def fuse_batch_tensor(
        self, perception_outputs: list[PerceptionOutput]
    ) -> torch.Tensor:
        """
        Fuse batch of perception outputs into a single stacked tensor.

        Args:
            perception_outputs: List of perception outputs

        Returns:
            Stacked tensor of shape (batch_size, 128)
        """
        embeddings = [
            self.fuse(output).fused_embedding for output in perception_outputs
        ]
        return torch.stack(embeddings, dim=0)

    def _validate_perception_output(self, perception_output: PerceptionOutput) -> None:
        """
        Validate perception output has correct shapes.

        Args:
            perception_output: Output to validate

        Raises:
            ValueError: If shapes are incorrect
        """
        # Check grid tensor shape
        expected_grid_shape = (GRID_TENSOR_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
        if perception_output.grid_tensor.shape != expected_grid_shape:
            raise ValueError(
                f"Grid tensor shape mismatch: expected {expected_grid_shape}, "
                f"got {perception_output.grid_tensor.shape}"
            )

        # Check global features shape
        if perception_output.global_features.shape != (GLOBAL_FEATURES_DIM,):
            raise ValueError(
                f"Global features shape mismatch: expected ({GLOBAL_FEATURES_DIM},), "
                f"got {perception_output.global_features.shape}"
            )

        # Check cat embedding shape
        if perception_output.cat_embedding.shape != (CAT_EMBEDDING_DIM,):
            raise ValueError(
                f"Cat embedding shape mismatch: expected ({CAT_EMBEDDING_DIM},), "
                f"got {perception_output.cat_embedding.shape}"
            )

    def enable_profiling(self) -> None:
        """Enable performance profiling."""
        self._profiling_enabled = True

    def disable_profiling(self) -> None:
        """Disable performance profiling."""
        self._profiling_enabled = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and error metrics."""
        metrics = {
            "error_counts": self._error_counts.copy(),
            "profiling_enabled": self._profiling_enabled,
        }

        if self._fusion_times:
            metrics["fusion_times"] = {
                "count": len(self._fusion_times),
                "mean_ms": sum(self._fusion_times) / len(self._fusion_times),
                "min_ms": min(self._fusion_times),
                "max_ms": max(self._fusion_times),
            }

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._fusion_times.clear()
        self._error_counts = {"shape": 0, "general": 0}

    def get_output_shape(self) -> tuple:
        """Get expected output tensor shape."""
        return (FUSION_OUTPUT_DIM,)

    def get_input_dim(self) -> int:
        """Get expected input dimension."""
        return FUSION_INPUT_DIM

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.fusion_mlp.parameters() if p.requires_grad)

    def to(self, device: torch.device) -> "StateFusionProcessor":
        """Move processor to device."""
        self.fusion_mlp.to(device)
        return self

    def train(self) -> "StateFusionProcessor":
        """Set processor to training mode."""
        self.fusion_mlp.train()
        return self

    def eval(self) -> "StateFusionProcessor":
        """Set processor to evaluation mode."""
        self.fusion_mlp.eval()
        return self


# Convenience function for quick fusion
def fuse_perception_output(
    perception_output: PerceptionOutput, config: Optional[FusionConfig] = None
) -> FusedStateOutput:
    """
    Convenience function to fuse a single perception output.

    Args:
        perception_output: Perception output to fuse
        config: Optional fusion configuration

    Returns:
        Fused state output
    """
    processor = StateFusionProcessor(config)
    return processor.fuse(perception_output)
