"""
Policy head processor for converting fused state embeddings into action probabilities.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_utils import decode_action
from .data_types import ActionInfo, PolicyConfig, PolicyOutput


class PolicyHead(nn.Module):
    """
    Neural network for converting fused state embeddings into action probabilities.

    Takes 128-dimensional fused embedding and outputs 700-dimensional action distribution
    using a 2-layer MLP: 128 → 256 → 700.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__()
        self.config = config or PolicyConfig()

        # Build MLP layers
        self.layer1 = nn.Linear(
            self.config.input_dim, self.config.hidden_dim, bias=self.config.use_bias
        )

        self.layer2 = nn.Linear(
            self.config.hidden_dim, self.config.output_dim, bias=self.config.use_bias
        )

        # Dropout layer (only active during training)
        self.dropout = (
            nn.Dropout(self.config.dropout_rate)
            if self.config.dropout_rate > 0
            else None
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights according to config."""
        for module in [self.layer1, self.layer2]:
            if self.config.weight_init == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif self.config.weight_init == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif self.config.weight_init == "normal":
                nn.init.normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy head.

        Args:
            fused_embedding: Input tensor of shape (batch_size, 128) or (128,)

        Returns:
            Action logits of shape (batch_size, 700) or (700,)
        """
        x = fused_embedding

        # First layer with ReLU
        x = self.layer1(x)
        x = F.relu(x)

        # Optional dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # Second layer (output logits)
        x = self.layer2(x)

        return x

    def get_action_probabilities(
        self, fused_embedding: torch.Tensor, temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get action probabilities with optional temperature scaling.

        Args:
            fused_embedding: Input tensor of shape (batch_size, 128) or (128,)
            temperature: Temperature for softmax (uses config default if None)

        Returns:
            Action probabilities of shape (batch_size, 700) or (700,)
        """
        logits = self.forward(fused_embedding)

        # Apply temperature scaling
        temp = temperature if temperature is not None else self.config.temperature
        if temp != 1.0:
            logits = logits / temp

        return F.softmax(logits, dim=-1)


class PolicyProcessor:
    """
    High-level processor for policy decisions with sampling strategies and action masking.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """
        Initialize the policy processor.

        Args:
            config: Configuration for policy architecture
        """
        self.config = config or PolicyConfig()
        self.policy_head = PolicyHead(self.config)
        self._profiling_enabled = False

        # Performance tracking
        self._inference_times = []
        self._error_counts = {"shape": 0, "general": 0}

    def forward(
        self,
        fused_embedding: torch.Tensor,
        temperature: Optional[float] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Process fused embedding into policy output with full action information.

        Args:
            fused_embedding: 128-dimensional fused state embedding
            temperature: Temperature for softmax (uses config default if None)
            action_mask: Optional mask to zero out invalid actions (shape: [700])

        Returns:
            PolicyOutput with logits, probabilities, and derived information

        Raises:
            ValueError: If input shapes are incompatible
        """
        start_time = time.time() if self._profiling_enabled else None

        try:
            # Validate input
            self._validate_input(fused_embedding)

            # Get logits and probabilities
            logits = self.policy_head(fused_embedding)

            # Apply temperature scaling
            temp = temperature if temperature is not None else self.config.temperature
            scaled_logits = logits / temp if temp != 1.0 else logits

            # Apply action mask if provided
            if action_mask is not None:
                self._validate_action_mask(action_mask)
                masked_logits = scaled_logits + torch.log(action_mask + 1e-8)
            else:
                masked_logits = scaled_logits

            # Get probabilities
            probs = F.softmax(masked_logits, dim=-1)

            # Calculate inference time
            inference_time_ms = None
            if self._profiling_enabled and start_time is not None:
                inference_time_ms = (time.time() - start_time) * 1000
                self._inference_times.append(inference_time_ms)

            # Create output
            return PolicyOutput(
                action_logits=logits,
                action_probs=probs,
                temperature=temp,
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            self._error_counts["general"] += 1
            if isinstance(e, ValueError):
                self._error_counts["shape"] += 1
            raise

    def select_action(
        self,
        fused_embedding: torch.Tensor,
        strategy: str = "deterministic",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Select an action using the specified strategy.

        Args:
            fused_embedding: 128-dimensional fused state embedding
            strategy: Selection strategy ("deterministic", "categorical", "top_k")
            temperature: Temperature for softmax
            top_k: Number of top actions for top-k sampling
            action_mask: Optional mask to zero out invalid actions

        Returns:
            PolicyOutput with selected action and action info
        """
        # Get base policy output
        policy_output = self.forward(fused_embedding, temperature, action_mask)

        # Select action based on strategy
        if strategy == "deterministic":
            selected_action = torch.argmax(policy_output.action_probs).item()
        elif strategy == "categorical":
            selected_action = torch.multinomial(policy_output.action_probs, 1).item()
        elif strategy == "top_k":
            if top_k is None:
                top_k = 10  # Default
            top_k_probs, top_k_indices = torch.topk(policy_output.action_probs, top_k)
            # Renormalize top-k probabilities
            top_k_probs = top_k_probs / top_k_probs.sum()
            # Sample from top-k
            selected_idx = torch.multinomial(top_k_probs, 1).item()
            selected_action = top_k_indices[selected_idx].item()
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of: deterministic, categorical, top_k"
            )

        # Decode action information
        action_info = decode_action(selected_action)

        # Update policy output with selection
        policy_output.selected_action = selected_action
        policy_output.action_info = action_info

        return policy_output

    def select_batch_actions(
        self,
        fused_embeddings: torch.Tensor,
        strategy: str = "deterministic",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> List[PolicyOutput]:
        """
        Select actions for a batch of fused embeddings.

        Args:
            fused_embeddings: Batch of embeddings [batch_size, 128]
            strategy: Selection strategy
            temperature: Temperature for softmax
            top_k: Number of top actions for top-k sampling
            action_masks: Optional masks [batch_size, 700]

        Returns:
            List of PolicyOutput objects, one per batch item
        """
        batch_size = fused_embeddings.size(0)
        results = []

        for i in range(batch_size):
            embedding = fused_embeddings[i]
            mask = action_masks[i] if action_masks is not None else None

            result = self.select_action(embedding, strategy, temperature, top_k, mask)
            results.append(result)

        return results

    def _validate_input(self, fused_embedding: torch.Tensor) -> None:
        """Validate fused embedding input."""
        if fused_embedding.dim() not in [1, 2]:
            raise ValueError(
                f"fused_embedding must be 1D or 2D, got {fused_embedding.dim()}D"
            )

        expected_dim = self.config.input_dim
        if fused_embedding.dim() == 1:
            if fused_embedding.size(0) != expected_dim:
                raise ValueError(
                    f"Expected embedding size {expected_dim}, got {fused_embedding.size(0)}"
                )
        else:  # 2D
            if fused_embedding.size(1) != expected_dim:
                raise ValueError(
                    f"Expected embedding size {expected_dim}, got {fused_embedding.size(1)}"
                )

    def _validate_action_mask(self, action_mask: torch.Tensor) -> None:
        """Validate action mask."""
        if action_mask.dim() != 1 or action_mask.size(0) != self.config.output_dim:
            raise ValueError(
                f"action_mask must be 1D with {self.config.output_dim} elements"
            )

        if not torch.all((action_mask == 0) | (action_mask == 1)):
            raise ValueError("action_mask must contain only 0s and 1s")

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

        if self._inference_times:
            metrics["inference_times"] = {
                "count": len(self._inference_times),
                "mean_ms": sum(self._inference_times) / len(self._inference_times),
                "min_ms": min(self._inference_times),
                "max_ms": max(self._inference_times),
            }

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._inference_times.clear()
        self._error_counts = {"shape": 0, "general": 0}

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.policy_head.parameters() if p.requires_grad)

    def to(self, device: torch.device) -> "PolicyProcessor":
        """Move processor to device."""
        self.policy_head.to(device)
        return self

    def train(self) -> "PolicyProcessor":
        """Set processor to training mode."""
        self.policy_head.train()
        return self

    def eval(self) -> "PolicyProcessor":
        """Set processor to evaluation mode."""
        self.policy_head.eval()
        return self


# Convenience functions
def create_action_mask(valid_actions: List[int]) -> torch.Tensor:
    """
    Create action mask from list of valid action indices.

    Args:
        valid_actions: List of valid action indices [0-699]

    Returns:
        Binary mask tensor with 1s for valid actions, 0s for invalid
    """
    mask = torch.zeros(700)
    mask[valid_actions] = 1.0
    return mask


def select_action_simple(
    fused_embedding: torch.Tensor, config: Optional[PolicyConfig] = None
) -> Tuple[int, ActionInfo]:
    """
    Simple convenience function for action selection.

    Args:
        fused_embedding: 128-dimensional fused state embedding
        config: Optional policy configuration

    Returns:
        Tuple of (action_index, action_info)
    """
    processor = PolicyProcessor(config)
    policy_output = processor.select_action(fused_embedding)
    return policy_output.selected_action, policy_output.action_info
