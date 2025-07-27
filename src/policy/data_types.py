"""
Data types and configuration for the policy head.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PolicyConfig:
    """
    Configuration for the policy head.

    Attributes:
        input_dim: Input embedding dimension (from state-fusion layer)
        hidden_dim: Hidden layer size for the MLP
        output_dim: Number of possible actions (700 for ChuChu Rocket)
        dropout_rate: Dropout rate during training (0.0 disables dropout)
        use_bias: Whether to use bias terms in linear layers
        weight_init: Weight initialization strategy
        temperature: Softmax temperature for exploration control
        gradient_clipping: Maximum gradient norm (None disables clipping)
    """

    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 700
    dropout_rate: float = 0.1
    use_bias: bool = True
    weight_init: str = "xavier"
    temperature: float = 1.0
    gradient_clipping: Optional[float] = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if self.weight_init not in ["xavier", "kaiming", "normal"]:
            raise ValueError("weight_init must be one of: xavier, kaiming, normal")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass
class ActionInfo:
    """
    Structured information about a game action.

    Attributes:
        action_type: Type of action ("place_up", "place_down", "place_left", "place_right", "erase")
        tile_row: Row coordinate of target tile (0-9)
        tile_col: Column coordinate of target tile (0-13)
        tile_idx: Linear tile index (0-139)
        action_idx: Action index in the 700-dimensional space
    """

    action_type: str
    tile_row: int
    tile_col: int
    tile_idx: int
    action_idx: int

    def __post_init__(self):
        """Validate action information."""
        valid_types = ["place_up", "place_down", "place_left", "place_right", "erase"]
        if self.action_type not in valid_types:
            raise ValueError(f"action_type must be one of: {valid_types}")

        if not (0 <= self.tile_row <= 9):
            raise ValueError("tile_row must be between 0 and 9")

        if not (0 <= self.tile_col <= 13):
            raise ValueError("tile_col must be between 0 and 13")

        if not (0 <= self.tile_idx <= 139):
            raise ValueError("tile_idx must be between 0 and 139")

        if not (0 <= self.action_idx <= 699):
            raise ValueError("action_idx must be between 0 and 699")

        # Validate consistency
        expected_tile_idx = self.tile_row * 14 + self.tile_col
        if self.tile_idx != expected_tile_idx:
            raise ValueError(
                f"tile_idx {self.tile_idx} doesn't match coordinates ({self.tile_row}, {self.tile_col})"
            )


@dataclass
class PolicyOutput:
    """
    Output from the policy head.

    Attributes:
        action_logits: Raw logits before softmax [700]
        action_probs: Softmax probabilities [700]
        selected_action: Sampled/selected action index
        action_info: Decoded action information
        confidence: Maximum probability in distribution
        entropy: Entropy of the action distribution (measure of uncertainty)
        temperature: Temperature used for softmax (if applicable)
        source_step: Step number from original game state
        source_tick: Tick number from original game state
        inference_time_ms: Time taken for inference in milliseconds
    """

    action_logits: torch.Tensor
    action_probs: torch.Tensor
    selected_action: Optional[int] = None
    action_info: Optional[ActionInfo] = None
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    temperature: Optional[float] = None
    source_step: Optional[int] = None
    source_tick: Optional[int] = None
    inference_time_ms: Optional[float] = None

    def __post_init__(self):
        """Validate and compute derived properties."""
        # Validate tensor shapes
        if self.action_logits.dim() != 1 or self.action_logits.size(0) != 700:
            raise ValueError(
                f"action_logits must be 1D with 700 elements, got {self.action_logits.shape}"
            )

        if self.action_probs.dim() != 1 or self.action_probs.size(0) != 700:
            raise ValueError(
                f"action_probs must be 1D with 700 elements, got {self.action_probs.shape}"
            )

        # Validate probability distribution
        prob_sum = self.action_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):  # Allow small numerical errors
            raise ValueError(f"action_probs must sum to 1.0, got {prob_sum}")

        # Compute derived properties if not provided
        if self.confidence is None:
            self.confidence = self.action_probs.max().item()

        if self.entropy is None:
            # Calculate entropy: -sum(p * log(p))
            log_probs = torch.log(
                self.action_probs + 1e-8
            )  # Add small epsilon for numerical stability
            self.entropy = -(self.action_probs * log_probs).sum().item()

    def to_device(self, device: torch.device) -> "PolicyOutput":
        """Move tensors to a different device."""
        return PolicyOutput(
            action_logits=self.action_logits.to(device),
            action_probs=self.action_probs.to(device),
            selected_action=self.selected_action,
            action_info=self.action_info,
            confidence=self.confidence,
            entropy=self.entropy,
            temperature=self.temperature,
            source_step=self.source_step,
            source_tick=self.source_tick,
            inference_time_ms=self.inference_time_ms,
        )

    def detach(self) -> "PolicyOutput":
        """Detach tensors from computation graph."""
        return PolicyOutput(
            action_logits=self.action_logits.detach(),
            action_probs=self.action_probs.detach(),
            selected_action=self.selected_action,
            action_info=self.action_info,
            confidence=self.confidence,
            entropy=self.entropy,
            temperature=self.temperature,
            source_step=self.source_step,
            source_tick=self.source_tick,
            inference_time_ms=self.inference_time_ms,
        )

    def get_top_k_actions(self, k: int = 5) -> list[Tuple[int, float]]:
        """
        Get top-k actions with their probabilities.

        Args:
            k: Number of top actions to return

        Returns:
            List of (action_idx, probability) tuples sorted by probability (descending)
        """
        top_k_values, top_k_indices = torch.topk(self.action_probs, k)
        return [
            (idx.item(), prob.item()) for idx, prob in zip(top_k_indices, top_k_values)
        ]


# Constants for action space
BOARD_HEIGHT = 10
BOARD_WIDTH = 14
TOTAL_TILES = BOARD_HEIGHT * BOARD_WIDTH  # 140
TOTAL_ACTIONS = 700

# Action type offsets in the 700-dimensional space
ACTION_TYPE_OFFSETS = {
    "place_up": 0,  # Actions 0-139
    "place_down": 140,  # Actions 140-279
    "place_left": 280,  # Actions 280-419
    "place_right": 420,  # Actions 420-559
    "erase": 560,  # Actions 560-699
}

# Reverse mapping for decoding
OFFSET_TO_ACTION_TYPE = {v: k for k, v in ACTION_TYPE_OFFSETS.items()}

# Validate action space dimensions
assert TOTAL_TILES == 140, f"Expected 140 tiles, got {TOTAL_TILES}"
assert (
    TOTAL_ACTIONS == 560 + 140
), f"Expected 700 actions (560 + 140), got {TOTAL_ACTIONS}"
assert (
    max(ACTION_TYPE_OFFSETS.values()) + TOTAL_TILES <= TOTAL_ACTIONS
), "Action space overflow"
