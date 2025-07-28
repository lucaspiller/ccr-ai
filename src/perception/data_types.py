"""
Data types and constants for the perception layer.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

# Constants for tensor shapes and dimensions
GRID_CHANNELS = 28
GRID_HEIGHT = 10
GRID_WIDTH = 14
GLOBAL_FEATURES_DIM = 16
CAT_EMBEDDING_DIM = 32

# Channel assignments for grid tensor
WALL_CHANNELS = (0, 2)  # Channels 0-1: vertical, horizontal walls
ROCKET_CHANNELS = (2, 10)  # Channels 2-9: player rocket masks
SPAWNER_CHANNELS = (10, 11)  # Channel 10: spawner positions
ARROW_CHANNELS = (11, 17)  # Channels 11-16: arrow geometry, owner, health
MOUSE_FLOW_CHANNELS = (17, 22)  # Channels 17-21: mouse flow + confidence
CAT_CHANNELS = (22, 26)  # Channels 22-25: cat positions by direction
SPECIAL_MICE_CHANNELS = (26, 28)  # Channels 26-27: gold/bonus mice

# Global feature indices
REMAINING_TICKS_IDX = 0
ARROW_BUDGET_IDX = 1
CAT_COUNT_IDX = 2
BONUS_STATE_INDICES = (3, 8)  # Indices 3-7: bonus state one-hot
PLAYER_SCORES_INDICES = (8, 16)  # Indices 8-15: player scores

# Normalization constants
MAX_TICKS_NORM = 10000.0
MAX_CATS_NORM = 16.0
MAX_SCORE_NORM = 100.0
ROCKET_DISTANCE_NORM = 20.0

# Bonus modes for one-hot encoding
BONUS_MODES = ["none", "mouse_mania", "cat_mania", "speed_up", "slow_down"]

# Performance targets
TARGET_ENCODING_TIME_MS = 1.0  # Target encoding time in milliseconds


@dataclass
class PerceptionConfig:
    """Configuration for perception layer components."""

    board_width: int = GRID_WIDTH
    board_height: int = GRID_HEIGHT
    max_cats: int = 16
    max_players: int = 8

    # Validation settings
    validate_input: bool = True
    strict_bounds_checking: bool = True

    # Performance settings
    enable_profiling: bool = False
    cache_cat_features: bool = False


@dataclass
class PerceptionOutput:
    """
    Structured output from perception layer processing.

    Contains all tensor representations needed for the neural network:
    - Grid tensor: 28-channel spatial representation (raw)
    - Grid embedding: CNN-processed spatial features (flattened)
    - Global features: 16-dimensional game state vector
    - Cat embedding: 32-dimensional set encoding of cats
    """

    grid_tensor: torch.Tensor  # [28, height, width] - raw grid
    grid_embedding: torch.Tensor  # [grid_embed_size] - CNN-processed
    global_features: torch.Tensor  # [16]
    cat_embedding: torch.Tensor  # [32]

    # Metadata for debugging and validation
    source_step: Optional[int] = None
    source_tick: Optional[int] = None
    cat_count: Optional[int] = None
    encoding_time_ms: Optional[float] = None
    _validate_shapes: bool = True

    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        if self._validate_shapes:
            self._validate_tensor_shapes()

    def _validate_tensor_shapes(self):
        """Validate that all tensors have expected shapes."""
        # Validate channels and global features dimensions
        if self.grid_tensor.shape[0] != GRID_CHANNELS:
            raise ValueError(
                f"Invalid grid tensor channels: {self.grid_tensor.shape[0]}. "
                f"Expected: {GRID_CHANNELS}"
            )

        if self.global_features.shape != (GLOBAL_FEATURES_DIM,):
            raise ValueError(
                f"Invalid global features shape: {self.global_features.shape}. "
                f"Expected: ({GLOBAL_FEATURES_DIM},)"
            )

        if self.cat_embedding.shape != (CAT_EMBEDDING_DIM,):
            raise ValueError(
                f"Invalid cat embedding shape: {self.cat_embedding.shape}. "
                f"Expected: ({CAT_EMBEDDING_DIM},)"
            )

    @property
    def total_features(self) -> int:
        """Total number of features across all components."""
        return (
            self.grid_tensor.numel()
            + self.global_features.numel()
            + self.cat_embedding.numel()
        )

    def get_combined_embedding(self) -> torch.Tensor:
        """
        Concatenate CNN grid embedding with global and cat features for fusion MLP.

        Returns:
            torch.Tensor: [grid_embed_size + 16 + 32] feature vector
        """
        # Use CNN-processed grid embedding instead of raw flattened grid
        combined = torch.cat(
            [self.grid_embedding, self.global_features, self.cat_embedding], dim=0
        )

        return combined

    def get_grid_channels(self, channel_range: Tuple[int, int]) -> torch.Tensor:
        """
        Extract specific channel range from grid tensor.

        Args:
            channel_range: (start, end) indices for channels

        Returns:
            torch.Tensor: Selected channels [end-start, height, width]
        """
        start, end = channel_range
        if not (0 <= start < end <= GRID_CHANNELS):
            raise ValueError(f"Invalid channel range: {channel_range}")

        return self.grid_tensor[start:end]

    def get_wall_channels(self) -> torch.Tensor:
        """Get wall encoding channels (0-1)."""
        return self.get_grid_channels(WALL_CHANNELS)

    def get_rocket_channels(self) -> torch.Tensor:
        """Get rocket position channels (2-9)."""
        return self.get_grid_channels(ROCKET_CHANNELS)

    def get_arrow_channels(self) -> torch.Tensor:
        """Get arrow encoding channels (11-16)."""
        return self.get_grid_channels(ARROW_CHANNELS)

    def get_mouse_flow_channels(self) -> torch.Tensor:
        """Get mouse flow channels (17-21)."""
        return self.get_grid_channels(MOUSE_FLOW_CHANNELS)

    def get_cat_channels(self) -> torch.Tensor:
        """Get cat position channels (22-25)."""
        return self.get_grid_channels(CAT_CHANNELS)

    def get_special_mice_channels(self) -> torch.Tensor:
        """Get special mice channels (26-27)."""
        return self.get_grid_channels(SPECIAL_MICE_CHANNELS)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/debugging."""
        return {
            "grid_tensor": self.grid_tensor.tolist(),
            "grid_embedding": self.grid_embedding.tolist(),
            "global_features": self.global_features.tolist(),
            "cat_embedding": self.cat_embedding.tolist(),
            "source_step": self.source_step,
            "source_tick": self.source_tick,
            "cat_count": self.cat_count,
            "encoding_time_ms": self.encoding_time_ms,
            "total_features": self.total_features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionOutput":
        """Create from dictionary."""
        return cls(
            grid_tensor=torch.tensor(data["grid_tensor"], dtype=torch.float32),
            grid_embedding=torch.tensor(data["grid_embedding"], dtype=torch.float32),
            global_features=torch.tensor(data["global_features"], dtype=torch.float32),
            cat_embedding=torch.tensor(data["cat_embedding"], dtype=torch.float32),
            source_step=data.get("source_step"),
            source_tick=data.get("source_tick"),
            cat_count=data.get("cat_count"),
            encoding_time_ms=data.get("encoding_time_ms"),
        )

    def summary(self) -> str:
        """Generate human-readable summary of perception output."""
        non_zero_channels = (self.grid_tensor.sum(dim=(1, 2)) > 0).sum().item()
        non_zero_grid_embed = (self.grid_embedding != 0).sum().item()
        non_zero_global = (self.global_features > 0).sum().item()
        non_zero_cat = (self.cat_embedding != 0).sum().item()

        summary = f"""Perception Output Summary:
  Grid Tensor: {self.grid_tensor.shape} ({non_zero_channels}/{GRID_CHANNELS} channels active)
  Grid Embedding: {self.grid_embedding.shape} ({non_zero_grid_embed} non-zero CNN features)
  Global Features: {self.global_features.shape} ({non_zero_global}/{GLOBAL_FEATURES_DIM} non-zero)
  Cat Embedding: {self.cat_embedding.shape} ({non_zero_cat}/{CAT_EMBEDDING_DIM} non-zero)
  Total Features: {self.total_features:,}
  Combined Shape: {self.get_combined_embedding().shape}"""

        if self.cat_count is not None:
            summary += f"\n  Cat Count: {self.cat_count}"
        if self.encoding_time_ms is not None:
            summary += f"\n  Encoding Time: {self.encoding_time_ms:.2f}ms"
        if self.source_step is not None:
            summary += f"\n  Source Step: {self.source_step}"

        return summary

    def to(self, device: torch.device) -> "PerceptionOutput":
        """Move all tensors to specified device."""
        return PerceptionOutput(
            grid_tensor=self.grid_tensor.to(device),
            grid_embedding=self.grid_embedding.to(device),
            global_features=self.global_features.to(device),
            cat_embedding=self.cat_embedding.to(device),
            source_step=self.source_step,
            source_tick=self.source_tick,
            cat_count=self.cat_count,
            encoding_time_ms=self.encoding_time_ms,
            _validate_shapes=False,  # Skip validation after device move
        )


@dataclass
class PerceptionMetrics:
    """Metrics for monitoring perception layer performance."""

    total_encodings: int = 0
    avg_encoding_time_ms: float = 0.0
    max_encoding_time_ms: float = 0.0
    min_encoding_time_ms: float = float("inf")

    total_cats_processed: int = 0
    avg_cats_per_encoding: float = 0.0
    max_cats_per_encoding: int = 0

    validation_errors: int = 0
    shape_mismatches: int = 0

    def update_timing(self, encoding_time_ms: float):
        """Update timing metrics with new measurement."""
        self.total_encodings += 1

        # Update running average
        prev_total_time = self.avg_encoding_time_ms * (self.total_encodings - 1)
        self.avg_encoding_time_ms = (
            prev_total_time + encoding_time_ms
        ) / self.total_encodings

        # Update min/max
        self.max_encoding_time_ms = max(self.max_encoding_time_ms, encoding_time_ms)
        self.min_encoding_time_ms = min(self.min_encoding_time_ms, encoding_time_ms)

    def update_cats(self, cat_count: int):
        """Update cat processing metrics."""
        self.total_cats_processed += cat_count
        self.max_cats_per_encoding = max(self.max_cats_per_encoding, cat_count)

        # Update average - this should be called after update_timing
        if self.total_encodings > 0:
            self.avg_cats_per_encoding = (
                self.total_cats_processed / self.total_encodings
            )

    def record_error(self, error_type: str = "general"):
        """Record validation or processing error."""
        self.validation_errors += 1
        if error_type == "shape":
            self.shape_mismatches += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary dictionary."""
        return {
            "total_encodings": self.total_encodings,
            "avg_encoding_time_ms": round(self.avg_encoding_time_ms, 3),
            "max_encoding_time_ms": round(self.max_encoding_time_ms, 3),
            "min_encoding_time_ms": (
                round(self.min_encoding_time_ms, 3)
                if self.min_encoding_time_ms != float("inf")
                else 0
            ),
            "avg_cats_per_encoding": round(self.avg_cats_per_encoding, 2),
            "max_cats_per_encoding": self.max_cats_per_encoding,
            "validation_errors": self.validation_errors,
            "shape_mismatches": self.shape_mismatches,
            "meets_performance_target": self.avg_encoding_time_ms
            <= TARGET_ENCODING_TIME_MS,
        }
