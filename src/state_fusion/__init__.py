"""
State-fusion layer for ChuChu Rocket AI.

This module provides feature fusion functionality that combines perception outputs
into a single 128-dimensional embedding for decision-making.
"""

from .data_types import FusedStateOutput, FusionConfig
from .processors import StateFusionProcessor

__all__ = ["FusionConfig", "FusedStateOutput", "StateFusionProcessor"]
