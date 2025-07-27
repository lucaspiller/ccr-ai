"""
State-fusion layer for ChuChu Rocket AI.

This module provides feature fusion functionality that combines perception outputs
into a single 128-dimensional embedding for decision-making.
"""

from .data_types import FusionConfig, FusedStateOutput
from .processors import StateFusionProcessor

__all__ = ["FusionConfig", "FusedStateOutput", "StateFusionProcessor"]