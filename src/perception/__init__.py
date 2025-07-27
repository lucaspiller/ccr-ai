"""
Perception layer for converting game state to neural network inputs.
"""

from .encoders import GridEncoder, GlobalFeatureExtractor
from .cat_encoder import CatSetEncoder, CatSetProcessor
from .processors import (
    GameStateProcessor,
    BatchGameStateProcessor,
    process_game_state,
    get_combined_embedding,
)
from .data_types import PerceptionOutput, PerceptionConfig, PerceptionMetrics

__all__ = [
    "GridEncoder",
    "GlobalFeatureExtractor",
    "CatSetEncoder",
    "CatSetProcessor",
    "GameStateProcessor",
    "BatchGameStateProcessor",
    "process_game_state",
    "get_combined_embedding",
    "PerceptionOutput",
    "PerceptionConfig",
    "PerceptionMetrics",
]
