"""
Perception layer for converting game state to neural network inputs.
"""

from .cat_encoder import CatSetEncoder, CatSetProcessor
from .data_types import PerceptionConfig, PerceptionMetrics, PerceptionOutput
from .encoders import GlobalFeatureExtractor, GridEncoder
from .processors import (BatchGameStateProcessor, GameStateProcessor,
                         get_combined_embedding, process_game_state)

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
