"""
Behaviour Cloning training module for ChuChu Rocket AI.

This module implements supervised learning on BFS-optimal solutions
as Stage A of the training pipeline.
"""

from .config import BCConfig
from .data_loader import (BehaviourCloningDataset, create_data_loaders,
                          get_device, load_puzzles_from_csv,
                          parse_bfs_solution)
from .evaluator import BCEvaluator
from .model_manager import ModelManager
from .trainer import BCTrainer

__all__ = [
    "BCConfig",
    "BehaviourCloningDataset",
    "create_data_loaders",
    "load_puzzles_from_csv",
    "parse_bfs_solution",
    "get_device",
    "BCTrainer",
    "ModelManager",
    "BCEvaluator",
]
