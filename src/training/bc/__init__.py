"""
Behaviour Cloning training module for ChuChu Rocket AI.

This module implements supervised learning on BFS-optimal solutions
as Stage A of the training pipeline.
"""

# BC Sequence Lite components
from .bc_sequence_lite_config import BCSequenceLiteConfig
from .bc_sequence_lite_data_loader import (BCSequenceLiteDataset,
                                           create_sequence_lite_data_loaders)
from .bc_sequence_lite_trainer import (BCSequenceLiteTrainer,
                                       train_bc_sequence_lite)
from .config import BCConfig
from .data_loader import (BehaviourCloningDataset, create_data_loaders,
                          get_device, load_puzzles_from_csv,
                          parse_bfs_solution)
from .evaluator import BCEvaluator
from .trainer import BCTrainer

__all__ = [
    "BCConfig",
    "BehaviourCloningDataset",
    "create_data_loaders",
    "load_puzzles_from_csv",
    "parse_bfs_solution",
    "get_device",
    "BCTrainer",
    "BCEvaluator",
    # BC Sequence Lite
    "BCSequenceLiteConfig",
    "BCSequenceLiteDataset",
    "create_sequence_lite_data_loaders",
    "BCSequenceLiteTrainer",
    "train_bc_sequence_lite",
]
