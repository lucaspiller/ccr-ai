"""
Puzzle generation module for CCG AI.

Provides puzzle generation functionality for PPO training and evaluation.
"""

from .puzzle_config import PuzzleConfig, PuzzleSpec, SpriteConfig, WallConfig
from .puzzle_generator import PuzzleGenerator

__all__ = ["PuzzleConfig", "PuzzleSpec", "SpriteConfig", "WallConfig", "PuzzleGenerator"]