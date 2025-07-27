"""
BFS solver for ChuChu Rocket puzzle mode.

Finds optimal arrow placements to route all mice to rockets.
"""

from .difficulty import DifficultyLabel, DifficultyScorer
from .solver import BFSResult, BFSSolver

__all__ = ["BFSSolver", "BFSResult", "DifficultyScorer", "DifficultyLabel"]
