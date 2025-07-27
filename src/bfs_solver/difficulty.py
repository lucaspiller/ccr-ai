"""
Difficulty scoring for ChuChu Rocket puzzle mode.
"""

from enum import Enum
from typing import List, Tuple

from ..game.board import Direction
from ..game.engine import GameEngine
from ..game.sprites import SpriteType


class DifficultyLabel(Enum):
    """Difficulty labels for puzzles."""

    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    BRUTAL = "Brutal"


class DifficultyScorer:
    """Scores puzzle difficulty based on solution and board characteristics."""

    @staticmethod
    def score_puzzle(
        engine: GameEngine, solution: List[Tuple[Tuple[int, int], Direction]]
    ) -> float:
        """Calculate difficulty score for a puzzle.

        Args:
            engine: Game engine with the puzzle
            solution: BFS solution (arrow placements)

        Returns:
            Difficulty score
        """
        # Count cats and holes
        num_cats = len(engine.sprite_manager.get_sprites_by_type(SpriteType.CAT))

        holes_count = 0
        for y in range(engine.board.height):
            for x in range(engine.board.width):
                from ..game.board import CellType

                if engine.board.get_cell_type(x, y) == CellType.HOLE:
                    holes_count += 1

        # Apply scoring formula from PRD
        score = 1.0 * len(solution) + 4.0 * num_cats + 2.0 * holes_count

        return score

    @staticmethod
    def get_difficulty_label(score: float) -> DifficultyLabel:
        """Get difficulty label from score.

        Args:
            score: Difficulty score

        Returns:
            Difficulty label
        """
        if score <= 10:
            return DifficultyLabel.EASY
        elif score <= 20:
            return DifficultyLabel.MEDIUM
        elif score <= 35:
            return DifficultyLabel.HARD
        else:
            return DifficultyLabel.BRUTAL

    @staticmethod
    def score_and_label(
        engine: GameEngine, solution: List[Tuple[Tuple[int, int], Direction]]
    ) -> Tuple[float, DifficultyLabel]:
        """Calculate both score and label.

        Args:
            engine: Game engine with the puzzle
            solution: BFS solution

        Returns:
            (score, label) tuple
        """
        score = DifficultyScorer.score_puzzle(engine, solution)
        label = DifficultyScorer.get_difficulty_label(score)
        return score, label
