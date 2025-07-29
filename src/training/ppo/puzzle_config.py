"""
Puzzle configuration dataclass for PPO training.

Defines the structure of puzzle configurations used in PPO environments and evaluators.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.game.levels import Level


@dataclass
class PuzzleSpec:
    """Specification for a puzzle difficulty level."""

    name: str
    width_range: Tuple[int, int]
    height_range: Tuple[int, int]
    mice_range: Tuple[int, int]
    cats_range: Tuple[int, int]
    arrow_budget: int
    holes_range: Tuple[int, int]
    walls_range: Tuple[int, int]
    rockets_range: Tuple[int, int]


@dataclass
class SpriteConfig:
    """Configuration for a sprite (mouse, cat, rocket, hole)."""

    x: int
    y: int
    direction: Optional[str] = None  # For mice and cats
    player_id: Optional[int] = None  # For rockets


@dataclass
class WallConfig:
    """Configuration for a wall segment."""

    x: int
    y: int
    direction: str  # "horizontal" or "vertical"


@dataclass
class PuzzleConfig:
    """Configuration for a puzzle instance."""

    width: int
    height: int
    arrow_budget: int
    walls: List[WallConfig]
    rockets: List[SpriteConfig]
    mice: List[SpriteConfig]
    cats: List[SpriteConfig]
    holes: List[SpriteConfig]
    puzzle_id: str
    difficulty: str
    _level: Optional[Level] = None  # Internal use only, not serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PuzzleConfig":
        """Create PuzzleConfig from dictionary format."""
        return cls(
            width=data["width"],
            height=data["height"],
            arrow_budget=data["arrow_budget"],
            walls=[WallConfig(**wall) for wall in data.get("walls", [])],
            rockets=[SpriteConfig(**rocket) for rocket in data.get("rockets", [])],
            mice=[SpriteConfig(**mouse) for mouse in data.get("mice", [])],
            cats=[SpriteConfig(**cat) for cat in data.get("cats", [])],
            holes=[SpriteConfig(**hole) for hole in data.get("holes", [])],
            puzzle_id=data.get("puzzle_id"),
            difficulty=data.get("difficulty"),
            _level=data.get("_level"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert PuzzleConfig to dictionary format (for serialization)."""
        result = {
            "width": self.width,
            "height": self.height,
            "arrow_budget": self.arrow_budget,
            "walls": [
                {"x": w.x, "y": w.y, "direction": w.direction} for w in self.walls
            ],
            "rockets": [
                {"x": r.x, "y": r.y, "player_id": r.player_id} for r in self.rockets
            ],
            "mice": [{"x": m.x, "y": m.y, "direction": m.direction} for m in self.mice],
            "cats": [{"x": c.x, "y": c.y, "direction": c.direction} for c in self.cats],
            "holes": [{"x": h.x, "y": h.y} for h in self.holes],
        }

        if self.puzzle_id is not None:
            result["puzzle_id"] = self.puzzle_id
        if self.difficulty is not None:
            result["difficulty"] = self.difficulty
        # Note: _level is intentionally excluded from serialization

        return result
