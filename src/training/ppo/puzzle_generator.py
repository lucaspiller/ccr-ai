"""
Shared puzzle generation functionality for PPO training and evaluation.

Handles puzzle spec definitions and generates levels/puzzle configs
to avoid code duplication between curriculum.py and ppo_evaluator.py.
"""

import random
from typing import Any, List

from src.game.board import CellType
from src.game.board_builder import BoardBuilder, BoardConfig
from src.game.engine import GameResult

from .puzzle_config import PuzzleConfig, PuzzleSpec, SpriteConfig, WallConfig


class PuzzleGenerator:
    """Generates puzzles and levels from puzzle specifications."""

    def __init__(self, execution_timeout: int = 1000):
        self.execution_timeout = execution_timeout

    def get_puzzle_specs(self) -> dict[str, PuzzleSpec]:
        """Get standard puzzle specifications for different difficulties."""
        return {
            "easy": PuzzleSpec(
                name="Easy",
                width_range=(5, 8),
                height_range=(5, 6),
                mice_range=(1, 5),
                cats_range=(0, 0),
                arrow_budget=5,
                holes_range=(0, 1),
                walls_range=(5, 10),
                rockets_range=(1, 2),
            ),
            "medium": PuzzleSpec(
                name="Medium",
                width_range=(8, 10),
                height_range=(6, 8),
                mice_range=(5, 7),
                cats_range=(1, 2),
                arrow_budget=4,
                holes_range=(1, 2),
                walls_range=(10, 15),
                rockets_range=(1, 5),
            ),
            "hard": PuzzleSpec(
                name="Hard",
                width_range=(10, 14),
                height_range=(8, 10),
                mice_range=(7, 10),
                cats_range=(2, 4),
                arrow_budget=3,
                holes_range=(2, 3),
                walls_range=(15, 25),
                rockets_range=(1, 5),
            ),
        }

    def generate_level(self, spec: PuzzleSpec, rng: random.Random) -> Any:
        """Generate a single level according to specification."""
        # Random board size within range
        width = rng.randint(*spec.width_range)
        height = rng.randint(*spec.height_range)

        # Random counts within ranges
        num_mice = rng.randint(*spec.mice_range)
        num_cats = rng.randint(*spec.cats_range)
        num_holes = rng.randint(*spec.holes_range)
        num_walls = rng.randint(*spec.walls_range)
        num_rockets = rng.randint(*spec.rockets_range)

        # Create board config
        config = BoardConfig(
            board_w=width,
            board_h=height,
            num_walls=num_walls,
            num_mice=num_mice,
            num_rockets=num_rockets,
            num_cats=num_cats,
            num_holes=num_holes,
            arrow_budget=spec.arrow_budget,
        )

        # Generate level using BoardBuilder
        board_builder = BoardBuilder(config, seed=rng.randint(0, 2**31 - 1))
        level = board_builder.generate_level(f"{spec.name} Puzzle")

        return level

    def generate_puzzle_config(
        self, spec: PuzzleSpec, rng: random.Random, puzzle_id: str, difficulty: str
    ) -> PuzzleConfig:
        """Generate a puzzle config from a spec using BoardBuilder."""
        # Random board size within range
        width = rng.randint(*spec.width_range)
        height = rng.randint(*spec.height_range)

        # Random counts within ranges
        num_mice = rng.randint(*spec.mice_range)
        num_cats = rng.randint(*spec.cats_range)
        num_holes = rng.randint(*spec.holes_range)
        num_walls = rng.randint(*spec.walls_range)
        num_rockets = rng.randint(*spec.rockets_range)

        # Create board config
        config = BoardConfig(
            board_w=width,
            board_h=height,
            num_walls=num_walls,
            num_mice=num_mice,
            num_rockets=num_rockets,
            num_cats=num_cats,
            num_holes=num_holes,
            arrow_budget=spec.arrow_budget,
        )

        # Generate level using BoardBuilder
        board_builder = BoardBuilder(config, seed=rng.randint(0, 2**31 - 1))
        level = board_builder.generate_level(f"{spec.name} Puzzle")

        # Verify the puzzle can't be completed without arrow placements
        engine = level.create_engine(max_steps=self.execution_timeout, puzzle_mode=True)
        engine.start_game()

        engine_steps = 0
        while (
            engine.result == GameResult.ONGOING
            and engine_steps < self.execution_timeout
        ):
            engine.step()
            engine_steps += 1

        if engine.result == GameResult.SUCCESS:
            # Regenerate if solvable without placements
            return self.generate_puzzle_config(spec, rng, puzzle_id, difficulty)

        # Convert to the expected dictionary format for the environment
        board = level.board

        # Extract walls
        walls = []
        for (x1, y1), (x2, y2) in board.walls:
            # Determine direction based on coordinates
            if x1 == x2:  # Horizontal wall
                walls.append(WallConfig(x=x1, y=min(y1, y2), direction="horizontal"))
            else:  # Vertical wall
                walls.append(WallConfig(x=min(x1, x2), y=y1, direction="vertical"))

        # Extract rockets
        rockets = []
        rocket_positions = board.find_cells_by_type(CellType.ROCKET)
        for i, (x, y) in enumerate(rocket_positions):
            rockets.append(SpriteConfig(x=x, y=y, player_id=0))

        # Extract mice
        mice = []
        for sprite_id, sprite in level.sprite_manager.sprites.items():
            if sprite.get_sprite_type().name == "MOUSE":
                mice.append(
                    SpriteConfig(x=int(sprite.x), y=int(sprite.y), direction="RIGHT")
                )

        # Extract cats
        cats = []
        for sprite_id, sprite in level.sprite_manager.sprites.items():
            if sprite.get_sprite_type().name == "CAT":
                cats.append(
                    SpriteConfig(x=int(sprite.x), y=int(sprite.y), direction="RIGHT")
                )

        # Extract holes
        holes = []
        hole_positions = board.find_cells_by_type(CellType.HOLE)
        for x, y in hole_positions:
            holes.append(SpriteConfig(x=x, y=y))

        puzzle = PuzzleConfig(
            width=board.width,
            height=board.height,
            arrow_budget=board.max_arrows,
            walls=walls,
            rockets=rockets,
            mice=mice,
            cats=cats,
            holes=holes,
            puzzle_id=f"{spec.name}_{rng.randint(0, 999):03d}",
            difficulty=difficulty,
            _level=level,  # Keep level for internal use
        )

        return puzzle

    def generate_evaluation_puzzles(
        self, fixed_seed: int = 12345
    ) -> dict[str, List[PuzzleConfig]]:
        """Generate fixed evaluation puzzle sets."""
        rng = random.Random(fixed_seed)
        specs = self.get_puzzle_specs()

        puzzles = {"easy": [], "medium": [], "hard": []}

        # Easy puzzles (100 puzzles)
        for i in range(100):
            puzzle = self.generate_puzzle_config(
                specs["easy"], rng, puzzle_id=f"easy_{i:03d}", difficulty="easy"
            )
            puzzles["easy"].append(puzzle)

        # Medium puzzles (200 puzzles)
        for i in range(200):
            puzzle = self.generate_puzzle_config(
                specs["medium"], rng, puzzle_id=f"medium_{i:03d}", difficulty="medium"
            )
            puzzles["medium"].append(puzzle)

        # Hard puzzles (200 puzzles)
        for i in range(200):
            puzzle = self.generate_puzzle_config(
                specs["hard"], rng, puzzle_id=f"hard_{i:03d}", difficulty="hard"
            )
            puzzles["hard"].append(puzzle)

        return puzzles
