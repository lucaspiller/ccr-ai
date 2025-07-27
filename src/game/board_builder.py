"""
BoardBuilder for generating ChuChu Rocket puzzle mode boards.

Generates static puzzle boards with no spawners for behavior cloning.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .board import Board, CellType, Direction
from .levels import Level, LevelBuilder


@dataclass
class BoardConfig:
    """Configuration for board generation."""

    board_w: int = 7
    board_h: int = 7
    num_walls: int = 10
    num_mice: int = 3
    num_rockets: int = 2
    num_cats: int = 1
    num_holes: int = 2
    arrow_budget: int = 3


class BoardBuilder:
    """Generates puzzle mode boards for ChuChu Rocket."""

    def __init__(self, config: BoardConfig, seed: Optional[int] = None):
        """Initialize board builder with configuration.

        Args:
            config: Board generation configuration
            seed: Random seed for deterministic generation
        """
        self.config = config
        self.rng = random.Random(seed)

    def generate_board(self) -> Board:
        """Generate a complete puzzle board.

        Returns:
            Board: Generated puzzle board with all elements placed
        """
        board = Board(self.config.board_w, self.config.board_h)
        board.max_arrows = self.config.arrow_budget

        # Generate board elements in order of constraint priority
        self._place_walls(board)
        self._place_rockets(board)
        self._place_holes(board)
        self._place_mice(board)
        self._place_cats(board)

        return board

    def generate_level(self, name: str = "Generated Puzzle") -> "Level":
        """Generate a complete puzzle level with board and sprites.

        Args:
            name: Name for the generated level

        Returns:
            Level: Complete level ready for engine creation
        """
        board = self.generate_board()

        # Create level builder and copy the board
        level_builder = LevelBuilder(self.config.board_w, self.config.board_h, name)
        level_builder.set_metadata(
            description=f"Generated puzzle board with {self.config.num_mice} mice, {self.config.num_cats} cats",
            difficulty="medium",
            tags=["puzzle", "generated"],
        )
        level_builder.board = board.copy()

        # Add sprites for mice
        mice_positions = getattr(board, "mice_positions", [])
        for i, (x, y) in enumerate(mice_positions):
            level_builder.add_mouse(x, y, f"mouse_{i}", Direction.RIGHT)

        # Add sprites for cats
        cat_positions = getattr(board, "cat_positions", [])
        for i, (x, y) in enumerate(cat_positions):
            level_builder.add_cat(x, y, f"cat_{i}", Direction.RIGHT)

        # Add rocket sprites (rockets are already in board as cells)
        rocket_positions = board.find_cells_by_type(CellType.ROCKET)
        for i, (x, y) in enumerate(rocket_positions):
            level_builder.add_rocket_sprite(x, y, f"rocket_{i}")

        return level_builder.build()

    def _place_walls(self, board: Board) -> None:
        """Place wall segments randomly on the board."""
        walls_placed = 0
        attempts = 0
        max_attempts = self.config.num_walls * 10

        while walls_placed < self.config.num_walls and attempts < max_attempts:
            attempts += 1

            # Pick random cell
            x = self.rng.randint(0, self.config.board_w - 1)
            y = self.rng.randint(0, self.config.board_h - 1)

            # Pick random direction for wall
            direction = self.rng.choice(list(Direction))
            nx, ny = x + direction.dx, y + direction.dy

            # Check if wall placement is valid
            if not board.is_valid_position(nx, ny):
                continue

            wall = ((x, y), (nx, ny))
            reverse_wall = ((nx, ny), (x, y))

            # Don't place duplicate walls
            if wall in board.walls or reverse_wall in board.walls:
                continue

            # Place the wall
            board.walls.add(wall)
            walls_placed += 1

    def _place_rockets(self, board: Board) -> None:
        """Place rockets on empty cells."""
        rockets_placed = 0
        attempts = 0
        max_attempts = self.config.num_rockets * 20

        while rockets_placed < self.config.num_rockets and attempts < max_attempts:
            attempts += 1

            x = self.rng.randint(0, self.config.board_w - 1)
            y = self.rng.randint(0, self.config.board_h - 1)

            if board.get_cell_type(x, y) == CellType.EMPTY:
                board.set_cell_type(x, y, CellType.ROCKET)
                rockets_placed += 1

    def _place_holes(self, board: Board) -> None:
        """Place holes on empty cells."""
        holes_placed = 0
        attempts = 0
        max_attempts = self.config.num_holes * 20

        while holes_placed < self.config.num_holes and attempts < max_attempts:
            attempts += 1

            x = self.rng.randint(0, self.config.board_w - 1)
            y = self.rng.randint(0, self.config.board_h - 1)

            if board.get_cell_type(x, y) == CellType.EMPTY:
                board.set_cell_type(x, y, CellType.HOLE)
                holes_placed += 1

    def _place_mice(self, board: Board) -> None:
        """Place mice on empty cells.

        Note: Mice are represented as special markers in the board for puzzle mode.
        In the actual game engine, these will be converted to initial mouse positions.
        """
        # For now, we'll use a custom approach to track mouse starting positions
        # This could be extended to use a special cell type or metadata
        mice_positions = []
        mice_placed = 0
        attempts = 0
        max_attempts = self.config.num_mice * 20

        while mice_placed < self.config.num_mice and attempts < max_attempts:
            attempts += 1

            x = self.rng.randint(0, self.config.board_w - 1)
            y = self.rng.randint(0, self.config.board_h - 1)

            if (
                board.get_cell_type(x, y) == CellType.EMPTY
                and (x, y) not in mice_positions
            ):
                mice_positions.append((x, y))
                mice_placed += 1

        # Store mice positions as board metadata
        board.mice_positions = mice_positions

    def _place_cats(self, board: Board) -> None:
        """Place cats on empty cells.

        Note: Similar to mice, cats are stored as metadata for puzzle mode.
        """
        cat_positions = []
        cats_placed = 0
        attempts = 0
        max_attempts = self.config.num_cats * 20

        while cats_placed < self.config.num_cats and attempts < max_attempts:
            attempts += 1

            x = self.rng.randint(0, self.config.board_w - 1)
            y = self.rng.randint(0, self.config.board_h - 1)

            mice_positions = getattr(board, "mice_positions", [])
            if (
                board.get_cell_type(x, y) == CellType.EMPTY
                and (x, y) not in mice_positions
                and (x, y) not in cat_positions
            ):
                cat_positions.append((x, y))
                cats_placed += 1

        # Store cat positions as board metadata
        board.cat_positions = cat_positions

    def get_empty_cells(self, board: Board) -> List[Tuple[int, int]]:
        """Get all empty cells on the board.

        Args:
            board: Board to analyze

        Returns:
            List of (x, y) coordinates for empty cells
        """
        empty_cells = []
        mice_positions = getattr(board, "mice_positions", [])
        cat_positions = getattr(board, "cat_positions", [])

        for y in range(board.height):
            for x in range(board.width):
                if (
                    board.get_cell_type(x, y) == CellType.EMPTY
                    and (x, y) not in mice_positions
                    and (x, y) not in cat_positions
                ):
                    empty_cells.append((x, y))

        return empty_cells
