from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class CellType(Enum):
    EMPTY = 0
    HOLE = 1
    ROCKET = 2
    SPAWNER = 3
    WALL = 4


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    def opposite(self) -> "Direction":
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]

    def turn_right(self) -> "Direction":
        right_turns = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
        }
        return right_turns[self]

    def turn_left(self) -> "Direction":
        left_turns = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP,
        }
        return left_turns[self]


class Board:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.arrows: Dict[Tuple[int, int], Direction] = {}
        self.arrow_placement_order: List[Tuple[int, int]] = (
            []
        )  # Track order for 3-arrow limit
        self.max_arrows = 3  # ChuChu Rocket arrow limit
        self.walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = (
            set()
        )  # Walls are edges between adjacent tiles

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell_type(self, x: int, y: int) -> CellType:
        if not self.is_valid_position(x, y):
            return None
        return CellType(self.grid[y, x])

    def set_cell_type(self, x: int, y: int, cell_type: CellType) -> None:
        if self.is_valid_position(x, y):
            self.grid[y, x] = cell_type.value

    def is_walkable(self, x: int, y: int) -> bool:
        cell_type = self.get_cell_type(x, y)
        return (
            cell_type == CellType.EMPTY
            or cell_type == CellType.ROCKET
            or cell_type == CellType.HOLE
        )

    def place_arrow(self, x: int, y: int, direction: Direction) -> bool:
        """Place an arrow at the given position."""
        if not self.is_valid_position(x, y):
            return False
        if self.get_cell_type(x, y) != CellType.EMPTY:
            return False
        if self.has_arrow(x, y):
            return False

        # Store the last removed arrow for querying
        self._last_removed_arrow = None

        # If we're at the arrow limit, remove the oldest arrow
        if len(self.arrows) >= self.max_arrows:
            oldest_pos = self.arrow_placement_order[0]
            self._last_removed_arrow = oldest_pos
            self.remove_arrow(oldest_pos[0], oldest_pos[1])

        # Place the new arrow
        self.arrows[(x, y)] = direction
        self.arrow_placement_order.append((x, y))
        return True

    def get_last_removed_arrow(self) -> Optional[Tuple[int, int]]:
        """Get the position of the last arrow that was automatically removed during placement."""
        return getattr(self, "_last_removed_arrow", None)

    def remove_arrow(self, x: int, y: int) -> bool:
        if (x, y) in self.arrows:
            del self.arrows[(x, y)]
            # Also remove from placement order list
            if (x, y) in self.arrow_placement_order:
                self.arrow_placement_order.remove((x, y))
            return True
        return False

    def get_arrow_direction(self, x: int, y: int) -> Optional[Direction]:
        return self.arrows.get((x, y))

    def has_arrow(self, x: int, y: int) -> bool:
        return (x, y) in self.arrows

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for direction in Direction:
            nx, ny = x + direction.dx, y + direction.dy
            if self.is_valid_position(nx, ny):
                # Check for wall between (x, y) and (nx, ny)
                if ((x, y), (nx, ny)) not in self.walls and (
                    (nx, ny),
                    (x, y),
                ) not in self.walls:
                    neighbors.append((nx, ny))
        return neighbors

    def get_walkable_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for direction in Direction:
            nx, ny = x + direction.dx, y + direction.dy
            if self.is_walkable(nx, ny):
                # Check for wall between (x, y) and (nx, ny)
                if ((x, y), (nx, ny)) not in self.walls and (
                    (nx, ny),
                    (x, y),
                ) not in self.walls:
                    neighbors.append((nx, ny))
        return neighbors

    def find_cells_by_type(self, cell_type: CellType) -> List[Tuple[int, int]]:
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.get_cell_type(x, y) == cell_type:
                    positions.append((x, y))
        return positions

    def copy(self) -> "Board":
        new_board = Board(self.width, self.height)
        new_board.grid = self.grid.copy()
        new_board.arrows = self.arrows.copy()
        new_board.arrow_placement_order = self.arrow_placement_order.copy()
        new_board.max_arrows = self.max_arrows
        new_board.walls = self.walls.copy()
        return new_board

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "grid": self.grid.tolist(),
            "arrows": {
                f"{x},{y}": direction.name for (x, y), direction in self.arrows.items()
            },
            "arrow_placement_order": [[x, y] for x, y in self.arrow_placement_order],
            "max_arrows": self.max_arrows,
            "walls": [((x1, y1), (x2, y2)) for ((x1, y1), (x2, y2)) in self.walls],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Board":
        board = cls(data["width"], data["height"])
        board.grid = np.array(data["grid"], dtype=int)
        for pos_str, direction_name in data["arrows"].items():
            x, y = map(int, pos_str.split(","))
            direction = Direction[direction_name]
            board.arrows[(x, y)] = direction

        # Restore arrow placement order and max arrows (with backward compatibility)
        if "arrow_placement_order" in data:
            board.arrow_placement_order = [
                tuple(pos) for pos in data["arrow_placement_order"]
            ]
        if "max_arrows" in data:
            board.max_arrows = data["max_arrows"]

        if "walls" in data:
            board.walls = set(
                tuple(tuple(pair) for pair in wall) for wall in data["walls"]
            )
        return board

    def __str__(self) -> str:
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) in self.arrows:
                    arrow_symbols = {
                        Direction.UP: "↑",
                        Direction.DOWN: "↓",
                        Direction.LEFT: "←",
                        Direction.RIGHT: "→",
                    }
                    row.append(arrow_symbols[self.arrows[(x, y)]])
                else:
                    cell_symbols = {
                        CellType.EMPTY: ".",
                        CellType.HOLE: "O",
                        CellType.ROCKET: "R",
                        CellType.SPAWNER: "S",
                        CellType.WALL: "#",
                    }
                    ct = self.get_cell_type(x, y)
                    row.append(cell_symbols.get(ct, "?"))
            result.append("".join(row))
        return "\n".join(result)
