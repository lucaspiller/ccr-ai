import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .board import Board, CellType, Direction
from .engine import GameEngine
from .sprites import SpriteManager, SpriteType


@dataclass
class LevelMetadata:
    name: str
    description: str
    difficulty: str
    author: str
    version: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LevelMetadata":
        return cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            difficulty=data.get("difficulty", "medium"),
            author=data.get("author", "Unknown"),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
        )


class Level:
    def __init__(
        self, metadata: LevelMetadata, board: Board, sprite_manager: SpriteManager
    ):
        self.metadata = metadata
        self.board = board
        self.sprite_manager = sprite_manager
        self.initial_arrows: Dict[Tuple[int, int], Direction] = {}

    def create_engine(
        self, max_steps: int = 1000, seed: Optional[int] = None
    ) -> GameEngine:
        board_copy = self.board.copy()
        sprite_manager_copy = self.sprite_manager.copy()

        for (x, y), direction in self.initial_arrows.items():
            board_copy.place_arrow(x, y, direction)

        return GameEngine(board_copy, sprite_manager_copy, max_steps, seed)

    def validate(self) -> List[str]:
        errors = []

        if self.board.width <= 0 or self.board.height <= 0:
            errors.append("Board dimensions must be positive")

        mice = self.sprite_manager.get_sprites_by_type(SpriteType.MOUSE)
        if not mice:
            errors.append("Level must contain at least one mouse")

        rockets = self.sprite_manager.get_sprites_by_type(SpriteType.ROCKET)
        if not rockets:
            errors.append("Level must contain at least one rocket")

        for sprite in self.sprite_manager.sprites.values():
            x, y = sprite.position
            if not self.board.is_valid_position(x, y):
                errors.append(f"Sprite {sprite.sprite_id} is outside board boundaries")
            elif not self.board.is_walkable(x, y):
                errors.append(
                    f"Sprite {sprite.sprite_id} is placed on non-walkable cell"
                )

        sprite_positions = {}
        for sprite in self.sprite_manager.sprites.values():
            pos = sprite.position
            if pos in sprite_positions:
                errors.append(f"Multiple sprites at position {pos}")
            sprite_positions[pos] = sprite.sprite_id

        for (x, y), direction in self.initial_arrows.items():
            if not self.board.is_valid_position(x, y):
                errors.append(f"Arrow at ({x}, {y}) is outside board boundaries")
            elif not self.board.is_walkable(x, y):
                errors.append(f"Arrow at ({x}, {y}) is on non-walkable cell")
            elif (x, y) in sprite_positions:
                errors.append(f"Arrow at ({x}, {y}) conflicts with sprite")

        return errors

    def get_difficulty_metrics(self) -> Dict[str, Any]:
        mice_count = len(self.sprite_manager.get_sprites_by_type(SpriteType.MOUSE))
        cat_count = len(self.sprite_manager.get_sprites_by_type(SpriteType.CAT))
        rocket_count = len(self.sprite_manager.get_sprites_by_type(SpriteType.ROCKET))

        walkable_cells = sum(
            1
            for y in range(self.board.height)
            for x in range(self.board.width)
            if self.board.is_walkable(x, y)
        )

        hole_count = len(self.board.find_cells_by_type(CellType.HOLE))
        wall_count = len(self.board.find_cells_by_type(CellType.WALL))

        return {
            "board_size": self.board.width * self.board.height,
            "walkable_ratio": walkable_cells / (self.board.width * self.board.height),
            "mice_count": mice_count,
            "cat_count": cat_count,
            "rocket_count": rocket_count,
            "hole_count": hole_count,
            "wall_count": wall_count,
            "sprite_density": len(self.sprite_manager.sprites) / walkable_cells,
            "cat_to_mouse_ratio": cat_count / max(mice_count, 1),
            "initial_arrows": len(self.initial_arrows),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "board": self.board.to_dict(),
            "sprite_manager": self.sprite_manager.to_dict(),
            "initial_arrows": {
                f"{x},{y}": direction.name
                for (x, y), direction in self.initial_arrows.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Level":
        metadata = LevelMetadata.from_dict(data["metadata"])
        board = Board.from_dict(data["board"])
        sprite_manager = SpriteManager.from_dict(data["sprite_manager"])

        level = cls(metadata, board, sprite_manager)

        for pos_str, direction_name in data.get("initial_arrows", {}).items():
            x, y = map(int, pos_str.split(","))
            direction = Direction[direction_name]
            level.initial_arrows[(x, y)] = direction

        return level

    def save_to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> "Level":
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> "Level":
        return Level.from_dict(self.to_dict())

    def __str__(self) -> str:
        return f"Level: {self.metadata.name} ({self.board.width}x{self.board.height})"


class LevelBuilder:
    def __init__(self, width: int, height: int, name: str = "Custom Level"):
        self.metadata = LevelMetadata(
            name=name,
            description="",
            difficulty="medium",
            author="LevelBuilder",
            version="1.0",
            tags=[],
        )
        self.board = Board(width, height)
        self.sprite_manager = SpriteManager()
        self.initial_arrows: Dict[Tuple[int, int], Direction] = {}

    def set_metadata(self, **kwargs) -> "LevelBuilder":
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        return self

    def set_cell(self, x: int, y: int, cell_type: CellType) -> "LevelBuilder":
        self.board.set_cell_type(x, y, cell_type)
        return self

    def add_wall(
        self, position: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> "LevelBuilder":
        ((x1, y1), (x2, y2)) = position
        self.board.walls.add(((x1, y1), (x2, y2)))
        return self

    def add_hole(self, position: Tuple[int, int]) -> "LevelBuilder":
        (x, y) = position
        self.set_cell(x, y, CellType.HOLE)
        return self

    def add_rocket(self, position: Tuple[int, int]) -> "LevelBuilder":
        (x, y) = position
        self.set_cell(x, y, CellType.ROCKET)
        self.sprite_manager.create_rocket(x, y)
        return self

    def add_spawner(
        self,
        position: Tuple[int, int],
        direction: Direction = Direction.RIGHT,
    ) -> "LevelBuilder":
        from .sprites import SpriteType

        (x, y) = position
        self.set_cell(x, y, CellType.SPAWNER)
        spawner = self.sprite_manager.create_spawner(x, y)
        spawner.set_spawn_direction(direction)
        return self

    def add_mouse(
        self,
        x: int,
        y: int,
        sprite_id: Optional[str] = None,
        direction: Direction = Direction.RIGHT,
    ) -> "LevelBuilder":
        mouse = self.sprite_manager.create_mouse(x, y, sprite_id)
        mouse.set_direction(direction)
        return self

    def add_cat(
        self,
        x: int,
        y: int,
        sprite_id: Optional[str] = None,
        direction: Direction = Direction.RIGHT,
    ) -> "LevelBuilder":
        cat = self.sprite_manager.create_cat(x, y, sprite_id)
        cat.set_direction(direction)
        return self

    def add_rocket_sprite(
        self, x: int, y: int, sprite_id: Optional[str] = None
    ) -> "LevelBuilder":
        self.sprite_manager.create_rocket(x, y, sprite_id)
        return self

    def add_initial_arrow(
        self, position: Tuple[int, int], direction: Direction
    ) -> "LevelBuilder":
        self.initial_arrows[position] = direction
        return self

    def create_border_walls(self) -> "LevelBuilder":
        # Top and bottom borders
        for x in range(self.board.width):
            self.board.walls.add(((x, 0), (x, -1)))  # Top border
            self.board.walls.add(
                ((x, self.board.height - 1), (x, self.board.height))
            )  # Bottom border

        # Left and right borders
        for y in range(self.board.height):
            self.board.walls.add(((0, y), (-1, y)))  # Left border
            self.board.walls.add(
                ((self.board.width - 1, y), (self.board.width, y))
            )  # Right border

        return self

    def fill_empty_space(self) -> "LevelBuilder":
        for y in range(self.board.height):
            for x in range(self.board.width):
                if self.board.get_cell_type(x, y) == CellType.EMPTY:
                    continue
        return self

    def build(self) -> Level:
        level = Level(self.metadata, self.board, self.sprite_manager)
        level.initial_arrows = self.initial_arrows.copy()
        return level

    def validate_and_build(self) -> Tuple[Level, List[str]]:
        level = self.build()
        errors = level.validate()
        return level, errors


class LevelCollection:
    def __init__(self, name: str = "Level Collection"):
        self.name = name
        self.levels: Dict[str, Level] = {}
        self.metadata = {
            "name": name,
            "description": "",
            "author": "",
            "version": "1.0",
            "created_at": "",
            "tags": [],
        }

    def add_level(self, level: Level) -> None:
        self.levels[level.metadata.name] = level

    def remove_level(self, name: str) -> bool:
        if name in self.levels:
            del self.levels[name]
            return True
        return False

    def get_level(self, name: str) -> Optional[Level]:
        return self.levels.get(name)

    def get_levels_by_difficulty(self, difficulty: str) -> List[Level]:
        return [
            level
            for level in self.levels.values()
            if level.metadata.difficulty == difficulty
        ]

    def get_levels_by_tag(self, tag: str) -> List[Level]:
        return [level for level in self.levels.values() if tag in level.metadata.tags]

    def validate_all(self) -> Dict[str, List[str]]:
        validation_results = {}
        for name, level in self.levels.items():
            errors = level.validate()
            if errors:
                validation_results[name] = errors
        return validation_results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "levels": {name: level.to_dict() for name, level in self.levels.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LevelCollection":
        collection = cls(data["metadata"]["name"])
        collection.metadata = data["metadata"]

        for name, level_data in data["levels"].items():
            level = Level.from_dict(level_data)
            collection.add_level(level)

        return collection

    def save_to_directory(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        collection_file = os.path.join(directory, "collection.json")
        with open(collection_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        for name, level in self.levels.items():
            filename = f"{name.replace(' ', '_').lower()}.json"
            filepath = os.path.join(directory, filename)
            level.save_to_file(filepath)

    @classmethod
    def load_from_directory(cls, directory: str) -> "LevelCollection":
        collection_file = os.path.join(directory, "collection.json")

        if os.path.exists(collection_file):
            with open(collection_file, "r") as f:
                metadata = json.load(f)
            collection = cls(metadata["name"])
            collection.metadata = metadata
        else:
            collection = cls(os.path.basename(directory))

        for filename in os.listdir(directory):
            if filename.endswith(".json") and filename != "collection.json":
                filepath = os.path.join(directory, filename)
                try:
                    level = Level.load_from_file(filepath)
                    collection.add_level(level)
                except Exception as e:
                    print(f"Warning: Could not load level from {filename}: {e}")

        return collection

    def __len__(self) -> int:
        return len(self.levels)

    def __iter__(self):
        return iter(self.levels.values())

    def __getitem__(self, name: str) -> Level:
        return self.levels[name]
