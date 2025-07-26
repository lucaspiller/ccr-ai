import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .board import Direction


class SpriteType(Enum):
    MOUSE = "mouse"
    GOLD_MOUSE = "gold_mouse"
    BONUS_MOUSE = "bonus_mouse"
    CAT = "cat"
    ROCKET = "rocket"
    SPAWNER = "spawner"


class SpriteState(Enum):
    ACTIVE = "active"
    CAPTURED = "captured"
    ESCAPED = "escaped"


class BonusMode(Enum):
    NONE = "none"
    MOUSE_MANIA = "mouse_mania"
    CAT_MANIA = "cat_mania"
    SPEED_UP = "speed_up"
    SLOW_DOWN = "slow_down"
    PLACE_ARROWS_AGAIN = "place_arrows_again"
    MOUSE_MONOPOLY = "mouse_monopoly"
    CAT_ATTACK = "cat_attack"
    EVERYBODY_MOVE = "everybody_move"


class BonusState:
    def __init__(self):
        self.mode = BonusMode.NONE
        self.remaining_ticks = 0
        self.duration_ticks = 0  # Total duration for reference
        self.previous_mode = BonusMode.NONE  # Track the last active mode

    def start_bonus(
        self, mode: BonusMode, duration_seconds: float, ticks_per_second: int = 60
    ):
        """Start a bonus mode for the specified duration"""
        self.mode = mode
        self.duration_ticks = int(duration_seconds * ticks_per_second)
        self.remaining_ticks = self.duration_ticks

    def tick(self):
        """Process one tick - decreases remaining time"""
        if self.remaining_ticks > 0:
            self.remaining_ticks -= 1
            if self.remaining_ticks <= 0:
                self.previous_mode = self.mode
                self.mode = BonusMode.NONE

    def is_active(self) -> bool:
        """Check if any bonus mode is currently active"""
        return self.mode != BonusMode.NONE

    def get_progress(self) -> float:
        """Get progress of current bonus (0.0 to 1.0, where 1.0 is complete)"""
        if not self.is_active() or self.duration_ticks == 0:
            return 1.0
        return 1.0 - (self.remaining_ticks / self.duration_ticks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "remaining_ticks": self.remaining_ticks,
            "duration_ticks": self.duration_ticks,
            "previous_mode": self.previous_mode.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BonusState":
        bonus_state = cls()
        bonus_state.mode = BonusMode(data.get("mode", "none"))
        bonus_state.remaining_ticks = data.get("remaining_ticks", 0)
        bonus_state.duration_ticks = data.get("duration_ticks", 0)
        bonus_state.previous_mode = BonusMode(data.get("previous_mode", "none"))
        return bonus_state


class Sprite(ABC):
    def __init__(self, x: int, y: int, sprite_id: str):
        # Float position is now authoritative (center of tile for rockets/spawners)
        self.x = float(x)
        self.y = float(y)
        self.sprite_id = sprite_id
        self.state = SpriteState.ACTIVE
        self.direction = Direction.RIGHT

        # Tick-based movement system
        self.move_interval_ticks = 8  # Default, overridden by subclasses
        self.ticks_since_last_move = 0

        # Track the direction we were moving when we started current movement
        self.movement_direction = Direction.RIGHT

    @property
    def tile_x(self) -> int:
        """Integer tile position derived from float position"""
        return int(self.x)

    @property
    def tile_y(self) -> int:
        """Integer tile position derived from float position"""
        return int(self.y)

    @property
    def tile_position(self) -> Tuple[int, int]:
        """Integer tile position for board lookups"""
        return (self.tile_x, self.tile_y)

    @property
    def position(self) -> Tuple[float, float]:
        """Float position (authoritative)"""
        return (self.x, self.y)

    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        self.x, self.y = pos

    def can_move_this_tick(self, bonus_state=None) -> bool:
        """Check if sprite should move on this tick based on its speed"""
        effective_interval = self.get_effective_move_interval(bonus_state)
        return self.ticks_since_last_move >= effective_interval

    def get_effective_move_interval(self, bonus_state=None) -> float:
        """Get the effective movement interval based on bonus modes"""
        base_interval = self.move_interval_ticks

        if bonus_state and bonus_state.is_active():
            if bonus_state.mode == BonusMode.SPEED_UP:
                # Double speed = half interval
                return base_interval / 2.0
            elif bonus_state.mode == BonusMode.SLOW_DOWN:
                # Half speed = double interval
                return base_interval * 2.0

        return base_interval

    def tick(self) -> None:
        """Advance one tick for this sprite"""
        self.ticks_since_last_move += 1

    def reset_move_timer(self) -> None:
        """Reset the movement timer after a move"""
        self.ticks_since_last_move = 0
        # Capture the direction we're moving in for smooth interpolation
        self.movement_direction = self.direction

    def move_to(self, x: float, y: float) -> None:
        """Move to absolute float position"""
        self.x = x
        self.y = y

    def move_by(self, dx: float, dy: float) -> None:
        """Move by relative float amount"""
        self.x += dx
        self.y += dy

    def set_direction(self, direction: Direction) -> None:
        self.direction = direction

    def set_state(self, state: SpriteState) -> None:
        self.state = state

    def is_active(self) -> bool:
        return self.state == SpriteState.ACTIVE

    @abstractmethod
    def get_sprite_type(self) -> SpriteType:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sprite_id": self.sprite_id,
            "x": self.x,
            "y": self.y,
            "state": self.state.value,
            "direction": self.direction.name,
            "move_interval_ticks": self.move_interval_ticks,
            "ticks_since_last_move": self.ticks_since_last_move,
            "movement_direction": self.movement_direction.name,
            "type": self.get_sprite_type().value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sprite":
        sprite_type = SpriteType(data["type"])

        if sprite_type == SpriteType.MOUSE:
            sprite = Mouse(data["x"], data["y"], data["sprite_id"])
        elif sprite_type == SpriteType.GOLD_MOUSE:
            sprite = GoldMouse(data["x"], data["y"], data["sprite_id"])
        elif sprite_type == SpriteType.BONUS_MOUSE:
            sprite = BonusMouse(data["x"], data["y"], data["sprite_id"])
        elif sprite_type == SpriteType.CAT:
            sprite = Cat(data["x"], data["y"], data["sprite_id"])
        elif sprite_type == SpriteType.ROCKET:
            sprite = Rocket(data["x"], data["y"], data["sprite_id"])
        elif sprite_type == SpriteType.SPAWNER:
            sprite = Spawner.from_dict(data)
            return sprite  # Return early since Spawner.from_dict handles all attributes
        else:
            raise ValueError(f"Unknown sprite type: {sprite_type}")

        sprite.state = SpriteState(data["state"])
        sprite.direction = Direction[data["direction"]]
        sprite.move_interval_ticks = data.get("move_interval_ticks", 8)
        sprite.ticks_since_last_move = data.get("ticks_since_last_move", 0)
        sprite.x = float(data["x"])
        sprite.y = float(data["y"])
        sprite.movement_direction = Direction[
            data.get("movement_direction", data["direction"])
        ]

        return sprite

    def copy(self) -> "Sprite":
        return self.from_dict(self.to_dict())

    def __str__(self) -> str:
        return f"{self.get_sprite_type().value}({self.sprite_id}) at ({self.x:.1f}, {self.y:.1f}) tile ({self.tile_x}, {self.tile_y}) facing {self.direction.name}"


class Mouse(Sprite):
    def __init__(self, x: int, y: int, sprite_id: str):
        super().__init__(x, y, sprite_id)
        self.move_interval_ticks = 8  # 1 tile every 8 ticks (7.5 tiles/sec)

    def get_sprite_type(self) -> SpriteType:
        return SpriteType.MOUSE

    def can_be_captured_by_cat(self) -> bool:
        return self.state == SpriteState.ACTIVE

    def capture(self) -> None:
        self.state = SpriteState.CAPTURED

    def escape(self) -> None:
        self.state = SpriteState.ESCAPED

    def get_rocket_value(self) -> int:
        return 1


class GoldMouse(Sprite):
    def __init__(self, x: int, y: int, sprite_id: str):
        super().__init__(x, y, sprite_id)
        self.move_interval_ticks = 8  # Same speed as regular mouse

    def get_sprite_type(self) -> SpriteType:
        return SpriteType.GOLD_MOUSE

    def can_be_captured_by_cat(self) -> bool:
        return self.state == SpriteState.ACTIVE

    def capture(self) -> None:
        self.state = SpriteState.CAPTURED

    def escape(self) -> None:
        self.state = SpriteState.ESCAPED

    def get_rocket_value(self) -> int:
        return 50


class BonusMouse(Sprite):
    def __init__(self, x: int, y: int, sprite_id: str):
        super().__init__(x, y, sprite_id)
        self.move_interval_ticks = 8  # Same speed as regular mouse

    def get_sprite_type(self) -> SpriteType:
        return SpriteType.BONUS_MOUSE

    def can_be_captured_by_cat(self) -> bool:
        return self.state == SpriteState.ACTIVE

    def capture(self) -> None:
        self.state = SpriteState.CAPTURED

    def escape(self) -> None:
        self.state = SpriteState.ESCAPED

    def get_rocket_value(self) -> int:
        return 0  # Triggers roulette instead


class Cat(Sprite):
    def __init__(self, x: int, y: int, sprite_id: str):
        super().__init__(x, y, sprite_id)
        self.move_interval_ticks = 16  # 1 tile every 16 ticks (3.75 tiles/sec)

    def get_sprite_type(self) -> SpriteType:
        return SpriteType.CAT

    def can_capture_mouse(self, mouse) -> bool:
        return (
            self.state == SpriteState.ACTIVE
            and hasattr(mouse, "can_be_captured_by_cat")
            and mouse.can_be_captured_by_cat()
            and self.tile_position == mouse.tile_position
        )

    def capture_mouse(self, mouse) -> bool:
        if self.can_capture_mouse(mouse):
            mouse.capture()
            return True
        return False


class Rocket(Sprite):
    def __init__(self, x: int, y: int, sprite_id: str):
        super().__init__(x, y, sprite_id)
        self.move_interval_ticks = float("inf")  # Rockets don't move
        self.mice_collected = 0

    def get_sprite_type(self) -> SpriteType:
        return SpriteType.ROCKET

    def can_collect_mouse(self, mouse) -> bool:
        return (
            self.state == SpriteState.ACTIVE
            and mouse.state == SpriteState.ACTIVE
            and self.tile_position == mouse.tile_position
            and hasattr(mouse, "can_be_captured_by_cat")  # Any mouse-like sprite
        )

    def collect_mouse(self, mouse, bonus_callback=None) -> bool:
        if self.can_collect_mouse(mouse):
            mouse.escape()
            # Check if this is a bonus mouse for bonus round triggering
            if mouse.get_sprite_type() == SpriteType.BONUS_MOUSE and bonus_callback:
                bonus_callback(mouse, self)

            # Add different values based on mouse type
            if hasattr(mouse, "get_rocket_value"):
                self.mice_collected += mouse.get_rocket_value()
            else:
                self.mice_collected += 1  # Default for regular Mouse
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"mice_collected": self.mice_collected})
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rocket":
        rocket = cls(data["x"], data["y"], data["sprite_id"])
        rocket.state = SpriteState(data["state"])
        rocket.direction = Direction[data["direction"]]
        rocket.move_interval_ticks = data.get("move_interval_ticks", float("inf"))
        rocket.ticks_since_last_move = data.get("ticks_since_last_move", 0)
        rocket.x = float(data["x"])
        rocket.y = float(data["y"])
        rocket.mice_collected = data.get("mice_collected", 0)
        return rocket

    def __str__(self) -> str:
        return f"Rocket({self.sprite_id}) at ({self.x:.1f}, {self.y:.1f}) tile ({self.tile_x}, {self.tile_y}) [{self.mice_collected} mice]"


class Spawner(Sprite):
    def __init__(self, x: int, y: int, sprite_id: str):
        super().__init__(x, y, sprite_id)
        self.move_interval_ticks = float("inf")  # Spawners don't move
        self.spawn_interval_ticks = 120  # Spawn every 2 seconds at 60 ticks/sec

        # Randomize initial spawn timing for visual variety (uses global random state)
        self.ticks_since_last_spawn = random.randint(0, self.spawn_interval_ticks - 1)

        self.spawn_direction = Direction.RIGHT
        self.spawn_count = 0  # Track number of spawns for cat timing

    def get_sprite_type(self) -> SpriteType:
        return SpriteType.SPAWNER

    def tick(self) -> None:
        super().tick()
        self.ticks_since_last_spawn += 1

    def can_spawn_this_tick(self, bonus_state=None) -> bool:
        interval = self.get_effective_spawn_interval(bonus_state)
        return self.ticks_since_last_spawn >= interval

    def get_effective_spawn_interval(self, bonus_state=None) -> int:
        """Get the spawn interval based on current bonus mode"""
        if bonus_state and bonus_state.is_active():
            if bonus_state.mode == BonusMode.MOUSE_MANIA:
                # Mouse Mania: 4 spawns per second = 1 spawn every 15 ticks (at 60 ticks/sec)
                return 15
            elif bonus_state.mode == BonusMode.CAT_MANIA:
                # Cat Mania: spawn every 120 ticks (2 seconds)
                return 120

        # Normal spawn interval
        return self.spawn_interval_ticks

    def reset_spawn_timer(self) -> None:
        self.ticks_since_last_spawn = 0

    def set_spawn_direction(self, direction: Direction) -> None:
        self.spawn_direction = direction

    def get_actual_spawn_type(self, sprite_manager, bonus_state=None) -> SpriteType:
        """Get the type to spawn based on sophisticated spawning rules and bonus modes"""
        self.spawn_count += 1

        # Check for bonus mode overrides
        if bonus_state and bonus_state.is_active():
            if bonus_state.mode == BonusMode.MOUSE_MANIA:
                # During Mouse Mania: only mice spawn (normal and gold, no cats or bonus)
                # 1/32 chance for gold mouse, but only if none exists on board
                if random.randint(1, 32) == 1:
                    existing_gold = sprite_manager.get_sprites_by_type(
                        SpriteType.GOLD_MOUSE
                    )
                    active_gold = [s for s in existing_gold if s.is_active()]
                    if not active_gold:
                        return SpriteType.GOLD_MOUSE

                # Default to regular mouse during Mouse Mania
                return SpriteType.MOUSE
            elif bonus_state.mode == BonusMode.CAT_MANIA:
                # During Cat Mania: only cats spawn
                return SpriteType.CAT

        # Normal spawning rules (when no bonus mode is active)
        # Every 5th spawn is a cat, but only if under the 16 cat limit
        if self.spawn_count % 5 == 0:
            existing_cats = sprite_manager.get_sprites_by_type(SpriteType.CAT)
            active_cats = [s for s in existing_cats if s.is_active()]
            if len(active_cats) < 16:
                return SpriteType.CAT
            # If cat limit reached, spawn a regular mouse instead
            return SpriteType.MOUSE

        # For mice, check for special types with restrictions
        # 1/32 chance for gold mouse, but only if none exists on board
        if random.randint(1, 32) == 1:
            existing_gold = sprite_manager.get_sprites_by_type(SpriteType.GOLD_MOUSE)
            active_gold = [s for s in existing_gold if s.is_active()]
            if not active_gold:
                return SpriteType.GOLD_MOUSE

        # 1/12 chance for bonus mouse, but only if none exists on board
        if random.randint(1, 12) == 1:
            existing_bonus = sprite_manager.get_sprites_by_type(SpriteType.BONUS_MOUSE)
            active_bonus = [s for s in existing_bonus if s.is_active()]
            if not active_bonus:
                return SpriteType.BONUS_MOUSE

        # Default to regular mouse
        return SpriteType.MOUSE

    def get_spawn_position(self) -> Tuple[float, float]:
        return (self.x + self.spawn_direction.dx, self.y + self.spawn_direction.dy)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "spawn_interval_ticks": self.spawn_interval_ticks,
                "ticks_since_last_spawn": self.ticks_since_last_spawn,
                "spawn_direction": self.spawn_direction.name,
                "spawn_count": self.spawn_count,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spawner":
        # For deserialization, create without randomization to preserve exact state
        spawner = cls.__new__(cls)  # Create without calling __init__
        Sprite.__init__(spawner, int(data["x"]), int(data["y"]), data["sprite_id"])

        # Set spawner-specific attributes
        spawner.move_interval_ticks = float("inf")
        spawner.spawn_interval_ticks = data.get("spawn_interval_ticks", 120)
        spawner.ticks_since_last_spawn = data.get("ticks_since_last_spawn", 0)
        spawner.spawn_direction = Direction[data.get("spawn_direction", "RIGHT")]
        spawner.spawn_count = data.get("spawn_count", 0)

        # Override with saved state
        spawner.state = SpriteState(data["state"])
        spawner.direction = Direction[data["direction"]]
        spawner.move_interval_ticks = data.get("move_interval_ticks", float("inf"))
        spawner.ticks_since_last_move = data.get("ticks_since_last_move", 0)
        spawner.x = float(data["x"])
        spawner.y = float(data["y"])
        spawner.movement_direction = Direction[
            data.get("movement_direction", data["direction"])
        ]

        return spawner

    def __str__(self) -> str:
        return f"Spawner({self.sprite_id}) at ({self.x:.1f}, {self.y:.1f}) tile ({self.tile_x}, {self.tile_y}) spawning mixed types"


class SpriteManager:
    def __init__(self):
        self.sprites: Dict[str, Sprite] = {}
        self._next_id = 0

    def add_sprite(self, sprite: Sprite) -> None:
        self.sprites[sprite.sprite_id] = sprite

    def remove_sprite(self, sprite_id: str) -> bool:
        if sprite_id in self.sprites:
            del self.sprites[sprite_id]
            return True
        return False

    def get_sprite(self, sprite_id: str) -> Optional[Sprite]:
        return self.sprites.get(sprite_id)

    def get_sprites_by_type(self, sprite_type: SpriteType) -> list[Sprite]:
        return [
            sprite
            for sprite in self.sprites.values()
            if sprite.get_sprite_type() == sprite_type
        ]

    def get_active_sprites(self) -> list[Sprite]:
        return [sprite for sprite in self.sprites.values() if sprite.is_active()]

    def get_sprites_at_tile(self, tile_x: int, tile_y: int) -> list[Sprite]:
        return [
            sprite
            for sprite in self.sprites.values()
            if sprite.tile_position == (tile_x, tile_y)
        ]

    def get_active_sprites_at_tile(self, tile_x: int, tile_y: int) -> list[Sprite]:
        return [
            sprite
            for sprite in self.sprites.values()
            if sprite.tile_position == (tile_x, tile_y) and sprite.is_active()
        ]

    def get_sprites_at_position(self, x: int, y: int) -> list[Sprite]:
        """Legacy method - delegates to tile-based lookup"""
        return self.get_sprites_at_tile(x, y)

    def get_active_sprites_at_position(self, x: int, y: int) -> list[Sprite]:
        """Legacy method - delegates to tile-based lookup"""
        return self.get_active_sprites_at_tile(x, y)

    def create_mouse(self, x: int, y: int, sprite_id: Optional[str] = None) -> Mouse:
        if sprite_id is None:
            sprite_id = f"mouse_{self._next_id}"
            self._next_id += 1

        mouse = Mouse(x, y, sprite_id)
        self.add_sprite(mouse)
        return mouse

    def create_gold_mouse(
        self, x: int, y: int, sprite_id: Optional[str] = None
    ) -> GoldMouse:
        if sprite_id is None:
            sprite_id = f"gold_mouse_{self._next_id}"
            self._next_id += 1

        gold_mouse = GoldMouse(x, y, sprite_id)
        self.add_sprite(gold_mouse)
        return gold_mouse

    def create_bonus_mouse(
        self, x: int, y: int, sprite_id: Optional[str] = None
    ) -> BonusMouse:
        if sprite_id is None:
            sprite_id = f"bonus_mouse_{self._next_id}"
            self._next_id += 1

        bonus_mouse = BonusMouse(x, y, sprite_id)
        self.add_sprite(bonus_mouse)
        return bonus_mouse

    def create_cat(self, x: int, y: int, sprite_id: Optional[str] = None) -> Cat:
        if sprite_id is None:
            sprite_id = f"cat_{self._next_id}"
            self._next_id += 1

        cat = Cat(x, y, sprite_id)
        self.add_sprite(cat)
        return cat

    def create_rocket(self, x: int, y: int, sprite_id: Optional[str] = None) -> Rocket:
        if sprite_id is None:
            sprite_id = f"rocket_{self._next_id}"
            self._next_id += 1

        rocket = Rocket(x, y, sprite_id)
        self.add_sprite(rocket)
        return rocket

    def create_spawner(
        self,
        x: int,
        y: int,
        sprite_id: Optional[str] = None,
    ) -> "Spawner":
        if sprite_id is None:
            sprite_id = f"spawner_{self._next_id}"
            self._next_id += 1

        spawner = Spawner(x, y, sprite_id)
        self.add_sprite(spawner)
        return spawner

    def clear(self) -> None:
        self.sprites.clear()
        self._next_id = 0

    def copy(self) -> "SpriteManager":
        new_manager = SpriteManager()
        new_manager._next_id = self._next_id
        for sprite in self.sprites.values():
            new_manager.add_sprite(sprite.copy())
        return new_manager

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sprites": {
                sprite_id: sprite.to_dict()
                for sprite_id, sprite in self.sprites.items()
            },
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpriteManager":
        manager = cls()
        manager._next_id = data.get("next_id", 0)

        for sprite_id, sprite_data in data.get("sprites", {}).items():
            sprite = Sprite.from_dict(sprite_data)
            manager.add_sprite(sprite)

        return manager

    def __len__(self) -> int:
        return len(self.sprites)

    def __iter__(self):
        return iter(self.sprites.values())
