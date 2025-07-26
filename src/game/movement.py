from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .board import Board, CellType, Direction
from .sprites import Cat, Mouse, Sprite, SpriteManager, SpriteState, SpriteType


class PathFinder:
    @staticmethod
    def find_path(
        board: Board, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        if not board.is_walkable(*start) or not board.is_walkable(*goal):
            return None

        if start == goal:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            for nx, ny in board.get_walkable_neighbors(x, y):
                if (nx, ny) == goal:
                    return path + [(nx, ny)]

                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    @staticmethod
    def get_next_position(
        board: Board, sprite: Sprite, bonus_state=None
    ) -> Tuple[float, float]:
        """Calculate the next fractional position for smooth movement"""
        # Calculate movement speed (fraction of a tile per tick)
        if sprite.move_interval_ticks == float("inf"):
            return sprite.position  # Static sprites don't move

        # Use effective movement interval based on bonus modes
        effective_interval = sprite.get_effective_move_interval(bonus_state)
        speed = 1.0 / effective_interval

        # Check if sprite is at tile center and needs direction update
        if PathFinder._is_at_tile_center(sprite):
            PathFinder._check_arrow_interaction(board, sprite)
            PathFinder._check_wall_collision(board, sprite)

        # Move in current direction
        dx = sprite.direction.dx * speed
        dy = sprite.direction.dy * speed

        new_x = sprite.x + dx
        new_y = sprite.y + dy

        return (new_x, new_y)

    @staticmethod
    def _is_at_tile_center(sprite: Sprite, tolerance: float = 0.1) -> bool:
        """Check if sprite is close to the center of its current tile"""
        center_x = float(sprite.tile_x)
        center_y = float(sprite.tile_y)
        distance = ((sprite.x - center_x) ** 2 + (sprite.y - center_y) ** 2) ** 0.5
        return distance < tolerance

    @staticmethod
    def _check_arrow_interaction(board: Board, sprite: Sprite) -> None:
        """Check if there's an arrow at sprite's tile that changes direction"""
        arrow_direction = board.get_arrow_direction(sprite.tile_x, sprite.tile_y)
        if arrow_direction:
            sprite.set_direction(arrow_direction)

    @staticmethod
    def _check_wall_collision(board: Board, sprite: Sprite) -> None:
        """Check if sprite will hit a wall and needs to turn"""
        tile_x, tile_y = sprite.tile_x, sprite.tile_y
        direction = sprite.direction

        # Check if next tile is blocked
        next_tile_x = tile_x + direction.dx
        next_tile_y = tile_y + direction.dy

        # Check if move is blocked by walls or invalid position
        can_move = board.is_walkable(next_tile_x, next_tile_y)
        if can_move:
            # Also check for wall between current and next tile
            if ((tile_x, tile_y), (next_tile_x, next_tile_y)) in board.walls or (
                (next_tile_x, next_tile_y),
                (tile_x, tile_y),
            ) in board.walls:
                can_move = False

        if not can_move:
            PathFinder._handle_wall_collision(board, sprite, direction)

    @staticmethod
    def _handle_wall_collision(
        board: Board, sprite: Sprite, direction: Direction
    ) -> None:
        """Handle wall collision by turning sprite"""
        tile_x, tile_y = sprite.tile_x, sprite.tile_y

        # ChuChu Rocket turning hierarchy: right, then left, then U-turn as fallback
        turn_order = [
            direction.turn_right(),
            direction.turn_left(),
            direction.opposite(),
        ]

        for new_direction in turn_order:
            next_tile_x = tile_x + new_direction.dx
            next_tile_y = tile_y + new_direction.dy

            if board.is_walkable(next_tile_x, next_tile_y):
                # Also check for wall between current and next tile
                if (
                    (tile_x, tile_y),
                    (next_tile_x, next_tile_y),
                ) not in board.walls and (
                    (next_tile_x, next_tile_y),
                    (tile_x, tile_y),
                ) not in board.walls:
                    sprite.set_direction(new_direction)
                    return

        # If no direction works, sprite stays facing current direction (shouldn't happen)


class CollisionDetector:
    @staticmethod
    def detect_sprite_collisions(
        sprite_manager: SpriteManager, collision_distance: float = 0.5
    ) -> List[Tuple[Sprite, Sprite]]:
        """Detect collisions based on distance between sprite positions"""
        collisions = []
        sprites = list(sprite_manager.get_active_sprites())

        for i in range(len(sprites)):
            for j in range(i + 1, len(sprites)):
                sprite1, sprite2 = sprites[i], sprites[j]

                # Calculate distance between sprites
                dx = sprite1.x - sprite2.x
                dy = sprite1.y - sprite2.y
                distance = (dx * dx + dy * dy) ** 0.5

                # Sprites collide if they're within collision distance
                if distance <= collision_distance:
                    collisions.append((sprite1, sprite2))

        return collisions

    @staticmethod
    def handle_collision(sprite1: Sprite, sprite2: Sprite, bonus_callback=None) -> bool:
        from .sprites import Cat, Rocket

        # Check if sprite is a mouse-like (has can_be_captured_by_cat method)
        def is_mouse_like(sprite):
            return hasattr(sprite, "can_be_captured_by_cat")

        # Cat catches mouse
        if isinstance(sprite1, Cat) and is_mouse_like(sprite2):
            return sprite1.capture_mouse(sprite2)
        elif is_mouse_like(sprite1) and isinstance(sprite2, Cat):
            return sprite2.capture_mouse(sprite1)

        # Rocket collects mouse
        elif isinstance(sprite1, Rocket) and is_mouse_like(sprite2):
            return sprite1.collect_mouse(sprite2, bonus_callback)
        elif is_mouse_like(sprite1) and isinstance(sprite2, Rocket):
            return sprite2.collect_mouse(sprite1, bonus_callback)

        # Cat enters rocket (damages it according to ChuChu Rocket rules)
        elif isinstance(sprite1, Cat) and isinstance(sprite2, Rocket):
            return CollisionDetector._handle_cat_rocket_collision(sprite1, sprite2)
        elif isinstance(sprite1, Rocket) and isinstance(sprite2, Cat):
            return CollisionDetector._handle_cat_rocket_collision(sprite2, sprite1)

        return False

    @staticmethod
    def _handle_cat_rocket_collision(cat: "Cat", rocket: "Rocket") -> bool:
        """Handle cat entering rocket - reduces mice count by 1/3 and removes cat"""
        if cat.state == SpriteState.ACTIVE and rocket.state == SpriteState.ACTIVE:
            # Reduce rocket's mice by 1/3 (rounded down)
            mice_lost = rocket.mice_collected // 3
            rocket.mice_collected = max(0, rocket.mice_collected - mice_lost)

            # Remove the cat from play
            cat.set_state(SpriteState.CAPTURED)
            return True

        return False

    @staticmethod
    def check_hole_falls(board: Board, sprite_manager: SpriteManager) -> List[Sprite]:
        fallen_sprites = []

        for sprite in sprite_manager.get_active_sprites():
            tile_x, tile_y = sprite.tile_x, sprite.tile_y
            if board.get_cell_type(tile_x, tile_y) == CellType.HOLE:
                sprite.set_state(SpriteState.CAPTURED)
                fallen_sprites.append(sprite)

        return fallen_sprites


class MovementEngine:
    def __init__(self, board: Board, sprite_manager: SpriteManager):
        self.board = board
        self.sprite_manager = sprite_manager
        self.path_finder = PathFinder()
        self.collision_detector = CollisionDetector()

    def move_sprite(self, sprite: Sprite, bonus_state=None) -> bool:
        if not sprite.is_active():
            return False

        # For fractional movement, sprites move every tick (no interval checking)
        old_position = sprite.position
        new_position = self.path_finder.get_next_position(
            self.board, sprite, bonus_state
        )

        # Update position if it changed
        if new_position != old_position:
            sprite.position = new_position
            return True

        return False

    def move_all_sprites(self, bonus_state=None) -> Dict[str, Tuple[float, float]]:
        movements = {}

        # First tick all sprites (for spawners and interval timing)
        for sprite in self.sprite_manager.get_active_sprites():
            sprite.tick()

        # Move all active sprites (fractional movement happens every tick)
        for sprite in self.sprite_manager.get_active_sprites():
            old_position = sprite.position
            if self.move_sprite(sprite, bonus_state):
                movements[sprite.sprite_id] = sprite.position

        return movements

    def process_collisions(self, bonus_callback=None) -> List[Tuple[str, str]]:
        collision_results = []
        collisions = self.collision_detector.detect_sprite_collisions(
            self.sprite_manager
        )

        for sprite1, sprite2 in collisions:
            if self.collision_detector.handle_collision(
                sprite1, sprite2, bonus_callback
            ):
                collision_results.append((sprite1.sprite_id, sprite2.sprite_id))

        return collision_results

    def process_hole_falls(self) -> List[str]:
        from .sprites import SpriteState

        fallen_sprites = self.collision_detector.check_hole_falls(
            self.board, self.sprite_manager
        )
        return [sprite.sprite_id for sprite in fallen_sprites]

    def process_spawns(self, bonus_state=None) -> List[str]:
        from .sprites import Spawner, SpriteType

        spawned_sprites = []

        spawners = [
            sprite
            for sprite in self.sprite_manager.get_active_sprites()
            if isinstance(sprite, Spawner)
        ]

        for spawner in spawners:
            if spawner.can_spawn_this_tick(bonus_state):
                spawn_x, spawn_y = spawner.get_spawn_position()
                spawn_tile_x, spawn_tile_y = int(spawn_x), int(spawn_y)

                # Check if spawn position is valid and empty
                if self.board.is_walkable(
                    spawn_tile_x, spawn_tile_y
                ) and not self.sprite_manager.get_active_sprites_at_tile(
                    spawn_tile_x, spawn_tile_y
                ):

                    # Get the actual type to spawn based on sophisticated rules
                    actual_spawn_type = spawner.get_actual_spawn_type(
                        self.sprite_manager, bonus_state
                    )

                    # Create new sprite based on actual spawn type
                    if actual_spawn_type == SpriteType.MOUSE:
                        new_sprite = self.sprite_manager.create_mouse(
                            spawn_tile_x, spawn_tile_y
                        )
                    elif actual_spawn_type == SpriteType.GOLD_MOUSE:
                        new_sprite = self.sprite_manager.create_gold_mouse(
                            spawn_tile_x, spawn_tile_y
                        )
                    elif actual_spawn_type == SpriteType.BONUS_MOUSE:
                        new_sprite = self.sprite_manager.create_bonus_mouse(
                            spawn_tile_x, spawn_tile_y
                        )
                    elif actual_spawn_type == SpriteType.CAT:
                        new_sprite = self.sprite_manager.create_cat(
                            spawn_tile_x, spawn_tile_y
                        )
                    else:
                        continue  # Unknown spawn type, skip

                    new_sprite.set_direction(spawner.spawn_direction)
                    spawned_sprites.append(new_sprite.sprite_id)

                    spawner.reset_spawn_timer()

        return spawned_sprites

    def simulate_step(self, bonus_callback=None, bonus_state=None) -> Dict[str, any]:
        movements = self.move_all_sprites(bonus_state)
        collisions = self.process_collisions(bonus_callback)
        hole_falls = self.process_hole_falls()
        spawns = self.process_spawns(bonus_state)

        return {
            "movements": movements,
            "collisions": collisions,
            "hole_falls": hole_falls,
            "spawns": spawns,
        }

    def can_sprite_move_to(self, sprite: Sprite, target_x: int, target_y: int) -> bool:
        if not self.board.is_walkable(target_x, target_y):
            return False

        path = self.path_finder.find_path(
            self.board, sprite.tile_position, (target_x, target_y)
        )
        return path is not None

    def get_reachable_positions(
        self, sprite: Sprite, max_distance: Optional[int] = None
    ) -> Set[Tuple[int, int]]:
        reachable = set()
        queue = deque([(sprite.tile_position, 0)])
        visited = {sprite.tile_position}

        while queue:
            (x, y), distance = queue.popleft()
            reachable.add((x, y))

            if max_distance is not None and distance >= max_distance:
                continue

            for nx, ny in self.board.get_walkable_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), distance + 1))

        return reachable

    def get_sprite_interactions(self, sprite: Sprite) -> List[Sprite]:
        interactions = []
        sprites_at_tile = self.sprite_manager.get_active_sprites_at_tile(
            sprite.tile_x, sprite.tile_y
        )

        for other_sprite in sprites_at_tile:
            if other_sprite.sprite_id != sprite.sprite_id:
                interactions.append(other_sprite)

        return interactions
