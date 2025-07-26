import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .board import Board, Direction
from .movement import MovementEngine
from .sprites import (BonusMode, BonusState, Cat, Mouse, Rocket, SpriteManager,
                      SpriteState, SpriteType)


class GameResult(Enum):
    ONGOING = "ongoing"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class GameEvent:
    def __init__(self, event_type: str, data: Dict[str, Any], step: int):
        self.event_type = event_type
        self.data = data
        self.step = step
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "step": self.step,
            "timestamp": self.timestamp,
        }


class GameEngine:
    def __init__(
        self,
        board: Board,
        sprite_manager: SpriteManager,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        self.board = board
        self.sprite_manager = sprite_manager
        self.movement_engine = MovementEngine(board, sprite_manager)
        self.max_steps = max_steps
        self.current_step = 0
        self.current_tick = 0  # Track ticks for 60 ticks/second system
        self.events: List[GameEvent] = []
        self.result = GameResult.ONGOING
        self.step_callbacks: List[Callable[["GameEngine"], None]] = []
        self.bonus_state = BonusState()

        # Set random seed for deterministic gameplay
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        random.seed(self.seed)

        # Store initial state for reset functionality
        self.initial_board = board.copy()
        self.initial_sprite_manager = sprite_manager.copy()

    def add_step_callback(self, callback: Callable[["GameEngine"], None]) -> None:
        self.step_callbacks.append(callback)

    def remove_step_callback(self, callback: Callable[["GameEngine"], None]) -> None:
        if callback in self.step_callbacks:
            self.step_callbacks.remove(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        event = GameEvent(event_type, data, self.current_step)
        self.events.append(event)

    def _check_win_condition(self) -> bool:
        from .sprites import SpriteType

        mice = self.sprite_manager.get_sprites_by_type(SpriteType.MOUSE)
        if not mice:
            return True

        active_mice = [mouse for mouse in mice if mouse.state == SpriteState.ACTIVE]
        return len(active_mice) == 0

    def _check_lose_condition(self) -> bool:
        from .sprites import SpriteType

        cats = self.sprite_manager.get_sprites_by_type(SpriteType.CAT)
        mice = self.sprite_manager.get_sprites_by_type(SpriteType.MOUSE)

        if not cats or not mice:
            return False

        captured_mice = [mouse for mouse in mice if mouse.state == SpriteState.CAPTURED]
        total_mice = len(mice)

        return len(captured_mice) > total_mice // 2

    def place_arrow(self, x: int, y: int, direction: Direction) -> bool:
        if self.result != GameResult.ONGOING:
            return False

        # Check if there are stationary sprites (spawners, rockets) at this position
        # Moving sprites (mice, cats) don't block arrow placement
        sprites_at_position = self.sprite_manager.get_sprites_at_position(x, y)
        for sprite in sprites_at_position:
            sprite_type = sprite.get_sprite_type()
            if sprite_type in [SpriteType.SPAWNER, SpriteType.ROCKET]:
                return False

        success = self.board.place_arrow(x, y, direction)
        if success:
            self._emit_event(
                "arrow_placed", {"x": x, "y": y, "direction": direction.name}
            )

            # Check if an old arrow was automatically removed
            removed_arrow_pos = self.board.get_last_removed_arrow()
            if removed_arrow_pos:
                self._emit_event(
                    "arrow_auto_removed",
                    {"x": removed_arrow_pos[0], "y": removed_arrow_pos[1]},
                )

        return success

    def remove_arrow(self, x: int, y: int) -> bool:
        if self.result != GameResult.ONGOING:
            return False

        success = self.board.remove_arrow(x, y)
        if success:
            self._emit_event("arrow_removed", {"x": x, "y": y})

        return success

    def step(self) -> Dict[str, Any]:
        if self.result != GameResult.ONGOING:
            return self.get_step_result()

        self.current_step += 1
        self.current_tick += 1

        # Create bonus callback for triggering bonus rounds
        def bonus_callback(bonus_mouse, rocket):
            # Randomly select bonus mode from available modes
            available_modes = [
                BonusMode.MOUSE_MANIA,
                BonusMode.CAT_MANIA,
                BonusMode.SPEED_UP,
                BonusMode.SLOW_DOWN,
            ]
            bonus_mode = random.choice(available_modes)

            # Start the selected bonus mode for 7 seconds
            self.bonus_state.start_bonus(bonus_mode, 7.0)
            self._emit_event(
                "bonus_round_started",
                {
                    "bonus_mode": bonus_mode.value,
                    "duration_seconds": 7.0,
                    "triggered_by_mouse": bonus_mouse.sprite_id,
                    "rocket_id": rocket.sprite_id,
                    "rocket_position": rocket.tile_position,
                },
            )

        # Tick the bonus state to handle timing
        was_active = self.bonus_state.is_active()
        self.bonus_state.tick()
        if was_active and not self.bonus_state.is_active():
            self._emit_event(
                "bonus_round_ended",
                {"bonus_mode": self.bonus_state.previous_mode.value},
            )

        step_result = self.movement_engine.simulate_step(
            bonus_callback, self.bonus_state
        )

        if step_result["movements"]:
            self._emit_event("sprites_moved", step_result["movements"])

        if step_result["collisions"]:
            self._emit_event(
                "collisions_occurred", {"collisions": step_result["collisions"]}
            )

        if step_result["hole_falls"]:
            self._emit_event(
                "sprites_fell", {"fallen_sprites": step_result["hole_falls"]}
            )

        if step_result["spawns"]:
            self._emit_event(
                "sprites_spawned", {"spawned_sprites": step_result["spawns"]}
            )

        if self.current_step >= self.max_steps:
            self.result = GameResult.TIMEOUT
            self._emit_event(
                "game_timeout", {"step": self.current_step, "tick": self.current_tick}
            )

        for callback in self.step_callbacks:
            callback(self)

        return self.get_step_result()

    def get_step_result(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "tick": self.current_tick,
            "result": self.result.value,
            "board_state": self.get_board_state(),
            "sprite_states": self.get_sprite_states(),
            "game_stats": self.get_game_stats(),
            "bonus_state": self.bonus_state.to_dict(),
        }

    def get_board_state(self) -> Dict[str, Any]:
        return {
            "grid": self.board.grid.tolist(),
            "arrows": {
                f"{x},{y}": direction.name
                for (x, y), direction in self.board.arrows.items()
            },
        }

    def get_sprite_states(self) -> Dict[str, Any]:
        return {
            sprite_id: {
                "type": sprite.get_sprite_type().value,
                "position": sprite.position,
                "state": sprite.state.value,
                "direction": sprite.direction.name,
                "move_progress": (
                    min(sprite.ticks_since_last_move / sprite.move_interval_ticks, 1.0)
                    if sprite.move_interval_ticks != float("inf")
                    else 0.0
                ),
            }
            for sprite_id, sprite in self.sprite_manager.sprites.items()
        }

    def get_game_stats(self) -> Dict[str, Any]:
        from .sprites import SpriteType

        mice = self.sprite_manager.get_sprites_by_type(SpriteType.MOUSE)
        cats = self.sprite_manager.get_sprites_by_type(SpriteType.CAT)
        rockets = self.sprite_manager.get_sprites_by_type(SpriteType.ROCKET)

        mice_stats = {
            "total": len(mice),
            "active": len([m for m in mice if m.state == SpriteState.ACTIVE]),
            "captured": len([m for m in mice if m.state == SpriteState.CAPTURED]),
            "escaped": len([m for m in mice if m.state == SpriteState.ESCAPED]),
        }

        cat_stats = {
            "total": len(cats),
            "active": len([c for c in cats if c.state == SpriteState.ACTIVE]),
        }

        rocket_stats = {
            "total": len(rockets),
            "mice_collected": sum(r.mice_collected for r in rockets),
        }

        return {
            "mice": mice_stats,
            "cats": cat_stats,
            "rockets": rocket_stats,
            "arrows_placed": len(self.board.arrows),
        }

    def run_simulation(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        if max_steps is not None:
            original_max = self.max_steps
            self.max_steps = min(self.max_steps, self.current_step + max_steps)

        while self.result == GameResult.ONGOING:
            self.step()

        if max_steps is not None:
            self.max_steps = original_max

        return {
            "final_result": self.result.value,
            "total_steps": self.current_step,
            "events": [event.to_dict() for event in self.events],
            "final_state": self.get_step_result(),
        }

    def reset(self) -> None:
        self.current_step = 0
        self.current_tick = 0
        self.events.clear()
        self.result = GameResult.ONGOING

        # Restore initial board and sprite state
        self.board = self.initial_board.copy()
        self.sprite_manager = self.initial_sprite_manager.copy()

        # Recreate movement engine with reset state
        self.movement_engine = MovementEngine(self.board, self.sprite_manager)

        # Reset bonus state
        self.bonus_state = BonusState()

    def copy(self) -> "GameEngine":
        new_board = self.board.copy()
        new_sprite_manager = self.sprite_manager.copy()
        new_engine = GameEngine(new_board, new_sprite_manager, self.max_steps)
        new_engine.current_step = self.current_step
        new_engine.result = self.result
        return new_engine

    def get_valid_arrow_positions(self) -> List[Tuple[int, int]]:
        valid_positions = []
        for y in range(self.board.height):
            for x in range(self.board.width):
                if self.board.is_walkable(x, y) and not self.board.has_arrow(x, y):
                    # Check if there are stationary sprites (spawners, rockets) at this position
                    # Moving sprites (mice, cats) don't block arrow placement
                    sprites_here = self.sprite_manager.get_sprites_at_position(x, y)
                    blocked = False
                    for sprite in sprites_here:
                        if sprite.get_sprite_type() in [
                            SpriteType.SPAWNER,
                            SpriteType.ROCKET,
                        ]:
                            blocked = True
                            break
                    if not blocked:
                        valid_positions.append((x, y))
        return valid_positions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "board": self.board.to_dict(),
            "sprite_manager": self.sprite_manager.to_dict(),
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "current_tick": self.current_tick,
            "result": self.result.value,
            "bonus_state": self.bonus_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameEngine":
        board = Board.from_dict(data["board"])
        sprite_manager = SpriteManager.from_dict(data["sprite_manager"])

        engine = cls(board, sprite_manager, data["max_steps"], data.get("seed"))
        engine.current_step = data["current_step"]
        engine.current_tick = data.get("current_tick", 0)
        engine.result = GameResult(data["result"])

        # Restore bonus state if present
        if "bonus_state" in data:
            engine.bonus_state = BonusState.from_dict(data["bonus_state"])

        for event_data in data["events"]:
            event = GameEvent(
                event_data["event_type"], event_data["data"], event_data["step"]
            )
            event.timestamp = event_data["timestamp"]
            engine.events.append(event)

        return engine
