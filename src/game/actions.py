from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .board import Direction
from .engine import GameEngine


class ActionType(Enum):
    PLACE_ARROW = "place_arrow"
    REMOVE_ARROW = "remove_arrow"
    WAIT = "wait"


class ActionResult(Enum):
    SUCCESS = "success"
    INVALID_POSITION = "invalid_position"
    POSITION_OCCUPIED = "position_occupied"
    NO_ARROW_TO_REMOVE = "no_arrow_to_remove"
    GAME_OVER = "game_over"
    INVALID_ACTION = "invalid_action"


class Action(ABC):
    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @abstractmethod
    def is_valid(self, engine: GameEngine) -> bool:
        pass

    @abstractmethod
    def execute(self, engine: GameEngine) -> ActionResult:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        pass

    def __str__(self) -> str:
        return f"{self.action_type.value}"


class PlaceArrowAction(Action):
    def __init__(self, x: int, y: int, direction: Direction):
        super().__init__(ActionType.PLACE_ARROW)
        self.x = x
        self.y = y
        self.direction = direction

    def is_valid(self, engine: GameEngine) -> bool:
        if engine.result.value != "ongoing":
            return False

        if not engine.board.is_valid_position(self.x, self.y):
            return False

        if not engine.board.is_walkable(self.x, self.y):
            return False

        if engine.board.has_arrow(self.x, self.y):
            return False

        sprites_at_position = engine.sprite_manager.get_sprites_at_position(
            self.x, self.y
        )
        return len(sprites_at_position) == 0

    def execute(self, engine: GameEngine) -> ActionResult:
        if not self.is_valid(engine):
            if engine.result.value != "ongoing":
                return ActionResult.GAME_OVER
            elif not engine.board.is_valid_position(
                self.x, self.y
            ) or not engine.board.is_walkable(self.x, self.y):
                return ActionResult.INVALID_POSITION
            elif engine.board.has_arrow(
                self.x, self.y
            ) or engine.sprite_manager.get_sprites_at_position(self.x, self.y):
                return ActionResult.POSITION_OCCUPIED
            else:
                return ActionResult.INVALID_ACTION

        success = engine.place_arrow(self.x, self.y, self.direction)
        return ActionResult.SUCCESS if success else ActionResult.INVALID_ACTION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "x": self.x,
            "y": self.y,
            "direction": self.direction.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaceArrowAction":
        return cls(x=data["x"], y=data["y"], direction=Direction[data["direction"]])

    def __str__(self) -> str:
        return f"PlaceArrow({self.x}, {self.y}, {self.direction.name})"


class RemoveArrowAction(Action):
    def __init__(self, x: int, y: int):
        super().__init__(ActionType.REMOVE_ARROW)
        self.x = x
        self.y = y

    def is_valid(self, engine: GameEngine) -> bool:
        if engine.result.value != "ongoing":
            return False

        if not engine.board.is_valid_position(self.x, self.y):
            return False

        return engine.board.has_arrow(self.x, self.y)

    def execute(self, engine: GameEngine) -> ActionResult:
        if not self.is_valid(engine):
            if engine.result.value != "ongoing":
                return ActionResult.GAME_OVER
            elif not engine.board.is_valid_position(self.x, self.y):
                return ActionResult.INVALID_POSITION
            elif not engine.board.has_arrow(self.x, self.y):
                return ActionResult.NO_ARROW_TO_REMOVE
            else:
                return ActionResult.INVALID_ACTION

        success = engine.remove_arrow(self.x, self.y)
        return ActionResult.SUCCESS if success else ActionResult.INVALID_ACTION

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type.value, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoveArrowAction":
        return cls(x=data["x"], y=data["y"])

    def __str__(self) -> str:
        return f"RemoveArrow({self.x}, {self.y})"


class WaitAction(Action):
    def __init__(self):
        super().__init__(ActionType.WAIT)

    def is_valid(self, engine: GameEngine) -> bool:
        return engine.result.value == "ongoing"

    def execute(self, engine: GameEngine) -> ActionResult:
        if not self.is_valid(engine):
            return ActionResult.GAME_OVER

        return ActionResult.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaitAction":
        return cls()

    def __str__(self) -> str:
        return "Wait"


class ActionValidator:
    @staticmethod
    def get_valid_actions(engine: GameEngine) -> List[Action]:
        if engine.result.value != "ongoing":
            return []

        valid_actions = []

        valid_actions.append(WaitAction())

        for y in range(engine.board.height):
            for x in range(engine.board.width):
                if engine.board.is_walkable(x, y) and not engine.board.has_arrow(x, y):
                    sprites_at_position = engine.sprite_manager.get_sprites_at_position(
                        x, y
                    )
                    if len(sprites_at_position) == 0:
                        for direction in Direction:
                            action = PlaceArrowAction(x, y, direction)
                            if action.is_valid(engine):
                                valid_actions.append(action)

                if engine.board.has_arrow(x, y):
                    action = RemoveArrowAction(x, y)
                    if action.is_valid(engine):
                        valid_actions.append(action)

        return valid_actions

    @staticmethod
    def get_valid_arrow_placements(engine: GameEngine) -> List[Tuple[int, int]]:
        valid_positions = []

        for y in range(engine.board.height):
            for x in range(engine.board.width):
                if engine.board.is_walkable(x, y) and not engine.board.has_arrow(x, y):
                    sprites_at_position = engine.sprite_manager.get_sprites_at_position(
                        x, y
                    )
                    if len(sprites_at_position) == 0:
                        valid_positions.append((x, y))

        return valid_positions

    @staticmethod
    def get_removable_arrows(engine: GameEngine) -> List[Tuple[int, int]]:
        return list(engine.board.arrows.keys())

    @staticmethod
    def validate_action_sequence(
        engine: GameEngine, actions: List[Action]
    ) -> List[Tuple[Action, ActionResult]]:
        results = []
        engine_copy = engine.copy()

        for action in actions:
            result = action.execute(engine_copy)
            results.append((action, result))

            if result != ActionResult.SUCCESS:
                break

            engine_copy.step()

        return results


class ActionHistory:
    def __init__(self):
        self.actions: List[Tuple[Action, ActionResult, int]] = []

    def add_action(self, action: Action, result: ActionResult, step: int) -> None:
        self.actions.append((action, result, step))

    def get_actions_by_type(
        self, action_type: ActionType
    ) -> List[Tuple[Action, ActionResult, int]]:
        return [
            (action, result, step)
            for action, result, step in self.actions
            if action.action_type == action_type
        ]

    def get_successful_actions(self) -> List[Tuple[Action, ActionResult, int]]:
        return [
            (action, result, step)
            for action, result, step in self.actions
            if result == ActionResult.SUCCESS
        ]

    def get_failed_actions(self) -> List[Tuple[Action, ActionResult, int]]:
        return [
            (action, result, step)
            for action, result, step in self.actions
            if result != ActionResult.SUCCESS
        ]

    def get_action_statistics(self) -> Dict[str, Any]:
        total_actions = len(self.actions)
        if total_actions == 0:
            return {"total": 0}

        successful = len(self.get_successful_actions())
        failed = len(self.get_failed_actions())

        action_type_counts = {}
        result_type_counts = {}

        for action, result, _ in self.actions:
            action_type = action.action_type.value
            result_type = result.value

            action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
            result_type_counts[result_type] = result_type_counts.get(result_type, 0) + 1

        return {
            "total": total_actions,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_actions,
            "action_types": action_type_counts,
            "result_types": result_type_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [
                {"action": action.to_dict(), "result": result.value, "step": step}
                for action, result, step in self.actions
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionHistory":
        history = cls()

        for action_data in data["actions"]:
            action_dict = action_data["action"]
            action_type = ActionType(action_dict["action_type"])

            if action_type == ActionType.PLACE_ARROW:
                action = PlaceArrowAction.from_dict(action_dict)
            elif action_type == ActionType.REMOVE_ARROW:
                action = RemoveArrowAction.from_dict(action_dict)
            elif action_type == ActionType.WAIT:
                action = WaitAction.from_dict(action_dict)
            else:
                continue

            result = ActionResult(action_data["result"])
            step = action_data["step"]

            history.add_action(action, result, step)

        return history

    def clear(self) -> None:
        self.actions.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)
