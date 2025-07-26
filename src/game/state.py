import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .board import Board
from .engine import GameEngine, GameResult
from .sprites import SpriteManager


@dataclass
class GameSnapshot:
    step: int
    board_state: Dict[str, Any]
    sprite_states: Dict[str, Any]
    game_stats: Dict[str, Any]
    result: str
    hash_value: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "board_state": self.board_state,
            "sprite_states": self.sprite_states,
            "game_stats": self.game_stats,
            "result": self.result,
            "hash_value": self.hash_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameSnapshot":
        return cls(
            step=data["step"],
            board_state=data["board_state"],
            sprite_states=data["sprite_states"],
            game_stats=data["game_stats"],
            result=data["result"],
            hash_value=data["hash_value"],
        )


class GameState:
    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.snapshots: List[GameSnapshot] = []
        self.state_history: Dict[str, int] = {}

    def _calculate_state_hash(self) -> str:
        state_data = {
            "board": self.engine.get_board_state(),
            "sprites": self.engine.get_sprite_states(),
        }
        state_json = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_json.encode()).hexdigest()

    def take_snapshot(self) -> GameSnapshot:
        hash_value = self._calculate_state_hash()

        snapshot = GameSnapshot(
            step=self.engine.current_step,
            board_state=self.engine.get_board_state(),
            sprite_states=self.engine.get_sprite_states(),
            game_stats=self.engine.get_game_stats(),
            result=self.engine.result.value,
            hash_value=hash_value,
        )

        self.snapshots.append(snapshot)

        if hash_value in self.state_history:
            self.state_history[hash_value] += 1
        else:
            self.state_history[hash_value] = 1

        return snapshot

    def get_current_snapshot(self) -> GameSnapshot:
        return self.take_snapshot()

    def get_snapshot_at_step(self, step: int) -> Optional[GameSnapshot]:
        for snapshot in self.snapshots:
            if snapshot.step == step:
                return snapshot
        return None

    def has_repeated_state(self) -> bool:
        current_hash = self._calculate_state_hash()
        return self.state_history.get(current_hash, 0) > 1

    def get_state_repetition_count(self) -> int:
        current_hash = self._calculate_state_hash()
        return self.state_history.get(current_hash, 0)

    def detect_cycles(self, min_cycle_length: int = 3) -> Optional[List[GameSnapshot]]:
        if len(self.snapshots) < min_cycle_length * 2:
            return None

        recent_snapshots = self.snapshots[-min_cycle_length * 2 :]

        for cycle_length in range(min_cycle_length, len(recent_snapshots) // 2 + 1):
            cycle_start = len(recent_snapshots) - cycle_length * 2
            if cycle_start < 0:
                continue

            first_cycle = recent_snapshots[cycle_start : cycle_start + cycle_length]
            second_cycle = recent_snapshots[
                cycle_start + cycle_length : cycle_start + cycle_length * 2
            ]

            if len(first_cycle) == len(second_cycle):
                if all(
                    s1.hash_value == s2.hash_value
                    for s1, s2 in zip(first_cycle, second_cycle)
                ):
                    return first_cycle

        return None

    def is_stuck(self, stuck_threshold: int = 10) -> bool:
        if len(self.snapshots) < stuck_threshold:
            return False

        recent_hashes = [s.hash_value for s in self.snapshots[-stuck_threshold:]]
        return len(set(recent_hashes)) <= 2

    def get_progress_metrics(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {}

        first_snapshot = self.snapshots[0]
        current_snapshot = self.snapshots[-1]

        initial_active_mice = first_snapshot.game_stats["mice"]["active"]
        current_active_mice = current_snapshot.game_stats["mice"]["active"]

        initial_arrows = first_snapshot.game_stats["arrows_placed"]
        current_arrows = current_snapshot.game_stats["arrows_placed"]

        return {
            "mice_progress": {
                "initial": initial_active_mice,
                "current": current_active_mice,
                "saved": initial_active_mice - current_active_mice,
                "progress_ratio": (initial_active_mice - current_active_mice)
                / max(initial_active_mice, 1),
            },
            "arrows_used": current_arrows - initial_arrows,
            "steps_taken": current_snapshot.step - first_snapshot.step,
            "unique_states": len(set(s.hash_value for s in self.snapshots)),
            "state_repetitions": sum(
                1 for count in self.state_history.values() if count > 1
            ),
        }

    def export_history(self) -> Dict[str, Any]:
        return {
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "state_history": self.state_history,
            "engine_state": self.engine.to_dict(),
        }

    def import_history(self, data: Dict[str, Any]) -> None:
        self.snapshots = [GameSnapshot.from_dict(s) for s in data["snapshots"]]
        self.state_history = data["state_history"]

        if "engine_state" in data:
            self.engine = GameEngine.from_dict(data["engine_state"])

    def save_to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.export_history(), f, indent=2)

    def load_from_file(self, filename: str) -> None:
        with open(filename, "r") as f:
            data = json.load(f)
        self.import_history(data)

    def clear_history(self) -> None:
        self.snapshots.clear()
        self.state_history.clear()

    def get_state_summary(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {"status": "no_snapshots"}

        current = self.snapshots[-1]
        cycles = self.detect_cycles()

        return {
            "current_step": current.step,
            "current_result": current.result,
            "total_snapshots": len(self.snapshots),
            "unique_states": len(set(s.hash_value for s in self.snapshots)),
            "has_cycles": cycles is not None,
            "cycle_length": len(cycles) if cycles else 0,
            "is_stuck": self.is_stuck(),
            "repetition_count": self.get_state_repetition_count(),
            "progress_metrics": self.get_progress_metrics(),
        }

    def analyze_state_transitions(self) -> Dict[str, Any]:
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data"}

        transitions = []
        for i in range(1, len(self.snapshots)):
            prev_snapshot = self.snapshots[i - 1]
            curr_snapshot = self.snapshots[i]

            transition = {
                "from_step": prev_snapshot.step,
                "to_step": curr_snapshot.step,
                "from_hash": prev_snapshot.hash_value,
                "to_hash": curr_snapshot.hash_value,
                "mice_change": (
                    curr_snapshot.game_stats["mice"]["active"]
                    - prev_snapshot.game_stats["mice"]["active"]
                ),
                "arrows_change": (
                    curr_snapshot.game_stats["arrows_placed"]
                    - prev_snapshot.game_stats["arrows_placed"]
                ),
            }
            transitions.append(transition)

        return {
            "total_transitions": len(transitions),
            "unique_transitions": len(
                set((t["from_hash"], t["to_hash"]) for t in transitions)
            ),
            "transitions": transitions,
        }

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> GameSnapshot:
        return self.snapshots[index]

    def __iter__(self):
        return iter(self.snapshots)
