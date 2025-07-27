"""
BFS solver for finding optimal arrow placements in ChuChu Rocket puzzle mode.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from ..game.board import Direction
from ..game.engine import GameEngine, GameResult
from ..game.sprites import SpriteState, SpriteType


@dataclass
class BFSResult:
    """Result of BFS solving."""

    solution: Optional[List[Tuple[Tuple[int, int], Direction]]]
    solution_length: int
    nodes_explored: int
    time_taken_ms: float
    success: bool


class BFSSolver:
    """BFS solver for ChuChu Rocket puzzle mode."""

    def __init__(self, depth_cap: int = 40, timeout_ms: float = 50):
        """Initialize BFS solver.

        Args:
            depth_cap: Maximum search depth
            timeout_ms: Timeout in milliseconds
        """
        self.depth_cap = depth_cap
        self.timeout_ms = timeout_ms

    def solve(self, engine: GameEngine) -> BFSResult:
        """Find optimal arrow placement solution using BFS.

        Args:
            engine: Game engine in placement phase

        Returns:
            BFSResult with solution if found
        """
        start_time = time.time()

        if not engine.puzzle_mode:
            return BFSResult(None, 0, 0, 0, False)

        # State: (arrow_placements tuple, game state hash)
        initial_state = self._get_state_key(engine)

        # BFS queue: (engine_copy, arrow_placements, depth)
        queue = deque([(engine.copy(), [], 0)])
        visited: Set[str] = {initial_state}
        nodes_explored = 0

        while queue:
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.timeout_ms:
                break

            current_engine, arrow_placements, depth = queue.popleft()
            nodes_explored += 1

            # Check depth limit
            if depth >= self.depth_cap:
                continue

            # Try to start the game and simulate
            test_engine = current_engine.copy()
            if not test_engine.start_game():
                continue

            # Simulate the game to completion
            result = self._simulate_to_end(test_engine)

            if result == GameResult.SUCCESS:
                # Found solution!
                elapsed_ms = (time.time() - start_time) * 1000
                return BFSResult(
                    solution=arrow_placements,
                    solution_length=len(arrow_placements),
                    nodes_explored=nodes_explored,
                    time_taken_ms=elapsed_ms,
                    success=True,
                )

            # If not solved, try placing more arrows
            if depth < self.depth_cap:
                self._expand_states(
                    current_engine, arrow_placements, depth, queue, visited
                )

        # No solution found
        elapsed_ms = (time.time() - start_time) * 1000
        return BFSResult(
            solution=None,
            solution_length=0,
            nodes_explored=nodes_explored,
            time_taken_ms=elapsed_ms,
            success=False,
        )

    def _get_state_key(self, engine: GameEngine) -> str:
        """Get a hashable key representing the current game state."""
        # Create state from arrow placements and sprite positions
        arrows = sorted(engine.board.arrows.items())

        mice_positions = []
        cats_positions = []

        for sprite in engine.sprite_manager.sprites.values():
            sprite_type = sprite.get_sprite_type()
            if sprite_type == SpriteType.MOUSE and sprite.state == SpriteState.ACTIVE:
                mice_positions.append((sprite.x, sprite.y, sprite.direction.name))
            elif sprite_type == SpriteType.CAT and sprite.state == SpriteState.ACTIVE:
                cats_positions.append((sprite.x, sprite.y, sprite.direction.name))

        mice_positions.sort()
        cats_positions.sort()

        return str((arrows, mice_positions, cats_positions))

    def _simulate_to_end(self, engine: GameEngine, max_steps: int = 500) -> GameResult:
        """Simulate the game until completion or timeout."""
        steps = 0
        while engine.result == GameResult.ONGOING and steps < max_steps:
            engine.step()
            steps += 1

        return engine.result

    def _expand_states(
        self,
        engine: GameEngine,
        arrow_placements: List[Tuple[Tuple[int, int], Direction]],
        depth: int,
        queue: deque,
        visited: Set[str],
    ) -> None:
        """Expand states by trying all possible arrow placements."""
        # Get all valid positions for arrow placement
        valid_positions = engine.get_valid_arrow_positions()

        # Try placing arrows in positions closest to mice first (heuristic)
        mice_positions = self._get_mice_positions(engine)
        if mice_positions:
            valid_positions = self._sort_by_distance_to_mice(
                valid_positions, mice_positions
            )

        # Try each direction for each valid position (limit to first 10 for performance)
        for x, y in valid_positions[:10]:
            for direction in Direction:
                # Try placing this arrow
                new_engine = engine.copy()
                if new_engine.place_arrow(x, y, direction):
                    new_arrow_placements = arrow_placements + [((x, y), direction)]
                    state_key = self._get_state_key(new_engine)

                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((new_engine, new_arrow_placements, depth + 1))

    def _get_mice_positions(self, engine: GameEngine) -> List[Tuple[float, float]]:
        """Get positions of all active mice."""
        mice_positions = []
        for sprite in engine.sprite_manager.sprites.values():
            if (
                sprite.get_sprite_type() == SpriteType.MOUSE
                and sprite.state == SpriteState.ACTIVE
            ):
                mice_positions.append((sprite.x, sprite.y))
        return mice_positions

    def _sort_by_distance_to_mice(
        self,
        positions: List[Tuple[int, int]],
        mice_positions: List[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        """Sort positions by minimum distance to any mouse (nearest first)."""

        def min_distance_to_mice(pos):
            x, y = pos
            if not mice_positions:
                return 0
            return min(abs(x - mx) + abs(y - my) for mx, my in mice_positions)

        return sorted(positions, key=min_distance_to_mice)
