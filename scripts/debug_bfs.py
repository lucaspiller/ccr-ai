#!/usr/bin/env python3
"""
Debug script for BFS solver - runs in verbose mode with detailed logging.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bfs_solver.solver import BFSSolver
from src.game.board import Direction
from src.game.board_builder import BoardBuilder, BoardConfig
from src.game.engine import GameResult
from src.game.sprites import SpriteState, SpriteType


class VerboseBFSSolver(BFSSolver):
    """BFS solver with verbose logging for debugging."""

    def solve(self, engine):
        """Solve with detailed logging."""
        start_time = time.time()
        print(f"=== BFS Solver Debug Session ===")
        print(f"Initial board state:")
        print(engine.board)
        print(f"Mice positions: {getattr(engine.board, 'mice_positions', [])}")
        print(f"Cat positions: {getattr(engine.board, 'cat_positions', [])}")
        print(f"Arrow budget: {engine.board.max_arrows}")
        print()

        if not engine.puzzle_mode:
            print("ERROR: Engine not in puzzle mode!")
            from src.bfs_solver.solver import BFSResult

            return BFSResult(None, 0, 0, 0, False)

        # Test if puzzle is solvable without arrows first
        print("Testing puzzle without any arrows...")
        test_engine = engine.copy()
        if test_engine.start_game():
            result = self._simulate_to_end(test_engine)
            print(f"No-arrow result: {result}")
            if result == GameResult.SUCCESS:
                print("Puzzle solvable without arrows!")
                from src.bfs_solver.solver import BFSResult

                return BFSResult(
                    solution=[],
                    solution_length=0,
                    nodes_explored=1,
                    time_taken_ms=(time.time() - start_time) * 1000,
                    success=True,
                )
        print()

        # State: (arrow_placements tuple, game state hash)
        initial_state = self._get_state_key(engine)
        print(f"Initial state key: {initial_state[:100]}...")

        # BFS queue: (engine_copy, arrow_placements, depth)
        queue = deque([(engine.copy(), [], 0)])
        visited: Set[str] = {initial_state}
        nodes_explored = 0

        print("Starting BFS exploration...")

        while queue:
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.timeout_ms:
                print(f"TIMEOUT after {elapsed_ms:.1f}ms")
                break

            current_engine, arrow_placements, depth = queue.popleft()
            nodes_explored += 1

            print(f"\n--- Node {nodes_explored} (depth {depth}) ---")
            print(f"Arrows placed: {arrow_placements}")

            # Check depth limit
            if depth >= self.depth_cap:
                print(f"Hit depth cap ({self.depth_cap})")
                continue

            # Try to start the game and simulate
            test_engine = current_engine.copy()
            if not test_engine.start_game():
                print("Failed to start game")
                continue

            print("Starting simulation...")
            # Simulate the game to completion
            result = self._simulate_to_end_verbose(test_engine)
            print(f"Simulation result: {result}")

            if result == GameResult.SUCCESS:
                # Found solution!
                elapsed_ms = (time.time() - start_time) * 1000
                print(f"\n🎉 SOLUTION FOUND! 🎉")
                print(f"Solution: {arrow_placements}")
                print(f"Length: {len(arrow_placements)}")
                print(f"Time: {elapsed_ms:.1f}ms")
                print(f"Nodes explored: {nodes_explored}")

                # Return result directly instead of calling super().solve() again
                from src.bfs_solver.solver import BFSResult

                return BFSResult(
                    solution=arrow_placements,
                    solution_length=len(arrow_placements),
                    nodes_explored=nodes_explored,
                    time_taken_ms=elapsed_ms,
                    success=True,
                )

            # If not solved, try placing more arrows
            if depth < self.depth_cap:
                print(f"Expanding from depth {depth}...")
                old_queue_size = len(queue)
                self._expand_states_verbose(
                    current_engine, arrow_placements, depth, queue, visited
                )
                new_states = len(queue) - old_queue_size
                print(f"Added {new_states} new states to explore")

        # No solution found
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"\n❌ No solution found")
        print(f"Time: {elapsed_ms:.1f}ms")
        print(f"Nodes explored: {nodes_explored}")
        print(f"Final queue size: {len(queue)}")

        # Return failure result directly
        from src.bfs_solver.solver import BFSResult

        return BFSResult(
            solution=None,
            solution_length=0,
            nodes_explored=nodes_explored,
            time_taken_ms=elapsed_ms,
            success=False,
        )

    def _simulate_to_end_verbose(self, engine, max_steps: int = 500):
        """Simulate with logging."""
        print(f"  Starting simulation (max {max_steps} steps)...")

        # Check initial state
        mice = engine.sprite_manager.get_sprites_by_type(SpriteType.MOUSE)
        active_mice = [m for m in mice if m.state == SpriteState.ACTIVE]
        print(f"  Initial: {len(active_mice)} active mice")

        steps = 0
        while engine.result == GameResult.ONGOING and steps < max_steps:
            engine.step()
            steps += 1

            if steps % 50 == 0:
                active_mice = [m for m in mice if m.state == SpriteState.ACTIVE]
                escaped_mice = [m for m in mice if m.state == SpriteState.ESCAPED]
                captured_mice = [m for m in mice if m.state == SpriteState.CAPTURED]
                print(
                    f"  Step {steps}: {len(active_mice)} active, {len(escaped_mice)} escaped, {len(captured_mice)} captured"
                )

        active_mice = [m for m in mice if m.state == SpriteState.ACTIVE]
        escaped_mice = [m for m in mice if m.state == SpriteState.ESCAPED]
        captured_mice = [m for m in mice if m.state == SpriteState.CAPTURED]
        print(
            f"  Final after {steps} steps: {len(active_mice)} active, {len(escaped_mice)} escaped, {len(captured_mice)} captured"
        )

        return engine.result

    def _expand_states_verbose(self, engine, arrow_placements, depth, queue, visited):
        """Expand with logging."""
        # Get all valid positions for arrow placement
        valid_positions = engine.get_valid_arrow_positions()
        print(
            f"  {len(valid_positions)} valid arrow positions: {valid_positions[:10]}{'...' if len(valid_positions) > 10 else ''}"
        )

        # Try placing arrows in positions closest to mice first (heuristic)
        mice_positions = self._get_mice_positions(engine)
        if mice_positions:
            print(f"  Mice at: {mice_positions}")
            valid_positions = self._sort_by_distance_to_mice(
                valid_positions, mice_positions
            )
            print(
                f"  Sorted by distance to mice: {valid_positions[:5]}{'...' if len(valid_positions) > 5 else ''}"
            )

        states_added = 0
        states_duplicate = 0

        # Try each direction for each valid position
        for x, y in valid_positions[:5]:  # Limit for debugging
            for direction in Direction:
                # Try placing this arrow
                new_engine = engine.copy()
                if new_engine.place_arrow(x, y, direction):
                    new_arrow_placements = arrow_placements + [((x, y), direction)]
                    state_key = self._get_state_key(new_engine)

                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((new_engine, new_arrow_placements, depth + 1))
                        states_added += 1
                    else:
                        states_duplicate += 1

        print(
            f"  Added {states_added} new states, skipped {states_duplicate} duplicates"
        )


def main():
    """Run debug BFS solver."""
    parser = argparse.ArgumentParser(description="Debug BFS solver with verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--depth-cap", type=int, default=10, help="Maximum search depth"
    )
    parser.add_argument(
        "--timeout", type=int, default=5000, help="Timeout in milliseconds"
    )

    args = parser.parse_args()

    print(f"Debugging BFS solver with seed {args.seed}")

    # Generate puzzle
    config = BoardConfig(
        board_w=9,
        board_h=9,
        num_walls=30,
        num_mice=5,
        num_rockets=2,
        num_cats=0,
        num_holes=2,
        arrow_budget=3,
    )

    seed = args.seed

    board_builder = BoardBuilder(config, seed=seed)
    level = board_builder.generate_level(f"Debug Puzzle {seed}")
    engine = level.create_engine(max_steps=1000, seed=seed, puzzle_mode=True)

    # Create verbose solver
    solver = VerboseBFSSolver(depth_cap=args.depth_cap, timeout_ms=args.timeout)

    # Solve
    result = solver.solve(engine)

    print(f"\n=== Final Result ===")
    print(f"Success: {result.success}")
    print(f"Solution: {result.solution}")
    print(f"Length: {result.solution_length}")
    print(f"Time: {result.time_taken_ms:.1f}ms")
    print(f"Nodes: {result.nodes_explored}")


if __name__ == "__main__":
    main()
