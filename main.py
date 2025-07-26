#!/usr/bin/env python3
"""
ChuChu Rocket AI Simulation Engine

A comprehensive simulation engine for ChuChu Rocket puzzle mode,
designed for AI agent development and testing.
"""

import argparse
import sys
from pathlib import Path

from src.game.board import CellType, Direction
from src.game.levels import LevelBuilder
from src.game.visualization import GameVisualizer, HeadlessVisualizer


def create_demo_level():
    """Create a simple demo level for testing."""
    builder = LevelBuilder(12, 9, "Demo Level")

    builder.create_border_walls()

    # Top left
    builder.add_wall(((0, 1), (0, 2)))
    builder.add_wall(((1, 1), (1, 2)))

    # Top right
    builder.add_wall(((10, 1), (10, 2)))
    builder.add_wall(((11, 1), (11, 2)))

    # Bottom left
    builder.add_wall(((0, 7), (0, 8)))
    builder.add_wall(((1, 7), (1, 8)))

    # Top right
    builder.add_wall(((10, 7), (10, 8)))
    builder.add_wall(((11, 7), (11, 8)))

    builder.add_spawner((5, 3), direction=Direction.UP)
    builder.add_spawner((6, 3), direction=Direction.UP)

    builder.add_hole((5, 4))
    builder.add_hole((6, 4))

    builder.add_spawner((5, 5), direction=Direction.DOWN)
    builder.add_spawner((6, 5), direction=Direction.DOWN)

    builder.add_rocket((0, 4))
    builder.add_rocket((11, 4))

    builder.add_initial_arrow((8, 0), Direction.DOWN)
    builder.add_initial_arrow((8, 8), Direction.UP)
    builder.add_initial_arrow((8, 4), Direction.RIGHT)

    level = builder.build()
    return level.create_engine(max_steps=3 * 60 * 60, seed=1337)


def run_visualization(headless=False):
    """Run the game with visualization."""
    engine = create_demo_level()

    if headless:
        visualizer = HeadlessVisualizer(engine)
        print("Running headless simulation...")
        result = visualizer.run_simulation(max_steps=30 * 60, verbose=True)
        print(f"Final result: {result}")
    else:
        try:
            visualizer = GameVisualizer(engine)
            visualizer.run()
        except ImportError:
            print("Pygame not available. Running in headless mode...")
            run_visualization(headless=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ChuChu Rocket AI Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive visualization
  python main.py --headless         # Headless simulation
        """,
    )

    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI visualization"
    )

    args = parser.parse_args()
    run_visualization(headless=args.headless)


if __name__ == "__main__":
    main()
