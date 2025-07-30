#!/usr/bin/env python3
"""
Interactive dual-pane visualizer for stepping through puzzles and seeing model activations.

Combines game simulation with real-time model activation analysis.
Left pane: Game board with manual stepping
Right pane: Model activation visualizations that update after each step

Usage:
    # Basic usage
    uv run python scripts/interactive_visualize.py --model model/bc_best.pth

    # Specify difficulty and seed
    uv run python scripts/interactive_visualize.py --model model/ppo_final.pth --difficulty medium --seed 42

    # Use larger cell size for better visibility
    uv run python scripts/interactive_visualize.py --model model/bc_best.pth --cell-size 50

Controls:
    SPACE - Step game forward
    R - Reset game
    ENTER - Start puzzle (placement → running phase)
    M - Toggle model panel visibility
    C - Cycle through CNN feature channels (8 at a time)
    B - Switch between CNN blocks (1→2→3→4→1...)
    Mouse - Click to place/rotate arrows (left) or remove arrows (right)
    ESC/Q - Quit
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer.interactive_visualizer import InteractiveModelVisualizer


def main():
    parser = argparse.ArgumentParser(description="Interactive model visualization")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (BC or PPO)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Puzzle difficulty to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible puzzles",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run model on (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=40,
        help="Size of each board cell in pixels (default: 40)",
    )

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Generate seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"Generated random seed: {args.seed}")

    try:
        print(f"Starting interactive visualizer...")
        print(f"Model: {args.model}")
        print(f"Difficulty: {args.difficulty}")
        print(f"Seed: {args.seed}")
        print(f"Device: {args.device}")
        print()

        # Create and run visualizer
        visualizer = InteractiveModelVisualizer(
            model_path=args.model,
            difficulty=args.difficulty,
            seed=args.seed,
            device=args.device,
            cell_size=args.cell_size,
        )

        visualizer.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
