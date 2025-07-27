#!/usr/bin/env python3
"""
Generate varied puzzle corpus with randomized board configurations.

This script generates puzzles with varied parameters to create a diverse dataset.
Suitable for overnight generation to build a large, varied corpus.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bfs_solver.difficulty import DifficultyScorer
from src.bfs_solver.solver import BFSSolver
from src.game.board import Direction
from src.game.board_builder import BoardBuilder, BoardConfig


def direction_to_string(direction: Direction) -> str:
    """Convert Direction enum to string representation."""
    return direction.name


def solution_to_string(solution: Optional[List]) -> str:
    """Convert solution list to string representation."""
    if not solution:
        return "[]"

    # Format as: [[x,y,direction], ...]
    parts = []
    for (x, y), direction in solution:
        parts.append(f"[{x},{y},{direction_to_string(direction)}]")

    return "[" + ",".join(parts) + "]"


def generate_random_config(rng: random.Random) -> BoardConfig:
    """Generate a random board configuration within PRD ranges."""

    # Board size - extended range for harder puzzles
    board_w = rng.choices([5, 6, 7, 8, 9, 10], weights=[25, 20, 20, 15, 10, 5])[0]
    board_h = rng.choices([5, 6, 7, 8, 9, 10], weights=[30, 25, 20, 15, 8, 2])[0]

    # Scale other parameters based on board size
    max_cells = board_w * board_h

    # Walls - scale with board size but keep reasonable
    wall_density = rng.uniform(0.05, 0.3)  # 5-30% of cells
    num_walls = min(int(max_cells * wall_density), 20)

    # Mice - extended range for complexity (1-10 as requested)
    num_mice = rng.choices(
        [1, 2, 3, 4, 5, 6, 7, 8, 10], weights=[15, 20, 20, 15, 10, 8, 5, 3, 2]
    )[0]

    # Rockets - extended range (1-5 as requested)
    num_rockets = rng.choices([1, 2, 3, 4, 5], weights=[20, 25, 20, 15, 8])[0]

    # Cats - more cats for harder puzzles (up to 2, but weighted toward more)
    num_cats = rng.choices([0, 1, 2], weights=[30, 50, 20])[0]

    # Holes - extended range (0-10 as requested)
    num_holes = rng.choices([0, 1, 2, 3, 4, 5], weights=[20, 25, 20, 15, 8, 5])[0]

    # Arrow budget
    arrow_budget = 3

    return BoardConfig(
        board_w=board_w,
        board_h=board_h,
        num_walls=num_walls,
        num_mice=num_mice,
        num_rockets=num_rockets,
        num_cats=num_cats,
        num_holes=num_holes,
        arrow_budget=arrow_budget,
    )


def generate_puzzle_row(
    seed: int,
    config: BoardConfig,
    solver: BFSSolver,
    difficulty_scorer: DifficultyScorer,
) -> Optional[Dict]:
    """Generate a single puzzle row."""
    try:
        # Generate board
        board_builder = BoardBuilder(config, seed=seed)
        level = board_builder.generate_level(f"Puzzle {seed}")
        engine = level.create_engine(max_steps=1000, seed=seed, puzzle_mode=True)

        # Solve with BFS
        result = solver.solve(engine)

        # Discard unsolvable or zero length solution puzzles
        if not result.success or result.solution_length == 0:
            return None

        # Score difficulty
        difficulty_score = difficulty_scorer.score_puzzle(engine, result.solution)
        difficulty_label = difficulty_scorer.get_difficulty_label(difficulty_score)

        return {
            "seed": seed,
            "board_w": config.board_w,
            "board_h": config.board_h,
            "num_walls": config.num_walls,
            "num_mice": config.num_mice,
            "num_rockets": config.num_rockets,
            "num_cats": config.num_cats,
            "num_holes": config.num_holes,
            "arrow_budget": config.arrow_budget,
            "bfs_solution": solution_to_string(result.solution),
            "difficulty_label": difficulty_label.value,
        }

    except Exception as e:
        print(f"Error generating puzzle for seed {seed}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate varied puzzle corpus for overnight runs"
    )
    parser.add_argument(
        "--output", type=str, default="varied_corpus.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--count", type=int, default=100000, help="Number of puzzles to attempt"
    )
    parser.add_argument("--start-seed", type=int, default=1, help="Starting seed value")
    parser.add_argument(
        "--append", action="store_true", help="Append to existing CSV file"
    )

    # Solver parameters
    parser.add_argument(
        "--depth-cap",
        type=int,
        default=30,
        help="BFS depth limit (higher for complex puzzles)",
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=500, help="BFS timeout in milliseconds"
    )

    # Performance monitoring
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N attempts",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=100,
        help="Flush CSV every N successful generations",
    )

    # Config generation
    parser.add_argument(
        "--config-seed", type=int, default=None, help="Seed for config randomization"
    )

    args = parser.parse_args()

    # Setup random number generator for config generation
    config_rng = random.Random(args.config_seed)

    # Create solver and difficulty scorer
    solver = BFSSolver(depth_cap=args.depth_cap, timeout_ms=args.timeout_ms)
    difficulty_scorer = DifficultyScorer()

    # CSV fieldnames
    fieldnames = [
        "seed",
        "board_w",
        "board_h",
        "num_walls",
        "num_mice",
        "num_rockets",
        "num_cats",
        "num_holes",
        "arrow_budget",
        "bfs_solution",
        "difficulty_label",
    ]

    # Open CSV file
    mode = "a" if args.append else "w"
    file_exists = Path(args.output).exists()

    print(f"Generating varied puzzles...")
    print(f"Output: {args.output} ({'append' if args.append else 'overwrite'})")
    print(f"BFS: depth_cap={args.depth_cap}, timeout={args.timeout_ms}ms")
    print(f"Config randomization seed: {args.config_seed}")
    print()

    start_time = time.time()
    generated_count = 0
    attempted_count = 0

    # Track difficulty distribution
    difficulty_stats = {"Easy": 0, "Medium": 0, "Hard": 0, "Brutal": 0}

    with open(args.output, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if new file or empty file
        if not args.append or not file_exists:
            writer.writeheader()

        while attempted_count < args.count:
            seed = args.start_seed + attempted_count
            attempted_count += 1

            # Generate random config for this seed
            config = generate_random_config(config_rng)

            # Generate puzzle
            row = generate_puzzle_row(seed, config, solver, difficulty_scorer)

            if row is not None:
                writer.writerow(row)
                generated_count += 1

                # Track difficulty distribution
                difficulty_stats[row["difficulty_label"]] += 1

                # Flush periodically for safety
                if generated_count % args.flush_interval == 0:
                    csvfile.flush()

            # Progress reporting
            if attempted_count % args.progress_interval == 0:
                elapsed = time.time() - start_time
                rate = generated_count / elapsed if elapsed > 0 else 0
                success_rate = (
                    generated_count / attempted_count * 100
                    if attempted_count > 0
                    else 0
                )

                print(
                    f"Progress: {attempted_count:,} attempted, {generated_count:,} generated "
                    f"({success_rate:.1f}% success), {rate:.1f}/sec"
                )
                print(f"  Difficulty distribution: {difficulty_stats}")

    # Final statistics
    elapsed = time.time() - start_time
    rate = generated_count / elapsed if elapsed > 0 else 0
    success_rate = generated_count / attempted_count * 100 if attempted_count > 0 else 0

    print(f"\n=== Generation Complete ===")
    print(f"Attempted: {attempted_count:,} puzzles")
    print(f"Generated: {generated_count:,} puzzles")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Rate: {rate:.1f} puzzles/sec")
    print(f"Output: {args.output}")
    print(f"Final difficulty distribution: {difficulty_stats}")

    # Calculate percentage distribution
    if generated_count > 0:
        print("\nDifficulty percentages:")
        for difficulty, count in difficulty_stats.items():
            percentage = count / generated_count * 100
            print(f"  {difficulty}: {count:,} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
