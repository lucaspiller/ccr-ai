#!/usr/bin/env python3
"""
Standalone script to visualize model activations on random puzzles.

Generates PNG visualizations showing:
- Original board with sprites, walls, and arrows (using game colors)
- CNN feature maps from convolutional layers
- Policy overlays showing arrow placement probabilities
- Cat embeddings (if cats present in puzzle)

Creates an HTML report to view all visualizations together.

Usage:
    # Basic usage - visualize single easy puzzle
    uv run python scripts/visualize_model.py --model model/bc_best.pth

    # Generate multiple medium puzzles with fixed seed
    uv run python scripts/visualize_model.py --model model/ppo_final.pth --difficulty medium --num-puzzles 5 --seed 42

    # Show interactively instead of saving files
    uv run python scripts/visualize_model.py --model model/bc_best.pth --show

Output:
    - PNG files saved to model/visualizations/ (default)
    - HTML report at model/visualizations/visualization_report.html
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.visualizer.visualizer import ModelVisualizer


def generate_html_viewer(output_dir: str, puzzle_info: list, model_path: str):
    """Generate HTML file to display all visualizations."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Visualization Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-100 min-h-screen font-sans">
    <div class="max-w-2xl mx-auto p-8">
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h1 class="text-3xl font-bold mb-4">Model Visualization Report</h1>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Model:</span>
                    <span class="text-slate-800">{os.path.basename(model_path)}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Generated:</span>
                    <span class="text-slate-800">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Total Puzzles:</span>
                    <span class="text-slate-800">{len(puzzle_info)}</span>
                </div>
            </div>
        </div>
"""

    for i, info in enumerate(puzzle_info):
        puzzle_id = info["puzzle_id"]
        board_size = info["board_size"]
        arrows = info["arrows"]
        walls = info["walls"]
        max_prob = info["max_prob"]
        difficulty = info["difficulty"]
        seed = info.get("seed", "random")

        html_content += f"""
    <div class="bg-white rounded-lg shadow-md overflow-hidden mb-8">
        <div class="bg-blue-600 text-white px-6 py-4">
            <h2 class="text-xl font-semibold">Puzzle {i+1}: {puzzle_id}</h2>
        </div>
        <div class="bg-slate-50 border-b border-slate-200">
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 p-6">
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Difficulty:</span>
                    <span class="text-slate-800">{difficulty.title()}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Board Size:</span>
                    <span class="text-slate-800">{board_size}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Walls:</span>
                    <span class="text-slate-800">{walls}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Arrows:</span>
                    <span class="text-slate-800">{arrows}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Max Policy Prob:</span>
                    <span class="text-slate-800">{max_prob:.3f}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-500 font-semibold">Seed:</span>
                    <span class="text-slate-800">{seed}</span>
                </div>
            </div>
        </div>
        <div class="grid grid-cols-1 gap-8 p-6">
"""

        # Add each visualization type
        viz_types = [
            (
                "Original Board",
                f"{puzzle_id}_original_board.png",
                "The original puzzle board with sprites, walls, and arrows",
            ),
            (
                "CNN Feature Maps",
                f"{puzzle_id}_cnn_features.png",
                "Feature maps from convolutional layers showing spatial pattern detection",
            ),
            (
                "Policy Overlay",
                f"{puzzle_id}_policy_overlay.png",
                "Arrow placement probabilities overlaid on the board",
            ),
            (
                "Cat Embeddings",
                f"{puzzle_id}_cat_embeddings.png",
                "Cat encoder activations (if cats present in puzzle)",
            ),
        ]

        for viz_name, filename, description in viz_types:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                html_content += f"""
            <div class="text-center p-4">
                <h3 class="text-lg font-semibold text-slate-800 mb-2">{viz_name}</h3>
                <p class="text-sm text-slate-500 mb-4">{description}</p>
                <img src="{filename}" alt="{viz_name} for {puzzle_id}" class="max-w-full h-auto border border-slate-200 rounded-lg bg-white mx-auto">
            </div>
"""

        html_content += """
        </div>
    </div>
"""

    html_content += """
</body>
</html>"""

    # Write HTML file
    html_path = os.path.join(output_dir, "visualization_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path


def main():
    parser = argparse.ArgumentParser(description="Visualize model activations")
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
        "--output-dir",
        type=str,
        default="model/visualizations",
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run model on (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=1,
        help="Number of puzzles to visualize",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving",
    )

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create output directory
    if not args.show:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving visualizations to: {args.output_dir}")

    try:
        # Generate seed if not provided
        if args.seed is None:
            args.seed = random.randint(0, 2**31 - 1)
            print(f"Generated random seed: {args.seed}")

        # Initialize visualizer
        print(f"Loading model: {args.model}")
        visualizer = ModelVisualizer(args.model, device=args.device)

        # Track puzzle info for HTML generation
        puzzle_info = []

        # Generate and visualize puzzles
        for i in range(args.num_puzzles):
            print(f"\n=== Puzzle {i+1}/{args.num_puzzles} ===")

            # Generate puzzle with seed
            puzzle_seed = args.seed + i
            level = visualizer.generate_random_puzzle(
                difficulty=args.difficulty, seed=puzzle_seed
            )

            # Create puzzle ID
            puzzle_id = f"{args.difficulty}_puzzle_{i:03d}_seed_{puzzle_seed}"

            # Visualize
            save_dir = args.output_dir if not args.show else None
            results = visualizer.visualize_state(
                level, save_dir=save_dir, puzzle_id=puzzle_id
            )

            # Print some stats
            board = level.board
            max_prob = torch.sigmoid(results["policy_logits"]).max().item()
            print(f"Board size: {board.width}x{board.height}")
            print(f"Arrows: {len(board.arrows)}, Walls: {len(board.walls)}")
            print(f"Policy max prob: {max_prob:.3f}")

            # Collect puzzle info for HTML
            puzzle_info.append(
                {
                    "puzzle_id": puzzle_id,
                    "board_size": f"{board.width}x{board.height}",
                    "arrows": len(board.arrows),
                    "walls": len(board.walls),
                    "max_prob": max_prob,
                    "difficulty": args.difficulty,
                    "seed": puzzle_seed,
                }
            )

        print(f"\n✅ Generated {args.num_puzzles} puzzle visualization(s)")
        print(f"🎲 Seed used: {args.seed} (add --seed {args.seed} to reproduce)")
        if not args.show:
            print(f"📁 Saved to: {args.output_dir}")

            # Generate HTML report
            html_path = generate_html_viewer(args.output_dir, puzzle_info, args.model)
            print(f"📄 HTML report: {html_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
