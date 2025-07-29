#!/usr/bin/env python3
"""
Script to evaluate BC-trained model solve rate on test puzzles.
"""

import argparse
import random

import numpy as np
import torch

from src.model.model_loader import ModelLoader

from .config import BCConfig
from .data_loader import create_data_loaders, load_puzzles_from_csv
from .evaluator import BCEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC model solve rate")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/bc_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/training_data.csv",
        help="Path to puzzle CSV file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per puzzle evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)",
    )

    # Set all random seeds for deterministic evaluation
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Additional deterministic settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parser.parse_args()

    # Load configuration
    config = BCConfig()
    config.csv_path = args.csv_path

    print(f"Loading puzzles from {args.csv_path}")
    puzzles = load_puzzles_from_csv(args.csv_path)
    print(f"Loaded {len(puzzles)} puzzles")

    # Get board dimensions from first puzzle to configure perception
    # Note: For variable board sizes, we need to use the maximum dimensions
    max_width = max(p["board_w"] for p in puzzles)
    max_height = max(p["board_h"] for p in puzzles)
    print(f"Max board dimensions: {max_width}×{max_height}")

    # Load model
    model_loader = ModelLoader(args.model_path, args.device)
    (perception_processor, state_fusion_processor, policy_processor) = (
        model_loader.get_bc_components()
    )
    config.device = model_loader.device

    # Create data loaders to get test split
    train_loader, val_loader, test_loader = create_data_loaders(
        config, perception_processor, state_fusion_processor
    )

    # For evaluation, use all loaded puzzles (they're already evaluation puzzles)
    test_puzzles = puzzles
    print(f"Evaluating on {len(test_puzzles)} test puzzles")

    # Initialize evaluator
    evaluator = BCEvaluator(
        config=config,
        perception_processor=perception_processor,
        state_fusion_processor=state_fusion_processor,
        policy_processor=policy_processor,
    )

    # Run evaluation
    print(f"Starting evaluation with max_steps={args.max_steps}")
    print("This may take several minutes...")

    metrics = evaluator.evaluate_puzzle_solving(test_puzzles, max_steps=args.max_steps)

    # Run probability analysis on validation data
    print("\nRunning probability analysis...")
    prob_metrics = evaluator.evaluate_multihot_probability_analysis(val_loader)
    metrics.update(prob_metrics)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Overall solve rate: {metrics.get('overall_solve_rate', 0.0):.4f}")

    if "overall_avg_steps" in metrics:
        print(f"Average steps (solved): {metrics['overall_avg_steps']:.2f}")

    if "overall_efficiency" in metrics:
        print(f"Average efficiency: {metrics['overall_efficiency']:.4f}")

    # Probability analysis
    if "avg_correct_tile_prob" in metrics:
        print(f"\n" + "=" * 50)
        print("PROBABILITY ANALYSIS")
        print("=" * 50)

        correct_prob = metrics["avg_correct_tile_prob"]
        topk_prob = metrics["avg_topk_prob"]
        prob_ratio = metrics["prob_ratio"]

        print(f"Average correct tile probability: {correct_prob:.4f}")
        print(f"Average top-k probability: {topk_prob:.4f}")
        print(f"Correct/Top-k ratio: {prob_ratio:.4f}")

    # Difficulty breakdown
    difficulties = ["easy", "medium", "hard"]
    for difficulty in difficulties:
        solve_key = f"{difficulty}_solve_rate"
        if solve_key in metrics:
            solve_rate = metrics[solve_key]
            print(f"\n{difficulty.title()} puzzles:")
            print(f"  Solve rate: {solve_rate:.4f}")

            steps_key = f"{difficulty}_avg_steps"
            if steps_key in metrics:
                print(f"  Avg steps: {metrics[steps_key]:.2f}")

            efficiency_key = f"{difficulty}_efficiency"
            if efficiency_key in metrics:
                print(f"  Efficiency: {metrics[efficiency_key]:.4f}")

    # Check success criteria from PRD
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 50)

    easy_solve_rate = metrics.get("easy_solve_rate", 0.0)
    medium_solve_rate = metrics.get("medium_solve_rate", 0.0)
    hard_solve_rate = metrics.get("hard_solve_rate", 0.0)

    print(
        f"Easy ≥95%:   {easy_solve_rate:.1%} {'✓' if easy_solve_rate >= 0.95 else '✗'}"
    )
    print(
        f"Medium ≥70%: {medium_solve_rate:.1%} {'✓' if medium_solve_rate >= 0.70 else '✗'}"
    )
    print(
        f"Hard ≥50%:   {hard_solve_rate:.1%} {'✓' if hard_solve_rate >= 0.50 else '✗'}"
    )

    # Overall efficiency check
    overall_efficiency = metrics.get("overall_efficiency", 0.0)
    print(
        f"Efficiency ≤1.5×: {overall_efficiency:.2f} {'✓' if overall_efficiency <= 1.5 else '✗'}"
    )


if __name__ == "__main__":
    main()
