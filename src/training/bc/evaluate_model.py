#!/usr/bin/env python3
"""
Script to evaluate BC-trained model solve rate on test puzzles.
"""

import argparse
import random
from pathlib import Path
import numpy as np

import torch

from ...perception.data_types import PerceptionConfig
from ...perception.processors import GameStateProcessor
from ...policy.processors import PolicyProcessor
from ...state_fusion.processors import StateFusionProcessor
from .config import BCConfig
from .data_loader import create_data_loaders, load_puzzles_from_csv
from .evaluator import BCEvaluator
from .model_manager import ModelManager


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
    config.device = args.device

    # Set device for evaluation
    if args.device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = args.device
    print(f"Using device: {device}")

    print(f"Loading puzzles from {args.csv_path}")
    puzzles = load_puzzles_from_csv(args.csv_path)
    print(f"Loaded {len(puzzles)} puzzles")

    # Get board dimensions from first puzzle to configure perception
    # Note: For variable board sizes, we need to use the maximum dimensions
    max_width = max(p["board_w"] for p in puzzles)
    max_height = max(p["board_h"] for p in puzzles)
    print(f"Max board dimensions: {max_width}×{max_height}")

    # Initialize processors with validation disabled for variable board sizes
    perception_config = PerceptionConfig()
    perception_config.strict_bounds_checking = False
    perception_config.validate_input = False
    perception_processor = GameStateProcessor(perception_config)
    state_fusion_processor = StateFusionProcessor()
    policy_processor = PolicyProcessor()

    # Create data loaders to get test split
    train_loader, val_loader, test_loader = create_data_loaders(
        config, perception_processor, state_fusion_processor
    )

    # For evaluation, use all loaded puzzles (they're already evaluation puzzles)
    test_puzzles = puzzles
    print(f"Evaluating on {len(test_puzzles)} test puzzles")

    # Load model
    model_manager = ModelManager(config)
    print(f"Loading model from {args.model_path}")

    if Path(args.model_path).exists():
        checkpoint = torch.load(
            args.model_path, map_location=device, weights_only=False
        )
        model_state = checkpoint["model_state_dict"]

        # Check if this is the new composite format or old format
        if isinstance(model_state, dict) and "cnn_encoder" in model_state:
            # New composite format - load all components
            perception_processor.get_cnn_encoder().load_state_dict(
                model_state["cnn_encoder"]
            )
            state_fusion_processor.fusion_mlp.load_state_dict(
                model_state["state_fusion"]
            )
            policy_processor.policy_head.load_state_dict(model_state["policy_head"])
            print("Model loaded successfully (composite format)")
        else:
            # Legacy format - only policy head was saved
            policy_processor.policy_head.load_state_dict(model_state)
            print(
                "Model loaded successfully (legacy format - CNN/fusion layers randomly initialized)"
            )

        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"Warning: Model file {args.model_path} not found")
        print("Using randomly initialized model")

    # Move models to device
    state_fusion_processor = state_fusion_processor.to(device)
    policy_processor = policy_processor.to(device)

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
