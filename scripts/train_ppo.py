#!/usr/bin/env python3
"""
PPO Training Script with Command Line Interface.

Provides command line interface for PPO training with support for continuous
overnight training and easy-only puzzle modes.
"""

import argparse
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training.ppo.config import PPOConfig
from src.training.ppo.train_ppo import PPOTrainingManager
from src.training.ppo.utils import format_training_time


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Self-Play Training for ChuChu Rocket"
    )

    parser.add_argument(
        "--config", type=str, help="Path to custom configuration file (optional)"
    )

    parser.add_argument(
        "--bc-model", type=str, help="Path to BC model checkpoint (overrides config)"
    )

    parser.add_argument(
        "--total-steps", type=int, help="Total environment steps (overrides config)"
    )

    parser.add_argument(
        "--parallel-envs",
        type=int,
        help="Number of parallel environments (overrides config)",
    )

    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate (overrides config)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device (overrides config)",
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced parameters",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint (model/ppo_checkpoint_latest.pth)",
    )

    # Continuous training options
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous training mode (ignores --total-steps)",
    )

    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Maximum training time in hours for continuous mode (default: 8.0)",
    )

    parser.add_argument(
        "--easy-only",
        action="store_true",
        help="Only train on easy puzzles (no curriculum progression)",
    )

    parser.add_argument(
        "--regenerate-puzzles",
        action="store_true",
        default=True,
        help="Regenerate puzzles when exhausted (default: True)",
    )

    parser.add_argument(
        "--no-regenerate-puzzles",
        action="store_false",
        dest="regenerate_puzzles",
        help="Disable puzzle regeneration",
    )

    return parser.parse_args()


def create_config_from_args(args) -> PPOConfig:
    """Create PPO configuration from command line arguments."""
    config = PPOConfig()

    # Override config with command line arguments
    if args.bc_model:
        config.bc_model_path = args.bc_model

    if args.total_steps:
        config.total_env_steps = args.total_steps

    if args.parallel_envs:
        config.num_parallel_envs = args.parallel_envs

    if args.learning_rate:
        config.learning_rate = args.learning_rate

    if args.device:
        config.device = args.device

    # Continuous training options
    if args.continuous:
        config.continuous_training = True
        config.max_training_hours = args.max_hours
        print(f"🌙 Continuous training enabled for {args.max_hours} hours")

    if args.easy_only:
        config.curriculum_easy_only = True
        print("🎯 Easy-only mode enabled")

    config.regenerate_puzzles_when_exhausted = args.regenerate_puzzles
    if args.regenerate_puzzles:
        print("🔄 Puzzle regeneration enabled")

    return config


def main():
    """Main entry point for PPO training."""
    print("ChuChu Rocket PPO Self-Play Training")
    print("=" * 50)

    # Parse command line arguments
    args = parse_args()

    # Create configuration
    config = create_config_from_args(args)

    # Create training manager
    training_manager = PPOTrainingManager(config, resume_requested=args.resume)

    try:
        if args.quick_test:
            # Run quick test
            results = training_manager.quick_test()
        else:
            # Run full training
            results = training_manager.run_training()

        # Print final status
        if results.get("success", False):
            print(
                f"\n✅ Training completed successfully in {format_training_time(results['total_time'])}"
            )

            sys.exit(0)
        else:
            print(
                f"\n❌ Training failed or was interrupted after {format_training_time(results['total_time'])}"
            )
            if "error" in results:
                print(f"Error: {results['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
