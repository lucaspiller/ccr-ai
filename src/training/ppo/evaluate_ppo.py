"""
Standalone PPO Model Evaluation Script.

This script loads a trained PPO model and evaluates it on a fixed set of puzzles
without any training. Useful for testing model performance and debugging.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

from src.training.ppo.config import PPOConfig

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.model.model_loader import ModelLoader

from .ppo_evaluator import PPOEvaluator


class PPOModelEvaluator:
    """Standalone evaluator for PPO models."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize evaluator with model path.

        Args:
            model_path: Path to saved PPO model
            device: Device to run evaluation on
        """
        model_loader = ModelLoader(model_path, device)
        (
            self.perception_processor,
            self.state_fusion_processor,
            self.policy_processor,
            self.value_head,
        ) = model_loader.get_ppo_components()
        self.device = model_loader.device
        self.parameter_count = model_loader.parameter_count

        self.config = PPOConfig()  # Load default PPO config
        self.config.verbose_env_logging = True

        print(f"PPO Model Evaluator initialized")
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {self.parameter_count}")

    def evaluate_puzzles(
        self,
        num_puzzles: int = 50,
        verbose: bool = True,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on all difficulty levels.

        Args:
            num_puzzles_per_difficulty: Number of puzzles per difficulty
            verbose: Whether to print detailed results
            difficulty: Specific difficulty to evaluate ('easy', 'medium', 'hard', or None for all)

        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating model on all difficulties ({num_puzzles} puzzles each)...")
        start_time = time.time()

        # Create evaluator
        evaluator = PPOEvaluator(
            config=self.config,
            perception_processor=self.perception_processor,
            state_fusion_processor=self.state_fusion_processor,
            policy_processor=self.policy_processor,
            value_head=self.value_head,
        )

        results = evaluator.evaluate(
            num_puzzles_per_difficulty=num_puzzles, difficulty=difficulty
        )

        evaluation_time = time.time() - start_time

        # Print summary
        if verbose:
            self._print_evaluation_summary(results, evaluation_time)

        return results

    def _print_evaluation_summary(
        self, results: Dict[str, Any], evaluation_time: float
    ):
        """Print evaluation summary."""
        print(f"\nEvaluation completed in {evaluation_time:.2f}s")
        print("=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        # Print difficulty-specific results
        for difficulty in ["easy", "medium", "hard"]:
            solve_rate_key = f"{difficulty}_solve_rate"
            if solve_rate_key in results:
                solve_rate = results[solve_rate_key]
                avg_reward = results.get(f"{difficulty}_avg_reward", 0.0)
                avg_steps = results.get(f"{difficulty}_avg_steps", 0.0)
                total = results.get(f"{difficulty}_total_puzzles", 0)
                successful = results.get(f"{difficulty}_successful_puzzles", 0)

                print(
                    f"{difficulty.upper():>6}: {solve_rate:6.3f} solve rate "
                    f"({successful:3d}/{total:3d}) "
                    f"| Avg Reward: {avg_reward:6.2f} "
                    f"| Avg Steps: {avg_steps:6.1f}"
                )

        print("-" * 70)

        # Print success criteria
        medium_target = 0.85  # From config
        hard_target = 0.60  # From config

        medium_rate = results.get("medium_solve_rate", 0.0)
        hard_rate = results.get("hard_solve_rate", 0.0)

        medium_status = "✅ MET" if medium_rate >= medium_target else "❌ NOT MET"
        hard_status = "✅ MET" if hard_rate >= hard_target else "❌ NOT MET"

        print("SUCCESS CRITERIA:")
        print(f"Medium ≥ {medium_target:.3f}: {medium_rate:.3f} {medium_status}")
        print(f"Hard   ≥ {hard_target:.3f}: {hard_rate:.3f} {hard_status}")

        if medium_rate >= medium_target and hard_rate >= hard_target:
            print("\n🏆 ALL SUCCESS CRITERIA MET! 🏆")
        else:
            print("\n⚠️  Some success criteria not yet met")

        print("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate PPO model on ChuChu Rocket puzzles"
    )

    parser.add_argument(
        "model_path", type=str, help="Path to saved PPO model (.pth file)"
    )

    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=100,
        help="Number of puzzles to evaluate (default: 100)",
    )

    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "all"],
        default="easy",
        help="Difficulty level to evaluate (default: easy)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run evaluation on (default: auto)",
    )

    parser.add_argument(
        "--output", type=str, help="Path to save evaluation results (JSON format)"
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Create evaluator
        evaluator = PPOModelEvaluator(model_path=args.model_path, device=args.device)

        # Run evaluation
        results = evaluator.evaluate_puzzles(
            num_puzzles=args.num_puzzles,
            verbose=not args.quiet,
            difficulty=("easy" if args.difficulty == "easy" else None),
        )

        # Print final summary
        if not args.quiet:
            easy_rate = results.get("easy_solve_rate", 0.0)
            print(f"\nFinal Result: {easy_rate:.1%} solve rate on easy puzzles")

        # Exit with appropriate code
        easy_rate = results.get("easy_solve_rate", 0.0)
        if easy_rate >= 0.8:  # 80% threshold
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
