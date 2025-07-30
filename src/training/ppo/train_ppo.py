"""
Main PPO training script.

Orchestrates the complete PPO self-play training pipeline, loading the BC model
and training with curriculum progression until success criteria are met.
"""

import os
import signal
import time
from typing import Any, Dict, Optional

from .config import PPOConfig
from .ppo_trainer import PPOTrainer
from .utils import (format_training_time,
                    print_training_header, validate_config)


class PPOTrainingManager:
    """Manages the complete PPO training process."""

    def __init__(self, config: PPOConfig, resume_requested: bool = False):
        self.config = config
        self.trainer: Optional[PPOTrainer] = None
        self.resume_requested = resume_requested

    def run_training(self) -> Dict[str, Any]:
        """Run the complete PPO training process.

        Returns:
            Dictionary with training results and statistics
        """
        start_time = time.time()

        try:
            # Validate configuration
            print("Validating configuration...")
            issues = validate_config(self.config)
            if issues:
                print("Configuration validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
                return {"error": "Configuration validation failed", "issues": issues}

            # Print training header
            print_training_header(self.config)

            # Initialize trainer
            print("Initializing PPO trainer...")
            self.trainer = PPOTrainer(self.config)

            # Handle resume from checkpoint if requested
            if self.resume_requested:
                checkpoint_path = os.path.join(
                    self.config.model_dir, f"{self.config.checkpoint_name}_latest.pth"
                )

                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        f"Resume requested but checkpoint not found: {checkpoint_path}"
                    )

                print("Resuming from checkpoint...")
                self._load_checkpoint(checkpoint_path)

            # Run training
            print("\nStarting PPO training...")
            training_results = self.trainer.train()

            # Training completed successfully
            total_time = time.time() - start_time
            print(f"\n🎉 PPO Training completed successfully! 🎉")
            print(f"Total training time: {format_training_time(total_time)}")

            # Print final results
            self._print_final_results(training_results)

            return {
                "success": True,
                "total_time": total_time,
                "training_results": training_results,
            }

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            total_time = time.time() - start_time
            return {
                "success": False,
                "interrupted": True,
                "total_time": total_time,
                "message": "Training interrupted by user",
            }

        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            import traceback

            traceback.print_exc()

            total_time = time.time() - start_time
            return {"success": False, "error": str(e), "total_time": total_time}

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint to resume training."""
        try:
            print(f"Loading checkpoint from {checkpoint_path}...")

            # Use ModelLoader to handle the checkpoint loading
            from ...model.model_loader import ModelLoader

            model_loader = ModelLoader(
                checkpoint_path, device=str(self.trainer.device), load_components=False
            )
            checkpoint_info = model_loader.load_model()

            # Get the loaded components
            (
                perception_processor,
                state_fusion_processor,
                policy_processor,
                value_head,
            ) = model_loader.get_ppo_components()

            # Replace trainer components with loaded ones
            self.trainer.perception_processor = perception_processor
            self.trainer.state_fusion_processor = state_fusion_processor
            self.trainer.policy_processor = policy_processor
            self.trainer.value_head = value_head

            # Recreate optimizer with loaded parameters
            self.trainer.optimizer = self.trainer._create_optimizer()
            self.trainer.scheduler = self.trainer._create_scheduler()

            # Load optimizer and scheduler states if available
            checkpoint_data = model_loader.checkpoint_data
            if "optimizer" in checkpoint_data:
                self.trainer.optimizer.load_state_dict(checkpoint_data["optimizer"])
                print("   Loaded optimizer state")
            if "scheduler" in checkpoint_data:
                self.trainer.scheduler.load_state_dict(checkpoint_data["scheduler"])
                print("   Loaded scheduler state")

            # Restore training state
            self.trainer.global_step = checkpoint_info["global_step"]
            self.trainer.best_eval_score = checkpoint_info.get("best_eval_score", 0.0)

            print(
                f"✅ Successfully resumed training from step {self.trainer.global_step:,}"
            )
            print(f"   Previous best eval score: {self.trainer.best_eval_score:.3f}")
            print(f"   Checkpoint type: {checkpoint_info['checkpoint_type']}")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            import traceback

            traceback.print_exc()
            print("Starting training from scratch...")

    def _print_final_results(self, results: Dict[str, Any]):
        """Print final training results."""
        print("\n" + "=" * 60)
        print("FINAL TRAINING RESULTS")
        print("=" * 60)

        if "final_evaluation" in results:
            eval_results = results["final_evaluation"]

            # Print solve rates
            for difficulty in ["easy", "medium", "hard"]:
                solve_rate_key = f"{difficulty}_solve_rate"
                if solve_rate_key in eval_results:
                    solve_rate = eval_results[solve_rate_key]
                    print(f"{difficulty.upper():>6} Solve Rate: {solve_rate:.3f}")

            # Check success criteria
            medium_target = self.config.target_medium_solve_rate
            hard_target = self.config.target_hard_solve_rate

            medium_rate = eval_results.get("medium_solve_rate", 0.0)
            hard_rate = eval_results.get("hard_solve_rate", 0.0)

            print("-" * 60)
            print("SUCCESS CRITERIA:")

            medium_met = medium_rate >= medium_target
            hard_met = hard_rate >= hard_target

            medium_status = "✅ MET" if medium_met else "❌ NOT MET"
            hard_status = "✅ MET" if hard_met else "❌ NOT MET"

            print(f"Medium ≥ {medium_target:.3f}: {medium_rate:.3f} {medium_status}")
            print(f"Hard   ≥ {hard_target:.3f}: {hard_rate:.3f} {hard_status}")

            if medium_met and hard_met:
                print("\n🏆 ALL SUCCESS CRITERIA MET! 🏆")
            else:
                print("\n⚠️  Some success criteria not met")

        print("=" * 60)

    def quick_test(self) -> Dict[str, Any]:
        """Run a quick test of the training pipeline."""
        print("Running quick PPO training test...")

        # Create test config with reduced parameters
        test_config = PPOConfig()
        test_config.total_env_steps = 10000  # Much smaller for testing
        test_config.rollout_length = 256
        test_config.eval_frequency = 5000
        test_config.num_parallel_envs = 4

        # Override config
        original_config = self.config
        self.config = test_config

        try:
            result = self.run_training()
            print("Quick test completed!")
            return result
        finally:
            # Restore original config
            self.config = original_config
