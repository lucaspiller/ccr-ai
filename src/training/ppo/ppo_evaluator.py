"""
PPO Evaluator for performance tracking and comparison with baselines.

Evaluates trained models on fixed test sets and tracks performance metrics
across different puzzle difficulties.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ...perception.processors import GameStateProcessor
from ...policy.processors import PolicyProcessor
from ...state_fusion.processors import StateFusionProcessor
from .config import PPOConfig
from .logger import logger
from .ppo_env import PPOEnvironment
from src.puzzle_generator import PuzzleConfig, PuzzleGenerator


@dataclass
class EvaluationResult:
    """Result of evaluating on a single puzzle."""

    puzzle_id: str
    difficulty: str
    success: bool
    reward: float
    steps: int
    placement_steps: int
    execution_steps: int
    arrows_used: int
    mice_saved: int
    cats_fed: int
    game_result: str


class PPOEvaluator:
    """Evaluator for PPO training progress."""

    def __init__(
        self,
        config: PPOConfig,
        perception_processor: GameStateProcessor,
        state_fusion_processor: StateFusionProcessor,
        policy_processor: PolicyProcessor,
        value_head: torch.nn.Module,
    ):
        self.logger = logger.bind(component="ppo_evaluator")

        self.config = config
        self.perception_processor = perception_processor
        self.state_fusion_processor = state_fusion_processor
        self.policy_processor = policy_processor
        self.value_head = value_head
        self.device = next(policy_processor.policy_head.parameters()).device

        self.eval_env = PPOEnvironment(
            config=config,
            perception_processor=perception_processor,
            state_fusion_processor=state_fusion_processor,
        )

        # Create puzzle generator
        self.puzzle_generator = PuzzleGenerator(
            execution_timeout=config.execution_timeout
        )

        # Load or generate evaluation puzzles
        self.evaluation_puzzles: Dict[str, List[PuzzleConfig]] = (
            self.puzzle_generator.generate_evaluation_puzzles()
        )

        # Evaluation history
        self.evaluation_history: List[Dict[str, Any]] = []

        self.logger.info(
            f"PPO Evaluator initialized with {len(self.evaluation_puzzles)} test puzzles"
        )

    def evaluate(
        self,
        num_puzzles_per_difficulty: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation on test puzzles.

        Args:
            num_puzzles_per_difficulty: Number of puzzles to evaluate per difficulty.
                                       If None, evaluates all puzzles.

        Returns:
            Dictionary of evaluation results and metrics
        """
        self.logger.debug("Running PPO evaluation...")
        start_time = time.time()

        # Set models to evaluation mode
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.eval()
        self.policy_processor.eval()
        self.value_head.eval()

        evaluation_results = []

        # Evaluate each difficulty
        difficulties = [difficulty] if difficulty else ["easy", "medium", "hard"]
        for difficulty in difficulties:
            difficulty_puzzles = self.evaluation_puzzles[difficulty]

            # Limit number of puzzles if specified
            if num_puzzles_per_difficulty is not None:
                difficulty_puzzles = difficulty_puzzles[:num_puzzles_per_difficulty]

            self.logger.info(
                f"Evaluating {len(difficulty_puzzles)} {difficulty} puzzles..."
            )

            for puzzle in difficulty_puzzles:
                result = self._evaluate_single_puzzle(puzzle)
                evaluation_results.append(result)

        # Compute aggregate metrics
        metrics = self._compute_evaluation_metrics(evaluation_results)

        # Add timing information
        evaluation_time = time.time() - start_time
        metrics["evaluation_time"] = evaluation_time
        metrics["puzzles_evaluated"] = len(evaluation_results)

        # Store in history
        self.evaluation_history.append(
            {
                "timestamp": time.time(),
                "metrics": metrics,
                "detailed_results": evaluation_results,
            }
        )

        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        self._print_evaluation_summary(metrics)

        return metrics

    def _evaluate_single_puzzle(self, puzzle: PuzzleConfig) -> EvaluationResult:
        """Evaluate model on a single puzzle."""
        puzzle_id = puzzle.puzzle_id
        difficulty = puzzle.difficulty

        self.logger.debug(
            f"  Evaluating {puzzle_id} ({difficulty}): {puzzle.width}x{puzzle.height} "
            f"board, {len(puzzle.mice)} mice, {len(puzzle.cats)} cats, "
            f"budget={puzzle.arrow_budget}"
        )

        with torch.no_grad():
            # Reset environment with this puzzle
            observation, action_mask = self.eval_env.reset(puzzle)
            observation = observation.to(self.device)
            action_mask = action_mask.to(self.device)

            total_reward = 0.0
            episode_done = False
            steps = 0
            action_history = []
            reward_history = []

            # Run episode (use placement timeout since we're in the evaluator action loop)
            while not episode_done and steps < self.config.placement_timeout:
                # Get policy logits and sample action
                policy_logits = self.policy_processor.policy_head(
                    observation.unsqueeze(0)
                )

                # Debug policy logits
                if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                    self.logger.error(
                        f"    ERROR: NaN/Inf in policy_logits at step {steps}"
                    )
                    self.logger.warn(
                        f"    policy_logits stats: min={policy_logits.min():.3f}, max={policy_logits.max():.3f}, mean={policy_logits.mean():.3f}"
                    )
                    self.logger.warn(
                        f"    observation stats: min={observation.min():.3f}, max={observation.max():.3f}, mean={observation.mean():.3f}"
                    )
                    self.logger.warn(
                        f"    observation has NaN: {torch.isnan(observation).any()}"
                    )
                    break

                # Debug action mask
                valid_actions = action_mask.sum().item()
                self.logger.debug(
                    f"\n    Step {steps}: {valid_actions} valid actions out of {len(action_mask)}"
                )
                if valid_actions == 0:
                    self.logger.error(
                        f"    ERROR: No valid actions available at step {steps}"
                    )
                    break

                # Apply action mask
                from ...util.action_utils import (apply_action_mask,
                                                  decode_action)

                masked_logits = apply_action_mask(
                    policy_logits, action_mask.unsqueeze(0)
                )

                # Check for NaN in masked logits (inf is expected for invalid actions)
                if torch.isnan(masked_logits).any():
                    self.logger.error(
                        f"    ERROR: NaN in masked_logits at step {steps}"
                    )
                    self.logger.warn(
                        f"    policy_logits stats: min={policy_logits.min():.3f}, max={policy_logits.max():.3f}"
                    )
                    self.logger.warn(
                        f"    masked_logits stats: min={masked_logits.min():.3f}, max={masked_logits.max():.3f}"
                    )
                    self.logger.warn(f"    action_mask sum: {action_mask.sum()}")
                    break

                # Check that we have valid actions to choose from
                valid_logits = masked_logits[~torch.isinf(masked_logits)]
                if len(valid_logits) == 0:
                    self.logger.error(
                        f"    ERROR: No valid actions (all masked) at step {steps}"
                    )
                    break

                # Use deterministic policy for evaluation (argmax)
                action = torch.argmax(masked_logits, dim=-1).item()

                # Log top-k logits for analysis
                top_k = 5
                top_values, top_indices = torch.topk(
                    masked_logits.squeeze(0), k=top_k, dim=0
                )
                self.logger.info(f"    Top-{top_k} actions:")
                for i, (idx, value) in enumerate(zip(top_indices, top_values)):
                    idx_val = idx.item()
                    try:
                        action_info = decode_action(idx_val)
                        desc = f"{action_info.action_type}@({action_info.x},{action_info.y}) ({idx_val})"
                    except:
                        desc = f"action_{idx_val}"

                    selected = "← SELECTED" if idx_val == action else ""
                    self.logger.info(
                        f"      {i+1}. {desc} (logit={value:.3f}) {selected}"
                    )

                # Log action details
                try:
                    action_info = decode_action(action)
                    action_desc = (
                        f"{action_info.action_type}@({action_info.x},{action_info.y})"
                    )
                except:
                    action_desc = f"action_{action}"

                # Step environment
                self.logger.info(
                    f"[{steps}] Executing action: {action_desc} (ID: {action})"
                )
                step_result = self.eval_env.step(action)

                # Log step details
                step_reward = step_result.reward
                total_reward += step_reward
                episode_done = step_result.done
                steps += 1

                action_history.append((action, action_desc))
                reward_history.append(step_reward)

                # Check if transitioning to execution
                if step_result.info.get("phase") == "transition_to_execution":
                    self.logger.info(
                        f"\n    Step {steps:2d}: {action_desc} → reward={step_reward:.2f}, transitioning to execution"
                    )
                    # Now run the execution phase
                    self.logger.debug(f"    Running game execution...")
                    self.logger.info(f"[{steps}] Executing dummy action")
                    execution_result = self.eval_env.step(
                        0
                    )  # Dummy action, will be ignored in execution phase

                    # Update with execution results
                    total_reward += (
                        execution_result.reward - step_reward
                    )  # Don't double count placement reward
                    episode_done = execution_result.done

                    if episode_done:
                        final_stats = execution_result.info.get("episode_stats", {})
                        game_result = execution_result.info.get(
                            "game_result", "unknown"
                        )
                        self.logger.info(
                            f"    Execution complete: {game_result} | Total reward: {total_reward:.2f} | "
                            f"Mice saved: {final_stats.get('mice_saved', 0)}"
                        )

                    observation = execution_result.observation.to(self.device)
                    action_mask = execution_result.action_mask.to(self.device)

                elif steps <= 5 or step_reward != 0 or episode_done:
                    self.logger.info(
                        f"\n    Step {steps:2d}: {action_desc} → reward={step_reward:.2f}, done={episode_done}"
                    )
                    observation = step_result.observation.to(self.device)
                    action_mask = step_result.action_mask.to(self.device)
                else:
                    observation = step_result.observation.to(self.device)
                    action_mask = step_result.action_mask.to(self.device)

                # Early termination if no valid actions
                if not action_mask.any():
                    if not episode_done:
                        self.logger.info(
                            f"    No valid actions remaining at step {steps}"
                        )
                    break

            # Extract episode statistics (handle case where loop never ran)
            if "execution_result" in locals():
                # Use execution result if we transitioned to execution
                episode_stats = execution_result.info.get("episode_stats", {})
                game_result = execution_result.info.get("game_result", "unknown")
            elif "step_result" in locals():
                # Fall back to step result
                episode_stats = step_result.info.get("episode_stats", {})
                game_result = step_result.info.get("game_result", "unknown")
            else:
                episode_stats = {}
                game_result = "failed_early"

            success = game_result == "success"

            # Print summary
            self.logger.info(
                f"    Result: {game_result} | Reward: {total_reward:.2f} | Steps: {steps} | "
                f"Arrows: {episode_stats.get('arrows_used', 0)} | "
                f"Mice saved: {episode_stats.get('mice_saved', 0)}\n"
            )

            if len(action_history) > 5:
                placements = len(
                    [desc for _, desc in action_history if "place" in desc]
                )
                erases = len([desc for _, desc in action_history if "erase" in desc])
                self.logger.info(
                    f"    Action summary: {placements} placements, {erases} erases"
                )

            return EvaluationResult(
                puzzle_id=puzzle_id,
                difficulty=difficulty,
                success=success,
                reward=total_reward,
                steps=steps,
                placement_steps=episode_stats.get("placement_steps", 0),
                execution_steps=episode_stats.get("execution_ticks", 0),
                arrows_used=episode_stats.get("arrows_used", 0),
                mice_saved=episode_stats.get("mice_saved", 0),
                cats_fed=episode_stats.get("cats_fed", 0),
                game_result=game_result,
            )

    def _compute_evaluation_metrics(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Compute aggregate metrics from evaluation results."""
        if not results:
            return {"error": "No evaluation results"}

        # Group by difficulty
        by_difficulty = {}
        for result in results:
            if result.difficulty not in by_difficulty:
                by_difficulty[result.difficulty] = []
            by_difficulty[result.difficulty].append(result)

        metrics = {}

        # Compute metrics for each difficulty
        for difficulty, difficulty_results in by_difficulty.items():
            if not difficulty_results:
                continue

            # Basic metrics
            total_puzzles = len(difficulty_results)
            successful_puzzles = sum(1 for r in difficulty_results if r.success)
            solve_rate = successful_puzzles / total_puzzles

            # Reward metrics
            rewards = [r.reward for r in difficulty_results]
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            # Step metrics
            steps = [r.steps for r in difficulty_results]
            avg_steps = np.mean(steps)
            std_steps = np.std(steps)

            # Efficiency metrics (for successful puzzles only)
            successful_results = [r for r in difficulty_results if r.success]
            if successful_results:
                successful_steps = [r.steps for r in successful_results]
                avg_successful_steps = np.mean(successful_steps)

                # Arrow usage efficiency
                arrows_used = [r.arrows_used for r in successful_results]
                avg_arrows_used = np.mean(arrows_used)
            else:
                avg_successful_steps = float("inf")
                avg_arrows_used = 0.0

            # Store metrics
            metrics[f"{difficulty}_solve_rate"] = solve_rate
            metrics[f"{difficulty}_avg_reward"] = avg_reward
            metrics[f"{difficulty}_std_reward"] = std_reward
            metrics[f"{difficulty}_avg_steps"] = avg_steps
            metrics[f"{difficulty}_std_steps"] = std_steps
            metrics[f"{difficulty}_avg_successful_steps"] = avg_successful_steps
            metrics[f"{difficulty}_avg_arrows_used"] = avg_arrows_used
            metrics[f"{difficulty}_total_puzzles"] = total_puzzles
            metrics[f"{difficulty}_successful_puzzles"] = successful_puzzles

        # Overall metrics
        all_solve_rates = [
            metrics.get(f"{diff}_solve_rate", 0.0)
            for diff in ["easy", "medium", "hard"]
            if f"{diff}_solve_rate" in metrics
        ]
        if all_solve_rates:
            metrics["overall_avg_solve_rate"] = np.mean(all_solve_rates)

        # Primary success criteria from PRD
        metrics["meets_medium_target"] = (
            metrics.get("medium_solve_rate", 0.0)
            >= self.config.target_medium_solve_rate
        )
        metrics["meets_hard_target"] = (
            metrics.get("hard_solve_rate", 0.0) >= self.config.target_hard_solve_rate
        )

        return metrics

    def _print_evaluation_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 60)

        for difficulty in ["easy", "medium", "hard"]:
            if f"{difficulty}_solve_rate" in metrics:
                solve_rate = metrics[f"{difficulty}_solve_rate"]
                avg_reward = metrics[f"{difficulty}_avg_reward"]
                avg_steps = metrics[f"{difficulty}_avg_steps"]
                total = metrics[f"{difficulty}_total_puzzles"]
                successful = metrics[f"{difficulty}_successful_puzzles"]

                self.logger.info(
                    f"{difficulty.upper():>6}: {solve_rate:6.3f} solve rate "
                    f"({successful:3d}/{total:3d}) "
                    f"| Reward: {avg_reward:6.2f} "
                    f"| Steps: {avg_steps:6.1f}"
                )

        self.logger.info("-" * 60)

        # Success criteria
        medium_target = self.config.target_medium_solve_rate
        hard_target = self.config.target_hard_solve_rate

        medium_rate = metrics.get("medium_solve_rate", 0.0)
        hard_rate = metrics.get("hard_solve_rate", 0.0)

        medium_status = "✓" if medium_rate >= medium_target else "✗"
        hard_status = "✓" if hard_rate >= hard_target else "✗"

        self.logger.info(
            f"TARGETS: Medium {medium_status} {medium_rate:.3f} ≥ {medium_target:.3f} "
            f"| Hard {hard_status} {hard_rate:.3f} ≥ {hard_target:.3f}"
        )

        self.logger.info("=" * 60)

    def evaluate_vs_bc_baseline(self) -> Dict[str, Any]:
        """Compare current PPO model with BC baseline."""
        # This would require loading the BC model and running the same evaluation
        # For now, return placeholder comparison
        return {
            "ppo_vs_bc_improvement": {
                "medium_solve_rate_delta": 0.0,  # PPO - BC
                "hard_solve_rate_delta": 0.0,
                "efficiency_improvement": 0.0,
            },
            "note": "BC baseline comparison not implemented yet",
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations."""
        return self.evaluation_history.copy()

    def save_evaluation_results(self, filepath: str):
        """Save detailed evaluation results to file."""
        if not self.evaluation_history:
            self.logger.info("No evaluation results to save")
            return

        # Get latest evaluation
        latest_eval = self.evaluation_history[-1]

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(latest_eval, f, indent=2, default=str)

        self.logger.info(f"Evaluation results saved to {filepath}")

    def quick_evaluation(self, num_puzzles: int = 20) -> Dict[str, Any]:
        """Run a quick evaluation with limited puzzles for debugging."""
        self.logger.info(
            f"Running quick evaluation with {num_puzzles} puzzles per difficulty..."
        )
        return self.evaluate(num_puzzles_per_difficulty=num_puzzles)


def test_evaluator():
    """Test the evaluator."""
    from ...perception.data_types import PerceptionConfig
    from .config import PPOConfig

    # Create config and processors
    config = PPOConfig()
    perception_config = PerceptionConfig(strict_bounds_checking=False)
    perception_processor = GameStateProcessor(perception_config)
    state_fusion_processor = StateFusionProcessor()
    policy_processor = PolicyProcessor()
    value_head = torch.nn.Linear(128, 1)  # Dummy value head

    # Create evaluator
    evaluator = PPOEvaluator(
        config=config,
        perception_processor=perception_processor,
        state_fusion_processor=state_fusion_processor,
        policy_processor=policy_processor,
        value_head=value_head,
    )

    # Run quick evaluation
    results = evaluator.quick_evaluation(num_puzzles=5)
    print("Test evaluation completed!")
    return results


if __name__ == "__main__":
    test_evaluator()
