"""
Evaluation utilities for behaviour cloning models.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...game.board import Direction
from ...game.board_builder import BoardBuilder, BoardConfig
from ...game.engine import GameEngine, GamePhase, GameResult
from ...perception.data_types import PerceptionConfig
from ...perception.processors import GameStateProcessor
from ...policy.processors import PolicyProcessor
from ...state_fusion.processors import StateFusionProcessor
from ...util.action_utils import apply_action_mask, decode_action
from .config import BCConfig
from .data_loader import get_device


class BCEvaluator:
    """Evaluates behaviour cloning model performance."""

    def __init__(
        self,
        config: BCConfig,
        perception_processor: GameStateProcessor,
        state_fusion_processor: StateFusionProcessor,
        policy_processor: PolicyProcessor,
    ):
        """Initialize evaluator.

        Args:
            config: Training configuration
            perception_processor: Perception layer processor
            state_fusion_processor: State fusion processor
            policy_processor: Policy head processor
        """
        self.config = config
        self.device = get_device(config.device)

        self.perception_processor = perception_processor
        self.perception_processor.get_cnn_encoder().to(self.device)
        self.state_fusion_processor = state_fusion_processor.to(self.device)
        self.policy_processor = policy_processor.to(self.device)

        # Set to evaluation mode
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.eval()
        self.policy_processor.eval()

    def evaluate_puzzle_solving(
        self, puzzles: List[Dict], max_steps: int = 100
    ) -> Dict[str, float]:
        """Evaluate model's puzzle solving performance.

        Args:
            puzzles: List of puzzle dictionaries to evaluate on
            max_steps: Maximum number of steps to attempt per puzzle

        Returns:
            Dictionary of evaluation metrics
        """
        results_by_difficulty = defaultdict(list)
        overall_results = []

        with torch.no_grad():
            for puzzle in puzzles:
                result = self._solve_single_puzzle(puzzle, max_steps)
                results_by_difficulty[puzzle["difficulty_label"]].append(result)
                overall_results.append(result)

        # Calculate metrics
        metrics = {}

        # Overall metrics
        metrics["overall_solve_rate"] = sum(r["solved"] for r in overall_results) / len(
            overall_results
        )
        solved_results = [r for r in overall_results if r["solved"]]
        if solved_results:
            metrics["overall_avg_steps"] = sum(
                r["steps_taken"] for r in solved_results
            ) / len(solved_results)
            metrics["overall_efficiency"] = sum(
                r["efficiency"] for r in solved_results
            ) / len(solved_results)
        else:
            metrics["overall_avg_steps"] = max_steps
            metrics["overall_efficiency"] = 0.0

        # Difficulty-specific metrics
        for difficulty, results in results_by_difficulty.items():
            if not results:
                continue

            solve_rate = sum(r["solved"] for r in results) / len(results)
            metrics[f"{difficulty.lower()}_solve_rate"] = solve_rate

            solved_subset = [r for r in results if r["solved"]]
            if solved_subset:
                avg_steps = sum(r["steps_taken"] for r in solved_subset) / len(
                    solved_subset
                )
                avg_efficiency = sum(r["efficiency"] for r in solved_subset) / len(
                    solved_subset
                )
                metrics[f"{difficulty.lower()}_avg_steps"] = avg_steps
                metrics[f"{difficulty.lower()}_efficiency"] = avg_efficiency
            else:
                metrics[f"{difficulty.lower()}_avg_steps"] = max_steps
                metrics[f"{difficulty.lower()}_efficiency"] = 0.0

        return metrics

    def _solve_single_puzzle(self, puzzle: Dict, max_steps: int) -> Dict:
        """Attempt to solve a single puzzle.

        Args:
            puzzle: Puzzle dictionary
            max_steps: Maximum steps to attempt

        Returns:
            Dictionary with solving results
        """
        # Create board and engine
        config = BoardConfig(
            board_w=puzzle["board_w"],
            board_h=puzzle["board_h"],
            num_walls=puzzle["num_walls"],
            num_mice=puzzle["num_mice"],
            num_rockets=puzzle["num_rockets"],
            num_cats=puzzle["num_cats"],
            num_holes=puzzle["num_holes"],
            arrow_budget=puzzle["arrow_budget"],
        )

        builder = BoardBuilder(config, seed=puzzle["seed"])
        level = builder.generate_level(f"Eval_Puzzle_{puzzle['seed']}")
        engine = level.create_engine(puzzle_mode=True, max_steps=max_steps)

        arrows_placed = 0
        solved = False
        placed_arrows = []  # Track what arrows we place

        print(
            f"\n--- Evaluating Puzzle {puzzle['seed']} ({puzzle['difficulty_label']}) ---"
        )
        print(
            f"Board: {puzzle['board_w']}×{puzzle['board_h']}, Budget: {puzzle['arrow_budget']}"
        )
        print(f"BFS solution: {puzzle['bfs_solution']}")

        # Phase 1: Placement phase - place arrows up to budget
        max_attempts = 1  # puzzle["arrow_budget"]  # Allow some failed attempts
        attempts = 0

        while (
            engine.phase == GamePhase.PLACEMENT
            and arrows_placed < puzzle["arrow_budget"]
            and attempts < max_attempts
        ):
            # Get current state
            game_state = engine.to_dict()

            # Process through perception and state fusion
            perception_output = self.perception_processor.process(game_state)

            # Move all perception output tensors to device
            perception_output = perception_output.to(self.device)

            fusion_output = self.state_fusion_processor.fuse(perception_output)

            # Get policy decision with consistent masking
            state_embedding = fusion_output.fused_embedding.unsqueeze(0).to(
                self.device
            )  # Add batch dimension

            # Use PolicyProcessor's masking method to ensure consistency
            masked_logits = self.policy_processor.forward_with_board_mask(
                state_embedding, puzzle["board_w"], puzzle["board_h"]
            )

            # Top-k sampling
            if attempts == 0:
                topk = torch.topk(masked_logits, k=5, dim=1).indices.tolist()
                print(f"  Top actions (up to 10):")
                for rank, a_id in enumerate(topk[0][:10]):
                    action_info = decode_action(a_id)
                    print(action_info)

            # Select action (greedy)
            action_idx = torch.argmax(masked_logits, dim=1).item()

            # Decode and apply action to game engine
            action_info = decode_action(action_idx)
            success = self._apply_action_to_engine(engine, action_info)
            attempts += 1

            if success:
                arrows_placed += 1
                placed_arrows.append(
                    f"({action_info.x},{action_info.y},{action_info.action_type})"
                )
                print(
                    f"  Placed arrow #{arrows_placed}: ({action_info.x}, {action_info.y}) {action_info.action_type}"
                )
            else:
                print(
                    f"  Failed attempt #{attempts}: ({action_info.x}, {action_info.y}) {action_info.action_type}"
                )

        if attempts >= max_attempts:
            print(
                f"  Warning: Reached max attempts ({max_attempts}) with only {arrows_placed} arrows placed"
            )

        # Phase 2: Start the game (transition to running phase)
        if engine.phase == GamePhase.PLACEMENT:
            print(f"  Starting game with {arrows_placed} arrows placed")
            engine.start_game()

        # Phase 3: Run simulation until completion
        simulation_steps = 0
        while engine.result == GameResult.ONGOING and simulation_steps < max_steps:
            engine.step()
            simulation_steps += 1

        # Check final result
        solved = engine.result == GameResult.SUCCESS

        print(f"  Simulation ran for {simulation_steps} steps")
        print(
            f"  Final result: {engine.result.name if hasattr(engine.result, 'name') else str(engine.result)}"
        )
        print(f"  Solved: {'YES' if solved else 'NO'}")
        print(f"  Arrows placed: {placed_arrows}")

        # Calculate efficiency vs optimal (BFS solution length)
        optimal_steps = len(puzzle["bfs_solution"])
        efficiency = optimal_steps / max(arrows_placed, 1) if solved else 0.0

        return {
            "solved": solved,
            "steps_taken": arrows_placed,  # Number of arrows placed
            "simulation_steps": simulation_steps,  # Steps during running phase
            "optimal_steps": optimal_steps,
            "efficiency": efficiency,
            "puzzle_seed": puzzle["seed"],
            "difficulty": puzzle["difficulty_label"],
            "final_result": (
                engine.result.name
                if hasattr(engine.result, "name")
                else str(engine.result)
            ),
        }

    def _apply_action_to_engine(self, engine: GameEngine, action_info) -> bool:
        """Apply decoded action to the game engine.

        Args:
            engine: Game engine instance
            action_info: Decoded action information

        Returns:
            True if action was successfully applied
        """
        x, y = action_info.x, action_info.y

        if action_info.action_type == "erase":
            return engine.remove_arrow(x, y)
        elif action_info.action_type == "place_up":
            return engine.place_arrow(x, y, Direction.UP)
        elif action_info.action_type == "place_down":
            return engine.place_arrow(x, y, Direction.DOWN)
        elif action_info.action_type == "place_left":
            return engine.place_arrow(x, y, Direction.LEFT)
        elif action_info.action_type == "place_right":
            return engine.place_arrow(x, y, Direction.RIGHT)
        else:
            print(f"Warning: Unknown action type: {action_info.action_type}")
            return False

    def evaluate_action_accuracy(self, test_loader) -> Dict[str, float]:
        """Evaluate action prediction accuracy on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of accuracy metrics
        """
        total_correct = 0
        total_samples = 0
        correct_by_action_type = defaultdict(int)
        total_by_action_type = defaultdict(int)

        with torch.no_grad():
            for (
                state_embeddings,
                action_targets,
                action_masks,
                arrow_budgets,
            ) in test_loader:
                state_embeddings = state_embeddings.to(self.device)
                action_targets = action_targets.to(self.device)
                action_masks = action_masks.to(self.device)

                # Forward pass - get raw logits from policy head
                logits = self.policy_processor.policy_head(state_embeddings)

                # Apply action mask before computing predictions
                masked_logits = apply_action_mask(logits, action_masks)
                predictions = torch.argmax(masked_logits, dim=1)

                # Overall accuracy
                correct = (predictions == action_targets).sum().item()
                total_correct += correct
                total_samples += action_targets.size(0)

                # Accuracy by action type
                for i in range(action_targets.size(0)):
                    pred = predictions[i].item()
                    target = action_targets[i].item()

                    # Determine action type
                    target_type = self._get_action_type(target)
                    total_by_action_type[target_type] += 1

                    if pred == target:
                        correct_by_action_type[target_type] += 1

        # Calculate metrics
        metrics = {
            "test_accuracy": total_correct / total_samples if total_samples > 0 else 0.0
        }

        # Add per-action-type accuracy
        for action_type in correct_by_action_type:
            if total_by_action_type[action_type] > 0:
                accuracy = (
                    correct_by_action_type[action_type]
                    / total_by_action_type[action_type]
                )
                metrics[f"{action_type}_accuracy"] = accuracy

        return metrics

    def evaluate_multihot_probability_analysis(self, test_loader) -> Dict[str, float]:
        """Evaluate model probability analysis on multi-hot targets.

        Analyzes:
        - correct_tile_probs: Average probability of ground truth actions
        - topk_probs: Average probability of top-k predictions

        This helps diagnose whether the model sees correct tiles but ranks them poorly
        vs. having completely wrong target encoding.

        Args:
            test_loader: Test data loader with multi-hot targets

        Returns:
            Dictionary with probability analysis metrics
        """
        total_correct_tile_prob = 0.0
        total_topk_prob = 0.0
        total_samples = 0
        k = 5  # Top-k to analyze

        with torch.no_grad():
            for (
                state_embeddings,
                action_targets,  # Multi-hot targets [batch_size, 700]
                action_masks,
                arrow_budgets,
                board_ws,
                board_hs,
            ) in test_loader:
                state_embeddings = state_embeddings.to(self.device)
                action_targets = action_targets.to(self.device)
                action_masks = action_masks.to(self.device)

                # Forward pass - get raw logits and convert to probabilities
                logits = self.policy_processor.policy_head(state_embeddings)
                probs = torch.sigmoid(logits)

                batch_size = action_targets.size(0)
                for i in range(batch_size):
                    sample_probs = probs[i]
                    sample_targets = action_targets[i]
                    sample_mask = action_masks[i]

                    # Get ground truth action indices
                    gt_indices = sample_targets.nonzero(as_tuple=True)[0]
                    if len(gt_indices) == 0:
                        continue  # Skip samples with no ground truth

                    # Correct tile probabilities: average prob of ground truth actions
                    correct_tile_probs = sample_probs[gt_indices]
                    avg_correct_prob = correct_tile_probs.mean().item()
                    total_correct_tile_prob += avg_correct_prob

                    # Top-k probabilities: get top-k from valid actions only
                    valid_indices = sample_mask.bool()
                    if valid_indices.sum() == 0:
                        continue

                    valid_probs = sample_probs[valid_indices]
                    actual_k = min(k, valid_probs.size(0))
                    topk_probs = torch.topk(valid_probs, actual_k).values
                    avg_topk_prob = topk_probs.mean().item()
                    total_topk_prob += avg_topk_prob

                    total_samples += 1

        # Calculate averages
        metrics = {}
        if total_samples > 0:
            metrics["avg_correct_tile_prob"] = total_correct_tile_prob / total_samples
            metrics["avg_topk_prob"] = total_topk_prob / total_samples
            metrics["prob_ratio"] = (
                (total_correct_tile_prob / total_samples)
                / (total_topk_prob / total_samples)
                if total_topk_prob > 0
                else 0.0
            )
        else:
            metrics["avg_correct_tile_prob"] = 0.0
            metrics["avg_topk_prob"] = 0.0
            metrics["prob_ratio"] = 0.0

        return metrics

    def _get_action_type(self, action_idx: int) -> str:
        """Get action type string from action index.

        Args:
            action_idx: Action index (0-699)

        Returns:
            Action type string
        """
        if 0 <= action_idx < 140:
            return "place_up"
        elif 140 <= action_idx < 280:
            return "place_down"
        elif 280 <= action_idx < 420:
            return "place_left"
        elif 420 <= action_idx < 560:
            return "place_right"
        elif 560 <= action_idx < 700:
            return "erase"
        else:
            return "unknown"

    def generate_evaluation_report(
        self, test_puzzles: List[Dict], test_loader
    ) -> Dict[str, float]:
        """Generate comprehensive evaluation report.

        Args:
            test_puzzles: List of test puzzle dictionaries
            test_loader: Test data loader

        Returns:
            Dictionary with all evaluation metrics
        """
        print("Evaluating action accuracy...")
        accuracy_metrics = self.evaluate_action_accuracy(test_loader)

        print("Evaluating puzzle solving performance...")
        solving_metrics = self.evaluate_puzzle_solving(test_puzzles)

        # Combine all metrics
        all_metrics = {**accuracy_metrics, **solving_metrics}

        # Print report
        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)

        print(f"\nAction Accuracy:")
        print(f"  Overall: {all_metrics['test_accuracy']:.4f}")
        for action_type in [
            "place_up",
            "place_down",
            "place_left",
            "place_right",
            "erase",
        ]:
            key = f"{action_type}_accuracy"
            if key in all_metrics:
                print(f"  {action_type}: {all_metrics[key]:.4f}")

        print(f"\nPuzzle Solving:")
        print(f"  Overall solve rate: {all_metrics['overall_solve_rate']:.4f}")
        print(f"  Overall efficiency: {all_metrics['overall_efficiency']:.4f}")

        for difficulty in ["Easy", "Medium", "Hard"]:
            solve_key = f"{difficulty.lower()}_solve_rate"
            if solve_key in all_metrics:
                print(f"  {difficulty} solve rate: {all_metrics[solve_key]:.4f}")

        print("=" * 50)

        return all_metrics
