"""
Curriculum system for PPO training.

Manages difficulty progression based on agent performance, generating puzzles
that match the current skill level and advancing when success thresholds are met.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PPOConfig
from .puzzle_generator import PuzzleGenerator


class CurriculumStage:
    """A single stage in the curriculum."""

    def __init__(self, name: str, target_solve_rate: float = 0.8):
        self.name = name
        self.target_solve_rate = target_solve_rate

        # Performance tracking
        self.performance_window = deque(maxlen=100)  # Last 100 episodes
        self.total_episodes = 0
        self.successful_episodes = 0

    def add_episode_result(self, success: bool, reward: float, steps: int):
        """Add an episode result to performance tracking."""
        self.performance_window.append(
            {"success": success, "reward": reward, "steps": steps}
        )
        self.total_episodes += 1
        if success:
            self.successful_episodes += 1

    def get_solve_rate(self) -> float:
        """Get current solve rate over the performance window."""
        if not self.performance_window:
            return 0.0

        successes = sum(1 for ep in self.performance_window if ep["success"])
        return successes / len(self.performance_window)

    def get_avg_reward(self) -> float:
        """Get average reward over the performance window."""
        if not self.performance_window:
            return 0.0

        return np.mean([ep["reward"] for ep in self.performance_window])

    def get_avg_steps(self) -> float:
        """Get average steps over the performance window."""
        if not self.performance_window:
            return 0.0

        return np.mean([ep["steps"] for ep in self.performance_window])

    def is_mastered(self) -> bool:
        """Check if this stage has been mastered."""
        return (
            len(self.performance_window) >= 50  # Need sufficient data
            and self.get_solve_rate() >= self.target_solve_rate
        )


class CurriculumManager:
    """Manages curriculum progression for PPO training."""

    def __init__(self, config: PPOConfig):
        self.config = config

        # Create puzzle generator
        self.puzzle_generator = PuzzleGenerator()

        # Get puzzle specs
        self.puzzle_specs = self.puzzle_generator.get_puzzle_specs()

        # Define curriculum stages
        self.stages = self._create_curriculum_stages()
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]

        # Random seed for board generation (use different seed each run for variety)
        self.rng = random.Random()

        # Puzzle regeneration settings
        self.max_puzzles_per_difficulty = 1000  # Generate new puzzles after this many

        # Performance tracking
        self.global_performance_window = deque(maxlen=1000)
        self.stage_transitions = []

        print(f"Curriculum initialized with {len(self.stages)} stages")
        print(f"Starting with stage: {self.current_stage.name}")
        if config.curriculum_easy_only:
            print("⚠️  Easy-only mode: curriculum advancement disabled")
        if config.regenerate_puzzles_when_exhausted:
            print(
                f"🔄 Puzzle regeneration enabled every {self.max_puzzles_per_difficulty} puzzles"
            )

    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create the curriculum stages as specified in PRD."""
        stages = []

        # Create stages with adjusted target solve rate for curriculum cats_range
        stages.append(CurriculumStage("Easy", target_solve_rate=0.8))
        stages.append(CurriculumStage("Medium", target_solve_rate=0.8))
        stages.append(CurriculumStage("Hard", target_solve_rate=0.6))

        return stages

    def update(self, rollout_stats: Dict[str, Any]):
        """Update curriculum based on recent performance."""
        # Extract episode results from rollout stats
        episode_rewards = rollout_stats.get("episode_rewards", [])
        episode_lengths = rollout_stats.get("episode_lengths", [])

        # Process each completed episode
        for reward, length in zip(episode_rewards, episode_lengths):
            # Determine if episode was successful (positive reward indicating puzzle solved)
            success = reward > 0

            # Add to current stage performance
            self.current_stage.add_episode_result(success, reward, length)

            # Add to global performance tracking
            self.global_performance_window.append(
                {
                    "stage": self.current_stage.name,
                    "success": success,
                    "reward": reward,
                    "steps": length,
                }
            )

        # Check for stage advancement
        self._check_stage_advancement()

    def _check_stage_advancement(self):
        """Check if we should advance to the next curriculum stage."""
        if (
            self.current_stage.is_mastered()
            and self.current_stage_idx < len(self.stages) - 1
        ):
            # Advance to next stage
            old_stage = self.current_stage.name
            self.current_stage_idx += 1
            self.current_stage = self.stages[self.current_stage_idx]

            # Record transition
            self.stage_transitions.append(
                {
                    "from_stage": old_stage,
                    "to_stage": self.current_stage.name,
                    "episodes_completed": sum(
                        stage.total_episodes
                        for stage in self.stages[: self.current_stage_idx]
                    ),
                    "solve_rate": self.stages[
                        self.current_stage_idx - 1
                    ].get_solve_rate(),
                }
            )

            print(f"\n🎓 CURRICULUM ADVANCEMENT 🎓")
            print(f"Advanced from {old_stage} to {self.current_stage.name}")
            print(
                f"Previous stage solve rate: {self.stages[self.current_stage_idx - 1].get_solve_rate():.3f}"
            )
            print(
                f"Total episodes completed: {sum(stage.total_episodes for stage in self.stages)}"
            )

    def get_puzzle_batch(self, batch_size: int) -> List[Any]:
        """Generate a batch of puzzle configs for the current curriculum stage."""
        puzzle_configs = []

        # Get the appropriate puzzle spec for current stage
        stage_name_lower = self.current_stage.name.lower()
        puzzle_spec = self.puzzle_specs[stage_name_lower]

        for _ in range(batch_size):
            puzzle_config = self.puzzle_generator.generate_puzzle_config(
                puzzle_spec,
                self.rng,
                puzzle_id=f"{stage_name_lower}_{self.current_stage.total_episodes:03d}",
                difficulty=self.current_stage.name.lower(),
            )
            puzzle_configs.append(puzzle_config)

        return puzzle_configs

    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about the current curriculum stage."""
        return {
            "stage_name": self.current_stage.name,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "solve_rate": self.current_stage.get_solve_rate(),
            "avg_reward": self.current_stage.get_avg_reward(),
            "avg_steps": self.current_stage.get_avg_steps(),
            "episodes_in_stage": self.current_stage.total_episodes,
            "is_mastered": self.current_stage.is_mastered(),
            "target_solve_rate": self.current_stage.target_solve_rate,
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global curriculum statistics."""
        if not self.global_performance_window:
            return {
                "total_episodes": 0,
                "overall_solve_rate": 0.0,
                "stage_distribution": {},
                "transitions": len(self.stage_transitions),
            }

        # Overall statistics
        total_episodes = len(self.global_performance_window)
        successful_episodes = sum(
            1 for ep in self.global_performance_window if ep["success"]
        )
        overall_solve_rate = successful_episodes / total_episodes

        # Stage distribution
        stage_counts = {}
        for ep in self.global_performance_window:
            stage = ep["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return {
            "total_episodes": total_episodes,
            "overall_solve_rate": overall_solve_rate,
            "stage_distribution": stage_counts,
            "transitions": len(self.stage_transitions),
            "transition_history": self.stage_transitions.copy(),
        }

    def reset_stage_performance(self):
        """Reset performance tracking for current stage (useful for debugging)."""
        self.current_stage.performance_window.clear()
        self.current_stage.total_episodes = 0
        self.current_stage.successful_episodes = 0
        print(f"Reset performance tracking for stage: {self.current_stage.name}")

    def force_advance_stage(self):
        """Force advancement to next stage (useful for debugging)."""
        if self.current_stage_idx < len(self.stages) - 1:
            old_stage = self.current_stage.name
            self.current_stage_idx += 1
            self.current_stage = self.stages[self.current_stage_idx]
            print(f"Forced advancement from {old_stage} to {self.current_stage.name}")
        else:
            print("Already at final curriculum stage")

    def set_stage(self, stage_name: str):
        """Set curriculum to specific stage by name."""
        for i, stage in enumerate(self.stages):
            if stage.name.lower() == stage_name.lower():
                self.current_stage_idx = i
                self.current_stage = stage
                print(f"Set curriculum to stage: {stage_name}")
                return

        available_stages = [stage.name for stage in self.stages]
        raise ValueError(
            f"Stage '{stage_name}' not found. Available: {available_stages}"
        )


# Example usage and testing
def test_curriculum():
    """Test curriculum system."""
    from .config import PPOConfig

    config = PPOConfig()
    curriculum = CurriculumManager(config)

    print("Testing curriculum system...")

    # Generate some puzzle configs
    puzzle_configs = curriculum.get_puzzle_batch(3)
    print(
        f"Generated {len(puzzle_configs)} puzzle configs for stage: {curriculum.current_stage.name}"
    )

    for i, puzzle_config in enumerate(puzzle_configs):
        print(
            f"Puzzle {i+1}: {puzzle_config.width}x{puzzle_config.height}, "
            f"{len(puzzle_config.mice)} mice, {len(puzzle_config.cats)} cats, "
            f"budget: {puzzle_config.arrow_budget}"
        )

    # Simulate some episode results
    print("\nSimulating episode results...")
    for i in range(60):
        # Simulate gradually improving performance
        success_rate = min(0.9, i / 50.0)
        success = random.random() < success_rate
        reward = 5.0 if success else -1.0
        steps = random.randint(20, 100)

        rollout_stats = {"episode_rewards": [reward], "episode_lengths": [steps]}
        curriculum.update(rollout_stats)

        if i % 10 == 0:
            stage_info = curriculum.get_current_stage_info()
            print(
                f"Episode {i}: Stage {stage_info['stage_name']}, "
                f"Solve rate: {stage_info['solve_rate']:.3f}"
            )

    # Print final stats
    global_stats = curriculum.get_global_stats()
    print(f"\nFinal stats:")
    print(f"Total episodes: {global_stats['total_episodes']}")
    print(f"Overall solve rate: {global_stats['overall_solve_rate']:.3f}")
    print(f"Stage transitions: {global_stats['transitions']}")
    print(f"Current stage: {curriculum.current_stage.name}")


if __name__ == "__main__":
    test_curriculum()
