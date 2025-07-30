"""
PPO Self-Play Training Module for ChuChu Rocket.

This module implements PPO (Proximal Policy Optimization) self-play training
to advance beyond the BC baseline. The training uses a curriculum system
that progresses from easy to hard puzzles based on performance.

Key Components:
- PPOTrainer: Main training loop with action masking and value head
- PPOEnvironment: Environment wrapper handling two-phase puzzle structure
- RolloutBuffer: Experience collection with GAE advantage computation
- CurriculumManager: Dynamic difficulty progression system
- PPOEvaluator: Performance tracking and baseline comparison

Usage:
    from src.training.ppo import PPOConfig, PPOTrainer

    config = PPOConfig()
    trainer = PPOTrainer(config)
    results = trainer.train()

Or run directly:
    python -m src.training.ppo.train_ppo
"""

from .config import PPOConfig
from .curriculum import CurriculumManager
from .ppo_env import PPOEnvironment, PPOEnvironmentManager
from .ppo_evaluator import EvaluationResult, PPOEvaluator
from .ppo_trainer import PPOTrainer
from .rollout_buffer import PPOExperience, RolloutBuffer
from .train_ppo import PPOTrainingManager

__all__ = [
    # Main classes
    "PPOConfig",
    "PPOTrainer",
    "PPOTrainingManager",
    # Environment
    "PPOEnvironment",
    "PPOEnvironmentManager",
    # Experience collection
    "RolloutBuffer",
    "PPOExperience",
    # Curriculum
    "CurriculumManager",
    # Evaluation
    "PPOEvaluator",
    "EvaluationResult",
]

# Version info
__version__ = "1.0.0"
__description__ = "PPO Self-Play Training for ChuChu Rocket puzzle solving"
