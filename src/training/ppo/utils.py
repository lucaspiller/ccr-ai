"""
Utility functions for PPO training.

Helper functions for model loading, action masking, scheduling, and other
common operations used throughout PPO training.
"""

import os
from typing import List

from .config import PPOConfig

def format_training_time(seconds: float) -> str:
    """Format training time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def validate_config(config: PPOConfig) -> List[str]:
    """Validate PPO configuration and return list of issues.

    Args:
        config: PPO configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check BC model exists
    if not os.path.exists(config.bc_model_path):
        issues.append(f"BC model not found: {config.bc_model_path}")

    # Check hyperparameter ranges
    if config.learning_rate <= 0:
        issues.append("Learning rate must be positive")

    if not (0 < config.discount_factor <= 1):
        issues.append("Discount factor must be in (0, 1]")

    if not (0 < config.gae_lambda <= 1):
        issues.append("GAE lambda must be in (0, 1]")

    if config.rollout_length <= 0:
        issues.append("Rollout length must be positive")

    if config.batch_size <= 0:
        issues.append("Batch size must be positive")

    if config.num_parallel_envs <= 0:
        issues.append("Number of parallel environments must be positive")

    # Check that batch size is reasonable relative to rollout length
    total_samples = config.rollout_length * config.num_parallel_envs
    if config.batch_size > total_samples:
        issues.append(
            f"Batch size ({config.batch_size}) larger than total samples ({total_samples})"
        )

    return issues


def print_training_header(config: PPOConfig):
    """Print training configuration header.

    Args:
        config: PPO configuration
    """
    print("=" * 80)
    print("PPO SELF-PLAY TRAINING")
    print("=" * 80)
    print(f"BC Model: {config.bc_model_path}")
    print(f"Total Steps: {config.total_env_steps:,}")
    print(f"Rollout Length: {config.rollout_length:,}")
    print(f"Parallel Envs: {config.num_parallel_envs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Discount Factor: {config.discount_factor}")
    print(f"GAE Lambda: {config.gae_lambda}")
    print("=" * 80)
