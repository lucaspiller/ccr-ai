"""
Utility functions for PPO training.

Helper functions for model loading, action masking, scheduling, and other
common operations used throughout PPO training.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...model.model_loader import ModelLoader
from ...perception.processors import GameStateProcessor
from ...policy.processors import PolicyProcessor
from ...state_fusion.processors import StateFusionProcessor
from .config import PPOConfig


def get_device(device_str: str = "auto") -> torch.device:
    """Get training device with automatic detection.

    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "mps")

    Returns:
        PyTorch device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
    else:
        device = torch.device(device_str)
        print(f"Using specified device: {device}")

    return device


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_ppo_parameters(
    perception_processor: GameStateProcessor,
    state_fusion_processor: StateFusionProcessor,
    policy_processor: PolicyProcessor,
    value_head: nn.Module,
) -> Dict[str, int]:
    """Count parameters in all PPO model components.

    Args:
        perception_processor: Perception processor
        state_fusion_processor: State fusion processor
        policy_processor: Policy processor
        value_head: Value head module

    Returns:
        Dictionary with parameter counts per component
    """
    counts = {
        "cnn_encoder": count_parameters(perception_processor.get_cnn_encoder()),
        "state_fusion": count_parameters(state_fusion_processor.fusion_mlp),
        "policy_head": count_parameters(policy_processor.policy_head),
        "value_head": count_parameters(value_head),
    }
    counts["total"] = sum(counts.values())

    return counts


def create_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_type: str,
    total_steps: int,
    warmup_steps: int = 0,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        schedule_type: Type of schedule ("cosine", "linear", "constant")
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    if schedule_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=kwargs.get("eta_min", optimizer.param_groups[0]["lr"] * 0.1),
        )
    elif schedule_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get("end_factor", 0.1),
            total_iters=total_steps,
        )
    elif schedule_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Add warmup if specified
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_steps],
        )

    return scheduler


def save_ppo_checkpoint(
    filepath: str,
    perception_processor: GameStateProcessor,
    state_fusion_processor: StateFusionProcessor,
    policy_processor: PolicyProcessor,
    value_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    config: PPOConfig,
    additional_data: Optional[Dict[str, Any]] = None,
):
    """Save PPO training checkpoint.

    Args:
        filepath: Path to save checkpoint
        perception_processor: Perception processor
        state_fusion_processor: State fusion processor
        policy_processor: Policy processor
        value_head: Value head module
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        global_step: Current training step
        config: PPO configuration
        additional_data: Additional data to save
    """
    checkpoint = {
        "global_step": global_step,
        "cnn_encoder": perception_processor.get_cnn_encoder().state_dict(),
        "state_fusion": state_fusion_processor.state_dict(),
        "policy_head": policy_processor.state_dict(),
        "value_head": value_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }

    if additional_data:
        checkpoint.update(additional_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance for value function evaluation.

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Explained variance (higher is better, 1.0 is perfect)
    """
    with torch.no_grad():
        var_y = torch.var(y_true)
        if var_y < 1e-8:
            return 0.0

        explained_var = 1 - torch.var(y_true - y_pred) / var_y
        return explained_var.item()


def linear_schedule(start_value: float, end_value: float, progress: float) -> float:
    """Linear interpolation between start and end values.

    Args:
        start_value: Starting value
        end_value: Ending value
        progress: Progress fraction (0.0 to 1.0)

    Returns:
        Interpolated value
    """
    progress = np.clip(progress, 0.0, 1.0)
    return start_value + progress * (end_value - start_value)


def cosine_schedule(start_value: float, end_value: float, progress: float) -> float:
    """Cosine interpolation between start and end values.

    Args:
        start_value: Starting value
        end_value: Ending value
        progress: Progress fraction (0.0 to 1.0)

    Returns:
        Interpolated value
    """
    progress = np.clip(progress, 0.0, 1.0)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return end_value + (start_value - end_value) * cosine_decay


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


def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes.

    Args:
        num: Number to format

    Returns:
        Formatted number string
    """
    if isinstance(num, float):
        if abs(num) < 1000:
            return f"{num:.2f}"
        elif abs(num) < 1_000_000:
            return f"{num/1000:.1f}K"
        else:
            return f"{num/1_000_000:.1f}M"
    else:
        if abs(num) < 1000:
            return str(num)
        elif abs(num) < 1_000_000:
            return f"{num//1000}K"
        else:
            return f"{num//1_000_000}M"


def create_directories(config: PPOConfig):
    """Create necessary directories for PPO training.

    Args:
        config: PPO configuration
    """
    directories = [
        config.model_dir,
        config.log_dir,
        os.path.join(config.log_dir, "tensorboard"),
        os.path.join(config.model_dir, "checkpoints"),
        os.path.join(config.model_dir, "visualizations"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


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
    print(f"Total Steps: {format_number(config.total_env_steps)}")
    print(f"Rollout Length: {format_number(config.rollout_length)}")
    print(f"Parallel Envs: {config.num_parallel_envs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Discount Factor: {config.discount_factor}")
    print(f"GAE Lambda: {config.gae_lambda}")
    print("=" * 80)


def print_model_summary(
    perception_processor: GameStateProcessor,
    state_fusion_processor: StateFusionProcessor,
    policy_processor: PolicyProcessor,
    value_head: nn.Module,
):
    """Print model architecture summary.

    Args:
        perception_processor: Perception processor
        state_fusion_processor: State fusion processor
        policy_processor: Policy processor
        value_head: Value head module
    """
    param_counts = count_ppo_parameters(
        perception_processor, state_fusion_processor, policy_processor, value_head
    )

    print("\nMODEL ARCHITECTURE")
    print("-" * 40)
    print(f"CNN Encoder:    {format_number(param_counts['cnn_encoder']):>8} params")
    print(f"State Fusion:   {format_number(param_counts['state_fusion']):>8} params")
    print(f"Policy Head:    {format_number(param_counts['policy_head']):>8} params")
    print(f"Value Head:     {format_number(param_counts['value_head']):>8} params")
    print("-" * 40)
    print(f"Total:          {format_number(param_counts['total']):>8} params")
    print("-" * 40)


# Test utilities
def test_utils():
    """Test utility functions."""
    print("Testing PPO utilities...")

    # Test device selection
    device = get_device("auto")
    print(f"Selected device: {device}")

    # Test formatting
    print(f"Format large number: {format_number(1_234_567)}")
    print(f"Format time: {format_training_time(3661.5)}")

    # Test scheduling
    progress_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("Linear schedule (1.0 -> 0.1):")
    for p in progress_values:
        val = linear_schedule(1.0, 0.1, p)
        print(f"  Progress {p:.2f}: {val:.3f}")

    print("Cosine schedule (1.0 -> 0.1):")
    for p in progress_values:
        val = cosine_schedule(1.0, 0.1, p)
        print(f"  Progress {p:.2f}: {val:.3f}")

    print("Utilities test completed!")


if __name__ == "__main__":
    test_utils()
