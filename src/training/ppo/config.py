"""
Configuration for PPO self-play training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """Configuration for PPO self-play training."""

    # Data parameters
    bc_model_path: str = "model/bc_final.pth"
    freeze_backbone: bool = True  # Freeze CNN and fusion layers, only train policy head

    # Environment parameters
    num_parallel_envs: int = 16
    placement_timeout: int = 10  # Max steps for placement phase
    execution_timeout: int = 1800  # Max ticks for execution phase (30 seconds at 60fps)
    early_terminate_on_success: bool = True  # Terminate when all mice scored

    # PPO Algorithm parameters
    learning_rate: float = 1e-4  # Reduced from 3e-4 for better stability
    lr_schedule: str = "cosine"  # cosine, linear, constant
    lr_warmup_steps: int = 10000  # 5% warmup

    discount_factor: float = 0.99
    gae_lambda: float = 0.95

    # Clipping parameters with decay
    clip_eps_start: float = 0.2
    clip_eps_end: float = 0.1
    value_clip_eps_start: float = 0.2
    value_clip_eps_end: float = 0.1

    # Entropy parameters with decay
    entropy_coeff_start: float = 0.01
    entropy_coeff_end: float = 0.001
    entropy_decay_threshold: float = 0.8  # Start decay when solve rate > 80%

    # KL divergence control
    kl_target: float = 0.01
    kl_adaptive_beta: bool = True  # If KL > 2×target, stop epoch early

    # Training schedule
    rollout_length: int = 2048  # Steps per update
    batch_size: int = 64  # Minibatch size during updates
    ppo_epochs: int = 4  # Reuse each batch N times
    max_grad_norm: float = 0.5  # Reduced from 1.0 for tighter gradient control

    # Value function parameters
    value_loss_coeff: float = 0.25  # Reduced from 0.5 to reduce value head impact
    value_head_init_std: float = (
        1e-4  # Small initialization to avoid large early advantages
    )

    # Training parameters
    total_env_steps: int = 10_000_000
    eval_frequency: int = 100_000  # Steps between evaluations
    log_frequency: int = 10_000  # Steps between training logs

    # Continuous training options
    continuous_training: bool = False  # Run indefinitely, ignore total_env_steps
    max_training_hours: float = (
        8.0  # Maximum training time in hours (for overnight runs)
    )
    regenerate_puzzles_when_exhausted: bool = (
        True  # Regenerate puzzle set when curriculum exhausted
    )

    # Curriculum parameters
    curriculum_window_steps: int = 10_000  # Sliding window for curriculum advancement
    curriculum_success_threshold: float = 0.8  # Advance when solve rate ≥ 80%
    curriculum_easy_only: bool = False  # Only train on easy puzzles (no progression)
    curriculum_max_difficulty: str = (
        "hard"  # Maximum difficulty level (easy, medium, hard)
    )

    # Reward function parameters (normalized for PPO stability)
    reward_mouse_saved: float = 1.0
    reward_cat_fed: float = -0.1
    reward_mouse_lost_hole: float = -0.1
    reward_arrow_cost: float = -0.05
    reward_success_bonus: float = 5.0
    reward_failure_penalty: float = 0

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4  # DataLoader workers

    # Model paths
    model_dir: str = "model"
    checkpoint_name: str = "ppo_checkpoint"
    best_model_name: str = "ppo_best.pth"
    final_model_name: str = "ppo_final.pth"

    # Logging
    log_dir: str = "logs/ppo"
    use_tensorboard: bool = True
    verbose_env_logging: bool = False  # Detailed environment logging (for evaluation)

    # Success criteria
    target_medium_solve_rate: float = 0.85  # +15pp over 70% BC baseline
    target_hard_solve_rate: float = 0.60  # +10pp over 50% BC baseline
    target_efficiency_ratio: float = 1.3  # Average steps ≤ 1.3× BFS optimal

    # Early stopping for continuous training
    early_stop_on_target: bool = False  # Stop when targets achieved
    patience_hours: float = 2.0  # Stop if no improvement for N hours

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 < self.discount_factor <= 1.0):
            raise ValueError("discount_factor must be in (0, 1]")
        if not (0.0 < self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in (0, 1]")

        if self.clip_eps_start <= 0 or self.clip_eps_end <= 0:
            raise ValueError("clip epsilon values must be positive")
        if self.entropy_coeff_start < 0 or self.entropy_coeff_end < 0:
            raise ValueError("entropy coefficients must be non-negative")

        if self.rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")

        if self.num_parallel_envs <= 0:
            raise ValueError("num_parallel_envs must be positive")
        if self.placement_timeout <= 0:
            raise ValueError("placement_timeout must be positive")
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")
