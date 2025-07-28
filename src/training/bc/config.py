"""
Configuration for behaviour cloning training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BCConfig:
    """Configuration for behaviour cloning training."""

    # Data parameters
    csv_path: str = "data/puzzles.csv"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 200
    patience: int = 50  # Early stopping patience
    min_epochs: int = 100  # Minimum epochs before early stopping

    # Model parameters
    gradient_clipping: float = 1.0
    weight_decay: float = 1e-4
    pos_weight: float = 100.0  # Positive weight for BCE loss
    label_smoothing: float = 0.05  # Label smoothing epsilon

    # Loss function flags
    use_focal_loss: bool = True  # Use Focal BCE instead of standard BCE
    focal_alpha: float = 5.0  # Focal loss alpha parameter (class balance)
    focal_gamma: float = 2.0  # Focal loss gamma parameter (hard example focus)
    use_negative_logit_penalty: bool = True  # Add penalty for positive logits
    negative_logit_penalty_weight: float = 1e-4  # Weight for negative logit penalty

    # Top-k suppression
    use_topk_suppression: bool = False  # Apply top-(k+2) suppression before loss
    topk_suppression_value: float = -2.0  # Value to set suppressed logits to

    # Evaluation parameters
    val_frequency_steps: int = 5000  # Steps between validation runs (legacy)
    val_frequency_epochs: int = 10  # Epochs between validation runs
    val_subset_fraction: float = (
        0.2  # Fraction of validation data to use for metrics (0.2 = 20%)
    )
    save_frequency: int = 5000  # Steps between checkpoints

    # Logging
    log_frequency: int = 1000  # Steps between training logs

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4  # DataLoader workers

    # Model paths
    model_dir: str = "model"
    checkpoint_name: str = "bc_checkpoint"
    best_model_name: str = "bc_best.pth"
    final_model_name: str = "bc_final.pth"

    # Success criteria
    target_easy_solve_rate: float = 0.95
    target_medium_solve_rate: float = 0.70
    target_hard_solve_rate: float = 0.50

    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 < self.train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1")
        if not (0.0 < self.val_split < 1.0):
            raise ValueError("val_split must be between 0 and 1")
        if not (0.0 < self.test_split < 1.0):
            raise ValueError("test_split must be between 0 and 1")
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
