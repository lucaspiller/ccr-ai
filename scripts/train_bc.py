#!/usr/bin/env python3
"""
Behaviour Cloning Training Script

Train the ChuChu Rocket AI using supervised learning on BFS optimal solutions.
This script provides command line interface for configuring all training parameters.

Usage:
    python scripts/train_bc.py --csv-path data/puzzles.csv --epochs 50 --batch-size 32
    python scripts/train_bc.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.bc import BCConfig, BCTrainer


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ChuChu Rocket AI using behaviour cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--csv-path",
        type=str,
        default="data/puzzles.csv",
        help="Path to CSV file with puzzle data",
    )
    data_group.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (0.0-1.0)",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (0.0-1.0)",
    )
    data_group.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for testing (0.0-1.0)",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    train_group.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    train_group.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    train_group.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (epochs)"
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    train_group.add_argument(
        "--gradient-clipping",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )

    # Logging and evaluation
    log_group = parser.add_argument_group("Logging and Evaluation")
    log_group.add_argument(
        "--log-frequency", type=int, default=1000, help="Steps between training logs"
    )
    log_group.add_argument(
        "--val-frequency-epochs",
        type=int,
        default=10,
        help="Epochs between validation runs",
    )
    log_group.add_argument(
        "--val-subset-fraction",
        type=float,
        default=0.2,
        help="Fraction of validation data to use for metrics (0.2 = 20%)",
    )
    log_group.add_argument(
        "--save-frequency",
        type=int,
        default=5000,
        help="Steps between model checkpoints",
    )

    # Hardware configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )
    hw_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of DataLoader workers"
    )

    # Model and output paths
    path_group = parser.add_argument_group("Model and Output Paths")
    path_group.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="Directory to save model checkpoints",
    )
    path_group.add_argument(
        "--checkpoint-name",
        type=str,
        default="bc_checkpoint",
        help="Base name for checkpoint files",
    )
    path_group.add_argument(
        "--best-model-name",
        type=str,
        default="bc_best.pth",
        help="Filename for best model",
    )
    path_group.add_argument(
        "--final-model-name",
        type=str,
        default="bc_final.pth",
        help="Filename for final model",
    )

    # Success criteria
    criteria_group = parser.add_argument_group("Success Criteria")
    criteria_group.add_argument(
        "--target-easy-solve-rate",
        type=float,
        default=0.95,
        help="Target solve rate for easy puzzles",
    )
    criteria_group.add_argument(
        "--target-medium-solve-rate",
        type=float,
        default=0.70,
        help="Target solve rate for medium puzzles",
    )
    criteria_group.add_argument(
        "--target-hard-solve-rate",
        type=float,
        default=0.50,
        help="Target solve rate for hard puzzles",
    )

    # Additional options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    misc_group.add_argument(
        "--evaluate-only", action="store_true", help="Only run evaluation, do not train"
    )
    misc_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    misc_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    errors = []

    # Check splits sum to 1.0
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 1e-6:
        errors.append(f"Data splits must sum to 1.0, got {total_split}")

    # Check file exists
    if not Path(args.csv_path).exists():
        errors.append(f"CSV file not found: {args.csv_path}")

    # Check positive values
    if args.batch_size <= 0:
        errors.append("batch-size must be positive")
    if args.learning_rate <= 0:
        errors.append("learning-rate must be positive")
    if args.epochs <= 0:
        errors.append("epochs must be positive")

    # Check ranges
    for split_name, split_val in [
        ("train-split", args.train_split),
        ("val-split", args.val_split),
        ("test-split", args.test_split),
    ]:
        if not (0.0 <= split_val <= 1.0):
            errors.append(f"{split_name} must be between 0.0 and 1.0")

    if errors:
        print("Argument validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def create_config_from_args(args):
    """Create BCConfig from command line arguments."""
    return BCConfig(
        # Data parameters
        csv_path=args.csv_path,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        # Logging and evaluation
        log_frequency=args.log_frequency,
        val_frequency_epochs=args.val_frequency_epochs,
        val_subset_fraction=args.val_subset_fraction,
        save_frequency=args.save_frequency,
        # Hardware
        device=args.device,
        num_workers=args.num_workers,
        # Model paths
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        best_model_name=args.best_model_name,
        final_model_name=args.final_model_name,
        # Success criteria
        target_easy_solve_rate=args.target_easy_solve_rate,
        target_medium_solve_rate=args.target_medium_solve_rate,
        target_hard_solve_rate=args.target_hard_solve_rate,
    )


def print_config_summary(config, args):
    """Print a summary of the training configuration."""
    print("=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)

    print(f"\nData Configuration:")
    print(f"  CSV Path: {config.csv_path}")
    print(
        f"  Data Splits: {config.train_split:.1%} train, {config.val_split:.1%} val, {config.test_split:.1%} test"
    )

    print(f"\nTraining Parameters:")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Max Epochs: {config.max_epochs}")
    print(f"  Early Stopping Patience: {config.patience}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Gradient Clipping: {config.gradient_clipping}")

    if config.use_focal_loss:
        print(f"  Focal Loss Alpha: {config.focal_alpha}")
        print(f"  Focal Loss Gamma: {config.focal_gamma}")
    else:
        print(f"  Using standard BCE loss")

    if config.use_negative_logit_penalty:
        print(
            f"  Negative Logit Penalty Weight: {config.negative_logit_penalty_weight}"
        )

    if config.use_topk_suppression:
        print(f"  Top-k Suppression: {config.topk_suppression_value}")

    print(f"\nHardware:")
    print(f"  Device: {config.device}")
    print(f"  DataLoader Workers: {config.num_workers}")

    print(f"\nModel Output:")
    print(f"  Model Directory: {config.model_dir}")
    print(f"  Checkpoint Name: {config.checkpoint_name}")

    print(f"\nSuccess Criteria:")
    print(f"  Easy Puzzles: {config.target_easy_solve_rate:.1%}")
    print(f"  Medium Puzzles: {config.target_medium_solve_rate:.1%}")
    print(f"  Hard Puzzles: {config.target_hard_solve_rate:.1%}")

    if args.resume:
        print(f"\nResuming from: {args.resume}")

    if args.evaluate_only:
        print(f"\nMode: EVALUATION ONLY")
    else:
        print(f"\nMode: TRAINING")

    print("=" * 60)


def main():
    """Main training function."""
    # Parse and validate arguments
    args = parse_args()
    validate_args(args)

    # Set random seed
    import random

    import numpy as np
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create configuration
    config = create_config_from_args(args)

    # Print configuration summary
    if args.verbose or not args.evaluate_only:
        print_config_summary(config, args)

    # Create trainer
    try:
        trainer = BCTrainer(config)
    except Exception as e:
        print(f"Error creating trainer: {e}")
        sys.exit(1)

    # Resume from checkpoint if specified
    if args.resume:
        try:
            checkpoint_info = trainer.model_manager.load_checkpoint(
                trainer.policy_processor.policy_head, trainer.optimizer, args.resume
            )
            trainer.current_epoch = checkpoint_info["epoch"]
            trainer.current_step = checkpoint_info["step"]
            print(
                f"Resumed from epoch {checkpoint_info['epoch']}, step {checkpoint_info['step']}"
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)

    if args.evaluate_only:
        # Run evaluation only
        print("\nRunning evaluation...")
        try:
            val_metrics = trainer.validate()
            print("\nValidation Results:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            sys.exit(1)
    else:
        # Run training
        print("\nStarting training...")
        try:
            final_metrics = trainer.train()

            print("\n" + "=" * 60)
            print("TRAINING COMPLETED")
            print("=" * 60)
            print("\nFinal Results:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            # Save current state
            trainer.model_manager.save_checkpoint(
                trainer.policy_processor,
                trainer.optimizer,
                trainer.current_epoch,
                trainer.current_step,
                {"interrupted": True},
            )
            print("Checkpoint saved before exit")
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
