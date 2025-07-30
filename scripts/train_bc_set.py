#!/usr/bin/env python3
"""
BC Set Training Script

Train the ChuChu Rocket AI using supervised learning on BFS optimal solutions.
BC-Set training learns global arrow placement patterns from final board states.

Usage:
    python scripts/train_bc_set.py --csv-path data/puzzles.csv --epochs 50
    python scripts/train_bc_set.py --help
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.training.bc import BCConfig, BCTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="BC Set Training")

    # Data arguments
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/puzzles.csv",
        help="Path to puzzles CSV file",
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of training epochs"
    )

    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (epochs)"
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not Path(args.csv_path).exists():
        print(f"Error: CSV file not found at {args.csv_path}")
        print("Please generate puzzles first or provide correct path.")
        return 1

    print("=== BC Set Training ===")
    print(f"CSV path: {args.csv_path}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.epochs}")
    print(f"Patience: {args.patience}")

    # Create configuration
    config = BCConfig(
        csv_path=args.csv_path,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        # BC-Set specific settings
        use_focal_loss=True,  # Use focal loss for BC-Set
    )

    print(f"Batch size: {config.batch_size}")
    print(f"Model directory: {config.model_dir}")

    try:
        # Create trainer
        print("\nInitializing trainer...")
        trainer = BCTrainer(config)

        # Train the model
        print("\nStarting training...")
        final_metrics = trainer.train()

        print("\n=== Training Results ===")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print(f"\nBC Set training completed!")
        print(f"Model checkpoints saved in: {config.model_dir}")
        print("\nNext steps:")
        print("1. Evaluate the model on test set")
        print("2. Use this model as input for BC Sequence Lite training")
        print("3. Run full episode evaluation to measure solve rates")

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
