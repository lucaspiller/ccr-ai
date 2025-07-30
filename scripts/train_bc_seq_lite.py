#!/usr/bin/env python3
"""
BC Sequence Lite training script.

This script implements the BC Sequence Lite approach:
1. Creates intermediate-state dataset from BFS solutions
2. Trains with freeze/unfreeze schedule:
   - Phase 1: Freeze perception + fusion, train policy head (3 epochs)
   - Phase 2: Unfreeze all, train with 10x lower LR (5-7 epochs)

Expected lift: From ~20% to 60-70% solve rate on comparable puzzle lengths.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.perception.data_types import PerceptionConfig
from src.perception.processors import GameStateProcessor
from src.state_fusion.processors import StateFusionProcessor
from src.training.bc.bc_sequence_lite_data_loader import \
    create_sequence_lite_data_loaders
from src.training.bc.bc_sequence_lite_trainer import train_bc_sequence_lite
from src.training.bc.config import BCConfig


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="BC Sequence Lite Training")

    # Data arguments
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/puzzles.csv",
        help="Path to puzzles CSV file",
    )

    parser.add_argument(
        "--bc-model",
        type=str,
        default=None,
        help="Path to pre-trained BC-Set model to continue training from",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
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

    # Create configuration
    config = BCConfig(
        csv_path=args.csv_path,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        # BC Sequence Lite specific settings
        use_focal_loss=False,  # Use plain cross-entropy as specified
    )

    print("=== BC Sequence Lite Training ===")
    print(f"CSV path: {config.csv_path}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Model directory: {config.model_dir}")

    try:
        # Initialize processors
        print("\nInitializing processors...")

        # Create perception processor with flexible board size support
        perception_config = PerceptionConfig(
            strict_bounds_checking=False,  # Allow variable board sizes
            validate_input=False,  # Disable input validation that assumes fixed size
        )
        perception_processor = GameStateProcessor(perception_config)

        # Create state fusion processor
        state_fusion_processor = StateFusionProcessor()

        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_sequence_lite_data_loaders(
            config, perception_processor, state_fusion_processor
        )

        if len(train_loader.dataset) == 0:
            print("Error: No training samples found. Check CSV data and BFS solutions.")
            return 1

        # Train the model
        print("\nStarting training...")
        metrics = train_bc_sequence_lite(
            config, train_loader, val_loader, args.bc_model
        )

        # Print final results
        print("\n=== Training Results ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print(f"\nBC Sequence Lite training completed!")
        print(f"Model checkpoints saved in: {config.model_dir}")
        print("\nNext steps:")
        print("1. Evaluate the model on test set")
        print("2. Run full episode evaluation to measure solve rates")
        print("3. Compare against BC-Set baseline")

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
