"""
Model saving and loading utilities for behaviour cloning.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import BCConfig


class ModelManager:
    """Manages model checkpointing and loading for behaviour cloning."""

    def __init__(self, config: BCConfig):
        """Initialize model manager.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = 0.0  # Track best validation metric

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            step: Current step number
            metrics: Training metrics dictionary
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        # Save latest checkpoint
        latest_path = self.model_dir / f"{self.config.checkpoint_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        # Save numbered checkpoint
        numbered_path = (
            self.model_dir
            / f"{self.config.checkpoint_name}_epoch_{epoch}_step_{step}.pth"
        )
        torch.save(checkpoint, numbered_path)

        # Save best model if this is the best
        if is_best:
            best_path = self.model_dir / self.config.best_model_name
            torch.save(checkpoint, best_path)
            self.best_metric = metrics.get("val_accuracy", 0.0)
            print(
                f"New best model saved at epoch {epoch}, step {step} with metric {self.best_metric:.4f}"
            )

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to checkpoint file (defaults to latest)
            map_location: Device to map tensors to

        Returns:
            Dictionary with checkpoint information
        """
        if checkpoint_path is None:
            checkpoint_path = (
                self.model_dir / f"{self.config.checkpoint_name}_latest.pth"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Handle different checkpoint formats
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)

        print(f"Loaded checkpoint from epoch {epoch}, step {step}")

        return {
            "epoch": epoch,
            "step": step,
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", None),
        }

    def load_best_model(
        self, model: nn.Module, map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load the best saved model.

        Args:
            model: Model to load state into
            map_location: Device to map tensors to

        Returns:
            Dictionary with checkpoint information
        """
        best_path = self.model_dir / self.config.best_model_name
        return self.load_checkpoint(
            model, checkpoint_path=best_path, map_location=map_location
        )

    def save_final_model(self, model: nn.Module, metrics: Dict[str, float]) -> None:
        """Save the final trained model.

        Args:
            model: Trained model to save
            metrics: Final training metrics
        """
        final_checkpoint = {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        final_path = self.model_dir / self.config.final_model_name
        torch.save(final_checkpoint, final_path)
        print(f"Final model saved to {final_path}")

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints.

        Returns:
            List of checkpoint filenames
        """
        checkpoints = []
        for file_path in self.model_dir.glob("*.pth"):
            checkpoints.append(file_path.name)
        return sorted(checkpoints)

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Remove old numbered checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        # Find all numbered checkpoints
        numbered_checkpoints = []
        for file_path in self.model_dir.glob(
            f"{self.config.checkpoint_name}_epoch_*.pth"
        ):
            numbered_checkpoints.append(file_path)

        # Sort by modification time (newest first)
        numbered_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for checkpoint_path in numbered_checkpoints[keep_last_n:]:
            checkpoint_path.unlink()
            print(f"Removed old checkpoint: {checkpoint_path.name}")

    def get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in megabytes.

        Args:
            model: Model to analyze

        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def is_better_metric(self, current_metric: float) -> bool:
        """Check if current metric is better than the best recorded.

        Args:
            current_metric: Current validation metric

        Returns:
            True if current metric is better
        """
        return current_metric > self.best_metric
