"""
Model saving and loading utilities for behaviour cloning.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ModelSaver:
    """Manages model checkpointing."""

    def __init__(self, model_dir: str = "model"):
        """Initialize model saver.

        Args:
            model_dir: Directory to save model checkpoints
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_bc_set_checkpoint(
        self,
        cnn_encoder: nn.Module,
        state_fusion: nn.Module,
        policy_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        step: int,
        best_val_jaccard: float,
        metrics: Dict[str, float],
        config: Optional[Any] = None,
        is_final: bool = False,
    ) -> None:
        """Save BC-Set checkpoint in standardized format.

        Args:
            cnn_encoder: CNN encoder module
            state_fusion: State fusion module
            policy_head: Policy head module
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            step: Current step number
            best_val_jaccard: Best validation Jaccard score
            metrics: Training metrics dictionary
            config: Training configuration (optional)
            is_final: Whether this is the final checkpoint
        """
        # Generate filename based on checkpoint type
        if is_final:
            filename = "bc_set_final.pth"
        else:
            filename = "bc_set_latest.pth"

        checkpoint_data = {
            # Common model components (transferable between training runs)
            "cnn_encoder": cnn_encoder.state_dict(),
            "state_fusion": state_fusion.state_dict(),
            "policy_head": policy_head.state_dict(),
            # BC-Set specific training state and metadata
            "bc_set": {
                "epoch": epoch,
                "step": step,
                "best_val_jaccard": best_val_jaccard,
                "config": config,
                "metrics": metrics,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "phase": "completed" if is_final else "training",
            },
        }

        checkpoint_path = self.model_dir / filename
        torch.save(checkpoint_data, checkpoint_path)

        if is_final:
            print(f"Final BC-Set checkpoint saved: {checkpoint_path}")

    def save_bc_seq_lite_checkpoint(
        self,
        cnn_encoder: nn.Module,
        state_fusion: nn.Module,
        policy_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        step: int,
        phase: str,
        best_val_accuracy: float,
        metrics: Dict[str, float],
        config: Optional[Any] = None,
        frozen_params: Optional[list] = None,
        is_final: bool = False,
    ) -> None:
        """Save BC Sequence Lite checkpoint in standardized format.

        Args:
            cnn_encoder: CNN encoder module
            state_fusion: State fusion module
            policy_head: Policy head module
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            step: Current step number
            phase: Current training phase
            best_val_accuracy: Best validation accuracy
            metrics: Training metrics dictionary
            config: Training configuration (optional)
            frozen_params: List of frozen parameter names
            is_final: Whether this is the final checkpoint
        """
        # Generate filename based on checkpoint type
        if is_final:
            filename = "bc_seq_lite_final.pth"
        else:
            filename = "bc_seq_lite_latest.pth"

        checkpoint_data = {
            # Common model components (transferable between training runs)
            "cnn_encoder": cnn_encoder.state_dict(),
            "state_fusion": state_fusion.state_dict(),
            "policy_head": policy_head.state_dict(),
            # BC Sequence Lite specific training state and metadata
            "bc_seq_lite": {
                "epoch": epoch,
                "step": step,
                "phase": "completed" if is_final else phase,
                "best_val_accuracy": best_val_accuracy,
                "config": config,
                "metrics": metrics,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "frozen_params": frozen_params or [],
            },
        }

        checkpoint_path = self.model_dir / filename
        torch.save(checkpoint_data, checkpoint_path)

        if is_final:
            print(f"Final BC Sequence Lite checkpoint saved: {checkpoint_path}")

    def save_ppo_checkpoint(
        self,
        cnn_encoder: nn.Module,
        state_fusion: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        global_step: int,
        best_eval_score: float,
        metrics: Dict[str, float],
        config: Optional[Any] = None,
        is_final: bool = False,
    ) -> None:
        """Save PPO checkpoint in standardized format.

        Args:
            cnn_encoder: CNN encoder module
            state_fusion: State fusion module
            policy_head: Policy head module
            value_head: Value head module
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            global_step: Current global step number
            best_eval_score: Best evaluation score
            metrics: Training metrics dictionary
            config: Training configuration (optional)
            is_final: Whether this is the final checkpoint
        """
        # Generate filename based on checkpoint type
        if is_final:
            filename = "ppo_final.pth"
        else:
            filename = "ppo_latest.pth"

        checkpoint_data = {
            # Common model components (transferable between training runs)
            "cnn_encoder": cnn_encoder.state_dict(),
            "state_fusion": state_fusion.state_dict(),
            "policy_head": policy_head.state_dict(),
            "value_head": value_head.state_dict(),
            # PPO specific training state and metadata
            "ppo": {
                "global_step": global_step,
                "best_eval_score": best_eval_score,
                "config": config,
                "metrics": metrics,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "phase": "completed" if is_final else "training",
            },
        }

        checkpoint_path = self.model_dir / filename
        torch.save(checkpoint_data, checkpoint_path)

        if is_final:
            print(f"Final PPO checkpoint saved: {checkpoint_path}")
