"""
PPO Trainer for ChuChu Rocket self-play training.

Implements PPO algorithm with action masking, value head, and curriculum learning.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ...model.model_loader import ModelLoader
from ...model.model_saver import ModelSaver
from .collect_rollout import collect_rollout as _collect_rollout
from .config import PPOConfig
from .curriculum import CurriculumManager
from .ppo_env import PPOEnvironmentManager
from .ppo_evaluator import PPOEvaluator
from .rollout_buffer import RolloutBuffer
from .update_policy import update_policy as _update_policy


class PPOTrainer:
    """PPO trainer for ChuChu Rocket puzzle solving."""

    def __init__(self, config: PPOConfig):
        """Initialize PPO trainer.

        Args:
            config: PPO training configuration
        """
        self.config = config
        self.device = self._get_device()

        # Initialize model components using unified loader
        self._load_model_components()

        # Create optimizer for all trainable parameters
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Initialize environment manager
        self.env_manager = PPOEnvironmentManager(
            config=config,
            perception_processor=self.perception_processor,
            state_fusion_processor=self.state_fusion_processor,
        )

        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            config=config,
            observation_dim=128,  # State embedding dimension
            num_envs=config.num_parallel_envs,
            device=self.device,
        )

        # Initialize curriculum manager
        self.curriculum = CurriculumManager(config)

        # Initialize evaluator
        self.evaluator = PPOEvaluator(
            config=config,
            perception_processor=self.perception_processor,
            state_fusion_processor=self.state_fusion_processor,
            policy_processor=self.policy_processor,
            value_head=self.value_head,
        )

        # Initialize model saver
        self.model_saver = ModelSaver(config.model_dir)

        # Training state
        self.global_step = 0
        self.current_rollout = 0
        self.best_eval_score = 0.0

        # Continuous training tracking
        self.training_start_time = None
        self.last_improvement_time = None
        self.target_achieved = False

        # Learning rate warmup state
        self.warmup_steps_completed = 0
        self.base_learning_rate = config.learning_rate

        # Adaptive gradient scaling
        self.large_grad_count = 0
        self.grad_norm_history = []

        # Hyperparameter schedules
        self.current_clip_eps = config.clip_eps_start
        self.current_value_clip_eps = config.value_clip_eps_start
        self.current_entropy_coeff = config.entropy_coeff_start

        # Training statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "kl_divergence": [],
            "explained_variance": [],
            "episode_rewards": [],
            "episode_lengths": [],
        }

        print(f"PPO Trainer initialized on {self.device}")
        print(f"Total trainable parameters: {self._count_parameters():,}")
        print()

    def _get_device(self) -> torch.device:
        """Get training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _load_model_components(self):
        """Load model components using unified model loader."""
        # Use unified model loader
        model_loader = ModelLoader(
            self.config.bc_model_path, device=str(self.device), load_components=False
        )
        checkpoint_info = model_loader.load_model()

        (
            self.perception_processor,
            self.state_fusion_processor,
            self.policy_processor,
            self.value_head,
        ) = model_loader.get_ppo_components()

        # Check for different training run metadata
        bc_set_info = model_loader.get_metadata("bc_set")
        bc_seq_lite_info = model_loader.get_metadata("bc_seq_lite")
        ppo_info = model_loader.get_metadata("ppo")

        if ppo_info:
            print(f"Source: PPO training (step {ppo_info['global_step']})")
            print(f"PPO best eval score: {ppo_info.get('best_eval_score', 'N/A'):.3f}")
        elif bc_seq_lite_info:
            print(
                f"Source: BC Sequence Lite training (epoch {bc_seq_lite_info['epoch']}, step {bc_seq_lite_info['step']})"
            )
            print(
                f"BC Seq Lite best accuracy: {bc_seq_lite_info.get('best_val_accuracy', 'N/A'):.3f}"
            )
        elif bc_set_info:
            print(
                f"Source: BC-Set training (epoch {bc_set_info['epoch']}, step {bc_set_info['step']})"
            )
            print(
                f"BC-Set best Jaccard: {bc_set_info.get('best_val_jaccard', 'N/A'):.3f}"
            )
        else:
            print("Source: Unknown training method")

        # Apply backbone freezing if configured
        if self.config.freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen: CNN and fusion layers are not trainable")

    def _freeze_backbone(self):
        """Freeze CNN encoder and state fusion layers."""
        # Freeze CNN encoder
        cnn_encoder = self.perception_processor.get_cnn_encoder()
        cnn_encoder.requires_grad_(False)

        # Freeze state fusion processor (freeze the fusion_mlp inside it)
        self.state_fusion_processor.fusion_mlp.requires_grad_(False)

        # Keep policy head and value head trainable
        self.policy_processor.policy_head.requires_grad_(True)
        self.value_head.requires_grad_(True)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for all trainable parameters."""
        parameters = []

        if not self.config.freeze_backbone:
            # Add CNN encoder parameters if not frozen
            parameters.extend(self.perception_processor.get_cnn_encoder().parameters())

            # Add state fusion parameters if not frozen
            parameters.extend(self.state_fusion_processor.fusion_mlp.parameters())

        # Always add policy head parameters (trainable in both modes)
        parameters.extend(self.policy_processor.policy_head.parameters())

        # Always add value head parameters (trainable in both modes)
        parameters.extend(self.value_head.parameters())

        # Filter out parameters that don't require gradients
        trainable_parameters = [p for p in parameters if p.requires_grad]

        return optim.AdamW(
            trainable_parameters, lr=self.config.learning_rate, weight_decay=1e-4
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Calculate number of updates
        num_updates = max(1, self.config.total_env_steps // self.config.rollout_length)

        print(
            f"Scheduler: total_env_steps={self.config.total_env_steps:,}, rollout_length={self.config.rollout_length}, num_updates={num_updates}"
        )

        if self.config.lr_schedule == "cosine":
            # Ensure T_max is at least 1 to avoid division by zero
            t_max = max(1, num_updates)
            print(f"Creating CosineAnnealingLR with T_max={t_max}")
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.lr_schedule == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=max(1, num_updates),
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0

        # CNN encoder
        total += sum(
            p.numel() for p in self.perception_processor.get_cnn_encoder().parameters()
        )

        # State fusion
        total += sum(
            p.numel() for p in self.state_fusion_processor.fusion_mlp.parameters()
        )

        # Policy head
        total += sum(p.numel() for p in self.policy_processor.policy_head.parameters())

        # Value head
        total += sum(p.numel() for p in self.value_head.parameters())

        return total

    def _check_model_stability(self) -> bool:
        """Check if model parameters are stable (no NaN/Inf).

        Returns:
            True if model is stable, False otherwise
        """
        for name, module in [
            ("cnn_encoder", self.perception_processor.get_cnn_encoder()),
            ("state_fusion", self.state_fusion_processor.fusion_mlp),
            ("policy_head", self.policy_processor.policy_head),
            ("value_head", self.value_head),
        ]:
            for param in module.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"Instability detected in {name}")
                    return False
        return True

    def _reset_unstable_components(self):
        """Reset components that have become unstable."""
        print("Resetting unstable model components...")

        # Reset value head (most likely to become unstable)
        print("Resetting value head...")
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

        # Reset optimizer state for value head
        value_head_params = list(self.value_head.parameters())
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in value_head_params:
                    if p in self.optimizer.state:
                        del self.optimizer.state[p]

        print("Model components reset successfully")

    def _reduce_learning_rate(self, factor: float = 0.5):
        """Reduce learning rate to improve stability.

        Args:
            factor: Factor to multiply learning rate by
        """
        old_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = old_lr * factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print(f"Reduced learning rate from {old_lr:.6f} to {new_lr:.6f}")

    def _should_continue_training(self) -> bool:
        """Check if training should continue based on various criteria."""
        current_time = time.time()
        training_hours = (current_time - self.training_start_time) / 3600

        # Check if we've hit the step limit (non-continuous mode)
        if (
            not self.config.continuous_training
            and self.global_step >= self.config.total_env_steps
        ):
            return False

        # Check maximum training time
        if training_hours >= self.config.max_training_hours:
            print(f"\n⌛ Maximum training time reached: {training_hours:.1f} hours")
            return False

        # Check if target was achieved and early stopping is enabled
        if self.config.early_stop_on_target and self.target_achieved:
            print("\n🎯 Target performance achieved, stopping training")
            return False

        # Check patience (no improvement for too long)
        if self.config.patience_hours > 0:
            hours_since_improvement = (current_time - self.last_improvement_time) / 3600
            if hours_since_improvement >= self.config.patience_hours:
                print(
                    f"\n😴 No improvement for {hours_since_improvement:.1f} hours, stopping training"
                )
                return False

        return True

    def _get_stop_reason(self) -> str:
        """Get the reason why training stopped."""
        current_time = time.time()
        training_hours = (current_time - self.training_start_time) / 3600

        if (
            not self.config.continuous_training
            and self.global_step >= self.config.total_env_steps
        ):
            return "step_limit_reached"
        elif training_hours >= self.config.max_training_hours:
            return "time_limit_reached"
        elif self.config.early_stop_on_target and self.target_achieved:
            return "target_achieved"
        elif self.config.patience_hours > 0:
            hours_since_improvement = (current_time - self.last_improvement_time) / 3600
            if hours_since_improvement >= self.config.patience_hours:
                return "no_improvement_timeout"
        else:
            return "manual_stop"

    def _apply_warmup(self):
        """Apply learning rate warmup during early training."""
        if self.warmup_steps_completed < self.config.lr_warmup_steps:
            # Linear warmup from 10% to 100% of base learning rate
            warmup_factor = 0.1 + 0.9 * (
                self.warmup_steps_completed / self.config.lr_warmup_steps
            )
            warmup_lr = self.base_learning_rate * warmup_factor

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

            self.warmup_steps_completed += 1

            if self.warmup_steps_completed % 1000 == 0:
                print(
                    f"Warmup progress: {self.warmup_steps_completed}/{self.config.lr_warmup_steps}, LR: {warmup_lr:.6f}"
                )

    def _monitor_gradient_norm(self, grad_norm: float):
        """Monitor gradient norms and apply adaptive learning rate reduction."""
        # Keep history of recent gradient norms
        self.grad_norm_history.append(grad_norm)
        if len(self.grad_norm_history) > 100:  # Keep last 100 measurements
            self.grad_norm_history.pop(0)

        # Check if current gradient is large
        if grad_norm > self.config.max_grad_norm * 2:
            self.large_grad_count += 1
            print(
                f"Warning: Large gradient norm: {grad_norm:.4f} (max: {self.config.max_grad_norm})"
            )

            # If we've had many large gradients recently, reduce learning rate
            if self.large_grad_count % 10 == 0:
                avg_recent_grad = sum(self.grad_norm_history[-20:]) / min(
                    20, len(self.grad_norm_history)
                )
                if avg_recent_grad > self.config.max_grad_norm * 1.5:
                    print(
                        f"Consistent large gradients detected (avg: {avg_recent_grad:.4f}), reducing learning rate"
                    )
                    self._reduce_learning_rate(factor=0.8)  # More gentle reduction
                    self.large_grad_count = 0  # Reset counter
        else:
            # Reset counter if gradients are normal
            if self.large_grad_count > 0:
                self.large_grad_count = max(0, self.large_grad_count - 1)

    def collect_rollout(self) -> Dict[str, Any]:
        """Collect a full rollout of experience."""
        return _collect_rollout(
            # Model components
            perception_processor=self.perception_processor,
            state_fusion_processor=self.state_fusion_processor,
            policy_processor=self.policy_processor,
            value_head=self.value_head,
            # Environment and data
            env_manager=self.env_manager,
            buffer=self.buffer,
            curriculum=self.curriculum,
            # Configuration
            config=self.config,
            device=self.device,
            # Monitoring functions
            check_model_stability_fn=self._check_model_stability,
            reset_unstable_components_fn=self._reset_unstable_components,
        )

    def update_policy(self) -> Dict[str, float]:
        """Update policy using PPO algorithm."""
        return _update_policy(
            # Model components
            perception_processor=self.perception_processor,
            state_fusion_processor=self.state_fusion_processor,
            policy_processor=self.policy_processor,
            value_head=self.value_head,
            optimizer=self.optimizer,
            # Training data
            buffer=self.buffer,
            # Configuration
            config=self.config,
            device=self.device,
            # Current hyperparameters
            current_clip_eps=self.current_clip_eps,
            current_value_clip_eps=self.current_value_clip_eps,
            current_entropy_coeff=self.current_entropy_coeff,
            # Monitoring functions
            check_model_stability_fn=self._check_model_stability,
            reset_unstable_components_fn=self._reset_unstable_components,
            monitor_gradient_norm_fn=self._monitor_gradient_norm,
        )

    def train(self) -> Dict[str, Any]:
        """Run full PPO training loop."""
        if self.config.continuous_training:
            print(
                f"🌙 Continuous training mode: will run for up to {self.config.max_training_hours} hours"
            )
        else:
            print(f"Total environment steps: {self.config.total_env_steps:,}")

        print(f"Rollout length: {self.config.rollout_length:,}")
        print(f"Parallel environments: {self.config.num_parallel_envs}\n")

        start_time = time.time()
        self.training_start_time = start_time
        self.last_improvement_time = start_time

        while self._should_continue_training():
            rollout_start = time.time()

            # Collect rollout
            rollout_stats = self.collect_rollout()
            rollout_time = time.time() - rollout_start

            # Update policy
            update_start = time.time()
            update_stats = self.update_policy()
            update_time = time.time() - update_start

            # Clear buffer for next rollout
            self.buffer.clear()

            # Update global step
            self.global_step += (
                self.config.rollout_length * self.config.num_parallel_envs
            )
            self.current_rollout += 1

            # Update hyperparameter schedules
            self._update_schedules(rollout_stats)

            # Update curriculum
            self.curriculum.update(rollout_stats)

            # Apply learning rate warmup if needed
            self._apply_warmup()

            # Learning rate scheduling
            try:
                self.scheduler.step()
            except ZeroDivisionError as e:
                print(f"Warning: Scheduler step failed with ZeroDivisionError: {e}")
                print(f"Scheduler type: {type(self.scheduler)}")
                print(
                    f"Current rollout: {self.current_rollout}, Global step: {self.global_step}"
                )
                # Continue without stepping scheduler

            # Logging
            if (
                self.current_rollout
                % (self.config.log_frequency // self.config.rollout_length)
                == 0
            ):
                self._log_training_step(
                    rollout_stats, update_stats, rollout_time, update_time
                )

                # Additional logging for continuous training
                if self.config.continuous_training:
                    self._log_continuous_training_status()

            # Evaluation
            if self.global_step % self.config.eval_frequency == 0:
                eval_results = self.evaluator.evaluate()
                self._handle_evaluation(eval_results)

            # Save checkpoint
            self._save_checkpoint()
            print(
                f"Step {self.global_step:,} completed in {time.time() - rollout_start:.2f} seconds\n"
            )

        # Final evaluation and save
        final_eval = self.evaluator.evaluate()
        self._save_checkpoint(is_final=True)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return {
            "training_time": training_time,
            "total_steps": self.global_step,
            "final_evaluation": final_eval,
            "target_achieved": self.target_achieved,
            "stop_reason": self._get_stop_reason(),
        }

    def _update_schedules(self, rollout_stats: Dict[str, Any]):
        """Update hyperparameter schedules."""
        # Progress through training
        progress = self.global_step / self.config.total_env_steps

        # Linear decay for clipping epsilons
        self.current_clip_eps = self.config.clip_eps_start + progress * (
            self.config.clip_eps_end - self.config.clip_eps_start
        )
        self.current_value_clip_eps = self.config.value_clip_eps_start + progress * (
            self.config.value_clip_eps_end - self.config.value_clip_eps_start
        )

        # Entropy coefficient decay after threshold
        current_solve_rate = rollout_stats.get("mean_episode_reward", 0.0)
        if current_solve_rate > self.config.entropy_decay_threshold:
            entropy_progress = min(
                1.0, (current_solve_rate - self.config.entropy_decay_threshold) / 0.2
            )
            self.current_entropy_coeff = (
                self.config.entropy_coeff_start
                + entropy_progress
                * (self.config.entropy_coeff_end - self.config.entropy_coeff_start)
            )

    def _log_training_step(
        self,
        rollout_stats: Dict[str, Any],
        update_stats: Dict[str, float],
        rollout_time: float,
        update_time: float,
    ):
        """Log training step information."""
        print(f"\nStep {self.global_step:,}")
        print(f"Rollout time: {rollout_time:.2f}s, Update time: {update_time:.2f}s")
        print(f"Episodes: {rollout_stats.get('num_episodes', 0)}")
        print(f"Mean reward: {rollout_stats.get('mean_episode_reward', 0.0):.3f}")
        print(f"Mean length: {rollout_stats.get('mean_episode_length', 0.0):.1f}")
        print(f"Policy loss: {update_stats['policy_loss']:.4f}")
        print(f"Value loss: {update_stats['value_loss']:.4f}")
        print(f"KL divergence: {update_stats['kl_divergence']:.6f}")
        print(f"Explained variance: {update_stats['explained_variance']:.3f}")
        print(f"Entropy coeff: {self.current_entropy_coeff:.4f}")
        print(f"Clip eps: {self.current_clip_eps:.3f}")
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _log_continuous_training_status(self):
        """Log additional status information for continuous training mode."""
        if not self.config.continuous_training:
            return

        current_time = time.time()
        training_hours = (current_time - self.training_start_time) / 3600
        hours_since_improvement = (current_time - self.last_improvement_time) / 3600

        print(
            f"🌙 Continuous training: {training_hours:.1f}h / {self.config.max_training_hours:.1f}h"
        )
        print(f"💤 Hours since improvement: {hours_since_improvement:.1f}h")
        print(f"🏆 Best eval score: {self.best_eval_score:.3f}")

        if self.config.curriculum_easy_only:
            print(f"🎯 Easy-only mode: curriculum locked")
        else:
            stage_info = self.curriculum.get_current_stage_info()
            print(
                f"📚 Current stage: {stage_info['stage_name']} ({stage_info['solve_rate']:.3f} solve rate)"
            )

        if self.target_achieved:
            print(f"🎆 TARGET ACHIEVED!")
        else:
            print(
                f"🎯 Target progress: Medium {self.best_eval_score:.3f}/{self.config.target_medium_solve_rate:.3f}"
            )

    def _handle_evaluation(self, eval_results: Dict[str, Any]):
        """Handle evaluation results."""
        print(f"\nEvaluation at step {self.global_step:,}:")
        print(f"Medium solve rate: {eval_results.get('medium_solve_rate', 0.0):.3f}")
        print(f"Hard solve rate: {eval_results.get('hard_solve_rate', 0.0):.3f}")

        # Check if we've achieved target performance
        medium_rate = eval_results.get("medium_solve_rate", 0.0)
        hard_rate = eval_results.get("hard_solve_rate", 0.0)

        if (
            medium_rate >= self.config.target_medium_solve_rate
            and hard_rate >= self.config.target_hard_solve_rate
        ):
            if not self.target_achieved:
                print(
                    f"\n🎆 TARGET ACHIEVED! Medium: {medium_rate:.3f}, Hard: {hard_rate:.3f}"
                )
                self.target_achieved = True

    def _save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint."""
        # Prepare metrics for saving
        metrics = {
            "policy_loss": (
                self.training_stats["policy_loss"][-1]
                if self.training_stats["policy_loss"]
                else 0.0
            ),
            "value_loss": (
                self.training_stats["value_loss"][-1]
                if self.training_stats["value_loss"]
                else 0.0
            ),
            "entropy_loss": (
                self.training_stats["entropy_loss"][-1]
                if self.training_stats["entropy_loss"]
                else 0.0
            ),
            "kl_divergence": (
                self.training_stats["kl_divergence"][-1]
                if self.training_stats["kl_divergence"]
                else 0.0
            ),
            "explained_variance": (
                self.training_stats["explained_variance"][-1]
                if self.training_stats["explained_variance"]
                else 0.0
            ),
        }

        # Use model saver to save checkpoint in standardized format
        self.model_saver.save_ppo_checkpoint(
            cnn_encoder=self.perception_processor.get_cnn_encoder(),
            state_fusion=self.state_fusion_processor.fusion_mlp,
            policy_head=self.policy_processor.policy_head,
            value_head=self.value_head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            best_eval_score=self.best_eval_score,
            metrics=metrics,
            config=self.config,
            is_final=is_final,
        )
