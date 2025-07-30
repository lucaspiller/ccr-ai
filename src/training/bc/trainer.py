"""
Main training loop for behaviour cloning.
"""

import time
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ...model.model_saver import ModelSaver
from ...perception.data_types import PerceptionConfig
from ...perception.processors import GameStateProcessor
from ...policy.processors import PolicyProcessor
from ...state_fusion.processors import StateFusionProcessor
from .config import BCConfig
from .data_loader import create_data_loaders, get_device
from .viz_board import viz_board


class BCTrainer:
    """Behaviour cloning trainer."""

    def __init__(self, config: BCConfig):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_device(config.device)

        # Initialize processors with relaxed bounds checking for variable board sizes
        perception_config = PerceptionConfig(strict_bounds_checking=False)
        self.perception_processor = GameStateProcessor(perception_config)
        self.state_fusion_processor = StateFusionProcessor()
        self.policy_processor = PolicyProcessor()

        # Create model saver
        self.model_saver = ModelSaver(config.model_dir)

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_jaccard = 0.0

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config, self.perception_processor, self.state_fusion_processor
        )

        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validation on {len(self.val_loader.dataset)} samples")
        print(f"Testing on {len(self.test_loader.dataset)} samples")

        # Move models to device
        self.perception_processor.get_cnn_encoder().to(self.device)
        self.state_fusion_processor.to(self.device)
        self.policy_processor.to(self.device)

        # Initialize model parameters to avoid MPS placeholder storage issues
        if self.device.type == "mps":
            self._initialize_mps_models()

        # Create optimizer - include CNN encoder parameters
        self.optimizer = optim.AdamW(
            list(self.perception_processor.get_cnn_encoder().parameters())
            + list(self.state_fusion_processor.fusion_mlp.parameters())
            + list(self.policy_processor.policy_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create learning rate scheduler - cosine decay to 1e-4 over 200 epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs, eta_min=1e-4
        )

        # Loss function for BC-Set: Binary cross-entropy with masking support
        self.criterion = self._create_masked_bce_loss()

    def rank_margin_loss(self, prob, gt_indices, margin=0.1):
        """Vectorized rank-margin loss: enforce best GT arrow scores >= δ higher than best wrong tile.

        Args:
            prob: Probabilities for all actions [batch_size, num_actions]
            gt_indices: Ground truth indices (boolean mask) [batch_size, num_actions]
            margin: Margin threshold (default 0.15)

        Returns:
            Rank margin loss (scalar)
        """
        # Get highest GT probability per sample [batch_size]
        gt_probs = torch.where(gt_indices, prob, torch.tensor(-1.0, device=prob.device))
        p_pos, _ = gt_probs.max(dim=1)

        # Get top-2 overall probabilities [batch_size, 2]
        top2_probs, _ = torch.topk(prob, k=2, dim=1)

        # For each sample, if top prob is GT, use second; otherwise use top
        is_top_gt = top2_probs[:, 0] == p_pos
        p_neg = torch.where(is_top_gt, top2_probs[:, 1], top2_probs[:, 0])

        # Only compute loss for samples with GT actions
        has_gt = gt_indices.sum(dim=1) > 0

        # Apply margin loss: 0 if margin satisfied, otherwise penalize
        loss_per_sample = F.relu(margin + p_neg - p_pos)
        loss_per_sample = torch.where(
            has_gt, loss_per_sample, torch.tensor(0.0, device=prob.device)
        )

        return loss_per_sample.mean()

    def _create_masked_bce_loss(self):
        """Create binary cross-entropy loss treating invalid actions as guaranteed negatives."""

        def masked_bce_loss(logits, targets, masks, arrow_budgets):
            # Force invalid actions to be negative targets (guaranteed 0)
            corrected_targets = targets.clone()
            corrected_targets[masks == 0] = 0.0

            # Apply label smoothing to corrected targets
            # positive targets: 1.0 -> (1.0 - eps + eps/2) = 1.0 - eps/2
            # negative targets: 0.0 -> (0.0 + eps/2) = eps/2
            eps = self.config.label_smoothing
            smoothed_targets = corrected_targets * (1.0 - eps) + eps * 0.5

            # Compute standard BCE loss (no reduction yet)
            pos_weight = torch.tensor(self.config.pos_weight, device=logits.device)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, smoothed_targets, reduction="none", pos_weight=pos_weight
            )

            if self.config.use_focal_loss:
                # Apply Focal Loss weighting: F = α (1-p)^γ BCE
                probs = torch.sigmoid(logits)

                # Compute focal weights: focus on hard examples
                # For positives: weight = (1-p)^γ, for negatives: weight = p^γ
                p_t = torch.where(
                    smoothed_targets >= 0.5, probs, 1 - probs
                )  # p if target=1, (1-p) if target=0
                focal_weight = (
                    self.config.focal_alpha * (1 - p_t) ** self.config.focal_gamma
                )

                # Apply focal weighting
                focal_loss = (focal_weight * bce_loss).mean()
            else:
                # Standard BCE loss
                focal_loss = bce_loss.mean()

            # Add rank margin loss
            probs = torch.sigmoid(logits)
            rank_loss = self.rank_margin_loss(
                probs, corrected_targets.bool(), margin=0.15
            )

            # Total loss: focal_loss + 5 * rank_loss
            total_loss = focal_loss + 5.0 * rank_loss

            # Add negative-logit penalty if enabled
            if self.config.use_negative_logit_penalty:
                negative_logit_penalty = (
                    self.config.negative_logit_penalty_weight
                    * logits.clamp_min(0).mean()
                )
                total_loss += negative_logit_penalty

            return total_loss

        return masked_bce_loss

    def _compute_set_metrics(self, logits, targets, masks):
        """Compute BC-Set metrics: Jaccard accuracy, precision, recall."""
        with torch.no_grad():
            # Convert logits to binary predictions (threshold at 0.5)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            # Apply mask to both predictions and targets
            valid_preds = predictions * masks
            valid_targets = targets * masks

            batch_size = targets.size(0)
            total_jaccard = 0.0
            total_precision = 0.0
            total_recall = 0.0

            for i in range(batch_size):
                pred_set = valid_preds[i]
                target_set = valid_targets[i]

                # Jaccard = |intersection| / |union|
                intersection = (pred_set * target_set).sum()
                union = (pred_set + target_set).clamp(max=1).sum()
                jaccard = (
                    intersection / union if union > 0 else torch.tensor(1.0)
                )  # Perfect if both empty

                # Precision = |intersection| / |predicted|
                pred_count = pred_set.sum()
                precision = (
                    intersection / pred_count if pred_count > 0 else torch.tensor(1.0)
                )

                # Recall = |intersection| / |target|
                target_count = target_set.sum()
                recall = (
                    intersection / target_count
                    if target_count > 0
                    else torch.tensor(1.0)
                )

                total_jaccard += jaccard.item()
                total_precision += precision.item()
                total_recall += recall.item()

            return total_jaccard, total_precision, total_recall

    def _compute_additional_metrics(self, logits, targets, masks, arrow_budgets):
        """Compute ROC-AUC and Precision@k metrics.

        Args:
            logits: Raw model logits [batch, 700]
            targets: Multi-hot target vectors [batch, 700]
            masks: Action masks [batch, 700]
            arrow_budgets: Arrow budget for each sample [batch] for Precision@k

        Returns:
            Tuple of (total_roc_auc, total_precision_at_k, num_valid_samples)
        """
        with torch.no_grad():
            batch_size = targets.size(0)
            total_roc_auc = 0.0
            total_precision_at_k = 0.0
            num_valid_samples = 0

            # Convert to probabilities
            probs = torch.sigmoid(logits)

            for i in range(batch_size):
                # Get valid actions for this sample
                valid_mask = masks[i] == 1
                if valid_mask.sum() == 0:
                    continue  # Skip if no valid actions

                valid_probs = probs[i][valid_mask].cpu().numpy()
                valid_targets = targets[i][valid_mask].cpu().numpy()

                # Skip if all targets are 0 or all targets are 1 (can't compute ROC-AUC)
                if len(np.unique(valid_targets)) < 2:
                    continue

                # ROC-AUC: Per-tile binary classification performance
                try:
                    roc_auc = roc_auc_score(valid_targets, valid_probs)
                    total_roc_auc += roc_auc
                except:
                    # Skip this sample if ROC-AUC computation fails
                    continue

                # Precision@k: Top-k accuracy where k = arrow_budget
                k = min(int(arrow_budgets[i]), len(valid_probs))
                if k > 0:
                    # Get indices of top-k predictions in valid actions
                    valid_indices = torch.where(valid_mask)[0]
                    all_probs = probs[i]

                    # Get top-k indices from all actions, then filter to valid ones
                    _, top_k_indices = torch.topk(
                        all_probs, k=min(k * 3, len(all_probs))
                    )  # Get extra to account for invalid actions

                    # Filter to valid actions and take first k
                    valid_top_k = []
                    for idx in top_k_indices:
                        if masks[i][idx] == 1:  # Valid action
                            valid_top_k.append(idx)
                            if len(valid_top_k) == k:
                                break

                    if len(valid_top_k) == k:
                        # Count how many of top-k predictions are correct
                        correct_in_top_k = sum(
                            targets[i][idx].item() for idx in valid_top_k
                        )
                        precision_at_k = correct_in_top_k / k
                        total_precision_at_k += precision_at_k

                num_valid_samples += 1

            return total_roc_auc, total_precision_at_k, num_valid_samples

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.perception_processor.get_cnn_encoder().train()
        self.state_fusion_processor.train()
        self.policy_processor.train()

        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (
            state_embeddings,
            action_targets,
            action_masks,
            arrow_budgets,
            board_ws,
            board_hs,
        ) in enumerate(progress_bar):
            # Move data to device
            state_embeddings = state_embeddings.to(self.device)
            action_targets = action_targets.to(self.device)
            action_masks = action_masks.to(self.device)
            arrow_budgets = arrow_budgets.to(self.device)

            # Forward pass - get raw logits from policy head
            logits = self.policy_processor.policy_head(state_embeddings)

            # Compute masked loss
            loss = self.criterion(logits, action_targets, action_masks, arrow_budgets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.perception_processor.get_cnn_encoder().parameters())
                    + list(self.state_fusion_processor.fusion_mlp.parameters())
                    + list(self.policy_processor.policy_head.parameters()),
                    self.config.gradient_clipping,
                )

            self.optimizer.step()

            # Update metrics - only loss for training
            total_loss += loss.item()
            total_samples += action_targets.size(0)

            self.current_step += 1

            # Update progress bar - only show loss and learning rate
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Log training metrics
            # if self.current_step % self.config.log_frequency == 0:
            #    current_avg_loss = total_loss / total_samples
            #    self._log_training_step(loss.item(), current_avg_loss)

            # Step-based validation (legacy - rarely used now)
            # if hasattr(self.config, 'val_frequency_steps') and self.current_step % self.config.val_frequency_steps == 0:
            #    val_metrics = self.validate()
            #    self._handle_validation_results(val_metrics)

            # Note: Removed step-based checkpoint saving, now only save during validation

        # Epoch metrics - only loss
        epoch_loss = total_loss / len(self.train_loader)

        return {
            "train_loss": epoch_loss,
        }

    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.eval()
        self.policy_processor.eval()

        total_loss = 0.0
        total_jaccard = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_roc_auc = 0.0
        total_precision_at_k = 0.0
        total_samples = 0
        total_additional_samples = 0

        # Calculate how many batches to evaluate based on subset fraction
        total_batches = len(self.val_loader)
        subset_batches = max(1, int(total_batches * self.config.val_subset_fraction))

        # Create random indices for subset evaluation
        import random

        batch_indices = random.sample(range(total_batches), subset_batches)
        batch_indices_set = set(batch_indices)

        # Select random sample for visualization (only once per validation)
        viz_batch_idx = random.randint(0, total_batches - 1)
        viz_sample_idx = None

        with torch.no_grad():
            for batch_idx, (
                state_embeddings,
                action_targets,
                action_masks,
                arrow_budgets,
                board_ws,
                board_hs,
            ) in enumerate(self.val_loader):
                # Move data to device
                state_embeddings = state_embeddings.to(self.device)
                action_targets = action_targets.to(self.device)
                action_masks = action_masks.to(self.device)
                arrow_budgets = arrow_budgets.to(self.device)

                # Forward pass - get raw logits from policy head
                logits = self.policy_processor.policy_head(state_embeddings)

                # Compute masked loss (always compute for all batches)
                loss = self.criterion(
                    logits, action_targets, action_masks, arrow_budgets
                )
                total_loss += loss.item()

                # Compute BC-Set metrics (always compute for all batches)
                batch_jaccard, batch_precision, batch_recall = (
                    self._compute_set_metrics(logits, action_targets, action_masks)
                )
                total_jaccard += batch_jaccard
                total_precision += batch_precision
                total_recall += batch_recall
                total_samples += action_targets.size(0)

                # Capture sample for visualization
                if batch_idx == viz_batch_idx and viz_sample_idx is None:
                    viz_sample_idx = random.randint(0, action_targets.size(0) - 1)
                    self._viz_sample_data = (
                        state_embeddings[viz_sample_idx].cpu(),
                        action_targets[viz_sample_idx].cpu(),
                        action_masks[viz_sample_idx].cpu(),
                        arrow_budgets[viz_sample_idx].cpu(),
                        board_ws[viz_sample_idx],
                        board_hs[viz_sample_idx],
                    )

                # Compute expensive metrics only on subset
                if batch_idx in batch_indices_set:
                    batch_roc_auc, batch_precision_at_k, batch_additional_samples = (
                        self._compute_additional_metrics(
                            logits, action_targets, action_masks, arrow_budgets
                        )
                    )
                    total_roc_auc += batch_roc_auc
                    total_precision_at_k += batch_precision_at_k
                    total_additional_samples += batch_additional_samples

        val_loss = total_loss / len(self.val_loader)
        val_jaccard = total_jaccard / total_samples if total_samples > 0 else 0.0
        val_precision = total_precision / total_samples if total_samples > 0 else 0.0
        val_recall = total_recall / total_samples if total_samples > 0 else 0.0
        val_roc_auc = (
            total_roc_auc / total_additional_samples
            if total_additional_samples > 0
            else 0.0
        )
        val_precision_at_k = (
            total_precision_at_k / total_additional_samples
            if total_additional_samples > 0
            else 0.0
        )

        # Show visualization for one random validation sample
        if hasattr(self, "_viz_sample_data"):
            state_emb, target, mask, budget, w, h = self._viz_sample_data
            with torch.no_grad():
                # Get raw logits
                raw_logits = self.policy_processor.policy_head(
                    state_emb.unsqueeze(0).to(self.device)
                )

                # Get masked logits using the utility function
                from ...util.action_utils import apply_action_mask

                masked_logits = apply_action_mask(
                    raw_logits, mask.unsqueeze(0).to(self.device)
                )

                # Create visualizations directory if it doesn't exist
                import os

                viz_dir = "model/visualizations"
                os.makedirs(viz_dir, exist_ok=True)

                # Generate filenames with epoch info
                raw_save_path = (
                    f"{viz_dir}/validation_epoch_{self.current_epoch:03d}_raw.png"
                )
                masked_save_path = (
                    f"{viz_dir}/validation_epoch_{self.current_epoch:03d}_masked.png"
                )

                # Visualize raw logits
                viz_board(
                    raw_logits.squeeze(0),
                    target,
                    mask,
                    w,
                    h,
                    title=f"RAW Logits - Epoch {self.current_epoch} (Budget: {budget:.0f})",
                    save_path=raw_save_path,
                )

                # Visualize masked logits
                viz_board(
                    masked_logits.squeeze(0),
                    target,
                    mask,
                    w,
                    h,
                    title=f"MASKED Logits - Epoch {self.current_epoch} (Budget: {budget:.0f})",
                    save_path=masked_save_path,
                )

        return {
            "val_loss": val_loss,
            "val_jaccard": val_jaccard,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_roc_auc": val_roc_auc,
            "val_precision_at_k": val_precision_at_k,
        }

    def train(self) -> Dict[str, float]:
        """Run full training loop.

        Returns:
            Final training metrics
        """
        print(f"Starting training on {self.device}")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Epoch-based validation (every N epochs)
            should_validate = (epoch % self.config.val_frequency_epochs == 0) or (
                epoch == self.config.max_epochs - 1
            )

            if should_validate:
                val_metrics = self.validate()

                # Print epoch summary with validation metrics
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Jaccard: {val_metrics['val_jaccard']:.4f}, "
                    f"Val ROC-AUC: {val_metrics['val_roc_auc']:.4f}, "
                    f"Val P@k: {val_metrics['val_precision_at_k']:.4f}, "
                    f"Steps: {self.current_step}, "
                )

                # Save latest model during evaluation
                self._save_checkpoint(
                    train_metrics["train_loss"],
                    0.0,
                    val_metrics,
                    is_final=False,
                )

                # Check for improvement
                if val_metrics["val_jaccard"] > self.best_val_jaccard:
                    self.best_val_jaccard = val_metrics["val_jaccard"]
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping - but only after min_epochs
                if (
                    patience_counter >= self.config.patience
                    and epoch >= self.config.min_epochs
                ):
                    print(
                        f"Early stopping at epoch {epoch} (patience: {self.config.patience}, min_epochs: {self.config.min_epochs})"
                    )
                    break

                # Check success criteria
                if self._check_success_criteria(val_metrics["val_jaccard"]):
                    print(f"Success criteria met at epoch {epoch}!")
                    break

            # Learning rate scheduling
            self.scheduler.step()

        # Training completed
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        final_metrics = {
            "best_val_jaccard": self.best_val_jaccard,
            "training_time_seconds": training_time,
            "total_epochs": self.current_epoch + 1,
            "total_steps": self.current_step,
        }

        # Save final checkpoint with new naming convention
        self._save_checkpoint(
            0.0,  # No final train loss needed
            0.0,  # No final train jaccard needed
            is_final=True,
        )

        return final_metrics

    def _log_training_step(self, loss: float, avg_loss: float) -> None:
        """Log training step metrics."""
        print(f"Step {self.current_step}: Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}")

    def _handle_validation_results(self, val_metrics: Dict[str, float]) -> None:
        """Handle validation results."""
        print(
            f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
            f"Jaccard: {val_metrics['val_jaccard']:.4f}"
        )

    def _save_checkpoint(
        self,
        train_loss: float,
        train_jaccard: float,
        val_metrics: Optional[Dict[str, float]] = None,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint."""
        metrics = {"train_loss": train_loss, "train_jaccard": train_jaccard}
        if val_metrics:
            metrics.update(val_metrics)

        # Use model saver to save checkpoint in standardized format
        self.model_saver.save_bc_set_checkpoint(
            cnn_encoder=self.perception_processor.get_cnn_encoder(),
            state_fusion=self.state_fusion_processor.fusion_mlp,
            policy_head=self.policy_processor.policy_head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            best_val_jaccard=self.best_val_jaccard,
            metrics=metrics,
            config=self.config,
            is_final=is_final,
        )

    def _initialize_mps_models(self) -> None:
        """Initialize models on MPS to avoid placeholder storage issues."""
        # Force initialization by running a dummy forward pass
        dummy_input = torch.randn(1, 128, device=self.device)

        # Initialize policy processor
        with torch.no_grad():
            _ = self.policy_processor.policy_head(dummy_input)

        # Initialize CNN encoder and state fusion processor with dummy perception output
        dummy_grid = torch.randn(1, 28, 10, 14, device=self.device)
        dummy_global = torch.randn(1, 16, device=self.device)
        dummy_cat = torch.randn(1, 32, device=self.device)

        # Initialize CNN encoder
        with torch.no_grad():
            dummy_grid_embedding = self.perception_processor.get_cnn_encoder()(
                dummy_grid
            )

        # Create dummy perception output
        from ...perception.data_types import PerceptionOutput

        dummy_perception = PerceptionOutput(
            grid_tensor=dummy_grid.squeeze(0),
            grid_embedding=dummy_grid_embedding.squeeze(0),
            global_features=dummy_global.squeeze(0),
            cat_embedding=dummy_cat.squeeze(0),
        )

        with torch.no_grad():
            _ = self.state_fusion_processor.fuse(dummy_perception)

    def _check_success_criteria(self, val_jaccard: float) -> bool:
        """Check if training success criteria are met."""
        # For BC-Set, success is achieving 70% Jaccard accuracy
        return val_jaccard >= 0.70


def main():
    """Main training function."""
    config = BCConfig()
    trainer = BCTrainer(config)
    final_metrics = trainer.train()

    print("\nTraining Summary:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
