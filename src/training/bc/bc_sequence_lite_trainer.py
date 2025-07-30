"""
BC Sequence Lite trainer with freeze/unfreeze logic.
"""

import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ...model.model_loader import ModelLoader
from ...model.model_saver import ModelSaver
from .config import BCConfig
from .data_loader import get_device


class BCSequenceLiteTrainer:
    """BC Sequence Lite trainer with perception/fusion freezing."""

    def __init__(
        self,
        config: BCConfig,
        perception_processor,
        state_fusion_processor,
        policy_processor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            perception_processor: Pre-trained perception processor
            state_fusion_processor: Pre-trained state fusion processor
            policy_processor: Pre-trained policy processor
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.config = config
        self.perception_processor = perception_processor
        self.state_fusion_processor = state_fusion_processor
        self.policy_processor = policy_processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Move components to device
        self.perception_processor.get_cnn_encoder().to(device)
        self.state_fusion_processor.to(device)
        self.policy_processor.to(device)

        # Set up optimizer (will be updated during training phases)
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0

        # Create model saver
        self.model_saver = ModelSaver(config.model_dir)

    def train(self) -> Dict[str, float]:
        """Train the model with freeze/unfreeze schedule.

        Returns:
            Dictionary with final training metrics
        """
        print("Starting BC Sequence Lite training...")
        print(f"Device: {self.device}")
        print(f"Total training samples: {len(self.train_loader.dataset)}")

        # Phase 1: Freeze perception + fusion, train policy head only (3 epochs)
        print("\n=== Phase 1: Training policy head only (perception/fusion frozen) ===")
        self._freeze_perception_and_fusion()
        self._setup_optimizer(lr=self.config.learning_rate)

        phase1_metrics = self._train_phase(num_epochs=3, phase_name="Phase1_Frozen")

        # Phase 2: Unfreeze all, train with 10x lower LR (5-7 epochs)
        print("\n=== Phase 2: Training all layers (unfrozen) ===")
        self._unfreeze_all()
        self._setup_optimizer(lr=self.config.learning_rate * 0.1)  # 10x lower LR

        phase2_metrics = self._train_phase(num_epochs=7, phase_name="Phase2_Unfrozen")

        # Save final checkpoint after training completion
        self._save_checkpoint(is_final=True)

        # Return combined metrics
        final_metrics = {**phase1_metrics, **phase2_metrics}

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Final checkpoint saved: model/bc_seq_lite_final.pth")

        return final_metrics

    def _freeze_perception_and_fusion(self):
        """Freeze perception and fusion layers, only train policy head."""
        print("Freezing perception and fusion layers...")

        # Count frozen/unfrozen parameters
        frozen_params = 0
        unfrozen_params = 0

        # Freeze CNN encoder
        cnn_encoder = self.perception_processor.get_cnn_encoder()
        for param in cnn_encoder.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

        # Freeze state fusion components
        for param in self.state_fusion_processor.fusion_mlp.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

        # Keep policy head unfrozen
        for param in self.policy_processor.policy_head.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()

        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {unfrozen_params:,}")

    def _unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        print("Unfreezing all layers...")

        total_params = 0

        # Unfreeze CNN encoder
        cnn_encoder = self.perception_processor.get_cnn_encoder()
        for param in cnn_encoder.parameters():
            param.requires_grad = True
            total_params += param.numel()

        # Unfreeze state fusion components
        for param in self.state_fusion_processor.fusion_mlp.parameters():
            param.requires_grad = True
            total_params += param.numel()

        # Keep policy head unfrozen
        for param in self.policy_processor.policy_head.parameters():
            param.requires_grad = True
            total_params += param.numel()

        print(f"Total trainable parameters: {total_params:,}")

    def _setup_optimizer(self, lr: float):
        """Setup optimizer and scheduler for current phase."""
        # Collect trainable parameters from components
        trainable_params = []

        # Policy head parameters (should always be trainable)
        policy_params = list(self.policy_processor.policy_head.parameters())
        trainable_policy_params = [p for p in policy_params if p.requires_grad]
        trainable_params.extend(trainable_policy_params)
        print(f"Policy head trainable params: {len(trainable_policy_params)}")

        # CNN encoder parameters (trainable in phase 2)
        cnn_encoder = self.perception_processor.get_cnn_encoder()
        cnn_params = list(cnn_encoder.parameters())
        trainable_cnn_params = [p for p in cnn_params if p.requires_grad]
        trainable_params.extend(trainable_cnn_params)
        print(f"CNN encoder trainable params: {len(trainable_cnn_params)}")

        # State fusion parameters (trainable in phase 2)
        fusion_params = list(self.state_fusion_processor.fusion_mlp.parameters())
        trainable_fusion_params = [p for p in fusion_params if p.requires_grad]
        trainable_params.extend(trainable_fusion_params)
        print(f"State fusion trainable params: {len(trainable_fusion_params)}")

        if len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found. Check model setup and freeze/unfreeze logic."
            )

        self.optimizer = AdamW(
            trainable_params, lr=lr, weight_decay=self.config.weight_decay
        )

        # Simple cosine annealing scheduler
        total_steps = (
            len(self.train_loader) * 10
        )  # Approximate total steps for this phase
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        print(
            f"Optimizer setup: lr={lr}, total_trainable_params={len(trainable_params)}"
        )

    def _train_phase(self, num_epochs: int, phase_name: str) -> Dict[str, float]:
        """Train for a specific phase."""
        phase_metrics = {}

        for epoch in range(num_epochs):
            self.epoch += 1

            # Training step
            train_metrics = self._train_epoch()

            # Validation step
            val_metrics = self._validate()

            # Logging
            print(f"Epoch {self.epoch} ({phase_name}):")
            print(
                f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
            )
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save latest model during evaluation
            self._save_checkpoint()

            # Track best accuracy but don't save separate best checkpoint
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Store final metrics
            phase_metrics.update(
                {
                    f"{phase_name}_final_train_loss": train_metrics["loss"],
                    f"{phase_name}_final_train_acc": train_metrics["accuracy"],
                    f"{phase_name}_final_val_loss": val_metrics["loss"],
                    f"{phase_name}_final_val_acc": val_metrics["accuracy"],
                }
            )

        return phase_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        # Set components to training mode
        self.perception_processor.get_cnn_encoder().train()
        self.state_fusion_processor.fusion_mlp.train()
        self.policy_processor.policy_head.train()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (state_embeddings, target_actions, action_masks) in enumerate(
            self.train_loader
        ):
            # Move to device
            state_embeddings = state_embeddings.to(self.device)
            target_actions = target_actions.to(self.device)
            action_masks = action_masks.to(self.device)

            # Forward pass - get policy logits directly from state embeddings
            policy_logits = self.policy_processor.policy_head(state_embeddings)

            # Apply action mask (set invalid actions to large negative value)
            masked_logits = policy_logits.clone()
            masked_logits[action_masks == 0] = -1e9

            # Cross-entropy loss (as specified - no focal BCE)
            loss = F.cross_entropy(masked_logits, target_actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping - collect all parameters
            all_params = []
            all_params.extend(self.perception_processor.get_cnn_encoder().parameters())
            all_params.extend(self.state_fusion_processor.fusion_mlp.parameters())
            all_params.extend(self.policy_processor.policy_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.config.gradient_clipping)

            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()

            # Calculate accuracy (on valid actions only)
            with torch.no_grad():
                predicted = torch.argmax(masked_logits, dim=1)
                correct_predictions += (predicted == target_actions).sum().item()
                total_predictions += target_actions.size(0)

            self.global_step += 1

            # Logging
            if batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions

        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        # Set components to evaluation mode
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.fusion_mlp.eval()
        self.policy_processor.policy_head.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for state_embeddings, target_actions, action_masks in self.val_loader:
                # Move to device
                state_embeddings = state_embeddings.to(self.device)
                target_actions = target_actions.to(self.device)
                action_masks = action_masks.to(self.device)

                # Forward pass
                policy_logits = self.policy_processor.policy_head(state_embeddings)

                # Apply action mask
                masked_logits = policy_logits.clone()
                masked_logits[action_masks == 0] = -1e9

                # Loss
                loss = F.cross_entropy(masked_logits, target_actions)
                total_loss += loss.item()

                # Accuracy
                predicted = torch.argmax(masked_logits, dim=1)
                correct_predictions += (predicted == target_actions).sum().item()
                total_predictions += target_actions.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions

        return {"loss": avg_loss, "accuracy": accuracy}

    def _save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint using model saver."""
        # Determine current phase
        current_phase = "training"
        if is_final:
            current_phase = "completed"
        elif hasattr(self, "_current_phase"):
            current_phase = self._current_phase

        # Get list of frozen parameters for tracking
        frozen_params = []
        if hasattr(self, "_frozen_params"):
            frozen_params = self._frozen_params

        # Use model saver to save checkpoint in standardized format
        self.model_saver.save_bc_seq_lite_checkpoint(
            cnn_encoder=self.perception_processor.get_cnn_encoder(),
            state_fusion=self.state_fusion_processor.fusion_mlp,
            policy_head=self.policy_processor.policy_head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            phase=current_phase,
            best_val_accuracy=self.best_val_accuracy,
            metrics={},  # Will be filled by training metrics
            frozen_params=frozen_params,
            is_final=is_final,
        )


def train_bc_sequence_lite(
    config: BCConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bc_model_path: str = None,
) -> Dict[str, float]:
    """Main training function for BC Sequence Lite.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        bc_model_path: Path to pre-trained BC-Set model checkpoint

    Returns:
        Dictionary with training metrics
    """
    # Get device
    device = get_device(config.device)

    # Load pre-trained BC-Set model
    if bc_model_path is None:
        # Look for default BC model paths with new naming convention
        possible_paths = ["model/bc_set_final.pth", "model/bc_set_latest.pth"]
        bc_model_path = next(
            (path for path in possible_paths if os.path.exists(path)), None
        )

        if bc_model_path is None:
            raise FileNotFoundError(
                "No BC-Set model found. Please train BC-Set first or provide bc_model_path. "
                f"Searched: {possible_paths}"
            )

    print(f"Loading BC-Set model from: {bc_model_path}")

    model_loader = ModelLoader(
        model_path=bc_model_path,
        device=device,
        load_components=True,
    )

    (
        perception_processor,
        state_fusion_processor,
        policy_processor,
    ) = model_loader.get_bc_components()

    bc_set_info = model_loader.get_metadata("bc_set")
    if bc_set_info:
        print(
            f"Source: BC-Set training (epoch {bc_set_info['epoch']}, step {bc_set_info['step']})"
        )
        print(f"BC-Set best Jaccard: {bc_set_info.get('best_val_jaccard', 'N/A'):.4f}")

    # Create trainer with loaded components
    trainer = BCSequenceLiteTrainer(
        config=config,
        perception_processor=perception_processor,
        state_fusion_processor=state_fusion_processor,
        policy_processor=policy_processor,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train the model
    metrics = trainer.train()

    return metrics
