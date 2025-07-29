"""
Standalone PPO policy update function with tqdm progress bar.

This module contains the policy update logic separated from the PPOTrainer class
for better modularity and easier debugging.
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def update_policy(
    # Model components
    perception_processor,
    state_fusion_processor,
    policy_processor,
    value_head,
    optimizer,
    # Training data
    buffer,
    # Configuration
    config,
    device,
    # Current hyperparameters
    current_clip_eps: float,
    current_value_clip_eps: float,
    current_entropy_coeff: float,
    # Monitoring functions
    check_model_stability_fn,
    reset_unstable_components_fn,
    monitor_gradient_norm_fn,
) -> Dict[str, float]:
    """Update policy using PPO algorithm with tqdm progress bar.

    Args:
        perception_processor: Perception processor module
        state_fusion_processor: State fusion processor module
        policy_processor: Policy processor module
        value_head: Value head module
        optimizer: PyTorch optimizer
        buffer: Rollout buffer with collected experience
        config: PPO configuration object
        device: PyTorch device
        current_clip_eps: Current clipping epsilon value
        current_value_clip_eps: Current value clipping epsilon
        current_entropy_coeff: Current entropy coefficient
        check_model_stability_fn: Function to check model stability
        reset_unstable_components_fn: Function to reset unstable components
        monitor_gradient_norm_fn: Function to monitor gradient norms

    Returns:
        Dictionary with training statistics
    """
    # Set models to training mode
    perception_processor.get_cnn_encoder().train()
    state_fusion_processor.train()
    policy_processor.train()
    value_head.train()

    update_stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
        "kl_divergence": 0.0,
        "explained_variance": 0.0,
        "gradient_norm": 0.0,
    }

    # Check model stability before policy update
    if not check_model_stability_fn():
        print(
            "Warning: Model instability detected before policy update, resetting unstable components"
        )
        reset_unstable_components_fn()
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "kl_divergence": 0.0,
            "explained_variance": 0.0,
            "gradient_norm": 0.0,
        }

    # Get batches from buffer
    batches = buffer.get_batches(config.batch_size)

    # Calculate total number of batches across all epochs for progress bar
    batches_per_epoch = len(batches)
    total_expected_batches = batches_per_epoch * config.ppo_epochs

    total_batches = 0
    early_stop = False

    # Create progress bar for the entire policy update
    with tqdm(
        total=total_expected_batches,
        desc="Policy Update",
        unit="batch",
        leave=False,
        ncols=100,
    ) as pbar:
        # Multiple epochs over the data
        for epoch in range(config.ppo_epochs):
            if early_stop:
                break

            epoch_kl = 0.0
            epoch_batches = 0

            for batch in batches:
                if early_stop:
                    break

                # Move batch to device
                observations = batch["observations"].to(device)
                actions = batch["actions"].to(device)
                old_log_probs = batch["log_probs"].to(device)
                old_values = batch["values"].to(device)
                advantages = batch["advantages"].to(device)
                returns = batch["returns"].to(device)
                action_masks = batch["action_masks"].to(device)

                # Forward pass
                policy_logits = policy_processor.policy_head(observations)
                values = value_head(observations).squeeze(-1)

                # Check for NaN/Inf in training forward pass
                if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                    print("Critical: NaN/Inf in policy_logits during training update")
                    raise RuntimeError(
                        "Policy network producing NaN/Inf values during training. Stopping."
                    )

                # Clamp policy logits to prevent extreme values during training
                policy_logits = torch.clamp(policy_logits, min=-10.0, max=10.0)

                # Apply action masks with more stable masking
                masked_logits = policy_logits.clone()
                masked_logits[action_masks == 0] = -1e9

                # Check for numerical issues in masked logits
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    print(
                        "Warning: NaN/Inf in masked logits during training, creating safe distribution"
                    )
                    mask_counts = action_masks.sum(dim=-1, keepdim=True)
                    safe_counts = torch.where(
                        mask_counts > 0, mask_counts, torch.ones_like(mask_counts)
                    )
                    masked_logits = torch.log(action_masks.float() / safe_counts + 1e-8)

                # Compute policy loss
                action_probs = F.softmax(masked_logits, dim=-1)

                # Handle NaN in action probabilities during policy update
                if torch.isnan(action_probs).any():
                    print(
                        "Warning: NaN in policy update action probabilities, fixing..."
                    )
                    # Use uniform distribution over valid actions
                    mask_counts = action_masks.sum(dim=-1, keepdim=True)
                    safe_counts = torch.where(
                        mask_counts > 0, mask_counts, torch.ones_like(mask_counts)
                    )
                    safe_masks = torch.where(
                        mask_counts > 0, action_masks, torch.ones_like(action_masks)
                    )
                    action_probs = safe_masks.float() / safe_counts

                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(actions)

                # Policy ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - current_clip_eps, 1.0 + current_clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                if config.value_clip_eps_start > 0:
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -current_value_clip_eps,
                        current_value_clip_eps,
                    )
                    value_loss1 = F.mse_loss(values, returns)
                    value_loss2 = F.mse_loss(values_clipped, returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, returns)

                # Entropy loss
                entropy = action_dist.entropy().mean()
                entropy_loss = -current_entropy_coeff * entropy

                # Total loss
                total_loss = (
                    policy_loss + config.value_loss_coeff * value_loss + entropy_loss
                )

                # Check for NaN in loss before backward pass
                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    print(
                        f"Warning: NaN/Inf in total_loss: {total_loss.item()}, skipping update"
                    )
                    pbar.set_postfix(
                        {
                            "epoch": f"{epoch+1}/{config.ppo_epochs}",
                            "status": "NaN loss - skipped",
                        }
                    )
                    pbar.update(1)
                    continue

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()

                # Check for NaN/Inf in gradients
                nan_grads = False
                for name, param in [
                    ("cnn_encoder", perception_processor.get_cnn_encoder()),
                    ("state_fusion", state_fusion_processor.fusion_mlp),
                    ("policy_head", policy_processor.policy_head),
                    ("value_head", value_head),
                ]:
                    for p in param.parameters():
                        if p.grad is not None and (
                            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        ):
                            print(
                                f"Warning: NaN/Inf gradients in {name}, skipping update"
                            )
                            nan_grads = True
                            break
                    if nan_grads:
                        break

                if nan_grads:
                    optimizer.zero_grad()  # Clear bad gradients
                    pbar.set_postfix(
                        {
                            "epoch": f"{epoch+1}/{config.ppo_epochs}",
                            "status": "NaN grads - skipped",
                        }
                    )
                    pbar.update(1)
                    continue

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]["params"], config.max_grad_norm
                )

                # Monitor gradient norm and apply adaptive scaling
                monitor_gradient_norm_fn(grad_norm)

                optimizer.step()

                # Check for NaN/Inf in parameters after update
                nan_params = False
                for name, param in [
                    ("cnn_encoder", perception_processor.get_cnn_encoder()),
                    ("state_fusion", state_fusion_processor.fusion_mlp),
                    ("policy_head", policy_processor.policy_head),
                    ("value_head", value_head),
                ]:
                    for p in param.parameters():
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            print(
                                f"Critical: NaN/Inf parameters in {name} after update! Stopping training."
                            )
                            nan_params = True
                            break
                    if nan_params:
                        break

                if nan_params:
                    print("Attempting to recover from parameter instability...")
                    reset_unstable_components_fn()
                    # Note: Learning rate reduction is handled by the reset function
                    pbar.set_postfix(
                        {
                            "epoch": f"{epoch+1}/{config.ppo_epochs}",
                            "status": "Reset model - recovered",
                        }
                    )
                    pbar.update(1)
                    # Skip this batch but continue training
                    continue

                # KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (old_log_probs - new_log_probs).mean()
                    epoch_kl += kl_div.item()
                    epoch_batches += 1

                # Update statistics
                update_stats["policy_loss"] += policy_loss.item()
                update_stats["value_loss"] += value_loss.item()
                update_stats["entropy_loss"] += entropy_loss.item()
                update_stats["kl_divergence"] += kl_div.item()
                update_stats["gradient_norm"] += grad_norm.item()
                total_batches += 1

                # Update progress bar with current metrics
                pbar.set_postfix(
                    {
                        "epoch": f"{epoch+1}/{config.ppo_epochs}",
                        "p_loss": f"{policy_loss.item():.4f}",
                        "v_loss": f"{value_loss.item():.4f}",
                        "kl": f"{kl_div.item():.6f}",
                        "grad": f"{grad_norm.item():.2f}",
                    }
                )
                pbar.update(1)

                # Explained variance
                with torch.no_grad():
                    y_true = returns
                    y_pred = values
                    var_y = torch.var(y_true)
                    explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                    update_stats["explained_variance"] += explained_var.item()

            # Check for early stopping based on KL divergence (at epoch level)
            if config.kl_adaptive_beta and epoch_batches > 0:
                avg_kl = epoch_kl / epoch_batches
                if avg_kl > 2.0 * config.kl_target:
                    early_stop = True
                    pbar.set_description(
                        f"Policy Update (Early Stop - High KL: {avg_kl:.6f})"
                    )
                    print(
                        f"Early stopping at epoch {epoch} due to high KL: {avg_kl:.6f}"
                    )
                    # Skip remaining batches in progress bar
                    remaining_batches = total_expected_batches - pbar.n
                    pbar.update(remaining_batches)
                    break

        # Average statistics
        if total_batches > 0:
            for key in update_stats:
                update_stats[key] /= total_batches

    return update_stats
