"""
Standalone PPO rollout collection function with tqdm progress bar.

This module contains the rollout collection logic separated from the PPOTrainer class
for better modularity and easier debugging.
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def collect_rollout(
    # Model components
    perception_processor,
    state_fusion_processor,
    policy_processor,
    value_head,
    # Environment and data
    env_manager,
    buffer,
    curriculum,
    # Configuration
    config,
    device,
    # Monitoring functions
    check_model_stability_fn,
    reset_unstable_components_fn,
) -> Dict[str, Any]:
    """Collect a full rollout of experience with tqdm progress bar.

    Args:
        perception_processor: Perception processor module
        state_fusion_processor: State fusion processor module
        policy_processor: Policy processor module
        value_head: Value head module
        env_manager: Environment manager for parallel environments
        buffer: Rollout buffer to store experience
        curriculum: Curriculum manager for puzzle selection
        config: PPO configuration object
        device: PyTorch device
        check_model_stability_fn: Function to check model stability
        reset_unstable_components_fn: Function to reset unstable components

    Returns:
        Dictionary with rollout statistics
    """
    # Set models to evaluation mode
    perception_processor.get_cnn_encoder().eval()
    state_fusion_processor.eval()
    policy_processor.eval()
    value_head.eval()

    rollout_stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_entropy": 0.0,
        "value_estimates": 0.0,
    }

    # Get puzzle configurations from curriculum
    puzzle_configs = curriculum.get_puzzle_batch(config.num_parallel_envs)

    # Reset environments
    observations, action_masks = env_manager.reset_all(puzzle_configs)
    observations = observations.to(device)
    action_masks = action_masks.to(device)

    # Check model stability before rollout
    if not check_model_stability_fn():
        print(
            "Warning: Model instability detected before rollout, resetting unstable components"
        )
        reset_unstable_components_fn()

    # Collect rollout with progress bar
    total_entropy = 0.0
    total_value = 0.0
    step_count = 0
    episode_count = 0
    total_episode_score = 0.0  # Running total of episode scores

    # Create progress bar for rollout collection
    with tqdm(
        total=config.rollout_length,
        desc="Collecting Rollout",
        unit="step",
        leave=False,
        ncols=100,
    ) as pbar:
        for step in range(config.rollout_length):
            with torch.no_grad():
                # Get policy logits and values
                policy_logits = policy_processor.policy_head(observations)
                values = value_head(observations).squeeze(-1)

                # Check for NaN/Inf in raw policy logits
                if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                    print(f"Critical: NaN/Inf in policy_logits at step {step}")
                    print(
                        f"Policy logits stats: min={policy_logits.min().item():.4f}, max={policy_logits.max().item():.4f}, mean={policy_logits.mean().item():.4f}"
                    )
                    # Check if observations contain NaN/Inf
                    if (
                        torch.isnan(observations).any()
                        or torch.isinf(observations).any()
                    ):
                        print("Observations contain NaN/Inf!")
                    raise RuntimeError(
                        "Policy network producing NaN/Inf values. Training stopped."
                    )

                # Clamp policy logits to prevent extreme values
                policy_logits = torch.clamp(policy_logits, min=-10.0, max=10.0)

                # Apply action masks with more stable masking
                # Instead of -inf, use a large negative value to prevent numerical issues
                masked_logits = policy_logits.clone()
                masked_logits[action_masks == 0] = -1e9

                # Check for NaN/Inf and handle gracefully
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    print(
                        "Warning: NaN/Inf detected in masked logits, creating safe uniform distribution"
                    )
                    # Create safe uniform distribution over valid actions
                    mask_counts = action_masks.sum(dim=-1, keepdim=True)
                    safe_counts = torch.where(
                        mask_counts > 0, mask_counts, torch.ones_like(mask_counts)
                    )
                    masked_logits = torch.log(action_masks.float() / safe_counts + 1e-8)

                # Sample actions
                action_probs = F.softmax(masked_logits, dim=-1)

                # Debug and fix NaN in action probabilities
                if torch.isnan(action_probs).any():
                    if step % 50 == 0:
                        print(
                            f"Warning: NaN detected in action probabilities at step {step}"
                        )

                        # Debug which environments have NaN
                        nan_envs = torch.isnan(action_probs).any(dim=1)
                        print(
                            f"Environments with NaN: {torch.where(nan_envs)[0].tolist()}"
                        )

                        # Check the source tensors
                        if torch.isnan(policy_logits).any():
                            print("NaN found in policy_logits")
                        if torch.isnan(masked_logits).any():
                            print("NaN found in masked_logits")
                        if torch.isnan(observations).any():
                            print("NaN found in observations")

                        # Check for inf values that become NaN after softmax
                        if torch.isinf(masked_logits).any():
                            print("Inf found in masked_logits")

                    # Create safe uniform distribution over valid actions
                    mask_counts = action_masks.sum(dim=-1, keepdim=True)
                    safe_counts = torch.where(
                        mask_counts > 0, mask_counts, torch.ones_like(mask_counts)
                    )
                    safe_masks = torch.where(
                        mask_counts > 0, action_masks, torch.ones_like(action_masks)
                    )
                    action_probs = safe_masks.float() / safe_counts

                action_dist = torch.distributions.Categorical(action_probs)
                actions = action_dist.sample()
                log_probs = action_dist.log_prob(actions)

                # Calculate entropy and value for logging
                entropy = action_dist.entropy().mean()
                total_entropy += entropy.item()
                total_value += values.mean().item()
                step_count += 1

            # Generate new puzzles for auto-reset
            auto_reset_puzzles = curriculum.get_puzzle_batch(config.num_parallel_envs)

            # Step environments with auto-reset
            next_obs, rewards, dones, infos, next_masks = env_manager.step_all(
                actions, auto_reset_puzzles
            )
            next_obs = next_obs.to(device)
            next_masks = next_masks.to(device)

            # Store experience in buffer (ensure all tensors are on the right device)
            buffer.add(
                observations=observations.to(device),
                actions=actions.to(device),
                log_probs=log_probs.to(device),
                values=values.to(device),
                rewards=rewards.to(device),
                dones=dones.to(device),
                action_masks=action_masks.to(device),
            )

            # Update for next step
            observations = next_obs
            action_masks = next_masks

            # Collect episode statistics
            step_episode_count = 0
            for info in infos:
                if "episode_stats" in info:
                    stats = info["episode_stats"]
                    if "final_reward" in info:
                        final_reward = info["final_reward"]
                        rollout_stats["episode_rewards"].append(final_reward)
                        rollout_stats["episode_lengths"].append(
                            stats["placement_steps"] + stats["execution_ticks"]
                        )
                        step_episode_count += 1
                        total_episode_score += final_reward

            episode_count += step_episode_count

            # Calculate average score across completed episodes
            avg_score = (
                total_episode_score / episode_count if episode_count > 0 else 0.0
            )

            # Update progress bar
            pbar.set_postfix(
                {
                    "episodes": episode_count,
                    "avg_score": f"{avg_score:.3f}",
                    "entropy": f"{entropy.item():.4f}",
                    "value": f"{values.mean().item():.3f}",
                    "reward": f"{rewards.mean().item() if len(rewards) > 0 else 0.0:.3f}",
                }
            )
            pbar.update(1)

    # Compute final values for advantage calculation
    with torch.no_grad():
        final_values = value_head(observations).squeeze(-1)

    # Compute advantages and returns
    buffer.compute_advantages_and_returns(final_values)

    # Update rollout statistics
    rollout_stats["policy_entropy"] = (
        total_entropy / step_count if step_count > 0 else 0.0
    )
    rollout_stats["value_estimates"] = (
        total_value / step_count if step_count > 0 else 0.0
    )

    # Get episode statistics from buffer
    buffer_stats = buffer.get_episode_stats()
    rollout_stats.update(buffer_stats)

    return rollout_stats
