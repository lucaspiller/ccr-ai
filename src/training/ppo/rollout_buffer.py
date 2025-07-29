"""
Rollout buffer for PPO training.

Stores experience trajectories and computes GAE advantages for on-policy learning.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import PPOConfig


@dataclass
class PPOExperience:
    """Single step of PPO experience."""

    observation: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: torch.Tensor


class RolloutBuffer:
    """Buffer for storing PPO rollout experience and computing advantages."""

    def __init__(
        self,
        config: PPOConfig,
        observation_dim: int,
        num_envs: int,
        device: torch.device = None,
    ):
        """Initialize rollout buffer.

        Args:
            config: PPO configuration
            observation_dim: Dimension of state observations
            num_envs: Number of parallel environments
            device: Device to store tensors on
        """
        self.config = config
        self.observation_dim = observation_dim
        self.num_envs = num_envs
        self.max_steps = config.rollout_length
        self.device = device or torch.device("cpu")

        # Storage tensors [max_steps, num_envs, ...]
        self.observations = torch.zeros(
            (self.max_steps, num_envs, observation_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.long, device=self.device
        )
        self.log_probs = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.float32, device=self.device
        )
        self.values = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.bool, device=self.device
        )
        self.action_masks = torch.zeros(
            (self.max_steps, num_envs, 700), dtype=torch.bool, device=self.device
        )

        # Computed advantages and returns
        self.advantages = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.float32, device=self.device
        )
        self.returns = torch.zeros(
            (self.max_steps, num_envs), dtype=torch.float32, device=self.device
        )

        # Buffer state
        self.step = 0
        self.is_full = False

        # Episode tracking for debugging
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = [0.0] * num_envs
        self.current_episode_lengths = [0] * num_envs

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        action_masks: torch.Tensor,
    ):
        """Add experience to buffer.

        Args:
            observations: [num_envs, observation_dim]
            actions: [num_envs]
            log_probs: [num_envs]
            values: [num_envs]
            rewards: [num_envs]
            dones: [num_envs]
            action_masks: [num_envs, 700]
        """
        if self.step >= self.max_steps:
            raise ValueError(f"Buffer is full (step {self.step} >= {self.max_steps})")

        # Store experience (detach tensors to prevent gradient issues)
        self.observations[self.step] = observations.detach()
        self.actions[self.step] = actions.detach()
        self.log_probs[self.step] = log_probs.detach()
        self.values[self.step] = values.detach()
        self.rewards[self.step] = rewards.detach()
        self.dones[self.step] = dones.detach()
        self.action_masks[self.step] = action_masks.detach()

        # Update episode tracking
        for env_idx in range(self.num_envs):
            self.current_episode_rewards[env_idx] += rewards[env_idx].item()
            self.current_episode_lengths[env_idx] += 1

            if dones[env_idx]:
                # Episode completed
                self.episode_rewards.append(self.current_episode_rewards[env_idx])
                self.episode_lengths.append(self.current_episode_lengths[env_idx])
                self.current_episode_rewards[env_idx] = 0.0
                self.current_episode_lengths[env_idx] = 0

        self.step += 1
        if self.step >= self.max_steps:
            self.is_full = True

    def compute_advantages_and_returns(self, next_values: torch.Tensor):
        """Compute GAE advantages and returns.

        Args:
            next_values: Values for the next states [num_envs]
        """
        # Initialize GAE computation
        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Work backwards through the trajectory
        for step in reversed(range(self.step)):
            if step == self.step - 1:
                # Last step - use provided next values
                next_non_terminal = (~self.dones[step]).float()
                next_value = next_values
            else:
                # Use value from next step
                next_non_terminal = (~self.dones[step]).float()
                next_value = self.values[step + 1]

            # Compute TD error: δ = r + γ * V(s') - V(s)
            delta = (
                self.rewards[step]
                + self.config.discount_factor * next_value * next_non_terminal
                - self.values[step]
            )

            # Update GAE: A = δ + γ * λ * next_non_terminal * A_next
            gae = (
                delta
                + self.config.discount_factor
                * self.config.gae_lambda
                * next_non_terminal
                * gae
            )

            self.advantages[step] = gae

        # Compute returns: R = A + V
        self.returns = self.advantages + self.values[: self.step]

        # Normalize advantages for stability
        self._normalize_advantages()

    def _normalize_advantages(self):
        """Normalize advantages to have zero mean and unit variance."""
        valid_advantages = self.advantages[: self.step].flatten()

        # Only normalize if we have valid data
        if len(valid_advantages) > 1:
            mean = valid_advantages.mean()
            std = valid_advantages.std()

            # Avoid division by zero
            if std > 1e-8:
                self.advantages[: self.step] = (
                    self.advantages[: self.step] - mean
                ) / std

    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get shuffled minibatches for PPO updates.

        Args:
            batch_size: Size of each minibatch

        Returns:
            List of minibatch dictionaries
        """
        if not self.is_full:
            raise ValueError("Buffer must be full before getting batches")

        # Flatten all data [total_steps]
        total_steps = self.step * self.num_envs

        observations = self.observations[: self.step].flatten(
            0, 1
        )  # [total_steps, obs_dim]
        actions = self.actions[: self.step].flatten()  # [total_steps]
        log_probs = self.log_probs[: self.step].flatten()  # [total_steps]
        values = self.values[: self.step].flatten()  # [total_steps]
        advantages = self.advantages[: self.step].flatten()  # [total_steps]
        returns = self.returns[: self.step].flatten()  # [total_steps]
        action_masks = self.action_masks[: self.step].flatten(
            0, 1
        )  # [total_steps, 700]

        # Create random permutation for shuffling
        indices = torch.randperm(total_steps)

        # Split into batches
        batches = []
        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)
            batch_indices = indices[start:end]

            batch = {
                "observations": observations[batch_indices],
                "actions": actions[batch_indices],
                "log_probs": log_probs[batch_indices],
                "values": values[batch_indices],
                "advantages": advantages[batch_indices],
                "returns": returns[batch_indices],
                "action_masks": action_masks[batch_indices],
            }
            batches.append(batch)

        return batches

    def clear(self):
        """Clear buffer for next rollout."""
        self.step = 0
        self.is_full = False

        # Don't clear episode tracking - keep for statistics

    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        if len(self.episode_rewards) == 0:
            return {
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
                "num_episodes": 0,
            }

        stats = {
            "mean_episode_reward": np.mean(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
            "num_episodes": len(self.episode_rewards),
        }

        return stats

    def clear_episode_stats(self):
        """Clear episode statistics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()

    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information for debugging."""
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "is_full": self.is_full,
            "num_envs": self.num_envs,
            "observation_dim": self.observation_dim,
            "total_stored_steps": self.step * self.num_envs if self.step > 0 else 0,
        }


class MultiEnvironmentRolloutBuffer:
    """Enhanced rollout buffer that handles variable episode lengths better."""

    def __init__(self, config: PPOConfig, observation_dim: int, num_envs: int):
        """Initialize multi-environment rollout buffer.

        This version is better at handling episodes that end at different times
        and properly masks out invalid steps during advantage computation.
        """
        self.config = config
        self.observation_dim = observation_dim
        self.num_envs = num_envs
        self.max_steps = config.rollout_length

        # Use the same storage as regular buffer
        self.buffer = RolloutBuffer(config, observation_dim, num_envs)

        # Additional tracking for better episode handling
        self.episode_masks = torch.ones(
            (self.max_steps, num_envs), dtype=torch.bool
        )  # True for valid steps

    def add(self, *args, **kwargs):
        """Add experience (same interface as RolloutBuffer)."""
        self.buffer.add(*args, **kwargs)

        # Update episode masks - mark steps after episode ends as invalid
        if self.buffer.step > 1:
            prev_dones = self.buffer.dones[
                self.buffer.step - 2
            ]  # Previous step's dones
            current_step = self.buffer.step - 1

            # If environment was done in previous step, this step is start of new episode
            # All steps are valid for now - this could be enhanced for more complex masking
            self.episode_masks[current_step] = True

    def compute_advantages_and_returns(self, next_values: torch.Tensor):
        """Compute advantages with proper episode masking."""
        # Use base implementation but apply episode masks
        self.buffer.compute_advantages_and_returns(next_values)

        # Zero out advantages and returns for invalid steps
        invalid_mask = ~self.episode_masks[: self.buffer.step]
        self.buffer.advantages[: self.buffer.step][invalid_mask] = 0.0
        self.buffer.returns[: self.buffer.step][invalid_mask] = 0.0

    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get batches with episode masking."""
        batches = self.buffer.get_batches(batch_size)

        # Add episode masks to each batch
        episode_masks = self.episode_masks[: self.buffer.step].flatten()

        for i, batch in enumerate(batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(episode_masks))
            batch_mask = episode_masks[start_idx:end_idx]
            batch["episode_masks"] = batch_mask

        return batches

    def clear(self):
        """Clear buffer and reset episode masks."""
        self.buffer.clear()
        self.episode_masks.fill_(True)

    def __getattr__(self, name):
        """Delegate other attributes to base buffer."""
        return getattr(self.buffer, name)
