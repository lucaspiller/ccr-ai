import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.perception.data_types import PerceptionConfig
from src.perception.processors import GameStateProcessor
from src.policy.processors import PolicyProcessor
from src.policy.value_head import ValueConfig, ValueHead
from src.state_fusion.processors import StateFusionProcessor


class ModelLoader:
    def __init__(
        self, model_path: str, device: str = "auto", load_components: bool = True
    ):
        """Initialize model loader with checkpoint path.

        Args:
            model_path: Path to saved model checkpoint (BC or PPO)
            device: Device to load model on
            load_components: Whether to immediately load model components
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.checkpoint_type = None
        self.checkpoint_data = None

        # Model components (initialized when loaded)
        self.perception_processor = None
        self.state_fusion_processor = None
        self.policy_processor = None
        self.value_head = None
        self.parameter_count = 0

        if load_components:
            self.load_model()
            print(f"Model: {model_path}")
            print(f"Device: {self.device}")
            print(f"Checkpoint type: {self.checkpoint_type}")
            print(f"Total parameters: {self.parameter_count:,}")

    def _get_device(self, device_str: str) -> torch.device:
        """Get evaluation device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_str)

    def _count_parameters(self) -> int:
        """Count total model parameters."""
        total = 0
        total += sum(
            p.numel() for p in self.perception_processor.get_cnn_encoder().parameters()
        )
        total += sum(
            p.numel() for p in self.state_fusion_processor.fusion_mlp.parameters()
        )
        total += sum(p.numel() for p in self.policy_processor.policy_head.parameters())
        total += sum(p.numel() for p in self.value_head.parameters())
        return total

    def _detect_checkpoint_type(self, checkpoint: Dict[str, Any]) -> str:
        """Detect whether checkpoint is BC or PPO format.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            Checkpoint type ('bc' or 'ppo')
        """
        # PPO checkpoints have these top-level keys
        ppo_indicators = {"global_step", "optimizer", "scheduler", "value_head"}

        # BC checkpoints typically have nested structure
        bc_indicators = {"model_state_dict", "state_dict"}

        if any(key in checkpoint for key in ppo_indicators):
            return "ppo"
        elif any(key in checkpoint for key in bc_indicators):
            return "bc"
        elif all(
            key in checkpoint for key in ["cnn_encoder", "state_fusion", "policy_head"]
        ):
            # Direct state dict format (could be either, assume BC if no value_head)
            return "ppo" if "value_head" in checkpoint else "bc"
        else:
            raise ValueError(
                f"Unable to detect checkpoint format. Available keys: {list(checkpoint.keys())}"
            )

    def _extract_state_dict(
        self, checkpoint: Dict[str, Any], checkpoint_type: str
    ) -> Dict[str, Any]:
        """Extract model state dictionary from checkpoint.

        Args:
            checkpoint: Loaded checkpoint
            checkpoint_type: Type of checkpoint ('bc' or 'ppo')

        Returns:
            Model state dictionary
        """
        if checkpoint_type == "ppo":
            # PPO checkpoints have state dicts at top level
            return checkpoint
        else:
            # BC checkpoints have nested structure
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            else:
                # Assume direct state dict
                return checkpoint

    def _create_value_head(
        self, input_dim: int = 128, hidden_dim: int = 64
    ) -> ValueHead:
        """Create value head for PPO (or dummy for BC).

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension

        Returns:
            Value head module
        """
        config = ValueConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            weight_init="orthogonal",
            weight_gain=0.01,  # Match PPO trainer initialization
        )
        return ValueHead(config)

    def load_model(self) -> Dict[str, Any]:
        """Load model from checkpoint with automatic format detection.

        Returns:
            Dictionary with checkpoint metadata
        """
        print(f"Loading model from {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        self.checkpoint_data = checkpoint

        # Detect checkpoint type
        self.checkpoint_type = self._detect_checkpoint_type(checkpoint)
        print(f"Detected checkpoint type: {self.checkpoint_type}")

        # Extract state dictionary
        state_dict = self._extract_state_dict(checkpoint, self.checkpoint_type)

        # Validate required components
        required_components = ["cnn_encoder", "state_fusion", "policy_head"]
        missing_components = [
            comp for comp in required_components if comp not in state_dict
        ]

        # For PPO checkpoints, value_head is required
        if self.checkpoint_type == "ppo" and "value_head" not in state_dict:
            missing_components.append("value_head")

        if missing_components:
            raise ValueError(f"Missing required components: {missing_components}")

        # Initialize processors
        perception_config = PerceptionConfig(
            strict_bounds_checking=False, validate_input=False
        )
        self.perception_processor = GameStateProcessor(perception_config)
        self.state_fusion_processor = StateFusionProcessor()
        self.policy_processor = PolicyProcessor()

        # Load CNN encoder
        cnn_encoder = self.perception_processor.get_cnn_encoder()
        cnn_encoder.load_state_dict(state_dict["cnn_encoder"])
        cnn_encoder.to(self.device)

        # Load state fusion
        self.state_fusion_processor.fusion_mlp.load_state_dict(
            state_dict["state_fusion"]
        )
        self.state_fusion_processor.to(self.device)

        # Load policy head
        self.policy_processor.policy_head.load_state_dict(state_dict["policy_head"])
        self.policy_processor.to(self.device)

        # Load or create value head
        if self.checkpoint_type == "ppo":
            # Load existing value head from PPO checkpoint
            self.value_head = self._create_value_head()

            # Handle prefixed keys in value_head state dict
            value_head_state = state_dict["value_head"]
            if any(key.startswith("value_head.") for key in value_head_state.keys()):
                # Remove "value_head." prefix from keys
                cleaned_state = {
                    key.replace("value_head.", ""): value
                    for key, value in value_head_state.items()
                }
                self.value_head.load_state_dict(cleaned_state)
            else:
                # Keys are already clean
                self.value_head.load_state_dict(value_head_state)
        else:
            # Create dummy value head for BC checkpoints
            self.value_head = self._create_value_head()

        self.value_head.to(self.device)

        # Set to evaluation mode
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.eval()
        self.policy_processor.eval()
        self.value_head.eval()

        self.parameter_count = self._count_parameters()
        print("Model loaded successfully")

        # Return metadata
        metadata = {
            "checkpoint_type": self.checkpoint_type,
            "global_step": checkpoint.get("global_step", 0),
            "parameter_count": self.parameter_count,
        }

        if self.checkpoint_type == "ppo":
            metadata.update(
                {
                    "best_eval_score": checkpoint.get("best_eval_score", 0.0),
                    "config": checkpoint.get("config"),
                }
            )

        return metadata

    def get_bc_components(
        self,
    ) -> Tuple[GameStateProcessor, StateFusionProcessor, PolicyProcessor]:
        """Get model components for BC training (no value head).

        Returns:
            Tuple of (perception_processor, state_fusion_processor, policy_processor)
        """
        if self.perception_processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return (
            self.perception_processor,
            self.state_fusion_processor,
            self.policy_processor,
        )

    def get_ppo_components(
        self,
    ) -> Tuple[GameStateProcessor, StateFusionProcessor, PolicyProcessor, ValueHead]:
        """Get model components for PPO training (includes value head).

        Returns:
            Tuple of (perception_processor, state_fusion_processor, policy_processor, value_head)

        Raises:
            RuntimeError: If model not loaded or value head is None
        """
        if self.perception_processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.value_head is None:
            raise RuntimeError(
                "Value head not available. Load a PPO checkpoint or create a fresh value head."
            )

        return (
            self.perception_processor,
            self.state_fusion_processor,
            self.policy_processor,
            self.value_head,
        )

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the loaded checkpoint.

        Returns:
            Dictionary with checkpoint information
        """
        if self.checkpoint_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        info = {
            "path": self.model_path,
            "type": self.checkpoint_type,
            "parameter_count": self.parameter_count,
            "device": str(self.device),
        }

        if self.checkpoint_type == "ppo":
            info.update(
                {
                    "global_step": self.checkpoint_data.get("global_step", 0),
                    "best_eval_score": self.checkpoint_data.get("best_eval_score", 0.0),
                    "has_optimizer": "optimizer" in self.checkpoint_data,
                    "has_scheduler": "scheduler" in self.checkpoint_data,
                    "config": self.checkpoint_data.get("config"),
                }
            )

        return info

    def get_device(self) -> torch.device:
        """Get the device used for model loading."""
        return self.device
