import os
from typing import Any, Dict, Optional, Tuple

import torch

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

        # Extract state dictionary
        state_dict = checkpoint

        # Validate required components
        required_components = ["cnn_encoder", "state_fusion", "policy_head"]
        missing_components = [
            comp for comp in required_components if comp not in state_dict
        ]
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
        if "value_head" in state_dict:
            # Load existing value head from PPO checkpoint
            self.value_head = self._create_value_head()
            value_head_state = state_dict["value_head"]
            self.value_head.load_state_dict(value_head_state)
        else:
            # Create empty value head
            self.value_head = self._create_value_head()

        self.value_head.to(self.device)

        # Set to evaluation mode
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.eval()
        self.policy_processor.eval()
        self.value_head.eval()

        self.parameter_count = self._count_parameters()
        print("Model loaded successfully")

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

        return (
            self.perception_processor,
            self.state_fusion_processor,
            self.policy_processor,
            self.value_head,
        )

    def get_metadata(self, training_run: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific training run from loaded checkpoint.

        Args:
            training_run: Name of training run ('bc_set', 'bc_seq_lite', 'ppo')

        Returns:
            Dictionary with training run metadata, or None if not found
        """
        if self.checkpoint_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Return specific run metadata
        return self.checkpoint_data.get(training_run)

    def get_device(self) -> torch.device:
        """Get the device used for model loading."""
        return self.device
