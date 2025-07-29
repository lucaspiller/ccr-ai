import os

import torch

from src.perception.data_types import PerceptionConfig
from src.perception.processors import GameStateProcessor
from src.policy.processors import PolicyProcessor
from src.state_fusion.processors import StateFusionProcessor


class ModelLoader:
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize evaluator with model path.

        Args:
            model_path: Path to saved PPO model
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.device = self._get_device(device)

        # Initialize components
        self._load_model()

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

    def _load_model(self):
        """Load PPO model from checkpoint."""
        print(f"Loading model from {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # Initialize processors
        perception_config = PerceptionConfig(
            strict_bounds_checking=False, validate_input=False
        )
        self.perception_processor = GameStateProcessor(perception_config)
        self.state_fusion_processor = StateFusionProcessor()
        self.policy_processor = PolicyProcessor()
        self.value_head = torch.nn.Linear(128, 1)  # Dummy value head as not yet implemented

        # Load state dicts
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint model state does not contain 'model_state_dict' key"
            )

        model_state = checkpoint["model_state_dict"]

        if "cnn_encoder" not in model_state:
            raise ValueError(
                "Checkpoint model state does not contain 'cnn_encoder' state dict"
            )

        self.cnn_encoder = self.perception_processor.get_cnn_encoder()
        self.cnn_encoder.load_state_dict(model_state["cnn_encoder"])

        if "state_fusion" not in model_state:
            raise ValueError(
                "Checkpoint model state does not contain 'state_fusion' state dict"
            )

        self.state_fusion_processor.fusion_mlp.load_state_dict(
            model_state["state_fusion"]
        )

        if "policy_head" not in model_state:
            raise ValueError(
                "Checkpoint model state does not contain 'policy_head' state dict"
            )

        self.policy_processor.policy_head.load_state_dict(model_state["policy_head"])

        # TODO when value head is implemented
        # if 'value_head' not in model_state:
        #    raise ValueError("Checkpoint model state does not contain 'value_head' state dict")
        # self.value_head.load_state_dict(model_state['value_head'])

        # Move components to device
        self.cnn_encoder.to(self.device)
        self.state_fusion_processor.to(self.device)
        self.policy_processor.to(self.device)
        self.value_head.to(self.device)

        # Set to evaluation mode
        self.perception_processor.get_cnn_encoder().eval()
        self.state_fusion_processor.eval()
        self.policy_processor.eval()
        self.value_head.eval()

        self.parameter_count = self._count_parameters()

        print("Model loaded successfully")

    def get_components(self):
        """Get loaded model components."""
        return (
            self.perception_processor,
            self.state_fusion_processor,
            self.policy_processor,
            self.value_head,
        )

    def get_device(self) -> torch.device:
        """Get the device used for evaluation."""
        return self.device
