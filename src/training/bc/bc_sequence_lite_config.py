"""
Configuration specific to BC Sequence Lite training.
"""

from dataclasses import dataclass

from .config import BCConfig


@dataclass
class BCSequenceLiteConfig(BCConfig):
    """Configuration for BC Sequence Lite training."""

    # Phase-specific parameters
    frozen_epochs: int = 3  # Epochs to train with perception/fusion frozen
    unfrozen_epochs: int = 7  # Epochs to train with all layers unfrozen
    unfrozen_lr_multiplier: float = 0.1  # LR multiplier for unfrozen phase (10x lower)

    # Override parent defaults for sequence training
    use_focal_loss: bool = False  # Use plain cross-entropy as specified
    max_epochs: int = 10  # Total epochs (frozen + unfrozen)
    patience: int = 15  # More patience for sequential training

    # Sequence-specific training settings
    sequence_shuffle: bool = True  # Shuffle samples across puzzles

    def __post_init__(self):
        """Validate BC Sequence Lite configuration."""
        super().__post_init__()

        if self.frozen_epochs + self.unfrozen_epochs != self.max_epochs:
            print(
                f"Warning: frozen_epochs ({self.frozen_epochs}) + unfrozen_epochs ({self.unfrozen_epochs}) "
                f"!= max_epochs ({self.max_epochs}). Adjusting max_epochs."
            )
            self.max_epochs = self.frozen_epochs + self.unfrozen_epochs

        if not (0.0 < self.unfrozen_lr_multiplier <= 1.0):
            raise ValueError("unfrozen_lr_multiplier must be between 0 and 1")

        # Ensure we're using cross-entropy loss
        if self.use_focal_loss:
            print(
                "Warning: BC Sequence Lite should use cross-entropy loss. Setting use_focal_loss=False."
            )
            self.use_focal_loss = False
