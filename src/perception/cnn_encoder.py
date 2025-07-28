"""
CNN encoder for processing 28-channel grid tensors into spatial embeddings.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Convolutional neural network encoder for grid tensor processing.

    Converts 28-channel grid tensor [28, H, W] into a flattened spatial embedding
    via 4× Conv-BN-ReLU layers as specified in the AI overview architecture.
    """

    def __init__(
        self, input_channels: int = 28, board_width: int = 14, board_height: int = 10
    ):
        """Initialize CNN encoder.

        Args:
            input_channels: Number of input channels (28 for ChuChu Rocket grid)
            board_width: Maximum board width for size calculations
            board_height: Maximum board height for size calculations
        """
        super().__init__()
        self.input_channels = input_channels
        self.board_width = board_width
        self.board_height = board_height

        # Conv-BN-ReLU layers: 28→32→128→256 channels
        self.conv_layers = nn.Sequential(
            # Block 1: 28→32 channels
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: 32→64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3: 64→128 channels (increased capacity for junction motifs)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 4: 128→256 channels (final feature extraction with high capacity)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Calculate flattened size for maximum board dimensions
        # This is used by state fusion layer to know input size
        self.flattened_size = 256 * board_width * board_height

    def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN encoder.

        Args:
            grid_tensor: Input tensor of shape [batch, 28, height, width]
                        or [28, height, width] for single sample

        Returns:
            torch.Tensor: Flattened grid embedding [batch, max_flattened_size]
                         or [max_flattened_size] for single sample
                         Padded to maximum board size for consistent dimensions
        """
        # Handle single sample (add batch dimension)
        single_sample = False
        if grid_tensor.dim() == 3:
            grid_tensor = grid_tensor.unsqueeze(0)
            single_sample = True

        batch_size, channels, height, width = grid_tensor.shape

        # Pad to maximum board size if needed
        if height < self.board_height or width < self.board_width:
            # Pad with zeros to reach maximum size
            pad_h = self.board_height - height
            pad_w = self.board_width - width
            # Pad: (left, right, top, bottom)
            grid_tensor = torch.nn.functional.pad(
                grid_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        # Apply convolutional layers
        features = self.conv_layers(
            grid_tensor
        )  # [batch, 256, board_height, board_width]

        # Flatten spatial dimensions to consistent size
        flattened = features.view(
            batch_size, -1
        )  # [batch, 256*board_height*board_width]

        # Remove batch dimension if input was single sample
        if single_sample:
            flattened = flattened.squeeze(0)

        return flattened

    def get_output_size(self, height: int, width: int) -> int:
        """Get output size for given board dimensions.

        Note: Always returns maximum size since we pad to max dimensions.

        Args:
            height: Board height (ignored - we always use max)
            width: Board width (ignored - we always use max)

        Returns:
            Size of flattened output tensor (always max board size)
        """
        return self.flattened_size  # Always maximum: 256 * board_width * board_height

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# For compatibility with existing code
GridCNNEncoder = CNNEncoder  # Alias
