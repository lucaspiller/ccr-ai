"""
Cat set encoder for processing variable-length cat lists into fixed embeddings.
"""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.game.board import Direction
from src.game.sprites import SpriteType


class CatSetEncoder(nn.Module):
    """
    Neural set encoder for converting variable-length cat lists to fixed embeddings.

    Converts a list of cats into a fixed 32-dimensional embedding via:
    1. Per-cat feature extraction (8 dims): position, direction, threat features
    2. Shared MLP (8→32→32) with ReLU
    3. Max pooling over all cats (permutation invariant)
    4. LayerNorm for stability
    """

    def __init__(
        self, board_width: int = 14, board_height: int = 10, max_cats: int = 16
    ):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.max_cats = max_cats

        # Shared MLP for per-cat features
        self.cat_mlp = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32))

        # Layer normalization for pooled output
        self.layer_norm = nn.LayerNorm(32)

    def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cat set encoder.

        Args:
            cat_features: [num_cats, 8] tensor of per-cat features
                         Can be empty tensor [0, 8] if no cats

        Returns:
            torch.Tensor: [32] fixed-size cat embedding
        """
        if cat_features.size(0) == 0:
            # No cats - return zero embedding
            return torch.zeros(32, dtype=cat_features.dtype, device=cat_features.device)

        # Apply shared MLP to each cat
        cat_embeddings = self.cat_mlp(cat_features)  # [num_cats, 32]

        # Max pooling over cats (permutation invariant)
        pooled_embedding, _ = torch.max(cat_embeddings, dim=0)  # [32]

        # Layer normalization for stability
        normalized_embedding = self.layer_norm(pooled_embedding)

        return normalized_embedding

    def extract_cat_features(
        self,
        sprites: Dict[str, Dict[str, Any]],
        board_arrows: Dict[str, str],
        board_width: int,
        board_height: int,
    ) -> torch.Tensor:
        """
        Extract per-cat features from sprite data.

        Args:
            sprites: Sprite data from game state
            board_arrows: Arrow positions and directions
            board_width: Board width for normalization
            board_height: Board height for normalization

        Returns:
            torch.Tensor: [num_cats, 8] cat features
        """
        cat_features = []

        # Get all rockets for distance calculations
        rockets = []
        for sprite_data in sprites.values():
            if sprite_data["type"] == SpriteType.ROCKET.value:
                rockets.append((int(sprite_data["x"]), int(sprite_data["y"])))

        # Extract features for each active cat
        for sprite_data in sprites.values():
            if (
                sprite_data["type"] == SpriteType.CAT.value
                and sprite_data["state"] == "active"
            ):

                features = self._extract_single_cat_features(
                    sprite_data, board_arrows, rockets, board_width, board_height
                )
                cat_features.append(features)

        if not cat_features:
            # No cats - return empty tensor with correct shape
            return torch.zeros(0, 8, dtype=torch.float32)

        return torch.stack(cat_features)

    def _extract_single_cat_features(
        self,
        cat_data: Dict[str, Any],
        board_arrows: Dict[str, str],
        rockets: List[Tuple[int, int]],
        board_width: int,
        board_height: int,
    ) -> torch.Tensor:
        """
        Extract 8-dimensional feature vector for a single cat.

        Features:
        0-1: Normalized position (x_norm, y_norm)
        2-5: Direction one-hot (UP, DOWN, LEFT, RIGHT)
        6: Shrunk arrow ahead (0/1)
        7: Distance to nearest enemy rocket (normalized)
        """
        features = torch.zeros(8, dtype=torch.float32)

        # Position normalization
        x = float(cat_data["x"])
        y = float(cat_data["y"])
        features[0] = x / max(board_width - 1, 1)  # [0, 1]
        features[1] = y / max(board_height - 1, 1)  # [0, 1]

        # Direction one-hot encoding
        direction = cat_data["direction"]
        direction_map = {"UP": 2, "DOWN": 3, "LEFT": 4, "RIGHT": 5}
        if direction in direction_map:
            features[direction_map[direction]] = 1.0

        # Shrunk arrow ahead feature
        features[6] = self._check_shrunk_arrow_ahead(
            int(x), int(y), direction, board_arrows
        )

        # Distance to nearest enemy rocket
        features[7] = self._compute_min_rocket_distance(int(x), int(y), rockets)

        return features

    def _check_shrunk_arrow_ahead(
        self, cat_x: int, cat_y: int, cat_direction: str, board_arrows: Dict[str, str]
    ) -> float:
        """
        Check if there's a shrunk arrow in the tile ahead of the cat.

        For now, assumes all arrows are healthy (returns 0.0).
        TODO: Implement arrow health tracking in board state.
        """
        # Get the tile in front of the cat
        direction_offsets = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        if cat_direction not in direction_offsets:
            return 0.0

        dx, dy = direction_offsets[cat_direction]
        ahead_x = cat_x + dx
        ahead_y = cat_y + dy

        # Check if there's an arrow at that position
        arrow_key = f"{ahead_x},{ahead_y}"
        if arrow_key in board_arrows:
            # For now, assume all arrows are healthy (no shrunk arrows)
            # TODO: Check arrow health when implemented in board state
            return 0.0

        return 0.0

    def _compute_min_rocket_distance(
        self, cat_x: int, cat_y: int, rockets: List[Tuple[int, int]]
    ) -> float:
        """
        Compute L1 distance to nearest rocket, normalized by division by 20.

        For single-player mode, all rockets are considered "enemy" rockets.
        """
        if not rockets:
            return 1.0  # Max distance if no rockets

        min_distance = float("inf")
        for rocket_x, rocket_y in rockets:
            l1_distance = abs(cat_x - rocket_x) + abs(cat_y - rocket_y)
            min_distance = min(min_distance, l1_distance)

        # Normalize by dividing by 20 (as specified in AI overview)
        return min(min_distance / 20.0, 1.0)


class CatSetProcessor:
    """
    High-level processor that combines cat feature extraction with neural encoding.
    """

    def __init__(self, board_width: int = 14, board_height: int = 10):
        self.board_width = board_width
        self.board_height = board_height
        self.encoder = CatSetEncoder(board_width, board_height)

    def process(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        Process game state to extract cat set embedding.

        Args:
            game_state: Output from GameEngine.to_dict()

        Returns:
            torch.Tensor: [32] cat set embedding
        """
        sprites = game_state["sprite_manager"]["sprites"]
        board_arrows = game_state["board"]["arrows"]

        # Extract cat features
        cat_features = self.encoder.extract_cat_features(
            sprites, board_arrows, self.board_width, self.board_height
        )

        # Encode to fixed embedding
        cat_embedding = self.encoder(cat_features)

        return cat_embedding

    def get_cat_count(self, game_state: Dict[str, Any]) -> int:
        """Get the number of active cats in the game state."""
        sprites = game_state["sprite_manager"]["sprites"]
        count = 0
        for sprite_data in sprites.values():
            if (
                sprite_data["type"] == SpriteType.CAT.value
                and sprite_data["state"] == "active"
            ):
                count += 1
        return count
