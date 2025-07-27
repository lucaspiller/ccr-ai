"""
Grid and feature encoders for converting game state to tensor representations.
"""

from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from src.game.board import Direction
from src.game.sprites import SpriteType


class GridEncoder:
    """Encodes game state into 28-channel spatial tensor representation."""

    def __init__(self, width: int = 14, height: int = 10):
        self.width = width
        self.height = height

    def encode(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        Encode complete game state into 28-channel grid tensor.

        Args:
            game_state: Output from GameEngine.to_dict()

        Returns:
            torch.Tensor of shape [28, height, width]
        """
        board_data = game_state["board"]
        sprite_data = game_state["sprite_manager"]["sprites"]

        # Initialize output tensor
        grid_tensor = torch.zeros(28, self.height, self.width, dtype=torch.float32)

        # Encode each component into specific channels
        grid_tensor[0:2] = self.encode_walls(board_data["walls"])
        grid_tensor[2:10] = self.encode_rockets(sprite_data)
        grid_tensor[10:11] = self.encode_spawners(sprite_data)
        grid_tensor[11:17] = self.encode_arrows(board_data["arrows"])
        grid_tensor[17:22] = self.encode_mouse_flow(sprite_data)
        grid_tensor[22:26] = self.encode_cats(sprite_data)
        grid_tensor[26:28] = self.encode_special_mice(sprite_data)

        return grid_tensor

    def encode_walls(
        self, walls: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Encode walls into channels 0-1.

        Channel 0: Vertical walls (between tile and right neighbor)
        Channel 1: Horizontal walls (between tile and tile below)

        Args:
            walls: List of wall edges [((x1,y1), (x2,y2))]

        Returns:
            torch.Tensor of shape [2, height, width]
        """
        wall_tensor = torch.zeros(2, self.height, self.width, dtype=torch.float32)

        for (x1, y1), (x2, y2) in walls:
            # Ensure coordinates are within bounds
            if not (
                0 <= x1 < self.width
                and 0 <= y1 < self.height
                and 0 <= x2 < self.width
                and 0 <= y2 < self.height
            ):
                continue

            if x1 == x2:  # Vertical wall
                # Wall between tiles at same x-coordinate
                left_x = min(x1, x2)
                wall_y = min(y1, y2) if y1 != y2 else y1
                if left_x < self.width and wall_y < self.height:
                    wall_tensor[0, wall_y, left_x] = 1.0

            elif y1 == y2:  # Horizontal wall
                # Wall between tiles at same y-coordinate
                wall_x = min(x1, x2) if x1 != x2 else x1
                upper_y = min(y1, y2)
                if wall_x < self.width and upper_y < self.height:
                    wall_tensor[1, upper_y, wall_x] = 1.0

        return wall_tensor

    def encode_rockets(self, sprites: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Encode rocket positions into channels 2-9 (8 players max).

        Args:
            sprites: Sprite data from sprite_manager

        Returns:
            torch.Tensor of shape [8, height, width]
        """
        rocket_tensor = torch.zeros(8, self.height, self.width, dtype=torch.float32)

        # For now, assume single player (player 0) - extend for multiplayer later
        for sprite_data in sprites.values():
            if sprite_data["type"] == SpriteType.ROCKET.value:
                x = int(sprite_data["x"])
                y = int(sprite_data["y"])
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Place in player 0 channel for now
                    rocket_tensor[0, y, x] = 1.0

        return rocket_tensor

    def encode_spawners(self, sprites: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Encode spawner positions into channel 10.

        Args:
            sprites: Sprite data from sprite_manager

        Returns:
            torch.Tensor of shape [1, height, width]
        """
        spawner_tensor = torch.zeros(1, self.height, self.width, dtype=torch.float32)

        for sprite_data in sprites.values():
            if sprite_data["type"] == SpriteType.SPAWNER.value:
                x = int(sprite_data["x"])
                y = int(sprite_data["y"])
                if 0 <= x < self.width and 0 <= y < self.height:
                    spawner_tensor[0, y, x] = 1.0

        return spawner_tensor

    def encode_arrows(self, arrows: Dict[str, str]) -> torch.Tensor:
        """
        Encode arrows into channels 11-16.

        Channels 11-14: Arrow direction geometry (↑↓←→)
        Channel 15: Arrow owner ID (0=none, 1-8=player+1)
        Channel 16: Arrow health (0=none, 1=healthy, 2=shrunk)

        Args:
            arrows: Arrow data {position_key: direction_name}

        Returns:
            torch.Tensor of shape [6, height, width]
        """
        arrow_tensor = torch.zeros(6, self.height, self.width, dtype=torch.float32)

        # Direction mapping to channels 11-14
        direction_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}

        for pos_key, direction_name in arrows.items():
            x, y = map(int, pos_key.split(","))
            if 0 <= x < self.width and 0 <= y < self.height:
                # Set direction channel (11-14)
                if direction_name in direction_map:
                    channel_idx = direction_map[direction_name]
                    arrow_tensor[channel_idx, y, x] = 1.0

                # Set owner ID (channel 15) - assume player 1 for now
                arrow_tensor[4, y, x] = 1.0  # player 0 + 1

                # Set health (channel 16) - assume healthy for now
                arrow_tensor[5, y, x] = 1.0  # healthy

        return arrow_tensor

    def encode_mouse_flow(self, sprites: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Encode mouse movement flow into channels 17-21.

        Channels 17-20: Directional flow (↑↓←→)
        Channel 21: Flow confidence (total magnitude)

        Args:
            sprites: Sprite data from sprite_manager

        Returns:
            torch.Tensor of shape [5, height, width]
        """
        flow_tensor = torch.zeros(5, self.height, self.width, dtype=torch.float32)

        # Direction mapping
        direction_map = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3,
        }

        # Collect mice by tile position
        mice_by_tile = {}
        for sprite_data in sprites.values():
            if sprite_data["type"] in [
                SpriteType.MOUSE.value,
                SpriteType.GOLD_MOUSE.value,
                SpriteType.BONUS_MOUSE.value,
            ]:
                x = int(sprite_data["x"])
                y = int(sprite_data["y"])
                if 0 <= x < self.width and 0 <= y < self.height:
                    if (x, y) not in mice_by_tile:
                        mice_by_tile[(x, y)] = []
                    mice_by_tile[(x, y)].append(sprite_data)

        # Compute flow for each tile
        for (x, y), mice in mice_by_tile.items():
            # Count mice moving in each direction
            direction_counts = {
                Direction.UP: 0,
                Direction.DOWN: 0,
                Direction.LEFT: 0,
                Direction.RIGHT: 0,
            }

            for mouse in mice:
                direction = Direction[mouse["direction"]]
                direction_counts[direction] += 1

            # Find majority direction
            max_count = max(direction_counts.values())
            total_count = sum(direction_counts.values())

            if max_count > 0:
                # Set the dominant direction
                for direction, count in direction_counts.items():
                    if count == max_count:
                        channel_idx = direction_map[direction]
                        flow_tensor[channel_idx, y, x] = count / max(total_count, 1)
                        break

                # Set confidence (normalized by expected maximum)
                flow_tensor[4, y, x] = min(total_count / 16.0, 1.0)  # cap at 16 mice

        return flow_tensor

    def encode_cats(self, sprites: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Encode cat positions by facing direction into channels 22-25.

        Args:
            sprites: Sprite data from sprite_manager

        Returns:
            torch.Tensor of shape [4, height, width]
        """
        cat_tensor = torch.zeros(4, self.height, self.width, dtype=torch.float32)

        direction_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}

        for sprite_data in sprites.values():
            if (
                sprite_data["type"] == SpriteType.CAT.value
                and sprite_data["state"] == "active"
            ):
                x = int(sprite_data["x"])
                y = int(sprite_data["y"])
                direction = sprite_data["direction"]

                if (
                    0 <= x < self.width
                    and 0 <= y < self.height
                    and direction in direction_map
                ):
                    channel_idx = direction_map[direction]
                    cat_tensor[channel_idx, y, x] = 1.0

        return cat_tensor

    def encode_special_mice(self, sprites: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Encode special mice into channels 26-27.

        Channel 26: Gold mouse mask
        Channel 27: Bonus mouse mask

        Args:
            sprites: Sprite data from sprite_manager

        Returns:
            torch.Tensor of shape [2, height, width]
        """
        special_tensor = torch.zeros(2, self.height, self.width, dtype=torch.float32)

        for sprite_data in sprites.values():
            x = int(sprite_data["x"])
            y = int(sprite_data["y"])

            if 0 <= x < self.width and 0 <= y < self.height:
                if sprite_data["type"] == SpriteType.GOLD_MOUSE.value:
                    special_tensor[0, y, x] = 1.0
                elif sprite_data["type"] == SpriteType.BONUS_MOUSE.value:
                    special_tensor[1, y, x] = 1.0

        return special_tensor


class GlobalFeatureExtractor:
    """Extracts global game features into 16-dimensional vector."""

    def extract(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        Extract global features from game state.

        Args:
            game_state: Output from GameEngine.to_dict()

        Returns:
            torch.Tensor of shape [16]
        """
        features = torch.zeros(16, dtype=torch.float32)

        # Feature 0: Remaining time ticks (normalized)
        max_steps = game_state.get("max_steps", 1000)
        current_step = game_state.get("current_step", 0)
        remaining_ticks = max_steps - current_step
        features[0] = remaining_ticks / 10000.0

        # Feature 1: Arrow budget remaining
        max_arrows = game_state["board"].get("max_arrows", 3)
        placed_arrows = len(game_state["board"]["arrows"])
        features[1] = (max_arrows - placed_arrows) / max(max_arrows, 1)

        # Feature 2: Live cat count
        sprites = game_state["sprite_manager"]["sprites"]
        live_cats = sum(
            1
            for s in sprites.values()
            if s["type"] == SpriteType.CAT.value and s["state"] == "active"
        )
        features[2] = live_cats / 16.0

        # Features 3-7: Bonus state one-hot
        bonus_mode = game_state["bonus_state"]["mode"]
        bonus_modes = ["none", "mouse_mania", "cat_mania", "speed_up", "slow_down"]
        if bonus_mode in bonus_modes:
            features[3 + bonus_modes.index(bonus_mode)] = 1.0

        # Features 8-15: Player scores (mice collected by rockets)
        # For now, just track player 0's score
        player_scores = [0] * 8
        for sprite_data in sprites.values():
            if sprite_data["type"] == SpriteType.ROCKET.value:
                player_scores[0] = sprite_data.get("mice_collected", 0)
                break

        for i, score in enumerate(player_scores):
            features[8 + i] = score / 100.0

        return features
