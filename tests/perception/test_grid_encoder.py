"""
Tests for GridEncoder and GlobalFeatureExtractor.
"""

import numpy as np
import pytest
import torch

from src.game.sprites import SpriteType
from src.perception.encoders import GlobalFeatureExtractor, GridEncoder


class TestGridEncoder:
    """Test cases for GridEncoder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = GridEncoder(width=14, height=10)

        # Sample game state structure
        self.sample_game_state = {
            "board": {
                "width": 14,
                "height": 10,
                "grid": [[0] * 14 for _ in range(10)],
                "arrows": {},
                "walls": [],
                "max_arrows": 3,
            },
            "sprite_manager": {"sprites": {}},
            "current_step": 100,
            "max_steps": 1000,
            "bonus_state": {"mode": "none", "remaining_ticks": 0, "duration_ticks": 0},
        }

    def test_encode_output_shape(self):
        """Test that encode returns correct tensor shape."""
        result = self.encoder.encode(self.sample_game_state)
        assert result.shape == (28, 10, 14)
        assert result.dtype == torch.float32

    def test_encode_walls_vertical(self):
        """Test vertical wall encoding."""
        walls = [((5, 3), (5, 4))]  # Vertical wall
        result = self.encoder.encode_walls(walls)

        assert result.shape == (2, 10, 14)
        assert result[0, 3, 5] == 1.0  # Vertical wall channel
        assert result[1, 3, 5] == 0.0  # Horizontal wall channel

    def test_encode_walls_horizontal(self):
        """Test horizontal wall encoding."""
        walls = [((3, 5), (4, 5))]  # Horizontal wall
        result = self.encoder.encode_walls(walls)

        assert result.shape == (2, 10, 14)
        assert result[0, 5, 3] == 0.0  # Vertical wall channel
        assert result[1, 5, 3] == 1.0  # Horizontal wall channel

    def test_encode_walls_out_of_bounds(self):
        """Test wall encoding handles out-of-bounds coordinates."""
        walls = [((20, 20), (21, 20))]  # Out of bounds
        result = self.encoder.encode_walls(walls)

        assert result.shape == (2, 10, 14)
        assert torch.sum(result) == 0.0  # No walls should be set

    def test_encode_rockets(self):
        """Test rocket encoding."""
        sprites = {
            "rocket1": {
                "type": SpriteType.ROCKET.value,
                "x": 5.0,
                "y": 3.0,
                "state": "active",
            }
        }
        result = self.encoder.encode_rockets(sprites)

        assert result.shape == (8, 10, 14)
        assert result[0, 3, 5] == 1.0  # Player 0 rocket
        assert torch.sum(result[1:]) == 0.0  # No other player rockets

    def test_encode_spawners(self):
        """Test spawner encoding."""
        sprites = {
            "spawner1": {
                "type": SpriteType.SPAWNER.value,
                "x": 2.0,
                "y": 7.0,
                "state": "active",
            }
        }
        result = self.encoder.encode_spawners(sprites)

        assert result.shape == (1, 10, 14)
        assert result[0, 7, 2] == 1.0

    def test_encode_arrows(self):
        """Test arrow encoding."""
        arrows = {"5,3": "UP", "8,6": "RIGHT"}
        result = self.encoder.encode_arrows(arrows)

        assert result.shape == (6, 10, 14)

        # Check UP arrow
        assert result[0, 3, 5] == 1.0  # UP direction channel
        assert result[4, 3, 5] == 1.0  # Owner ID channel
        assert result[5, 3, 5] == 1.0  # Health channel

        # Check RIGHT arrow
        assert result[3, 6, 8] == 1.0  # RIGHT direction channel
        assert result[4, 6, 8] == 1.0  # Owner ID channel
        assert result[5, 6, 8] == 1.0  # Health channel

    def test_encode_mouse_flow_single_direction(self):
        """Test mouse flow encoding with mice moving in one direction."""
        sprites = {
            "mouse1": {
                "type": SpriteType.MOUSE.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            },
            "mouse2": {
                "type": SpriteType.MOUSE.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            },
        }
        result = self.encoder.encode_mouse_flow(sprites)

        assert result.shape == (5, 10, 14)
        assert result[0, 3, 5] == 1.0  # UP flow channel (2/2 = 1.0)
        assert result[4, 3, 5] > 0.0  # Confidence channel

    def test_encode_mouse_flow_mixed_directions(self):
        """Test mouse flow with mice moving in different directions."""
        sprites = {
            "mouse1": {
                "type": SpriteType.MOUSE.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            },
            "mouse2": {
                "type": SpriteType.MOUSE.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "DOWN",
                "state": "active",
            },
            "mouse3": {
                "type": SpriteType.MOUSE.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            },
        }
        result = self.encoder.encode_mouse_flow(sprites)

        assert result.shape == (5, 10, 14)
        # UP should be majority (2 out of 3)
        assert result[0, 3, 5] > result[1, 3, 5]  # UP > DOWN
        assert result[4, 3, 5] > 0.0  # Confidence > 0

    def test_encode_cats(self):
        """Test cat encoding by direction."""
        sprites = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 4.0,
                "y": 2.0,
                "direction": "LEFT",
                "state": "active",
            },
            "cat2": {
                "type": SpriteType.CAT.value,
                "x": 8.0,
                "y": 7.0,
                "direction": "DOWN",
                "state": "active",
            },
        }
        result = self.encoder.encode_cats(sprites)

        assert result.shape == (4, 10, 14)
        assert result[2, 2, 4] == 1.0  # LEFT direction channel
        assert result[1, 7, 8] == 1.0  # DOWN direction channel

    def test_encode_special_mice(self):
        """Test special mice encoding."""
        sprites = {
            "gold_mouse": {
                "type": SpriteType.GOLD_MOUSE.value,
                "x": 3.0,
                "y": 4.0,
                "state": "active",
            },
            "bonus_mouse": {
                "type": SpriteType.BONUS_MOUSE.value,
                "x": 7.0,
                "y": 1.0,
                "state": "active",
            },
        }
        result = self.encoder.encode_special_mice(sprites)

        assert result.shape == (2, 10, 14)
        assert result[0, 4, 3] == 1.0  # Gold mouse channel
        assert result[1, 1, 7] == 1.0  # Bonus mouse channel

    def test_encode_empty_state(self):
        """Test encoding of empty game state."""
        result = self.encoder.encode(self.sample_game_state)

        assert result.shape == (28, 10, 14)
        assert torch.sum(result) == 0.0  # All channels should be empty


class TestGlobalFeatureExtractor:
    """Test cases for GlobalFeatureExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = GlobalFeatureExtractor()

        self.sample_game_state = {
            "current_step": 100,
            "max_steps": 1000,
            "board": {"arrows": {"5,3": "UP"}, "max_arrows": 3},
            "sprite_manager": {
                "sprites": {
                    "rocket1": {"type": SpriteType.ROCKET.value, "mice_collected": 5},
                    "cat1": {"type": SpriteType.CAT.value, "state": "active"},
                    "cat2": {"type": SpriteType.CAT.value, "state": "active"},
                }
            },
            "bonus_state": {"mode": "speed_up"},
        }

    def test_extract_output_shape(self):
        """Test that extract returns correct tensor shape."""
        result = self.extractor.extract(self.sample_game_state)
        assert result.shape == (16,)
        assert result.dtype == torch.float32

    def test_remaining_ticks_feature(self):
        """Test remaining ticks normalization."""
        result = self.extractor.extract(self.sample_game_state)
        expected = (1000 - 100) / 10000.0
        assert abs(result[0] - expected) < 1e-6

    def test_arrow_budget_feature(self):
        """Test arrow budget calculation."""
        result = self.extractor.extract(self.sample_game_state)
        expected = (3 - 1) / 3  # 2 arrows remaining out of 3
        assert abs(result[1] - expected) < 1e-6

    def test_cat_count_feature(self):
        """Test live cat count normalization."""
        result = self.extractor.extract(self.sample_game_state)
        expected = 2 / 16.0  # 2 cats out of max 16
        assert abs(result[2] - expected) < 1e-6

    def test_bonus_state_one_hot(self):
        """Test bonus state one-hot encoding."""
        result = self.extractor.extract(self.sample_game_state)

        # speed_up should be at index 6 (3 + 3)
        assert result[6] == 1.0
        # Other bonus states should be 0
        assert result[3] == 0.0  # none
        assert result[4] == 0.0  # mouse_mania
        assert result[5] == 0.0  # cat_mania
        assert result[7] == 0.0  # slow_down

    def test_player_scores(self):
        """Test player score normalization."""
        result = self.extractor.extract(self.sample_game_state)
        expected = 5 / 100.0  # 5 mice collected, normalized
        assert abs(result[8] - expected) < 1e-6

        # Other player scores should be 0
        for i in range(9, 16):
            assert result[i] == 0.0

    def test_no_bonus_state(self):
        """Test handling of no active bonus."""
        game_state = self.sample_game_state.copy()
        game_state["bonus_state"]["mode"] = "none"

        result = self.extractor.extract(game_state)
        assert result[3] == 1.0  # none bonus state
        assert sum(result[4:8]) == 0.0  # no other bonus states


if __name__ == "__main__":
    pytest.main([__file__])
