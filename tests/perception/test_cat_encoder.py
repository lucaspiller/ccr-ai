"""
Tests for CatSetEncoder and CatSetProcessor.
"""

import pytest
import torch
import torch.nn as nn

from src.game.sprites import SpriteType
from src.perception.cat_encoder import CatSetEncoder, CatSetProcessor


class TestCatSetEncoder:
    """Test cases for CatSetEncoder neural module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = CatSetEncoder(board_width=14, board_height=10)

    def test_encoder_initialization(self):
        """Test encoder initializes correctly."""
        assert self.encoder.board_width == 14
        assert self.encoder.board_height == 10
        assert self.encoder.max_cats == 16
        assert isinstance(self.encoder.cat_mlp, nn.Sequential)
        assert isinstance(self.encoder.layer_norm, nn.LayerNorm)

    def test_forward_empty_cats(self):
        """Test forward pass with no cats."""
        empty_features = torch.zeros(0, 8)
        result = self.encoder(empty_features)

        assert result.shape == (32,)
        assert torch.all(result == 0.0)

    def test_forward_single_cat(self):
        """Test forward pass with single cat."""
        # Create dummy cat features
        cat_features = torch.randn(1, 8)
        result = self.encoder(cat_features)

        assert result.shape == (32,)
        assert not torch.all(result == 0.0)  # Should have non-zero values

    def test_forward_multiple_cats(self):
        """Test forward pass with multiple cats."""
        cat_features = torch.randn(3, 8)
        result = self.encoder(cat_features)

        assert result.shape == (32,)
        assert not torch.all(result == 0.0)

    def test_permutation_invariance(self):
        """Test that encoder is permutation invariant."""
        cat_features = torch.randn(3, 8)

        # Compute embedding for original order
        result1 = self.encoder(cat_features)

        # Shuffle the cats and compute again
        shuffled_indices = torch.randperm(3)
        shuffled_features = cat_features[shuffled_indices]
        result2 = self.encoder(shuffled_features)

        # Results should be identical
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_max_pooling_behavior(self):
        """Test that max pooling works correctly."""
        # Create features where one cat has maximum values in each dimension
        cat_features = torch.zeros(3, 8)
        cat_features[0, :4] = 1.0  # First cat high in first half
        cat_features[1, 4:] = 1.0  # Second cat high in second half
        # Third cat all zeros

        result = self.encoder(cat_features)
        assert result.shape == (32,)
        # Result should reflect the maximum values from both cats

    def test_extract_cat_features_no_cats(self):
        """Test feature extraction with no cats."""
        sprites = {}
        board_arrows = {}

        features = self.encoder.extract_cat_features(sprites, board_arrows, 14, 10)
        assert features.shape == (0, 8)

    def test_extract_cat_features_single_cat(self):
        """Test feature extraction with single cat."""
        sprites = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            }
        }
        board_arrows = {}

        features = self.encoder.extract_cat_features(sprites, board_arrows, 14, 10)
        assert features.shape == (1, 8)

        # Check position normalization
        assert abs(features[0, 0] - 5.0 / 13) < 1e-6  # x normalized
        assert abs(features[0, 1] - 3.0 / 9) < 1e-6  # y normalized

        # Check direction one-hot (UP should be at index 2)
        assert features[0, 2] == 1.0
        assert features[0, 3] == 0.0  # DOWN
        assert features[0, 4] == 0.0  # LEFT
        assert features[0, 5] == 0.0  # RIGHT

    def test_extract_cat_features_multiple_cats(self):
        """Test feature extraction with multiple cats."""
        sprites = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 2.0,
                "y": 1.0,
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
            "mouse1": {  # Should be ignored
                "type": SpriteType.MOUSE.value,
                "x": 4.0,
                "y": 4.0,
                "direction": "RIGHT",
                "state": "active",
            },
        }
        board_arrows = {}

        features = self.encoder.extract_cat_features(sprites, board_arrows, 14, 10)
        assert features.shape == (2, 8)

        # Check first cat (LEFT direction)
        assert features[0, 4] == 1.0  # LEFT

        # Check second cat (DOWN direction)
        assert features[1, 3] == 1.0  # DOWN

    def test_extract_cat_features_inactive_cat(self):
        """Test that inactive cats are ignored."""
        sprites = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "captured",  # Inactive
            }
        }
        board_arrows = {}

        features = self.encoder.extract_cat_features(sprites, board_arrows, 14, 10)
        assert features.shape == (0, 8)

    def test_single_cat_feature_extraction(self):
        """Test individual cat feature extraction."""
        cat_data = {"x": 7.0, "y": 4.0, "direction": "RIGHT", "state": "active"}
        board_arrows = {}
        rockets = [(2, 2), (10, 8)]  # Two rockets

        features = self.encoder._extract_single_cat_features(
            cat_data, board_arrows, rockets, 14, 10
        )

        assert features.shape == (8,)

        # Check position normalization
        assert abs(features[0] - 7.0 / 13) < 1e-6
        assert abs(features[1] - 4.0 / 9) < 1e-6

        # Check direction (RIGHT at index 5)
        assert features[5] == 1.0

        # Check rocket distance (should be min of distances to both rockets)
        dist1 = abs(7 - 2) + abs(4 - 2)  # Distance to first rocket: 5 + 2 = 7
        dist2 = abs(7 - 10) + abs(4 - 8)  # Distance to second rocket: 3 + 4 = 7
        min_dist = min(dist1, dist2)  # 7
        expected_normalized = min_dist / 20.0
        assert abs(features[7] - expected_normalized) < 1e-6

    def test_min_rocket_distance_no_rockets(self):
        """Test rocket distance calculation with no rockets."""
        distance = self.encoder._compute_min_rocket_distance(5, 5, [])
        assert distance == 1.0  # Should return max distance

    def test_min_rocket_distance_single_rocket(self):
        """Test rocket distance calculation with single rocket."""
        rockets = [(3, 7)]
        distance = self.encoder._compute_min_rocket_distance(5, 5, rockets)

        expected = (abs(5 - 3) + abs(5 - 7)) / 20.0  # L1 distance = 4, normalized = 0.2
        assert abs(distance - expected) < 1e-6

    def test_min_rocket_distance_multiple_rockets(self):
        """Test rocket distance calculation with multiple rockets."""
        rockets = [(1, 1), (8, 8), (5, 6)]  # Third rocket is closest
        distance = self.encoder._compute_min_rocket_distance(5, 5, rockets)

        # Closest is (5, 6) with distance 1
        expected = 1.0 / 20.0
        assert abs(distance - expected) < 1e-6

    def test_check_shrunk_arrow_ahead(self):
        """Test shrunk arrow ahead detection."""
        # Currently returns 0.0 since arrow health not implemented
        result = self.encoder._check_shrunk_arrow_ahead(5, 5, "UP", {"5,4": "UP"})
        assert result == 0.0

        # Test with no arrow ahead
        result = self.encoder._check_shrunk_arrow_ahead(5, 5, "UP", {})
        assert result == 0.0


class TestCatSetProcessor:
    """Test cases for CatSetProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = CatSetProcessor(board_width=14, board_height=10)

        # Sample game state
        self.sample_game_state = {
            "sprite_manager": {"sprites": {}},
            "board": {"arrows": {}},
        }

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        assert self.processor.board_width == 14
        assert self.processor.board_height == 10
        assert isinstance(self.processor.encoder, CatSetEncoder)

    def test_process_no_cats(self):
        """Test processing with no cats."""
        result = self.processor.process(self.sample_game_state)
        assert result.shape == (32,)
        assert torch.all(result == 0.0)

    def test_process_with_cats(self):
        """Test processing with cats."""
        game_state = self.sample_game_state.copy()
        game_state["sprite_manager"]["sprites"] = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
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

        result = self.processor.process(game_state)
        assert result.shape == (32,)
        assert not torch.all(result == 0.0)

    def test_get_cat_count_no_cats(self):
        """Test cat counting with no cats."""
        count = self.processor.get_cat_count(self.sample_game_state)
        assert count == 0

    def test_get_cat_count_with_cats(self):
        """Test cat counting with cats."""
        game_state = self.sample_game_state.copy()
        game_state["sprite_manager"]["sprites"] = {
            "cat1": {"type": SpriteType.CAT.value, "state": "active"},
            "cat2": {"type": SpriteType.CAT.value, "state": "active"},
            "cat3": {
                "type": SpriteType.CAT.value,
                "state": "captured",  # Should not be counted
            },
            "mouse1": {
                "type": SpriteType.MOUSE.value,
                "state": "active",  # Should not be counted
            },
        }

        count = self.processor.get_cat_count(game_state)
        assert count == 2

    def test_deterministic_processing(self):
        """Test that processing is deterministic."""
        game_state = self.sample_game_state.copy()
        game_state["sprite_manager"]["sprites"] = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            }
        }

        # Process twice
        result1 = self.processor.process(game_state)
        result2 = self.processor.process(game_state)

        assert torch.equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
