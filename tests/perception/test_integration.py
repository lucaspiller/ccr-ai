"""
Integration tests for GridEncoder with real game state.
"""

import pytest
import torch

from src.game.board import Board, Direction
from src.game.engine import GameEngine
from src.game.sprites import SpriteManager
from src.perception.encoders import GlobalFeatureExtractor, GridEncoder


class TestGridEncoderIntegration:
    """Integration tests for GridEncoder with actual game engine."""

    def setup_method(self):
        """Set up test game state."""
        # Create a simple game setup
        self.board = Board(width=14, height=10)
        self.sprite_manager = SpriteManager()

        # Add some game elements
        self.mouse = self.sprite_manager.create_mouse(5, 3)
        self.cat = self.sprite_manager.create_cat(8, 7)
        self.rocket = self.sprite_manager.create_rocket(2, 2)
        self.spawner = self.sprite_manager.create_spawner(10, 8)

        # Place an arrow
        self.board.place_arrow(6, 4, Direction.UP)

        # Create game engine and get state
        self.engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        self.game_state = self.engine.to_dict()

        # Create encoders
        self.grid_encoder = GridEncoder(width=14, height=10)
        self.global_extractor = GlobalFeatureExtractor()

    def test_real_game_state_structure(self):
        """Test that real game state has expected structure."""
        assert self.game_state["board"]["width"] == 14
        assert self.game_state["board"]["height"] == 10
        assert len(self.game_state["sprite_manager"]["sprites"]) == 4
        assert len(self.game_state["board"]["arrows"]) == 1
        assert "bonus_state" in self.game_state
        assert "current_step" in self.game_state

    def test_grid_encoder_with_real_state(self):
        """Test GridEncoder processes real game state correctly."""
        grid_tensor = self.grid_encoder.encode(self.game_state)

        # Check tensor properties
        assert grid_tensor.shape == (28, 10, 14)
        assert grid_tensor.dtype == torch.float32

        # Should have non-zero values in relevant channels
        non_zero_channels = (grid_tensor.sum(dim=(1, 2)) > 0).sum().item()
        assert non_zero_channels > 0

        # Check specific channels have expected content
        assert grid_tensor[2].sum().item() == 1.0  # 1 rocket (channel 2, player 0)
        assert grid_tensor[10].sum().item() == 1.0  # 1 spawner
        assert grid_tensor[11].sum().item() == 1.0  # 1 UP arrow
        assert grid_tensor[17:21].sum().item() > 0  # Mouse flow present
        assert grid_tensor[22:26].sum().item() == 1.0  # 1 cat

    def test_global_extractor_with_real_state(self):
        """Test GlobalFeatureExtractor processes real game state correctly."""
        global_features = self.global_extractor.extract(self.game_state)

        # Check tensor properties
        assert global_features.shape == (16,)
        assert global_features.dtype == torch.float32

        # Should have some non-zero features
        non_zero_features = (global_features > 0).sum().item()
        assert non_zero_features > 0

        # Check specific features
        assert global_features[0] > 0  # Remaining ticks
        assert global_features[1] > 0  # Arrow budget (2/3 remaining)
        assert global_features[2] > 0  # Cat count (1/16)
        assert global_features[3] == 1.0  # No bonus state active

    def test_encoder_consistency(self):
        """Test that encoding is deterministic."""
        grid_tensor1 = self.grid_encoder.encode(self.game_state)
        grid_tensor2 = self.grid_encoder.encode(self.game_state)

        assert torch.equal(grid_tensor1, grid_tensor2)

        global_features1 = self.global_extractor.extract(self.game_state)
        global_features2 = self.global_extractor.extract(self.game_state)

        assert torch.equal(global_features1, global_features2)

    def test_sprite_positioning(self):
        """Test that sprites are encoded at correct positions."""
        grid_tensor = self.grid_encoder.encode(self.game_state)

        # Check rocket at (2, 2)
        assert grid_tensor[2, 2, 2] == 1.0

        # Check spawner at (10, 8)
        assert grid_tensor[10, 8, 10] == 1.0

        # Check arrow at (6, 4) pointing UP
        assert grid_tensor[11, 4, 6] == 1.0  # UP direction channel

        # Check mouse creates flow at (5, 3)
        mouse_tile_flow = grid_tensor[17:21, 3, 5].sum().item()
        assert mouse_tile_flow > 0

        # Check cat at (8, 7) - should be in appropriate direction channel
        cat_tile_sum = grid_tensor[22:26, 7, 8].sum().item()
        assert cat_tile_sum == 1.0

    def test_step_and_reencode(self):
        """Test encoding after game step."""
        # Get initial encoding
        initial_grid = self.grid_encoder.encode(self.game_state)

        # Step the game for 300 ticks (5 seconds)
        for _ in range(300):
            self.engine.step()

        new_state = self.engine.to_dict()

        # Encode new state
        new_grid = self.grid_encoder.encode(new_state)

        # Grids should be same shape but may have different content
        assert new_grid.shape == initial_grid.shape

        # Static elements (rockets, spawners, arrows) should be unchanged
        assert torch.equal(new_grid[2], initial_grid[2])  # Rockets
        assert torch.equal(new_grid[10], initial_grid[10])  # Spawners
        assert torch.equal(new_grid[11:17], initial_grid[11:17])  # Arrows

    def test_multiple_mice_flow(self):
        """Test mouse flow encoding with multiple mice."""
        # Add more mice at same position with different directions
        mouse2 = self.sprite_manager.create_mouse(5, 3)  # Same position as first mouse
        mouse2.direction = Direction.DOWN

        mouse3 = self.sprite_manager.create_mouse(5, 3)
        mouse3.direction = Direction.UP  # Same as first mouse

        # Get new state and encode
        new_state = self.engine.to_dict()
        grid_tensor = self.grid_encoder.encode(new_state)

        # Should have stronger flow in UP direction (2 mice vs 1)
        up_flow = grid_tensor[17, 3, 5].item()
        down_flow = grid_tensor[18, 3, 5].item()
        confidence = grid_tensor[21, 3, 5].item()

        assert up_flow > down_flow  # UP should be majority
        assert confidence > 0  # Should have confidence value

    def test_performance_benchmark(self):
        """Basic performance test for encoding speed."""
        import time

        # Time multiple encodings
        start_time = time.time()
        for _ in range(100):
            self.grid_encoder.encode(self.game_state)
            self.global_extractor.extract(self.game_state)
        end_time = time.time()

        avg_time_ms = (end_time - start_time) * 1000 / 100

        # Should be under 10ms per encoding (very generous)
        assert avg_time_ms < 10.0, f"Encoding too slow: {avg_time_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
