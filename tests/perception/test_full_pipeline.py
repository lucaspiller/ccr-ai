"""
Full pipeline integration tests for the perception layer.
"""

import pytest
import torch
import time
from src.perception import (
    GameStateProcessor,
    PerceptionConfig,
    process_game_state,
    get_combined_embedding,
)
from src.game.board import Board, Direction
from src.game.sprites import SpriteManager
from src.game.engine import GameEngine


class TestFullPerceptionPipeline:
    """Integration tests for complete perception pipeline."""

    def setup_method(self):
        """Set up test game scenarios."""
        self.config = PerceptionConfig(
            board_width=14, board_height=10, validate_input=True, enable_profiling=True
        )
        self.processor = GameStateProcessor(self.config)

    def create_complex_game_state(self):
        """Create a complex game state with multiple sprite types."""
        board = Board(width=14, height=10)
        sprite_manager = SpriteManager()

        # Add various sprites
        mouse1 = sprite_manager.create_mouse(3, 4)
        mouse1.direction = Direction.UP

        mouse2 = sprite_manager.create_mouse(3, 4)  # Same position, different direction
        mouse2.direction = Direction.DOWN

        gold_mouse = sprite_manager.create_gold_mouse(7, 2)
        bonus_mouse = sprite_manager.create_bonus_mouse(9, 6)

        cat1 = sprite_manager.create_cat(5, 5)
        cat1.direction = Direction.LEFT

        cat2 = sprite_manager.create_cat(10, 8)
        cat2.direction = Direction.RIGHT

        rocket = sprite_manager.create_rocket(1, 1)
        spawner = sprite_manager.create_spawner(12, 9)

        # Add walls directly to the walls set
        board.walls.add(((3, 3), (3, 4)))  # Vertical wall
        board.walls.add(((5, 2), (6, 2)))  # Horizontal wall

        # Add arrows
        board.place_arrow(6, 6, Direction.UP)
        board.place_arrow(8, 4, Direction.RIGHT)

        # Create engine and step it a few times
        engine = GameEngine(board, sprite_manager, max_steps=1000)
        for _ in range(5):  # Let sprites move
            engine.step()

        return engine.to_dict()

    def test_complex_state_processing(self):
        """Test processing of complex game state."""
        game_state = self.create_complex_game_state()

        # Process the state
        output = self.processor.process(game_state)

        # Verify output structure
        assert output.grid_tensor.shape == (28, 10, 14)
        assert output.global_features.shape == (16,)
        assert output.cat_embedding.shape == (32,)
        assert output.cat_count == 2  # Two cats
        assert output.encoding_time_ms is not None

        # Verify non-zero channels are populated
        non_zero_channels = (output.grid_tensor.sum(dim=(1, 2)) > 0).sum().item()
        assert (
            non_zero_channels > 5
        )  # Should have walls, rockets, spawners, arrows, mice, cats

        # Verify specific channel content
        rocket_channels = output.get_rocket_channels()
        assert rocket_channels.sum() > 0  # Should have rocket

        mouse_flow_channels = output.get_mouse_flow_channels()
        assert mouse_flow_channels.sum() > 0  # Should have mouse flow

        cat_channels = output.get_cat_channels()
        assert cat_channels.sum() == 2  # Should have 2 cats

    def test_combined_embedding_properties(self):
        """Test properties of combined embedding."""
        game_state = self.create_complex_game_state()
        output = self.processor.process(game_state)

        combined = output.get_combined_embedding()

        # Check shape
        expected_size = 28 * 10 * 14 + 16 + 32
        assert combined.shape == (expected_size,)

        # Should have reasonable number of non-zero values
        non_zero_count = (combined != 0).sum().item()
        assert non_zero_count > 50  # Complex state should have many features

        # Values should be in reasonable range
        assert combined.min() >= -10.0  # No extreme negative values
        assert combined.max() <= 10.0  # No extreme positive values

    def test_state_evolution_tracking(self):
        """Test tracking state changes over time."""
        board = Board(width=14, height=10)
        sprite_manager = SpriteManager()

        # Create moving cat
        cat = sprite_manager.create_cat(5, 5)
        cat.direction = Direction.RIGHT

        engine = GameEngine(board, sprite_manager, max_steps=1000)

        # Process initial state
        initial_state = engine.to_dict()
        initial_output = self.processor.process(initial_state)

        # Step the game multiple times
        for _ in range(20):
            engine.step()

        # Process new state
        new_state = engine.to_dict()
        new_output = self.processor.process(new_state)

        # States should be different
        assert not torch.equal(initial_output.grid_tensor, new_output.grid_tensor)
        assert not torch.equal(initial_output.cat_embedding, new_output.cat_embedding)

        # But structure should remain the same
        assert initial_output.grid_tensor.shape == new_output.grid_tensor.shape
        assert initial_output.cat_count == new_output.cat_count  # Cat count unchanged

        # Global features should change (time progresses)
        assert not torch.equal(
            initial_output.global_features, new_output.global_features
        )

    def test_performance_with_complex_state(self):
        """Test performance with complex game state."""
        game_state = self.create_complex_game_state()

        # Benchmark processing
        results = self.processor.benchmark(game_state, num_iterations=50)

        # Should meet performance targets
        assert results["mean_ms"] < 5.0  # Should be fast even for complex states
        assert results["min_ms"] > 0.0
        assert results["max_ms"] < 20.0  # No extremely slow outliers

        # Performance should be consistent
        assert results["std_ms"] < results["mean_ms"]  # Low variance

    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        # Empty board
        board = Board(width=14, height=10)
        sprite_manager = SpriteManager()
        engine = GameEngine(board, sprite_manager, max_steps=1000)

        empty_state = engine.to_dict()
        empty_output = self.processor.process(empty_state)

        assert empty_output.cat_count == 0
        assert torch.all(empty_output.cat_embedding == 0.0)

        # Maximum cats
        for i in range(15):  # Near maximum
            x = i % 14
            y = i % 10
            sprite_manager.create_cat(x, y)

        max_cat_state = engine.to_dict()
        max_cat_output = self.processor.process(max_cat_state)

        assert max_cat_output.cat_count == 15
        assert not torch.all(max_cat_output.cat_embedding == 0.0)

    def test_convenience_function_integration(self):
        """Test convenience functions with real game states."""
        game_state = self.create_complex_game_state()

        # Test process_game_state function
        output1 = process_game_state(game_state)
        output2 = self.processor.process(game_state)

        # Should produce equivalent results for deterministic parts
        assert torch.allclose(output1.grid_tensor, output2.grid_tensor)
        assert torch.allclose(output1.global_features, output2.global_features)
        # Cat embeddings will differ due to different random initialization
        assert output1.cat_embedding.shape == output2.cat_embedding.shape
        assert output1.cat_count == output2.cat_count

        # Test get_combined_embedding function
        embedding1 = get_combined_embedding(game_state)
        embedding2 = output1.get_combined_embedding()

        # Should have same shape
        assert embedding1.shape == embedding2.shape

    def test_deterministic_processing(self):
        """Test that processing is deterministic with same processor."""
        game_state = self.create_complex_game_state()

        # Process the same state multiple times with same processor
        outputs = [self.processor.process(game_state) for _ in range(5)]

        # All outputs should be identical when using same processor
        reference = outputs[0]
        for output in outputs[1:]:
            assert torch.equal(reference.grid_tensor, output.grid_tensor)
            assert torch.equal(reference.global_features, output.global_features)
            assert torch.equal(reference.cat_embedding, output.cat_embedding)
            assert reference.cat_count == output.cat_count

    def test_channel_specific_content(self):
        """Test that specific channels contain expected content."""
        game_state = self.create_complex_game_state()
        output = self.processor.process(game_state)

        # Check walls (channels 0-1)
        wall_channels = output.get_wall_channels()
        assert wall_channels.sum() > 0  # Should have walls

        # Check rockets (channels 2-9)
        rocket_channels = output.get_rocket_channels()
        assert rocket_channels[0].sum() == 1  # Should have 1 rocket in player 0 channel

        # Check arrows (channels 11-16)
        arrow_channels = output.get_arrow_channels()
        assert arrow_channels[:4].sum() > 0  # Should have arrow directions
        assert arrow_channels[4].sum() > 0  # Should have arrow owners
        assert arrow_channels[5].sum() > 0  # Should have arrow health

        # Check special mice (channels 26-27)
        special_mice = output.get_special_mice_channels()
        assert special_mice[0].sum() == 1  # Should have 1 gold mouse
        assert special_mice[1].sum() == 1  # Should have 1 bonus mouse

    def test_global_features_content(self):
        """Test content of global features."""
        game_state = self.create_complex_game_state()
        output = self.processor.process(game_state)

        global_feat = output.global_features

        # Feature 0: Remaining ticks (should be positive)
        assert global_feat[0] > 0

        # Feature 1: Arrow budget (should be less than 1.0 since we placed arrows)
        assert 0 <= global_feat[1] <= 1.0

        # Feature 2: Cat count (should be 2/16)
        expected_cat_ratio = 2.0 / 16.0
        assert abs(global_feat[2] - expected_cat_ratio) < 1e-6

        # Features 3-7: Bonus state (should be one-hot)
        bonus_features = global_feat[3:8]
        assert bonus_features.sum() == 1.0  # Exactly one should be active
        assert global_feat[3] == 1.0  # "none" bonus state

    def test_memory_efficiency(self):
        """Test memory usage doesn't grow over time."""
        game_state = self.create_complex_game_state()

        # Process many times to check for memory leaks
        initial_tensors = []
        for i in range(100):
            output = self.processor.process(game_state)
            if i % 20 == 0:  # Sample every 20 iterations
                initial_tensors.append(output.get_combined_embedding().clone())

        # All sampled tensors should be identical (no accumulation)
        reference = initial_tensors[0]
        for tensor in initial_tensors[1:]:
            assert torch.equal(reference, tensor)

    def test_different_board_sizes(self):
        """Test with different board configurations."""
        # Test with different config
        custom_config = PerceptionConfig(
            board_width=12,
            board_height=8,
            validate_input=False,  # Allow different sizes
        )
        processor = GameStateProcessor(custom_config)

        # Create state with different dimensions
        board = Board(width=12, height=8)
        sprite_manager = SpriteManager()
        sprite_manager.create_mouse(5, 3)

        engine = GameEngine(board, sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        output = processor.process(game_state)

        # Should adapt to different board size
        assert output.grid_tensor.shape == (28, 8, 12)
        assert output.global_features.shape == (16,)
        assert output.cat_embedding.shape == (32,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
