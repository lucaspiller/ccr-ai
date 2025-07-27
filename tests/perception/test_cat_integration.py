"""
Integration tests for CatSetEncoder with real game engine states.
"""

import pytest
import torch
from src.perception.cat_encoder import CatSetProcessor
from src.game.board import Board, Direction
from src.game.sprites import SpriteManager
from src.game.engine import GameEngine


class TestCatSetIntegration:
    """Integration tests for CatSetEncoder with actual game engine."""

    def setup_method(self):
        """Set up test game state."""
        self.board = Board(width=14, height=10)
        self.sprite_manager = SpriteManager()
        self.processor = CatSetProcessor(board_width=14, board_height=10)

    def test_no_cats_integration(self):
        """Test cat encoder with game state containing no cats."""
        # Create game with only non-cat sprites
        mouse = self.sprite_manager.create_mouse(5, 3)
        rocket = self.sprite_manager.create_rocket(2, 2)

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Process cat embedding
        cat_embedding = self.processor.process(game_state)
        cat_count = self.processor.get_cat_count(game_state)

        assert cat_embedding.shape == (32,)
        assert torch.all(cat_embedding == 0.0)
        assert cat_count == 0

    def test_single_cat_integration(self):
        """Test cat encoder with single cat in game state."""
        # Create game with one cat
        cat = self.sprite_manager.create_cat(8, 6)
        cat.direction = Direction.LEFT
        rocket = self.sprite_manager.create_rocket(2, 2)

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Process cat embedding
        cat_embedding = self.processor.process(game_state)
        cat_count = self.processor.get_cat_count(game_state)

        assert cat_embedding.shape == (32,)
        assert not torch.all(cat_embedding == 0.0)
        assert cat_count == 1

        # Check that embedding is consistent
        cat_embedding2 = self.processor.process(game_state)
        assert torch.equal(cat_embedding, cat_embedding2)

    def test_multiple_cats_integration(self):
        """Test cat encoder with multiple cats in different positions."""
        # Create game with multiple cats
        cat1 = self.sprite_manager.create_cat(3, 2)
        cat1.direction = Direction.UP

        cat2 = self.sprite_manager.create_cat(9, 7)
        cat2.direction = Direction.DOWN

        cat3 = self.sprite_manager.create_cat(5, 5)
        cat3.direction = Direction.RIGHT

        rocket = self.sprite_manager.create_rocket(1, 1)

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Process cat embedding
        cat_embedding = self.processor.process(game_state)
        cat_count = self.processor.get_cat_count(game_state)

        assert cat_embedding.shape == (32,)
        assert not torch.all(cat_embedding == 0.0)
        assert cat_count == 3

    def test_cat_permutation_invariance_integration(self):
        """Test that cat embedding is invariant to sprite creation order."""
        # Create first configuration
        board1 = Board(width=14, height=10)
        sm1 = SpriteManager()
        cat1a = sm1.create_cat(3, 2)
        cat1a.direction = Direction.UP
        cat1b = sm1.create_cat(9, 7)
        cat1b.direction = Direction.DOWN

        engine1 = GameEngine(board1, sm1, max_steps=1000)
        state1 = engine1.to_dict()
        embedding1 = self.processor.process(state1)

        # Create second configuration with reversed order
        board2 = Board(width=14, height=10)
        sm2 = SpriteManager()
        cat2a = sm2.create_cat(9, 7)  # Create in opposite order
        cat2a.direction = Direction.DOWN
        cat2b = sm2.create_cat(3, 2)
        cat2b.direction = Direction.UP

        engine2 = GameEngine(board2, sm2, max_steps=1000)
        state2 = engine2.to_dict()
        embedding2 = self.processor.process(state2)

        # Embeddings should be identical due to permutation invariance
        assert torch.allclose(embedding1, embedding2, atol=1e-6)

    def test_cat_rocket_distance_integration(self):
        """Test that rocket distance features work correctly."""
        # Place cat far from rocket
        cat = self.sprite_manager.create_cat(13, 9)  # Far corner
        rocket = self.sprite_manager.create_rocket(0, 0)  # Opposite corner

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Extract raw cat features to check distance calculation
        sprites = game_state["sprite_manager"]["sprites"]
        board_arrows = game_state["board"]["arrows"]

        cat_features = self.processor.encoder.extract_cat_features(
            sprites, board_arrows, 14, 10
        )

        assert cat_features.shape == (1, 8)

        # Check distance feature (index 7)
        # L1 distance from (13,9) to (0,0) = 13 + 9 = 22
        # Normalized: 22/20 = 1.1, clamped to 1.0
        distance_feature = cat_features[0, 7].item()
        assert abs(distance_feature - 1.0) < 1e-6

    def test_cat_movement_and_encoding(self):
        """Test cat encoding after movement."""
        # Create cat and let it move
        cat = self.sprite_manager.create_cat(5, 5)
        cat.direction = Direction.RIGHT

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)

        # Get initial embedding
        initial_state = engine.to_dict()
        initial_embedding = self.processor.process(initial_state)
        initial_count = self.processor.get_cat_count(initial_state)

        # Step the game to move the cat
        for _ in range(10):  # Step multiple times to ensure movement
            engine.step()

        # Get new embedding
        new_state = engine.to_dict()
        new_embedding = self.processor.process(new_state)
        new_count = self.processor.get_cat_count(new_state)

        # Cat count should remain the same
        assert new_count == initial_count == 1

        # Embedding should be different due to position change
        assert not torch.equal(initial_embedding, new_embedding)
        assert new_embedding.shape == (32,)

    def test_cat_with_arrows_integration(self):
        """Test cat encoding in presence of arrows."""
        # Create cat and place arrow in front of it
        cat = self.sprite_manager.create_cat(5, 5)
        cat.direction = Direction.UP

        # Place arrow in front of cat
        self.board.place_arrow(
            5, 4, Direction.LEFT
        )  # Arrow at (5,4), cat at (5,5) facing UP

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Process embedding
        cat_embedding = self.processor.process(game_state)

        assert cat_embedding.shape == (32,)
        assert not torch.all(cat_embedding == 0.0)

        # Extract features to check arrow detection
        sprites = game_state["sprite_manager"]["sprites"]
        board_arrows = game_state["board"]["arrows"]

        cat_features = self.processor.encoder.extract_cat_features(
            sprites, board_arrows, 14, 10
        )

        # Should detect cat with proper features
        assert cat_features.shape == (1, 8)
        # Arrow ahead feature (index 6) should be 0 since arrows are healthy
        assert cat_features[0, 6] == 0.0

    def test_max_cats_handling(self):
        """Test handling of many cats (approaching max limit)."""
        # Create many cats
        positions = [(i, j) for i in range(0, 14, 2) for j in range(0, 10, 2)]
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        for i, (x, y) in enumerate(positions[:10]):  # Create 10 cats
            cat = self.sprite_manager.create_cat(x, y)
            cat.direction = directions[i % 4]

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Process embedding
        cat_embedding = self.processor.process(game_state)
        cat_count = self.processor.get_cat_count(game_state)

        assert cat_embedding.shape == (32,)
        assert not torch.all(cat_embedding == 0.0)
        assert cat_count == 10

    def test_performance_with_many_cats(self):
        """Test performance with maximum cats."""
        import time

        # Create near-maximum cats (15 out of 16 limit)
        for i in range(15):
            x = i % 14
            y = i % 10
            cat = self.sprite_manager.create_cat(x, y)
            cat.direction = Direction.UP

        engine = GameEngine(self.board, self.sprite_manager, max_steps=1000)
        game_state = engine.to_dict()

        # Time the encoding
        start_time = time.time()
        for _ in range(100):
            cat_embedding = self.processor.process(game_state)
        end_time = time.time()

        avg_time_ms = (end_time - start_time) * 1000 / 100

        # Should still be fast even with many cats
        assert avg_time_ms < 10.0, f"Cat encoding too slow: {avg_time_ms:.2f}ms"
        assert cat_embedding.shape == (32,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
