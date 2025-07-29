"""
Tests for GameStateProcessor and related components.
"""

import pytest
import torch

from src.game.sprites import SpriteType
from src.perception.data_types import (
    PerceptionConfig,
    PerceptionMetrics,
    PerceptionOutput,
)
from src.perception.processors import (
    BatchGameStateProcessor,
    GameStateProcessor,
    get_combined_embedding,
    process_game_state,
)


class TestPerceptionOutput:
    """Test cases for PerceptionOutput dataclass."""

    def test_perception_output_creation(self):
        """Test creation and validation of PerceptionOutput."""
        grid_tensor = torch.zeros(28, 10, 14)
        grid_embedding = torch.randn([35840])
        global_features = torch.zeros(16)
        cat_embedding = torch.zeros(32)

        output = PerceptionOutput(
            grid_tensor=grid_tensor,
            grid_embedding=grid_embedding,
            global_features=global_features,
            cat_embedding=cat_embedding,
        )

        assert output.grid_tensor.shape == (28, 10, 14)
        assert output.global_features.shape == (16,)
        assert output.cat_embedding.shape == (32,)

    def test_perception_output_validation(self):
        """Test that invalid shapes raise errors."""
        with pytest.raises(ValueError, match="Invalid grid tensor channels"):
            PerceptionOutput(
                grid_tensor=torch.zeros(27, 10, 14),  # Wrong channel count
                grid_embedding=torch.randn([35840]),
                global_features=torch.zeros(16),
                cat_embedding=torch.zeros(32),
            )

        with pytest.raises(ValueError, match="Invalid global features shape"):
            PerceptionOutput(
                grid_tensor=torch.zeros(28, 10, 14),
                grid_embedding=torch.randn([35840]),
                global_features=torch.zeros(15),  # Wrong dimension
                cat_embedding=torch.zeros(32),
            )

        with pytest.raises(ValueError, match="Invalid cat embedding shape"):
            PerceptionOutput(
                grid_tensor=torch.zeros(28, 10, 14),
                grid_embedding=torch.randn([35840]),
                global_features=torch.zeros(16),
                cat_embedding=torch.zeros(31),  # Wrong dimension
            )

    def test_channel_extraction(self):
        """Test extraction of specific channel ranges."""
        grid_tensor = torch.zeros(28, 10, 14)
        grid_tensor[2:10] = 1.0  # Set rocket channels

        output = PerceptionOutput(
            grid_tensor=grid_tensor,
            grid_embedding=torch.randn([35840]),
            global_features=torch.zeros(16),
            cat_embedding=torch.zeros(32),
        )

        rocket_channels = output.get_rocket_channels()
        assert rocket_channels.shape == (8, 10, 14)
        assert torch.all(rocket_channels == 1.0)

        wall_channels = output.get_wall_channels()
        assert wall_channels.shape == (2, 10, 14)
        assert torch.all(wall_channels == 0.0)

    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        grid_tensor = torch.randn(28, 10, 14)
        global_features = torch.randn(16)
        cat_embedding = torch.randn(32)

        output = PerceptionOutput(
            grid_tensor=grid_tensor,
            grid_embedding=torch.randn([35840]),
            global_features=global_features,
            cat_embedding=cat_embedding,
            source_step=100,
            cat_count=3,
        )

        # Convert to dict and back
        data = output.to_dict()
        reconstructed = PerceptionOutput.from_dict(data)

        assert torch.allclose(output.grid_tensor, reconstructed.grid_tensor)
        assert torch.allclose(output.global_features, reconstructed.global_features)
        assert torch.allclose(output.cat_embedding, reconstructed.cat_embedding)
        assert output.source_step == reconstructed.source_step
        assert output.cat_count == reconstructed.cat_count


class TestPerceptionMetrics:
    """Test cases for PerceptionMetrics."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerceptionMetrics()
        assert metrics.total_encodings == 0
        assert metrics.avg_encoding_time_ms == 0.0
        assert metrics.validation_errors == 0

    def test_timing_updates(self):
        """Test timing metric updates."""
        metrics = PerceptionMetrics()

        metrics.update_timing(1.0)
        assert metrics.total_encodings == 1
        assert metrics.avg_encoding_time_ms == 1.0
        assert metrics.max_encoding_time_ms == 1.0
        assert metrics.min_encoding_time_ms == 1.0

        metrics.update_timing(3.0)
        assert metrics.total_encodings == 2
        assert metrics.avg_encoding_time_ms == 2.0
        assert metrics.max_encoding_time_ms == 3.0
        assert metrics.min_encoding_time_ms == 1.0

    def test_cat_metrics(self):
        """Test cat processing metrics."""
        metrics = PerceptionMetrics()

        metrics.update_timing(1.0)  # Need encoding count for average first
        metrics.update_cats(5)
        assert metrics.total_cats_processed == 5
        assert metrics.max_cats_per_encoding == 5
        assert metrics.avg_cats_per_encoding == 5.0

        metrics.update_timing(1.0)
        metrics.update_cats(3)
        assert metrics.total_cats_processed == 8
        assert metrics.max_cats_per_encoding == 5
        assert metrics.avg_cats_per_encoding == 4.0


class TestGameStateProcessor:
    """Test cases for GameStateProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PerceptionConfig(
            board_width=14, board_height=10, validate_input=True
        )
        self.processor = GameStateProcessor(self.config)

        # Sample valid game state
        self.sample_game_state = {
            "board": {
                "width": 14,
                "height": 10,
                "arrows": {},
                "walls": [],
                "max_arrows": 3,
            },
            "sprite_manager": {"sprites": {}},
            "bonus_state": {"mode": "none", "remaining_ticks": 0},
            "current_step": 100,
            "current_tick": 200,
            "max_steps": 1000,
        }

    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.config.board_width == 14
        assert self.processor.config.board_height == 10
        assert hasattr(self.processor, "grid_encoder")
        assert hasattr(self.processor, "global_extractor")
        assert hasattr(self.processor, "cat_processor")

    def test_process_empty_state(self):
        """Test processing empty game state."""
        output = self.processor.process(self.sample_game_state)

        assert isinstance(output, PerceptionOutput)
        assert output.grid_tensor.shape == (28, 10, 14)
        assert output.global_features.shape == (16,)
        assert output.cat_embedding.shape == (32,)
        assert output.source_step == 100
        assert output.source_tick == 200
        assert output.cat_count == 0

    def test_process_with_sprites(self):
        """Test processing game state with sprites."""
        game_state = self.sample_game_state.copy()
        game_state["sprite_manager"]["sprites"] = {
            "cat1": {
                "type": SpriteType.CAT.value,
                "x": 5.0,
                "y": 3.0,
                "direction": "UP",
                "state": "active",
            },
            "rocket1": {
                "type": SpriteType.ROCKET.value,
                "x": 2.0,
                "y": 2.0,
                "state": "active",
                "mice_collected": 0,
            },
        }

        output = self.processor.process(game_state)

        assert output.cat_count == 1
        assert not torch.all(output.cat_embedding == 0.0)
        assert output.grid_tensor[2, 2, 2] == 1.0  # Rocket position

    def test_input_validation(self):
        """Test input validation."""
        # Missing required field
        invalid_state = {"board": {}}
        with pytest.raises(ValueError, match="Missing required field"):
            self.processor.process(invalid_state)

        # Wrong board dimensions
        invalid_state = self.sample_game_state.copy()
        invalid_state["board"]["width"] = 25
        with pytest.raises(ValueError, match="Board width mismatch"):
            self.processor.process(invalid_state)

    def test_validation_disabled(self):
        """Test processor with validation disabled."""
        config = PerceptionConfig(validate_input=False)
        processor = GameStateProcessor(config)

        # This should not raise an error even with missing fields
        minimal_state = {
            "board": {"width": 14, "height": 10, "arrows": {}, "walls": []},
            "sprite_manager": {"sprites": {}},
            "bonus_state": {"mode": "none"},
        }

        output = processor.process(minimal_state)
        assert isinstance(output, PerceptionOutput)

    def test_profiling(self):
        """Test performance profiling."""
        self.processor.enable_profiling()

        output = self.processor.process(self.sample_game_state)
        assert output.encoding_time_ms is not None
        assert output.encoding_time_ms > 0

        metrics = self.processor.get_metrics()
        assert metrics.total_encodings == 1
        assert metrics.avg_encoding_time_ms > 0

    def test_batch_processing(self):
        """Test batch processing."""
        states = [self.sample_game_state.copy() for _ in range(3)]
        outputs = self.processor.process_batch(states)

        assert len(outputs) == 3
        for output in outputs:
            assert isinstance(output, PerceptionOutput)

    def test_benchmarking(self):
        """Test benchmarking functionality."""
        results = self.processor.benchmark(self.sample_game_state, num_iterations=10)

        assert "mean_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
        assert "std_ms" in results
        assert results["iterations"] == 10
        assert isinstance(results["meets_target"], bool)

    def test_output_shapes(self):
        """Test output shape information."""
        shapes = self.processor.get_output_shapes()

        assert shapes["grid_tensor"] == (28, 10, 14)
        assert shapes["global_features"] == (16,)
        assert shapes["cat_embedding"] == (32,)
        assert shapes["combined_embedding"] == (35888,)


class TestBatchGameStateProcessor:
    """Test cases for BatchGameStateProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batch_processor = BatchGameStateProcessor(batch_size=4)

        self.sample_state = {
            "board": {"width": 14, "height": 10, "arrows": {}, "walls": []},
            "sprite_manager": {"sprites": {}},
            "bonus_state": {"mode": "none"},
        }

    def test_batch_processing(self):
        """Test batch processing functionality."""
        states = [self.sample_state.copy() for _ in range(6)]

        batched_tensor = self.batch_processor.process_batch(states)

        assert batched_tensor.shape[0] == 6
        assert batched_tensor.shape[1] == 35888

    def test_streaming_processing(self):
        """Test streaming batch processing."""
        states = [self.sample_state.copy() for _ in range(10)]

        batches = self.batch_processor.process_streaming(states)

        # Should create 3 batches: [4, 4, 2]
        assert len(batches) == 3
        assert batches[0].shape[0] == 4
        assert batches[1].shape[0] == 4
        assert batches[2].shape[0] == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_state = {
            "board": {"width": 14, "height": 10, "arrows": {}, "walls": []},
            "sprite_manager": {"sprites": {}},
            "bonus_state": {"mode": "none"},
        }

    def test_process_game_state_function(self):
        """Test process_game_state convenience function."""
        output = process_game_state(self.sample_state)

        assert isinstance(output, PerceptionOutput)
        assert output.grid_tensor.shape == (28, 10, 14)

    def test_convenience_functions_with_config(self):
        """Test convenience functions with custom config."""
        config = PerceptionConfig(validate_input=False)

        output = process_game_state(self.sample_state, config)
        embedding = get_combined_embedding(self.sample_state, config)

        assert isinstance(output, PerceptionOutput)
        assert isinstance(embedding, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
