"""
Main game state processor for converting raw game state to perception output.
"""

import time
from typing import Any, Dict, Optional
import torch

from .encoders import GridEncoder, GlobalFeatureExtractor
from .cat_encoder import CatSetProcessor
from .data_types import (
    PerceptionOutput,
    PerceptionConfig,
    PerceptionMetrics,
    GRID_WIDTH,
    GRID_HEIGHT,
)


class GameStateProcessor:
    """
    Main processor for converting GameEngine.to_dict() to neural network inputs.

    Orchestrates all perception components:
    - GridEncoder: 28-channel spatial representation
    - GlobalFeatureExtractor: 16-dimensional game features
    - CatSetProcessor: 32-dimensional cat set embedding

    Produces PerceptionOutput with all tensor representations needed for the neural network.
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        """
        Initialize the game state processor.

        Args:
            config: Configuration for perception components
        """
        self.config = config or PerceptionConfig()

        # Initialize component processors
        self.grid_encoder = GridEncoder(
            width=self.config.board_width, height=self.config.board_height
        )

        self.global_extractor = GlobalFeatureExtractor()

        self.cat_processor = CatSetProcessor(
            board_width=self.config.board_width, board_height=self.config.board_height
        )

        # Performance monitoring
        self.metrics = PerceptionMetrics()
        self._profiling_enabled = self.config.enable_profiling

    def process(self, game_state: Dict[str, Any]) -> PerceptionOutput:
        """
        Process complete game state into neural network ready tensors.

        Args:
            game_state: Output from GameEngine.to_dict()

        Returns:
            PerceptionOutput: Structured tensor representations

        Raises:
            ValueError: If game state is invalid or malformed
        """
        start_time = time.time() if self._profiling_enabled else None

        try:
            # Validate input if enabled
            if self.config.validate_input:
                self._validate_game_state(game_state)

            # Process each component
            grid_tensor = self.grid_encoder.encode(game_state)
            global_features = self.global_extractor.extract(game_state)
            cat_embedding = self.cat_processor.process(game_state)

            # Extract metadata
            cat_count = self.cat_processor.get_cat_count(game_state)
            source_step = game_state.get("current_step")
            source_tick = game_state.get("current_tick")

            # Calculate encoding time
            encoding_time_ms = None
            if self._profiling_enabled and start_time is not None:
                encoding_time_ms = (time.time() - start_time) * 1000
                self.metrics.update_timing(encoding_time_ms)
                self.metrics.update_cats(cat_count)

            # Create structured output
            output = PerceptionOutput(
                grid_tensor=grid_tensor,
                global_features=global_features,
                cat_embedding=cat_embedding,
                source_step=source_step,
                source_tick=source_tick,
                cat_count=cat_count,
                encoding_time_ms=encoding_time_ms,
                _validate_shapes=self.config.strict_bounds_checking,
            )

            return output

        except Exception as e:
            self.metrics.record_error("general")
            if isinstance(e, ValueError):
                self.metrics.record_error("shape")
            raise

    def process_batch(
        self, game_states: list[Dict[str, Any]]
    ) -> list[PerceptionOutput]:
        """
        Process multiple game states in batch.

        Args:
            game_states: List of game state dictionaries

        Returns:
            List[PerceptionOutput]: Processed outputs for each state
        """
        return [self.process(state) for state in game_states]

    def _validate_game_state(self, game_state: Dict[str, Any]) -> None:
        """
        Validate that game state has required structure.

        Args:
            game_state: Game state dictionary to validate

        Raises:
            ValueError: If game state is missing required fields
        """
        required_fields = ["board", "sprite_manager", "bonus_state"]
        for field in required_fields:
            if field not in game_state:
                raise ValueError(f"Missing required field in game state: {field}")

        # Validate board structure
        board = game_state["board"]
        board_required = ["width", "height", "arrows", "walls"]
        for field in board_required:
            if field not in board:
                raise ValueError(f"Missing required board field: {field}")

        # Validate sprite manager structure
        sprite_manager = game_state["sprite_manager"]
        if "sprites" not in sprite_manager:
            raise ValueError("Missing 'sprites' field in sprite_manager")

        # Validate bonus state structure
        bonus_state = game_state["bonus_state"]
        if "mode" not in bonus_state:
            raise ValueError("Missing 'mode' field in bonus_state")

        # Check board dimensions match config
        if self.config.strict_bounds_checking:
            board_width = board.get("width", 0)
            board_height = board.get("height", 0)

            if board_width != self.config.board_width:
                raise ValueError(
                    f"Board width mismatch: {board_width} != {self.config.board_width}"
                )

            if board_height != self.config.board_height:
                raise ValueError(
                    f"Board height mismatch: {board_height} != {self.config.board_height}"
                )

    def get_metrics(self) -> PerceptionMetrics:
        """Get performance metrics for monitoring."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = PerceptionMetrics()

    def enable_profiling(self) -> None:
        """Enable performance profiling."""
        self._profiling_enabled = True
        self.config.enable_profiling = True

    def disable_profiling(self) -> None:
        """Disable performance profiling."""
        self._profiling_enabled = False
        self.config.enable_profiling = False

    def get_output_shapes(self) -> Dict[str, tuple]:
        """Get expected output tensor shapes."""
        return {
            "grid_tensor": (28, self.config.board_height, self.config.board_width),
            "global_features": (16,),
            "cat_embedding": (32,),
            "combined_embedding": (
                28 * self.config.board_height * self.config.board_width + 16 + 32,
            ),
        }

    def benchmark(
        self, game_state: Dict[str, Any], num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark processing performance.

        Args:
            game_state: Sample game state for benchmarking
            num_iterations: Number of iterations to run

        Returns:
            Dict with timing statistics
        """
        self.enable_profiling()
        self.reset_metrics()

        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.process(game_state)
            times.append((time.time() - start_time) * 1000)

        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (
                sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)
            )
            ** 0.5,
            "iterations": num_iterations,
            "meets_target": sum(times) / len(times) <= 1.0,  # 1ms target
        }


class BatchGameStateProcessor:
    """
    Optimized processor for handling batches of game states efficiently.
    """

    def __init__(self, config: Optional[PerceptionConfig] = None, batch_size: int = 32):
        """
        Initialize batch processor.

        Args:
            config: Perception configuration
            batch_size: Target batch size for processing
        """
        self.processor = GameStateProcessor(config)
        self.batch_size = batch_size

    def process_batch(self, game_states: list[Dict[str, Any]]) -> torch.Tensor:
        """
        Process batch of game states into stacked tensor.

        Args:
            game_states: List of game state dictionaries

        Returns:
            torch.Tensor: Stacked combined embeddings [batch_size, features]
        """
        # Process each state individually
        outputs = [self.processor.process(state) for state in game_states]

        # Stack combined embeddings
        combined_embeddings = [output.get_combined_embedding() for output in outputs]
        batched_tensor = torch.stack(combined_embeddings, dim=0)

        return batched_tensor

    def process_streaming(
        self, game_states: list[Dict[str, Any]]
    ) -> list[torch.Tensor]:
        """
        Process streaming game states in optimal batch sizes.

        Args:
            game_states: List of game state dictionaries

        Returns:
            List[torch.Tensor]: Batched tensors
        """
        batches = []
        for i in range(0, len(game_states), self.batch_size):
            batch_states = game_states[i : i + self.batch_size]
            batch_tensor = self.process_batch(batch_states)
            batches.append(batch_tensor)

        return batches


# Convenience functions for quick processing
def process_game_state(
    game_state: Dict[str, Any], config: Optional[PerceptionConfig] = None
) -> PerceptionOutput:
    """
    Convenience function to process a single game state.

    Args:
        game_state: Game state dictionary
        config: Optional configuration

    Returns:
        PerceptionOutput: Processed perception output
    """
    processor = GameStateProcessor(config)
    return processor.process(game_state)


def get_combined_embedding(
    game_state: Dict[str, Any], config: Optional[PerceptionConfig] = None
) -> torch.Tensor:
    """
    Convenience function to get combined embedding directly.

    Args:
        game_state: Game state dictionary
        config: Optional configuration

    Returns:
        torch.Tensor: Combined embedding ready for neural network
    """
    output = process_game_state(game_state, config)
    return output.get_combined_embedding()
