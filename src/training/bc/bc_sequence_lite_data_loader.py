"""
BC Sequence Lite data loader for sequential placement training.
Creates intermediate states during arrow placement sequence.
"""

import ast
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ...game.board_builder import BoardBuilder, BoardConfig
from ...game.engine import Direction, GameEngine
from ...perception.data_types import PerceptionConfig
from ...perception.processors import GameStateProcessor
from ...state_fusion.processors import StateFusionProcessor
from ...util.action_utils import (create_action_mask, encode_action,
                                  get_tile_index)
from .config import BCConfig
from .data_loader import get_device, parse_bfs_solution


class BCSequenceLiteDataset(Dataset):
    """Dataset for BC Sequence Lite training from intermediate states."""

    def __init__(
        self,
        puzzles: List[Dict],
        perception_processor: GameStateProcessor,
        state_fusion_processor: StateFusionProcessor,
    ):
        """Initialize dataset with puzzle data and processors.

        Args:
            puzzles: List of puzzle dictionaries from CSV
            perception_processor: Processor for converting game state to tensors
            state_fusion_processor: Processor for fusing perception outputs
        """
        self.puzzles = puzzles
        self.perception_processor = perception_processor
        self.state_fusion_processor = state_fusion_processor
        self.state_action_pairs = []

        # Process all puzzles to generate intermediate state-action pairs
        self._process_puzzles()

    def _process_puzzles(self):
        """Process all puzzles to generate intermediate state-action pairs."""
        for puzzle in self.puzzles:
            pairs = self._generate_intermediate_state_pairs(puzzle)
            self.state_action_pairs.extend(pairs)

        # Shuffle order across puzzles as specified in requirements
        random.shuffle(self.state_action_pairs)

    def _generate_intermediate_state_pairs(
        self, puzzle: Dict
    ) -> List[Tuple[torch.Tensor, int, torch.Tensor]]:
        """Generate intermediate state-action pairs for sequential placement.

        Creates one sample per arrow placement step:
        - State: board state before placing the arrow
        - Action: the arrow to place next

        Args:
            puzzle: Puzzle dictionary with configuration and solution

        Returns:
            List of (state_tensor, action_index, action_mask) tuples
        """
        # Create board configuration from puzzle
        config = BoardConfig(
            board_w=puzzle["board_w"],
            board_h=puzzle["board_h"],
            num_walls=puzzle["num_walls"],
            num_mice=puzzle["num_mice"],
            num_rockets=puzzle["num_rockets"],
            num_cats=puzzle["num_cats"],
            num_holes=puzzle["num_holes"],
            arrow_budget=puzzle["arrow_budget"],
        )

        # Generate the board
        builder = BoardBuilder(config, seed=puzzle["seed"])
        level = builder.generate_level(f"Puzzle_{puzzle['seed']}")

        # Create action mask for this board size
        action_mask = create_action_mask(puzzle["board_w"], puzzle["board_h"])

        # Parse BFS solution
        solution = puzzle["bfs_solution"]

        # Direction mapping
        direction_enum_map = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT,
        }

        direction_map = {
            "UP": "place_up",
            "DOWN": "place_down",
            "LEFT": "place_left",
            "RIGHT": "place_right",
        }

        from ...util.action_utils import ACTION_TYPE_OFFSETS

        pairs = []

        # Create engine for sequential placement
        engine = level.create_engine(puzzle_mode=True)

        # For each arrow in the BFS solution
        for step_idx, (x, y, direction) in enumerate(solution):
            # Capture state BEFORE placing this arrow
            current_game_state = engine.to_dict()

            # Convert to perception input
            perception_output = self.perception_processor.process(current_game_state)

            # Get state fusion output (the 128-d embedding)
            fusion_output = self.state_fusion_processor.fuse(perception_output)

            # Calculate target action for this step
            action_type = direction_map[direction]
            tile_idx = get_tile_index(x, y)
            action_index = ACTION_TYPE_OFFSETS[action_type] + tile_idx

            # Validate action
            if action_mask[action_index] != 1:
                print(
                    f"Warning: Invalid action {action_index} for puzzle {puzzle['seed']} step {step_idx}"
                )
                continue

            # Add training sample: (state_before, target_action, mask)
            pairs.append(
                (
                    fusion_output.fused_embedding.detach().cpu(),
                    action_index,
                    action_mask.clone(),
                )
            )

            # Now place the arrow to advance to next state
            direction_enum = direction_enum_map[direction]
            success = engine.place_arrow(x, y, direction_enum)
            if not success:
                print(
                    f"Warning: Failed to place arrow at ({x}, {y}) {direction} for puzzle {puzzle['seed']} step {step_idx}"
                )
                break  # Can't continue if placement failed

        return pairs

    def __len__(self) -> int:
        """Return number of state-action pairs."""
        return len(self.state_action_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get a state-action pair with mask.

        Args:
            idx: Index of the pair

        Returns:
            Tuple of (state_embedding, target_action, action_mask)
        """
        state_embedding, target_action, action_mask = self.state_action_pairs[idx]
        return state_embedding, target_action, action_mask


def create_sequence_lite_data_loaders(
    config: BCConfig,
    perception_processor: GameStateProcessor,
    state_fusion_processor: StateFusionProcessor,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders for BC Sequence Lite.

    Args:
        config: Training configuration
        perception_processor: Processor for perception layer
        state_fusion_processor: Processor for state fusion layer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load puzzles from CSV
    from .data_loader import load_puzzles_from_csv

    puzzles = load_puzzles_from_csv(config.csv_path)

    # Set random seed for reproducible splits
    random.seed(42)

    # Puzzle-level split (not episode-level) as specified
    random.shuffle(puzzles)
    n = len(puzzles)
    train_end = int(n * config.train_split)
    val_end = train_end + int(n * config.val_split)

    train_puzzles = puzzles[:train_end]
    val_puzzles = puzzles[train_end:val_end]
    test_puzzles = puzzles[val_end:]

    # Create datasets
    train_dataset = BCSequenceLiteDataset(
        train_puzzles, perception_processor, state_fusion_processor
    )
    val_dataset = BCSequenceLiteDataset(
        val_puzzles, perception_processor, state_fusion_processor
    )
    test_dataset = BCSequenceLiteDataset(
        test_puzzles, perception_processor, state_fusion_processor
    )

    print(f"BC Sequence Lite Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples from {len(train_puzzles)} puzzles")
    print(f"  Val: {len(val_dataset)} samples from {len(val_puzzles)} puzzles")
    print(f"  Test: {len(test_dataset)} samples from {len(test_puzzles)} puzzles")

    # Determine actual device to adjust settings
    device = get_device(config.device)
    use_multiprocessing = config.num_workers > 0 and device.type != "mps"
    use_pin_memory = device.type == "cuda"

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Additional shuffling at batch level
        num_workers=config.num_workers if use_multiprocessing else 0,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if use_multiprocessing else 0,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if use_multiprocessing else 0,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
