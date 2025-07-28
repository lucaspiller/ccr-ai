"""
Data loading and preprocessing for behaviour cloning training.
"""

import ast
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ...game.board_builder import BoardBuilder, BoardConfig
from ...game.engine import Direction, GameEngine
from ...perception.data_types import PerceptionConfig
from ...perception.processors import GameStateProcessor
from ...state_fusion.processors import StateFusionProcessor
from ...util.action_utils import (create_action_mask, encode_action,
                                  get_tile_index)
from .config import BCConfig


class BehaviourCloningDataset(Dataset):
    """Dataset for behaviour cloning from BFS solutions."""

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

        # Process all puzzles to generate state-action pairs
        self._process_puzzles()

    def _process_puzzles(self):
        """Process all puzzles to generate state-action pairs."""
        for puzzle in self.puzzles:
            pairs = self._generate_state_action_pairs(puzzle)
            self.state_action_pairs.extend(pairs)

    def _generate_state_action_pairs(
        self, puzzle: Dict
    ) -> List[Tuple[torch.Tensor, int, torch.Tensor]]:
        """Generate state-action pairs for a single puzzle.

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

        # Create game engine
        engine = level.create_engine(puzzle_mode=True)

        # Create action mask for this board size
        action_mask = create_action_mask(puzzle["board_w"], puzzle["board_h"])

        # Parse BFS solution
        solution = puzzle["bfs_solution"]

        # BC-Set: Generate final state with all arrows placed, create multi-hot target

        # First, place all BFS arrows to create final state
        direction_enum_map = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT,
        }

        for x, y, direction in solution:
            direction_enum = direction_enum_map[direction]
            success = engine.place_arrow(x, y, direction_enum)
            if not success:
                print(
                    f"Warning: Failed to place arrow at ({x}, {y}) {direction} for puzzle {puzzle['seed']}"
                )
                # Continue anyway - partial solutions still useful

        # Get final game state with all arrows placed
        final_game_state = engine.to_dict()

        # Convert to perception input
        perception_output = self.perception_processor.process(final_game_state)

        # Get state fusion output (the 128-d embedding)
        fusion_output = self.state_fusion_processor.fuse(perception_output)

        # Create multi-hot target vector (700-dim binary)
        multi_hot_target = torch.zeros(700, dtype=torch.float32)

        # Set 1s for each BFS arrow location+direction
        direction_map = {
            "UP": "place_up",
            "DOWN": "place_down",
            "LEFT": "place_left",
            "RIGHT": "place_right",
        }

        from ...util.action_utils import ACTION_TYPE_OFFSETS

        for x, y, direction in solution:
            action_type = direction_map[direction]

            # Convert BFS coordinates to tile index using x,y convention
            tile_idx = get_tile_index(x, y)

            ## Calculate action index
            action_index = ACTION_TYPE_OFFSETS[action_type] + tile_idx

            # Validate and set target
            if action_mask[action_index] == 1:  # Valid action
                multi_hot_target[action_index] = 1.0
            else:
                print(
                    f"Warning: Invalid BFS action {action_index} for board {puzzle['board_w']}×{puzzle['board_h']} ({puzzle['seed']})"
                )

        # Debug: Check if we actually set any targets
        num_targets = multi_hot_target.sum().item()
        if num_targets == 0:
            print(
                f"WARNING: No targets set for puzzle {puzzle['seed']} with solution {solution}"
            )
        elif num_targets != len(solution):
            print(
                f"WARNING: Expected {len(solution)} targets but got {num_targets} for puzzle {puzzle['seed']}"
            )

        # Return single training sample: (final_state, multi_hot_target, mask, arrow_budget, board_w, board_h)
        pairs = [
            (
                fusion_output.fused_embedding.detach().cpu(),
                multi_hot_target,
                action_mask.clone(),
                torch.tensor(puzzle["arrow_budget"], dtype=torch.float32),
                puzzle["board_w"],
                puzzle["board_h"],
            )
        ]

        return pairs

    def __len__(self) -> int:
        """Return number of state-action pairs."""
        return len(self.state_action_pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Get a state-multihot pair with mask, arrow budget, and board dimensions.

        Args:
            idx: Index of the pair

        Returns:
            Tuple of (state_embedding, multi_hot_target, action_mask, arrow_budget, board_w, board_h)
        """
        (
            state_embedding,
            multi_hot_target,
            action_mask,
            arrow_budget,
            board_w,
            board_h,
        ) = self.state_action_pairs[idx]

        return (
            state_embedding,
            multi_hot_target,
            action_mask,
            arrow_budget,
            board_w,
            board_h,
        )


def parse_bfs_solution(solution_str: str) -> List[Tuple[int, int, str]]:
    """Parse BFS solution string from CSV.

    Args:
        solution_str: String representation of solution like "[[6,0,LEFT],[0,6,DOWN]]"

    Returns:
        List of (x, y, direction) tuples
    """
    try:
        # Replace unquoted direction strings with quoted ones
        import re

        # Match patterns like UP, DOWN, LEFT, RIGHT that are not already quoted
        direction_pattern = r"\b(UP|DOWN|LEFT|RIGHT)\b"
        fixed_solution_str = re.sub(direction_pattern, r'"\1"', solution_str)

        # Parse the string as a Python list
        solution_list = ast.literal_eval(fixed_solution_str)

        parsed_solution = []
        for action in solution_list:
            if len(action) != 3:
                raise ValueError(f"Invalid action format: {action}")
            x, y, direction = action
            parsed_solution.append((int(x), int(y), str(direction)))

        return parsed_solution
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse BFS solution '{solution_str}': {e}")


def load_puzzles_from_csv(csv_path: str) -> List[Dict]:
    """Load puzzles from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of puzzle dictionaries
    """
    puzzles = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Parse numeric fields
            puzzle = {
                "seed": int(row["seed"]),
                "board_w": int(row["board_w"]),
                "board_h": int(row["board_h"]),
                "num_walls": int(row["num_walls"]),
                "num_mice": int(row["num_mice"]),
                "num_rockets": int(row["num_rockets"]),
                "num_cats": int(row["num_cats"]),
                "num_holes": int(row["num_holes"]),
                "arrow_budget": int(row["arrow_budget"]),
                "difficulty_label": row["difficulty_label"],
                "bfs_solution": parse_bfs_solution(row["bfs_solution"]),
            }
            puzzles.append(puzzle)

    return puzzles


def create_data_loaders(
    config: BCConfig,
    perception_processor: GameStateProcessor,
    state_fusion_processor: StateFusionProcessor,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        config: Training configuration
        perception_processor: Processor for perception layer
        state_fusion_processor: Processor for state fusion layer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load puzzles from CSV
    puzzles = load_puzzles_from_csv(config.csv_path)

    # Set random seed for reproducible splits
    random.seed(42)

    # Simple global split (not stratified by difficulty for small datasets)
    random.shuffle(puzzles)
    n = len(puzzles)
    train_end = int(n * config.train_split)
    val_end = train_end + int(n * config.val_split)

    train_puzzles = puzzles[:train_end]
    val_puzzles = puzzles[train_end:val_end]
    test_puzzles = puzzles[val_end:]

    # Shuffle the combined datasets
    random.shuffle(train_puzzles)
    random.shuffle(val_puzzles)
    random.shuffle(test_puzzles)

    # Create datasets
    train_dataset = BehaviourCloningDataset(
        train_puzzles, perception_processor, state_fusion_processor
    )
    val_dataset = BehaviourCloningDataset(
        val_puzzles, perception_processor, state_fusion_processor
    )
    test_dataset = BehaviourCloningDataset(
        test_puzzles, perception_processor, state_fusion_processor
    )

    # Determine actual device to adjust settings
    device = get_device(config.device)
    use_multiprocessing = config.num_workers > 0 and device.type != "mps"
    use_pin_memory = device.type == "cuda"

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
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


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device from configuration string.

    Args:
        device_str: Device string ("auto", "cpu", "cuda", "mps")

    Returns:
        PyTorch device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Set MPS allocator to avoid memory fragmentation issues
            try:
                # Test MPS availability with a simple operation
                test_tensor = torch.randn(1, device="mps")
                del test_tensor
                return torch.device("mps")
            except Exception as e:
                print(f"Warning: MPS device available but not working properly: {e}")
                print("Falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    else:
        device = torch.device(device_str)
        # Validate the device works if it's MPS
        if device.type == "mps":
            try:
                test_tensor = torch.randn(1, device=device)
                del test_tensor
            except Exception as e:
                print(f"Error: Cannot use MPS device: {e}")
                print("Consider using --device cpu")
                raise
        return device
