"""
Reusable action encoding/decoding utilities for ChuChu Rocket AI.

This module provides a centralized location for action space utilities that can be
used consistently across all layers (policy, training, evaluation).

Coordinates use standard x,y convention:
- x: column coordinate (0-13, left to right)
- y: row coordinate (0-9, top to bottom)
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

# Board dimensions
BOARD_WIDTH = 14
BOARD_HEIGHT = 10
TOTAL_TILES = BOARD_WIDTH * BOARD_HEIGHT  # 140
TOTAL_ACTIONS = 700  # 140 tiles × 5 actions per tile

# Action type offsets in the 700-dimensional space
ACTION_TYPE_OFFSETS = {
    "place_up": 0,  # Actions 0-139
    "place_down": 140,  # Actions 140-279
    "place_left": 280,  # Actions 280-419
    "place_right": 420,  # Actions 420-559
    "erase": 560,  # Actions 560-699
}

# Reverse mapping for decoding
OFFSET_TO_ACTION_TYPE = {v: k for k, v in ACTION_TYPE_OFFSETS.items()}


@dataclass
class ActionInfo:
    """
    Structured information about a game action.

    Attributes:
        action_type: Type of action ("place_up", "place_down", "place_left", "place_right", "erase")
        x: Column coordinate (0-13)
        y: Row coordinate (0-9)
        tile_idx: Linear tile index (0-139)
        action_idx: Action index in the 700-dimensional space
    """

    action_type: str
    x: int
    y: int
    tile_idx: int
    action_idx: int

    def __post_init__(self):
        """Validate action information."""
        valid_types = ["place_up", "place_down", "place_left", "place_right", "erase"]
        if self.action_type not in valid_types:
            raise ValueError(f"action_type must be one of: {valid_types}")

        if not (0 <= self.x < BOARD_WIDTH):
            raise ValueError(f"x must be between 0 and {BOARD_WIDTH-1}")

        if not (0 <= self.y < BOARD_HEIGHT):
            raise ValueError(f"y must be between 0 and {BOARD_HEIGHT-1}")

        if not (0 <= self.tile_idx < TOTAL_TILES):
            raise ValueError(f"tile_idx must be between 0 and {TOTAL_TILES-1}")

        if not (0 <= self.action_idx < TOTAL_ACTIONS):
            raise ValueError(f"action_idx must be between 0 and {TOTAL_ACTIONS-1}")

        # Validate consistency
        expected_tile_idx = self.y * BOARD_WIDTH + self.x
        if self.tile_idx != expected_tile_idx:
            raise ValueError(
                f"tile_idx {self.tile_idx} doesn't match coordinates ({self.x}, {self.y})"
            )


def get_tile_index(x: int, y: int) -> int:
    """
    Convert board coordinates to linear tile index.

    Args:
        x: Column coordinate (0-13)
        y: Row coordinate (0-9)

    Returns:
        Linear tile index (0-139)

    Raises:
        ValueError: If coordinates are out of bounds
    """
    if not (0 <= x < BOARD_WIDTH):
        raise ValueError(f"x {x} out of bounds [0, {BOARD_WIDTH-1}]")

    if not (0 <= y < BOARD_HEIGHT):
        raise ValueError(f"y {y} out of bounds [0, {BOARD_HEIGHT-1}]")

    return y * BOARD_WIDTH + x


def get_tile_coords(tile_idx: int) -> Tuple[int, int]:
    """
    Convert linear tile index to board coordinates.

    Args:
        tile_idx: Linear tile index (0-139)

    Returns:
        Tuple of (x, y) coordinates

    Raises:
        ValueError: If tile index is out of bounds
    """
    if not (0 <= tile_idx < TOTAL_TILES):
        raise ValueError(f"tile_idx {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")

    y = tile_idx // BOARD_WIDTH
    x = tile_idx % BOARD_WIDTH
    return x, y


def encode_action(action_type: str, x: int, y: int) -> int:
    """
    Convert structured action to action index.

    Args:
        action_type: Type of action ("place_up", "place_down", "place_left", "place_right", "erase")
        x: Column coordinate (0-13)
        y: Row coordinate (0-9)

    Returns:
        Action index in 700-dimensional space

    Raises:
        ValueError: If action type is invalid or coordinates are out of bounds
    """
    if action_type not in ACTION_TYPE_OFFSETS:
        valid_types = list(ACTION_TYPE_OFFSETS.keys())
        raise ValueError(
            f"Invalid action_type '{action_type}'. Must be one of: {valid_types}"
        )

    tile_idx = get_tile_index(x, y)
    offset = ACTION_TYPE_OFFSETS[action_type]
    action_idx = offset + tile_idx

    return action_idx


def decode_action(action_idx: int) -> ActionInfo:
    """
    Convert action index to structured action information.

    Args:
        action_idx: Action index (0-699)

    Returns:
        ActionInfo with decoded action details

    Raises:
        ValueError: If action index is out of bounds
    """
    if not (0 <= action_idx < TOTAL_ACTIONS):
        raise ValueError(
            f"action_idx {action_idx} out of bounds [0, {TOTAL_ACTIONS-1}]"
        )

    # Find which action type this index belongs to
    action_type = None
    tile_idx = None

    for offset in sorted(ACTION_TYPE_OFFSETS.values(), reverse=True):
        if action_idx >= offset:
            action_type = OFFSET_TO_ACTION_TYPE[offset]
            tile_idx = action_idx - offset
            break

    if action_type is None or tile_idx is None:
        raise ValueError(f"Could not decode action index {action_idx}")

    # Validate tile index is within bounds
    if not (0 <= tile_idx < TOTAL_TILES):
        raise ValueError(
            f"Decoded tile index {tile_idx} out of bounds [0, {TOTAL_TILES-1}]"
        )

    # Get tile coordinates
    x, y = get_tile_coords(tile_idx)

    return ActionInfo(
        action_type=action_type,
        x=x,
        y=y,
        tile_idx=tile_idx,
        action_idx=action_idx,
    )


def create_action_mask(board_w: int, board_h: int) -> torch.Tensor:
    """
    Create action mask for given board size.

    Args:
        board_w: Board width
        board_h: Board height

    Returns:
        Binary mask tensor [700] where 1 = valid action, 0 = invalid action
    """
    # Validate board size
    if board_w <= 0 or board_h <= 0:
        raise ValueError(f"Board size must be positive, got {board_w}×{board_h}")
    if board_w > BOARD_WIDTH or board_h > BOARD_HEIGHT:
        raise ValueError(
            f"Board size {board_w}×{board_h} exceeds maximum {BOARD_WIDTH}×{BOARD_HEIGHT}"
        )

    # Start with all actions invalid
    mask = torch.zeros(TOTAL_ACTIONS, dtype=torch.float32)

    # Enable valid tiles for all 5 action types
    for y in range(board_h):
        for x in range(board_w):
            tile_idx = get_tile_index(x, y)

            for action_type_offset in ACTION_TYPE_OFFSETS.values():
                mask[action_type_offset + tile_idx] = 1.0

    return mask


def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply action mask to logits before computing probabilities.

    Args:
        logits: Raw action logits [batch_size, 700] or [700]
        action_mask: Binary action mask [700] or [batch_size, 700]

    Returns:
        Masked logits with invalid actions set to -inf
    """
    # Expand mask to match logits dimensions if needed
    if logits.dim() == 2 and action_mask.dim() == 1:
        action_mask = action_mask.unsqueeze(0).expand_as(logits)

    # Set invalid actions to -inf (probability = 0 after softmax)
    masked_logits = logits.clone()
    masked_logits[action_mask == 0] = float("-inf")

    return masked_logits


def get_valid_actions(board_w: int, board_h: int) -> torch.Tensor:
    """
    Get indices of all valid actions for given board size.

    Args:
        board_w: Board width
        board_h: Board height

    Returns:
        Tensor of valid action indices
    """
    mask = create_action_mask(board_w, board_h)
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)


def get_placement_actions(tile_idx: int) -> List[int]:
    """
    Get all arrow placement action indices for a specific tile.

    Args:
        tile_idx: Linear tile index (0-139)

    Returns:
        List of 4 action indices for placing arrows (up, down, left, right)

    Raises:
        ValueError: If tile index is out of bounds
    """
    if not (0 <= tile_idx < TOTAL_TILES):
        raise ValueError(f"tile_idx {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")

    return [
        ACTION_TYPE_OFFSETS["place_up"] + tile_idx,
        ACTION_TYPE_OFFSETS["place_down"] + tile_idx,
        ACTION_TYPE_OFFSETS["place_left"] + tile_idx,
        ACTION_TYPE_OFFSETS["place_right"] + tile_idx,
    ]


def get_erase_action(tile_idx: int) -> int:
    """
    Get erase action index for a specific tile.

    Args:
        tile_idx: Linear tile index (0-139)

    Returns:
        Action index for erasing arrows at the tile

    Raises:
        ValueError: If tile index is out of bounds
    """
    if not (0 <= tile_idx < TOTAL_TILES):
        raise ValueError(f"tile_idx {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")

    return ACTION_TYPE_OFFSETS["erase"] + tile_idx


def get_actions_for_tile(x: int, y: int) -> List[int]:
    """
    Get all possible action indices for a specific tile.

    Args:
        x: Column coordinate (0-13)
        y: Row coordinate (0-9)

    Returns:
        List of 5 action indices (4 placements + 1 erase)

    Raises:
        ValueError: If coordinates are out of bounds
    """
    tile_idx = get_tile_index(x, y)
    placement_actions = get_placement_actions(tile_idx)
    erase_action = get_erase_action(tile_idx)

    return placement_actions + [erase_action]


def get_actions_by_type(action_type: str) -> List[int]:
    """
    Get all action indices for a specific action type.

    Args:
        action_type: Type of action ("place_up", "place_down", "place_left", "place_right", "erase")

    Returns:
        List of 140 action indices for that action type

    Raises:
        ValueError: If action type is invalid
    """
    if action_type not in ACTION_TYPE_OFFSETS:
        valid_types = list(ACTION_TYPE_OFFSETS.keys())
        raise ValueError(
            f"Invalid action_type '{action_type}'. Must be one of: {valid_types}"
        )

    offset = ACTION_TYPE_OFFSETS[action_type]
    return list(range(offset, offset + TOTAL_TILES))


def is_placement_action(action_idx: int) -> bool:
    """
    Check if an action index corresponds to arrow placement.

    Args:
        action_idx: Action index (0-699)

    Returns:
        True if action places an arrow, False if it's an erase action

    Raises:
        ValueError: If action index is out of bounds
    """
    if not (0 <= action_idx < TOTAL_ACTIONS):
        raise ValueError(
            f"action_idx {action_idx} out of bounds [0, {TOTAL_ACTIONS-1}]"
        )

    return action_idx < ACTION_TYPE_OFFSETS["erase"]


def is_erase_action(action_idx: int) -> bool:
    """
    Check if an action index corresponds to erasing arrows.

    Args:
        action_idx: Action index (0-699)

    Returns:
        True if action erases arrows, False if it's a placement action

    Raises:
        ValueError: If action index is out of bounds
    """
    if not (0 <= action_idx < TOTAL_ACTIONS):
        raise ValueError(
            f"action_idx {action_idx} out of bounds [0, {TOTAL_ACTIONS-1}]"
        )

    return action_idx >= ACTION_TYPE_OFFSETS["erase"]


def get_action_type_from_index(action_idx: int) -> str:
    """
    Get the action type string from an action index.

    Args:
        action_idx: Action index (0-699)

    Returns:
        Action type string

    Raises:
        ValueError: If action index is out of bounds
    """
    action_info = decode_action(action_idx)
    return action_info.action_type


def validate_action_space() -> bool:
    """
    Validate that the action space encoding/decoding is consistent.

    Returns:
        True if all validations pass

    Raises:
        AssertionError: If any validation fails
    """
    # Test encoding/decoding round trip for all actions
    for action_idx in range(TOTAL_ACTIONS):
        action_info = decode_action(action_idx)
        reconstructed_idx = encode_action(
            action_info.action_type, action_info.x, action_info.y
        )
        assert (
            reconstructed_idx == action_idx
        ), f"Round trip failed for action {action_idx}"

    # Test all tile coordinates
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            tile_idx = get_tile_index(x, y)
            reconstructed_x, reconstructed_y = get_tile_coords(tile_idx)
            assert (reconstructed_x, reconstructed_y) == (
                x,
                y,
            ), f"Coordinate round trip failed for ({x}, {y})"

    # Test action type ranges
    expected_ranges = {
        "place_up": (0, 139),
        "place_down": (140, 279),
        "place_left": (280, 419),
        "place_right": (420, 559),
        "erase": (560, 699),
    }

    for action_type, (start, end) in expected_ranges.items():
        actions = get_actions_by_type(action_type)
        assert len(actions) == 140, f"Wrong number of actions for {action_type}"
        assert actions[0] == start, f"Wrong start index for {action_type}"
        assert actions[-1] == end, f"Wrong end index for {action_type}"

    print("Action space validation passed!")
    return True


# Validate action space dimensions
assert TOTAL_TILES == 140, f"Expected 140 tiles, got {TOTAL_TILES}"
assert (
    TOTAL_ACTIONS == 560 + 140
), f"Expected 700 actions (560 + 140), got {TOTAL_ACTIONS}"
assert (
    max(ACTION_TYPE_OFFSETS.values()) + TOTAL_TILES <= TOTAL_ACTIONS
), "Action space overflow"


# Run validation when module is imported
if __name__ == "__main__":
    validate_action_space()
