"""
Utilities for encoding and decoding actions in the policy head.
"""

from typing import Tuple, List
from .data_types import ActionInfo, ACTION_TYPE_OFFSETS, OFFSET_TO_ACTION_TYPE, BOARD_HEIGHT, BOARD_WIDTH, TOTAL_TILES


def get_tile_index(row: int, col: int) -> int:
    """
    Convert board coordinates to linear tile index.
    
    Args:
        row: Row coordinate (0-9)
        col: Column coordinate (0-13)
    
    Returns:
        Linear tile index (0-139)
    
    Raises:
        ValueError: If coordinates are out of bounds
    """
    if not (0 <= row < BOARD_HEIGHT):
        raise ValueError(f"Row {row} out of bounds [0, {BOARD_HEIGHT-1}]")
    
    if not (0 <= col < BOARD_WIDTH):
        raise ValueError(f"Column {col} out of bounds [0, {BOARD_WIDTH-1}]")
    
    return row * BOARD_WIDTH + col


def get_tile_coords(tile_idx: int) -> Tuple[int, int]:
    """
    Convert linear tile index to board coordinates.
    
    Args:
        tile_idx: Linear tile index (0-139)
    
    Returns:
        Tuple of (row, col) coordinates
    
    Raises:
        ValueError: If tile index is out of bounds
    """
    if not (0 <= tile_idx < TOTAL_TILES):
        raise ValueError(f"Tile index {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")
    
    row = tile_idx // BOARD_WIDTH
    col = tile_idx % BOARD_WIDTH
    return row, col


def encode_action(action_type: str, row: int, col: int) -> int:
    """
    Convert structured action to action index.
    
    Args:
        action_type: Type of action ("place_up", "place_down", "place_left", "place_right", "erase")
        row: Row coordinate (0-9)
        col: Column coordinate (0-13)
    
    Returns:
        Action index in 700-dimensional space
    
    Raises:
        ValueError: If action type is invalid or coordinates are out of bounds
    """
    if action_type not in ACTION_TYPE_OFFSETS:
        valid_types = list(ACTION_TYPE_OFFSETS.keys())
        raise ValueError(f"Invalid action_type '{action_type}'. Must be one of: {valid_types}")
    
    tile_idx = get_tile_index(row, col)
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
    if not (0 <= action_idx < 700):
        raise ValueError(f"Action index {action_idx} out of bounds [0, 699]")
    
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
        raise ValueError(f"Decoded tile index {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")
    
    # Get tile coordinates
    row, col = get_tile_coords(tile_idx)
    
    return ActionInfo(
        action_type=action_type,
        tile_row=row,
        tile_col=col,
        tile_idx=tile_idx,
        action_idx=action_idx
    )


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
        raise ValueError(f"Tile index {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")
    
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
        raise ValueError(f"Tile index {tile_idx} out of bounds [0, {TOTAL_TILES-1}]")
    
    return ACTION_TYPE_OFFSETS["erase"] + tile_idx


def get_actions_for_tile(row: int, col: int) -> List[int]:
    """
    Get all possible action indices for a specific tile.
    
    Args:
        row: Row coordinate (0-9)
        col: Column coordinate (0-13)
    
    Returns:
        List of 5 action indices (4 placements + 1 erase)
    
    Raises:
        ValueError: If coordinates are out of bounds
    """
    tile_idx = get_tile_index(row, col)
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
        raise ValueError(f"Invalid action_type '{action_type}'. Must be one of: {valid_types}")
    
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
    if not (0 <= action_idx < 700):
        raise ValueError(f"Action index {action_idx} out of bounds [0, 699]")
    
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
    if not (0 <= action_idx < 700):
        raise ValueError(f"Action index {action_idx} out of bounds [0, 699]")
    
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
    for action_idx in range(700):
        action_info = decode_action(action_idx)
        reconstructed_idx = encode_action(action_info.action_type, action_info.tile_row, action_info.tile_col)
        assert reconstructed_idx == action_idx, f"Round trip failed for action {action_idx}"
    
    # Test all tile coordinates
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            tile_idx = get_tile_index(row, col)
            reconstructed_row, reconstructed_col = get_tile_coords(tile_idx)
            assert (reconstructed_row, reconstructed_col) == (row, col), f"Coordinate round trip failed for ({row}, {col})"
    
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


# Run validation when module is imported
if __name__ == "__main__":
    validate_action_space()