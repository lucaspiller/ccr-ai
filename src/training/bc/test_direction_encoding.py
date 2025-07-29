#!/usr/bin/env python3
"""
Integration test to verify direction encoding end-to-end.

Tests:
1. Action-ID → (x,y,dir) decoding with validation board
2. Label creation: CSV puzzle → encode → decode roundtrip
3. Direction ordering consistency across all components
"""

import ast
from pathlib import Path

import numpy as np
import torch

from ...game.board import Direction
from ...game.board_builder import BoardBuilder, BoardConfig
from ...perception.data_types import PerceptionConfig
from ...perception.processors import GameStateProcessor
from ...policy.processors import PolicyProcessor
from ...state_fusion.processors import StateFusionProcessor
from ...util.action_utils import decode_action, encode_action
from .config import BCConfig
from .data_loader import create_data_loaders, get_device
from .model_manager import ModelManager
from .viz_board import viz_board


def test_action_decoding_with_validation_board():
    """Test 1: Action-ID → (x,y,dir) decoding with validation board"""
    print("=" * 60)
    print("TEST 1: Action-ID → (x,y,dir) decoding with validation board")
    print("=" * 60)

    # Test board dimensions from CSV
    board_w, board_h = 8, 6

    # Create dummy logits for testing decoding (no model needed)
    dummy_logits = torch.randn(1, 700)

    # Get top 10 actions
    top_values, top_indices = torch.topk(dummy_logits, k=10, dim=1)

    print(f"Board dimensions: {board_w}×{board_h}")
    print(f"\nTop 10 action IDs and their decoded positions:")
    print("Rank | Action ID | x | y | Direction | Valid Position?")
    print("-" * 55)

    results = []
    for rank, action_id in enumerate(top_indices[0]):
        action_info = decode_action(action_id.item())
        x, y = action_info.x, action_info.y
        action_type = action_info.action_type

        # Check if position is valid for this board
        valid_pos = (0 <= x < board_w) and (0 <= y < board_h)

        print(
            f"{rank+1:4d} | {action_id.item():9d} | {x:1d} | {y:1d} | {action_type:10s} | {'YES' if valid_pos else 'NO'}"
        )

        results.append(
            {
                "rank": rank + 1,
                "action_id": action_id.item(),
                "x": x,
                "y": y,
                "action_type": action_type,
                "valid": valid_pos,
            }
        )

    # Create visualization with decoded actions overlaid
    print(f"\nCreating visualization with top actions overlaid...")

    # Create target tensor for visualization (mark top actions as positive)
    target_tensor = torch.zeros(700)
    for result in results[:5]:  # Mark top 5
        target_tensor[result["action_id"]] = 1.0

    # Create mask (all valid for this simple test)
    mask_tensor = torch.ones(700)

    # Generate visualization
    try:
        viz_board(
            dummy_logits.squeeze(0).cpu(),
            target_tensor,
            mask_tensor,
            board_w,
            board_h,
            title=f"Action Decoding Test - Board {board_w}×{board_h}",
            save_path="test_action_decoding.png",
        )
        print(f"Visualization saved to: test_action_decoding.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Continuing with other tests...")

    return results


def test_label_creation_roundtrip():
    """Test 2: Label creation roundtrip - CSV puzzle → encode → decode"""
    print("\n" + "=" * 60)
    print("TEST 2: Label creation roundtrip - CSV puzzle → encode → decode")
    print("=" * 60)

    # Test data from provided CSV row
    test_puzzle = {
        "seed": 1344,
        "board_w": 8,
        "board_h": 6,
        "num_walls": 3,
        "num_mice": 2,
        "num_rockets": 2,
        "num_cats": 2,
        "num_holes": 3,
        "arrow_budget": 3,
        "bfs_solution": "[[5,2,DOWN],[5,3,LEFT]]",
        "difficulty_label": "Medium",
    }

    print(f"Original BFS solution string: {test_puzzle['bfs_solution']}")

    # Parse the BFS solution - manually since it contains Direction enum references
    bfs_str = test_puzzle["bfs_solution"]
    print(f"Parsing BFS solution: {bfs_str}")

    # Replace Direction enum names with strings
    bfs_str_clean = (
        bfs_str.replace("DOWN", '"DOWN"')
        .replace("LEFT", '"LEFT"')
        .replace("UP", '"UP"')
        .replace("RIGHT", '"RIGHT"')
    )

    try:
        bfs_solution = ast.literal_eval(bfs_str_clean)
        print(f"Parsed BFS solution: {bfs_solution}")
    except Exception as e:
        print(f"Error parsing BFS solution: {e}")
        return False

    # Convert to Direction enum format
    direction_map = {
        "UP": Direction.UP,
        "DOWN": Direction.DOWN,
        "LEFT": Direction.LEFT,
        "RIGHT": Direction.RIGHT,
    }

    converted_solution = []
    for x, y, dir_str in bfs_solution:
        if dir_str in direction_map:
            converted_solution.append((x, y, direction_map[dir_str]))
        else:
            print(f"Warning: Unknown direction '{dir_str}'")

    print(f"Converted solution: {converted_solution}")

    # Encode each action to action ID
    encoded_actions = []
    for x, y, direction in converted_solution:
        try:
            action_type = f"place_{direction.name.lower()}"
            action_id = encode_action(action_type, x, y)
            encoded_actions.append(action_id)
            print(f"({x}, {y}, {direction.name}) → Action ID: {action_id}")
        except Exception as e:
            print(f"Error encoding ({x}, {y}, {direction.name}): {e}")

    print(f"Encoded action IDs: {encoded_actions}")

    # Create 700-hot vector
    target_vector = torch.zeros(700)
    for action_id in encoded_actions:
        if 0 <= action_id < 700:
            target_vector[action_id] = 1.0
        else:
            print(f"Warning: Action ID {action_id} out of range!")

    print(f"700-hot vector has {target_vector.sum().item()} positive entries")

    # Decode back to verify roundtrip
    decoded_actions = []
    positive_indices = torch.nonzero(target_vector, as_tuple=True)[0]

    for action_id in positive_indices:
        action_info = decode_action(action_id.item())
        decoded_actions.append((action_info.x, action_info.y, action_info.action_type))
        print(
            f"Action ID {action_id.item()} → ({action_info.x}, {action_info.y}, {action_info.action_type})"
        )

    print(f"Decoded actions: {decoded_actions}")

    # Verify roundtrip consistency
    print(f"\nROUNDTRIP VERIFICATION:")
    print(f"Original: {[(x, y, d.name) for x, y, d in converted_solution]}")
    print(f"Decoded:  {decoded_actions}")

    # Check if they match
    original_tuples = [
        (x, y, f"place_{d.name.lower()}") for x, y, d in converted_solution
    ]
    roundtrip_success = original_tuples == decoded_actions

    print(f"Roundtrip successful: {'YES' if roundtrip_success else 'NO'}")

    if not roundtrip_success:
        print("MISMATCH DETAILS:")
        for i, (orig, decoded) in enumerate(zip(original_tuples, decoded_actions)):
            if orig != decoded:
                print(f"  Position {i}: {orig} → {decoded}")

    return roundtrip_success


def test_direction_ordering_consistency():
    """Test 3: Verify direction ordering consistency"""
    print("\n" + "=" * 60)
    print("TEST 3: Direction ordering consistency")
    print("=" * 60)

    # Check Direction enum values
    print("Direction enum values:")
    directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    for direction in directions:
        print(f"  {direction.name}: {direction.value}")

    # Test encoding/decoding consistency for each direction
    test_x, test_y = 3, 2  # Test position
    board_w, board_h = 8, 6

    print(f"\nTesting encoding/decoding at position ({test_x}, {test_y}):")
    print("Direction | Enum Value | Action ID | Decoded Direction | Match?")
    print("-" * 65)

    all_consistent = True
    for direction in directions:
        try:
            # Encode
            action_type = f"place_{direction.name.lower()}"
            action_id = encode_action(action_type, test_x, test_y)

            # Decode
            decoded_info = decode_action(action_id)
            decoded_x, decoded_y = decoded_info.x, decoded_info.y
            decoded_type = decoded_info.action_type

            # Check consistency
            expected_type = f"place_{direction.name.lower()}"
            position_match = (decoded_x == test_x) and (decoded_y == test_y)
            type_match = decoded_type == expected_type

            consistent = position_match and type_match
            if not consistent:
                all_consistent = False

            print(
                f"{direction.name:9s} | {direction.value} | {action_id:9d} | {decoded_type:15s} | {'YES' if consistent else 'NO'}"
            )

            if not consistent:
                print(f"  Expected: ({test_x}, {test_y}, {expected_type})")
                print(f"  Got:      ({decoded_x}, {decoded_y}, {decoded_type})")

        except Exception as e:
            print(f"{direction.name:9s} | ERROR: {e}")
            all_consistent = False

    # Test action ID ranges
    print(f"\nAction ID ranges (for board {board_w}×{board_h}):")
    print("Action Type | Expected Range | Actual Sample IDs")
    print("-" * 50)

    action_types = [
        ("place_up", Direction.UP),
        ("place_down", Direction.DOWN),
        ("place_left", Direction.LEFT),
        ("place_right", Direction.RIGHT),
    ]

    for action_name, direction in action_types:
        sample_ids = []
        for x in range(min(3, board_w)):
            for y in range(min(3, board_h)):
                try:
                    action_id = encode_action(action_name, x, y)
                    sample_ids.append(action_id)
                except:
                    pass

        if sample_ids:
            min_id = min(sample_ids)
            max_id = max(sample_ids)
            print(f"{action_name:11s} | {min_id:6d}-{max_id:6d}     | {sample_ids[:5]}")

    print(
        f"\nOverall direction ordering consistent: {'YES' if all_consistent else 'NO'}"
    )
    return all_consistent


def main():
    """Run all direction encoding tests"""
    print("DIRECTION ENCODING INTEGRATION TESTS")
    print("=" * 60)

    # Run all tests
    test1_results = test_action_decoding_with_validation_board()
    test2_success = test_label_creation_roundtrip()
    test3_success = test_direction_ordering_consistency()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(f"Test 1 - Action decoding: Completed (check visualization)")
    print(f"Test 2 - Label roundtrip: {'PASSED' if test2_success else 'FAILED'}")
    print(f"Test 3 - Direction consistency: {'PASSED' if test3_success else 'FAILED'}")

    if test2_success and test3_success:
        print(f"\n✅ All critical tests PASSED - Direction encoding appears correct")
    else:
        print(f"\n❌ Some tests FAILED - Direction encoding has issues")

    return test2_success and test3_success


if __name__ == "__main__":
    main()
