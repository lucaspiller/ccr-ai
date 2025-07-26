import numpy as np
import pytest

from src.game.board import Board, CellType, Direction


class TestBoard:
    def test_board_creation(self):
        board = Board(5, 3)
        assert board.width == 5
        assert board.height == 3
        assert board.grid.shape == (3, 5)
        assert np.all(board.grid == 0)

    def test_valid_position(self):
        board = Board(5, 3)
        assert board.is_valid_position(0, 0)
        assert board.is_valid_position(4, 2)
        assert not board.is_valid_position(-1, 0)
        assert not board.is_valid_position(5, 0)
        assert not board.is_valid_position(0, 3)

    def test_cell_type_operations(self):
        board = Board(3, 3)

        assert board.get_cell_type(1, 1) == CellType.EMPTY

        board.set_cell_type(1, 1, CellType.WALL)
        assert board.get_cell_type(1, 1) == CellType.WALL

        board.set_cell_type(2, 2, CellType.HOLE)
        assert board.get_cell_type(2, 2) == CellType.HOLE

    def test_walkable_cells(self):
        board = Board(3, 3)

        assert board.is_walkable(1, 1)

        board.set_cell_type(1, 1, CellType.WALL)
        assert not board.is_walkable(1, 1)

        board.set_cell_type(2, 2, CellType.ROCKET)
        assert board.is_walkable(2, 2)

    def test_arrow_operations(self):
        board = Board(3, 3)

        assert not board.has_arrow(1, 1)
        assert board.get_arrow_direction(1, 1) is None

        assert board.place_arrow(1, 1, Direction.UP)
        assert board.has_arrow(1, 1)
        assert board.get_arrow_direction(1, 1) == Direction.UP

        assert not board.place_arrow(1, 1, Direction.DOWN)

        assert board.remove_arrow(1, 1)
        assert not board.has_arrow(1, 1)
        assert not board.remove_arrow(1, 1)

    def test_arrow_placement_restrictions(self):
        board = Board(3, 3)
        board.set_cell_type(1, 1, CellType.WALL)

        assert not board.place_arrow(1, 1, Direction.UP)
        assert not board.place_arrow(-1, 0, Direction.UP)

    def test_neighbors(self):
        board = Board(3, 3)

        neighbors = board.get_neighbors(1, 1)
        expected = [(1, 0), (1, 2), (0, 1), (2, 1)]
        assert set(neighbors) == set(expected)

        corner_neighbors = board.get_neighbors(0, 0)
        expected_corner = [(0, 1), (1, 0)]
        assert set(corner_neighbors) == set(expected_corner)

    def test_walkable_neighbors(self):
        board = Board(3, 3)
        board.set_cell_type(1, 0, CellType.WALL)

        walkable = board.get_walkable_neighbors(1, 1)
        expected = [(1, 2), (0, 1), (2, 1)]
        assert set(walkable) == set(expected)

    def test_find_cells_by_type(self):
        board = Board(3, 3)
        board.set_cell_type(0, 0, CellType.WALL)
        board.set_cell_type(2, 2, CellType.WALL)

        walls = board.find_cells_by_type(CellType.WALL)
        assert set(walls) == {(0, 0), (2, 2)}

        empty = board.find_cells_by_type(CellType.EMPTY)
        assert len(empty) == 7

    def test_board_copy(self):
        board = Board(3, 3)
        board.set_cell_type(1, 1, CellType.WALL)
        board.place_arrow(2, 2, Direction.LEFT)

        board_copy = board.copy()

        assert board_copy.width == board.width
        assert board_copy.height == board.height
        assert board_copy.get_cell_type(1, 1) == CellType.WALL
        assert board_copy.has_arrow(2, 2)
        assert board_copy.get_arrow_direction(2, 2) == Direction.LEFT

        board_copy.set_cell_type(0, 0, CellType.HOLE)
        assert board.get_cell_type(0, 0) == CellType.EMPTY

    def test_serialization(self):
        board = Board(3, 3)
        board.set_cell_type(1, 1, CellType.WALL)
        board.place_arrow(2, 2, Direction.RIGHT)

        data = board.to_dict()
        restored_board = Board.from_dict(data)

        assert restored_board.width == board.width
        assert restored_board.height == board.height
        assert restored_board.get_cell_type(1, 1) == CellType.WALL
        assert restored_board.has_arrow(2, 2)
        assert restored_board.get_arrow_direction(2, 2) == Direction.RIGHT

    def test_string_representation(self):
        board = Board(3, 3)
        board.set_cell_type(0, 0, CellType.WALL)
        board.place_arrow(1, 1, Direction.UP)

        board_str = str(board)
        lines = board_str.split("\n")

        assert len(lines) == 3
        assert "#" in lines[0]
        assert "↑" in lines[1]


class TestDirection:
    def test_direction_properties(self):
        assert Direction.UP.dx == 0
        assert Direction.UP.dy == -1
        assert Direction.DOWN.dx == 0
        assert Direction.DOWN.dy == 1
        assert Direction.LEFT.dx == -1
        assert Direction.LEFT.dy == 0
        assert Direction.RIGHT.dx == 1
        assert Direction.RIGHT.dy == 0

    def test_opposite_directions(self):
        assert Direction.UP.opposite() == Direction.DOWN
        assert Direction.DOWN.opposite() == Direction.UP
        assert Direction.LEFT.opposite() == Direction.RIGHT
        assert Direction.RIGHT.opposite() == Direction.LEFT
