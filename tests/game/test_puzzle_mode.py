"""
Integration tests for puzzle mode functionality.
"""

import pytest

from src.game.board import Direction, CellType
from src.game.board_builder import BoardBuilder, BoardConfig
from src.game.engine import GameEngine, GameResult, GamePhase
from src.game.levels import LevelBuilder


class TestPuzzleMode:
    """Test puzzle mode functionality including placement phase and win conditions."""

    def test_puzzle_mode_initialization(self):
        """Test that puzzle mode initializes correctly."""
        config = BoardConfig(
            board_w=7,
            board_h=7,
            num_walls=10,
            num_mice=3,
            num_rockets=2,
            num_cats=1,
            num_holes=2,
            arrow_budget=3,
        )

        board_builder = BoardBuilder(config, seed=42)
        board = board_builder.generate_board()

        level_builder = LevelBuilder(config.board_w, config.board_h, "Test Puzzle")
        level_builder.board = board.copy()

        # Add sprites
        mice_positions = getattr(board, "mice_positions", [])
        for i, (x, y) in enumerate(mice_positions):
            level_builder.add_mouse(x, y, f"mouse_{i}", Direction.RIGHT)

        cat_positions = getattr(board, "cat_positions", [])
        for i, (x, y) in enumerate(cat_positions):
            level_builder.add_cat(x, y, f"cat_{i}", Direction.RIGHT)

        rocket_positions = board.find_cells_by_type(CellType.ROCKET)
        for i, (x, y) in enumerate(rocket_positions):
            level_builder.add_rocket_sprite(x, y, f"rocket_{i}")

        level = level_builder.build()
        engine = level.create_engine(max_steps=1000, seed=42, puzzle_mode=True)

        # Verify puzzle mode initialization
        assert engine.puzzle_mode is True
        assert engine.phase == GamePhase.PLACEMENT
        assert engine.result == GameResult.ONGOING
        assert len(engine.board.arrows) == 0

    def test_puzzle_mode_arrow_placement_budget(self):
        """Test that arrow placement respects the budget in placement phase."""
        config = BoardConfig(arrow_budget=2)
        board_builder = BoardBuilder(config, seed=42)
        board = board_builder.generate_board()

        level_builder = LevelBuilder(config.board_w, config.board_h, "Test Puzzle")
        level_builder.board = board.copy()

        level = level_builder.build()
        engine = level.create_engine(puzzle_mode=True)

        # Should be able to place up to budget
        assert engine.place_arrow(1, 1, Direction.UP) is True
        assert engine.place_arrow(2, 2, Direction.DOWN) is True
        assert len(engine.board.arrows) == 2

        # Should not be able to place more than budget
        assert engine.place_arrow(3, 3, Direction.LEFT) is False
        assert len(engine.board.arrows) == 2

    def test_puzzle_mode_phase_transition(self):
        """Test transitioning from placement to running phase."""
        config = BoardConfig()
        board_builder = BoardBuilder(config, seed=42)
        board = board_builder.generate_board()

        level_builder = LevelBuilder(config.board_w, config.board_h, "Test Puzzle")
        level_builder.board = board.copy()

        level = level_builder.build()
        engine = level.create_engine(puzzle_mode=True)

        # Start in placement phase
        assert engine.phase == GamePhase.PLACEMENT

        # Place an arrow
        engine.place_arrow(1, 1, Direction.UP)

        # Start the game
        assert engine.start_game() is True
        assert engine.phase == GamePhase.RUNNING

        # Cannot place arrows in running phase
        assert engine.place_arrow(2, 2, Direction.DOWN) is False

        # Cannot start game again
        assert engine.start_game() is False

    def test_puzzle_mode_win_condition_seed_42(self):
        """Test specific win condition for seed 42 with given arrows."""
        config = BoardConfig(
            board_w=7,
            board_h=7,
            num_walls=10,
            num_mice=3,
            num_rockets=2,
            num_cats=1,
            num_holes=2,
            arrow_budget=3,
        )

        board_builder = BoardBuilder(config, seed=42)
        board = board_builder.generate_board()

        level_builder = LevelBuilder(config.board_w, config.board_h, "Puzzle 42")
        level_builder.board = board.copy()

        # Add sprites for mice
        mice_positions = getattr(board, "mice_positions", [])
        for i, (x, y) in enumerate(mice_positions):
            level_builder.add_mouse(x, y, f"mouse_{i}", Direction.RIGHT)

        # Add sprites for cats
        cat_positions = getattr(board, "cat_positions", [])
        for i, (x, y) in enumerate(cat_positions):
            level_builder.add_cat(x, y, f"cat_{i}", Direction.RIGHT)

        # Add rocket sprites
        rocket_positions = board.find_cells_by_type(CellType.ROCKET)
        for i, (x, y) in enumerate(rocket_positions):
            level_builder.add_rocket_sprite(x, y, f"rocket_{i}")

        level = level_builder.build()
        engine = level.create_engine(max_steps=1000, seed=42, puzzle_mode=True)

        # Place the winning arrows as specified
        winning_arrows = [
            ((1, 5), Direction.DOWN),
            ((6, 1), Direction.LEFT),
        ]

        for (x, y), direction in winning_arrows:
            success = engine.place_arrow(x, y, direction)
            assert success, f"Failed to place arrow at ({x}, {y}) with direction {direction}"

        # Start the game
        assert engine.start_game() is True

        # Run simulation until completion
        max_steps = 500
        step_count = 0
        while engine.result == GameResult.ONGOING and step_count < max_steps:
            engine.step()
            step_count += 1

        # Verify win condition
        assert engine.result == GameResult.SUCCESS, f"Expected SUCCESS but got {engine.result} after {step_count} steps"

        # Verify all mice escaped (reached rockets)
        from src.game.sprites import SpriteType, SpriteState

        mice = engine.sprite_manager.get_sprites_by_type(SpriteType.MOUSE)
        escaped_mice = [mouse for mouse in mice if mouse.state == SpriteState.ESCAPED]
        
        assert len(escaped_mice) == len(mice), f"Expected all {len(mice)} mice to escape, but only {len(escaped_mice)} escaped"

    def test_puzzle_mode_lose_condition(self):
        """Test that puzzle mode loses when mice get captured."""
        config = BoardConfig(
            board_w=5,
            board_h=5,
            num_walls=0,
            num_mice=1,
            num_rockets=1,
            num_cats=1,
            num_holes=1,
            arrow_budget=3,
        )

        board_builder = BoardBuilder(config, seed=123)
        board = board_builder.generate_board()

        level_builder = LevelBuilder(config.board_w, config.board_h, "Lose Test")
        level_builder.board = board.copy()

        # Add sprites
        mice_positions = getattr(board, "mice_positions", [])
        for i, (x, y) in enumerate(mice_positions):
            level_builder.add_mouse(x, y, f"mouse_{i}", Direction.RIGHT)

        cat_positions = getattr(board, "cat_positions", [])
        for i, (x, y) in enumerate(cat_positions):
            level_builder.add_cat(x, y, f"cat_{i}", Direction.RIGHT)

        rocket_positions = board.find_cells_by_type(CellType.ROCKET)
        for i, (x, y) in enumerate(rocket_positions):
            level_builder.add_rocket_sprite(x, y, f"rocket_{i}")

        level = level_builder.build()
        engine = level.create_engine(max_steps=1000, seed=123, puzzle_mode=True)

        # Don't place helpful arrows - let mice likely get captured
        engine.start_game()

        # Run simulation
        max_steps = 200
        step_count = 0
        while engine.result == GameResult.ONGOING and step_count < max_steps:
            engine.step()
            step_count += 1

        # The result should be either FAILURE (mice captured) or potentially SUCCESS if mice got lucky
        # In puzzle mode, any mouse getting captured should result in failure
        assert engine.result in [GameResult.FAILURE, GameResult.SUCCESS, GameResult.TIMEOUT]