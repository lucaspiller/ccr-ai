import pytest

from src.game.board import Board, CellType, Direction
from src.game.engine import GameEngine, GameResult
from src.game.sprites import SpriteManager, SpriteType


class TestGameEngine:
    def create_simple_engine(self):
        board = Board(5, 5)
        sprite_manager = SpriteManager()

        board.set_cell_type(4, 2, CellType.ROCKET)
        sprite_manager.create_mouse(1, 2, "mouse1")
        sprite_manager.create_rocket(4, 2, "rocket1")

        return GameEngine(board, sprite_manager, max_steps=100)

    def test_engine_creation(self):
        engine = self.create_simple_engine()

        assert engine.current_step == 0
        assert engine.result == GameResult.ONGOING
        assert engine.max_steps == 100
        assert len(engine.events) == 0

    def test_arrow_placement(self):
        engine = self.create_simple_engine()

        assert engine.place_arrow(2, 2, Direction.RIGHT)
        assert engine.board.has_arrow(2, 2)
        assert len(engine.events) == 1
        assert engine.events[0].event_type == "arrow_placed"

    def test_arrow_removal(self):
        engine = self.create_simple_engine()

        engine.place_arrow(2, 2, Direction.RIGHT)
        assert engine.remove_arrow(2, 2)
        assert not engine.board.has_arrow(2, 2)

        events = [e for e in engine.events if e.event_type == "arrow_removed"]
        assert len(events) == 1

    def test_invalid_arrow_placement(self):
        engine = self.create_simple_engine()

        # Invalid position (out of bounds)
        assert not engine.place_arrow(-1, 0, Direction.RIGHT)

        # Valid position with mouse - should be allowed now
        assert engine.place_arrow(1, 2, Direction.RIGHT)

        # Invalid position with rocket - should be blocked
        assert not engine.place_arrow(4, 2, Direction.RIGHT)

    def test_step_execution(self):
        engine = self.create_simple_engine()

        initial_step = engine.current_step
        result = engine.step()

        assert engine.current_step == initial_step + 1
        assert "step" in result
        assert "result" in result
        assert "board_state" in result

    def test_game_state_tracking(self):
        engine = self.create_simple_engine()

        board_state = engine.get_board_state()
        assert "grid" in board_state
        assert "arrows" in board_state

        sprite_states = engine.get_sprite_states()
        assert "mouse1" in sprite_states
        assert "rocket1" in sprite_states

        stats = engine.get_game_stats()
        assert "mice" in stats
        assert "cats" in stats
        assert "rockets" in stats

    def test_win_condition(self):
        board = Board(3, 3)
        sprite_manager = SpriteManager()

        board.set_cell_type(2, 1, CellType.ROCKET)
        mouse = sprite_manager.create_mouse(1, 1, "mouse1")
        rocket = sprite_manager.create_rocket(2, 1, "rocket1")

        engine = GameEngine(board, sprite_manager, max_steps=10)

        mouse.move_to(2, 1)
        rocket.collect_mouse(mouse)

        engine.step()

        # Win condition removed - game now continues until timeout
        assert engine.result == GameResult.ONGOING

    def test_timeout_condition(self):
        engine = self.create_simple_engine()
        engine.max_steps = 2

        engine.step()
        engine.step()
        engine.step()

        assert engine.result == GameResult.TIMEOUT

    def test_engine_reset(self):
        engine = self.create_simple_engine()

        # Store initial state
        initial_positions = {}
        for sprite in engine.sprite_manager.sprites.values():
            initial_positions[sprite.sprite_id] = sprite.position

        # Modify state
        engine.place_arrow(1, 1, Direction.UP)
        engine.step()
        engine.step()

        # Verify state has changed
        assert engine.current_step > 0
        assert len(engine.board.arrows) > 0
        assert len(engine.events) > 0

        # Reset
        engine.reset()

        # Verify reset worked
        assert engine.current_step == 0
        assert engine.result == GameResult.ONGOING
        assert len(engine.events) == 0
        assert len(engine.board.arrows) == 0  # Arrows should be cleared

        # Verify sprites are reset to initial positions and state
        for sprite in engine.sprite_manager.sprites.values():
            assert sprite.is_active()
            assert sprite.position == initial_positions[sprite.sprite_id]

    def test_engine_copy(self):
        engine = self.create_simple_engine()
        engine.place_arrow(1, 1, Direction.UP)
        engine.step()

        copy = engine.copy()

        assert copy.current_step == engine.current_step
        assert copy.result == engine.result
        assert copy.board.has_arrow(1, 1)
        assert len(copy.sprite_manager.sprites) == len(engine.sprite_manager.sprites)

        copy.step()
        assert copy.current_step != engine.current_step

    def test_valid_arrow_positions(self):
        engine = self.create_simple_engine()

        valid_positions = engine.get_valid_arrow_positions()

        assert (1, 1) in valid_positions
        # Position (1, 2) has a mouse, but should still be valid for arrow placement
        assert (1, 2) in valid_positions
        # Position (4, 2) has a rocket, so should be invalid
        assert (4, 2) not in valid_positions

    def test_step_callbacks(self):
        engine = self.create_simple_engine()
        callback_called = []

        def test_callback(eng):
            callback_called.append(eng.current_step)

        engine.add_step_callback(test_callback)
        engine.step()
        engine.step()

        assert len(callback_called) == 2
        assert callback_called == [1, 2]

        engine.remove_step_callback(test_callback)
        engine.step()

        assert len(callback_called) == 2

    def test_simulation_run(self):
        engine = self.create_simple_engine()

        result = engine.run_simulation(max_steps=5)

        assert "final_result" in result
        assert "total_steps" in result
        assert "events" in result
        assert "final_state" in result

        assert engine.current_step <= 5

    def test_serialization(self):
        engine = self.create_simple_engine()
        engine.place_arrow(1, 1, Direction.UP)
        engine.step()

        data = engine.to_dict()
        restored = GameEngine.from_dict(data)

        assert restored.current_step == engine.current_step
        assert restored.result == engine.result
        assert restored.max_steps == engine.max_steps
        assert restored.board.has_arrow(1, 1)


class TestGameEvents:
    def test_event_creation(self):
        from src.game.engine import GameEvent

        event = GameEvent("test_event", {"key": "value"}, 5)

        assert event.event_type == "test_event"
        assert event.data == {"key": "value"}
        assert event.step == 5
        assert event.timestamp > 0

    def test_event_serialization(self):
        from src.game.engine import GameEvent

        event = GameEvent("test_event", {"key": "value"}, 5)
        data = event.to_dict()

        assert data["event_type"] == "test_event"
        assert data["data"] == {"key": "value"}
        assert data["step"] == 5
        assert "timestamp" in data
