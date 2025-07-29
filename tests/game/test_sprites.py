import pytest

from src.game.board import Direction
from src.game.sprites import Cat, Mouse, Rocket, SpriteManager, SpriteState, SpriteType


class TestSprites:
    def test_mouse_creation(self):
        mouse = Mouse(2, 3, "mouse1")
        assert mouse.x == 2
        assert mouse.y == 3
        assert mouse.sprite_id == "mouse1"
        assert mouse.position == (2, 3)
        assert mouse.state == SpriteState.ACTIVE
        assert mouse.get_sprite_type() == SpriteType.MOUSE

    def test_cat_creation(self):
        cat = Cat(1, 1, "cat1")
        assert cat.get_sprite_type() == SpriteType.CAT
        assert cat.is_active()

    def test_rocket_creation(self):
        rocket = Rocket(5, 5, "rocket1")
        assert rocket.get_sprite_type() == SpriteType.ROCKET
        assert rocket.mice_collected == 0

    def test_sprite_movement(self):
        mouse = Mouse(0, 0, "mouse1")

        mouse.move_to(3, 4)
        assert mouse.position == (3, 4)

        mouse.move_by(1, -1)
        assert mouse.position == (4, 3)

    def test_sprite_direction(self):
        mouse = Mouse(0, 0, "mouse1")
        assert mouse.direction == Direction.RIGHT

        mouse.set_direction(Direction.UP)
        assert mouse.direction == Direction.UP

    def test_mouse_capture(self):
        mouse = Mouse(0, 0, "mouse1")
        cat = Cat(0, 0, "cat1")

        assert mouse.can_be_captured_by_cat()
        assert cat.can_capture_mouse(mouse)

        assert cat.capture_mouse(mouse)
        assert mouse.state == SpriteState.CAPTURED
        assert not mouse.can_be_captured_by_cat()

    def test_rocket_collection(self):
        rocket = Rocket(0, 0, "rocket1")
        mouse = Mouse(0, 0, "mouse1")

        assert rocket.can_collect_mouse(mouse)
        assert rocket.collect_mouse(mouse)

        assert rocket.mice_collected == 1
        assert mouse.state == SpriteState.ESCAPED

    def test_rocket_infinite_capacity(self):
        rocket = Rocket(0, 0, "rocket1")

        # Rockets have infinite capacity in real ChuChu Rocket
        for i in range(10):  # Test with more mice
            mouse = Mouse(0, 0, f"mouse{i}")
            rocket.collect_mouse(mouse)

        assert rocket.mice_collected == 10

        extra_mouse = Mouse(0, 0, "extra")
        assert rocket.can_collect_mouse(extra_mouse)  # Can always collect more

    def test_sprite_serialization(self):
        mouse = Mouse(2, 3, "mouse1")
        mouse.set_direction(Direction.LEFT)
        mouse.set_state(SpriteState.CAPTURED)

        data = mouse.to_dict()
        restored = Mouse.from_dict(data)

        assert restored.position == mouse.position
        assert restored.sprite_id == mouse.sprite_id
        assert restored.direction == mouse.direction
        assert restored.state == mouse.state

    def test_sprite_copy(self):
        rocket = Rocket(1, 2, "rocket1")
        rocket.mice_collected = 2

        copy = rocket.copy()

        assert copy.position == rocket.position
        assert copy.mice_collected == rocket.mice_collected
        assert copy.sprite_id == rocket.sprite_id

        copy.move_to(5, 5)
        assert rocket.position == (1, 2)


class TestSpriteManager:
    def test_sprite_manager_creation(self):
        manager = SpriteManager()
        assert len(manager) == 0

    def test_add_remove_sprites(self):
        manager = SpriteManager()
        mouse = Mouse(0, 0, "mouse1")

        manager.add_sprite(mouse)
        assert len(manager) == 1
        assert manager.get_sprite("mouse1") == mouse

        assert manager.remove_sprite("mouse1")
        assert len(manager) == 0
        assert not manager.remove_sprite("nonexistent")

    def test_create_sprites(self):
        manager = SpriteManager()

        mouse = manager.create_mouse(1, 2)
        cat = manager.create_cat(3, 4)
        rocket = manager.create_rocket(5, 6)

        assert len(manager) == 3
        assert mouse.position == (1, 2)
        assert cat.position == (3, 4)
        assert rocket.position == (5, 6)

    def test_get_sprites_by_type(self):
        manager = SpriteManager()

        manager.create_mouse(0, 0)
        manager.create_mouse(1, 1)
        manager.create_cat(2, 2)

        mice = manager.get_sprites_by_type(SpriteType.MOUSE)
        cats = manager.get_sprites_by_type(SpriteType.CAT)

        assert len(mice) == 2
        assert len(cats) == 1

    def test_get_sprites_at_position(self):
        manager = SpriteManager()

        mouse1 = manager.create_mouse(1, 1)
        mouse2 = manager.create_mouse(1, 1)
        cat = manager.create_cat(2, 2)

        sprites_at_1_1 = manager.get_sprites_at_position(1, 1)
        sprites_at_2_2 = manager.get_sprites_at_position(2, 2)
        sprites_at_0_0 = manager.get_sprites_at_position(0, 0)

        assert len(sprites_at_1_1) == 2
        assert len(sprites_at_2_2) == 1
        assert len(sprites_at_0_0) == 0

    def test_get_active_sprites(self):
        manager = SpriteManager()

        mouse1 = manager.create_mouse(0, 0)
        mouse2 = manager.create_mouse(1, 1)
        mouse2.set_state(SpriteState.CAPTURED)

        active = manager.get_active_sprites()
        assert len(active) == 1
        assert active[0] == mouse1

    def test_sprite_manager_copy(self):
        manager = SpriteManager()
        mouse = manager.create_mouse(1, 2, "mouse1")

        copy = manager.copy()

        assert len(copy) == 1
        copy_mouse = copy.get_sprite("mouse1")
        assert copy_mouse.position == mouse.position

        copy_mouse.move_to(5, 5)
        assert mouse.position == (1, 2)

    def test_sprite_manager_serialization(self):
        manager = SpriteManager()
        manager.create_mouse(1, 2)
        manager.create_cat(3, 4)

        data = manager.to_dict()
        restored = SpriteManager.from_dict(data)

        assert len(restored) == 2
        mice = restored.get_sprites_by_type(SpriteType.MOUSE)
        cats = restored.get_sprites_by_type(SpriteType.CAT)

        assert len(mice) == 1
        assert len(cats) == 1

    def test_sprite_manager_iteration(self):
        manager = SpriteManager()
        manager.create_mouse(0, 0)
        manager.create_cat(1, 1)

        sprites = list(manager)
        assert len(sprites) == 2

    def test_clear(self):
        manager = SpriteManager()
        manager.create_mouse(0, 0)
        manager.create_cat(1, 1)

        manager.clear()
        assert len(manager) == 0
