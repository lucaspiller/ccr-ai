import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pygame

from .actions import PlaceArrowAction, RemoveArrowAction, WaitAction
from .board import CellType, Direction
from .engine import GameEngine, GamePhase, GameResult
from .sprites import BonusState, SpriteState, SpriteType


class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    DARK_GRAY = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    BROWN = (139, 69, 19)
    PINK = (255, 192, 203)


class VisualizationMode(Enum):
    AUTO = "auto"
    MANUAL = "manual"


class GameVisualizer:
    def __init__(self, engine: GameEngine, cell_size: int = 40, fps: int = 60):
        self.engine = engine
        self.cell_size = cell_size
        self.fps = fps

        pygame.init()

        # Add margin around the board
        self.board_margin = 20
        self.board_width = engine.board.width * cell_size
        self.board_height = engine.board.height * cell_size
        self.info_panel_width = 300

        # Add margin to window size
        self.window_width = (
            self.board_width + self.info_panel_width + (2 * self.board_margin)
        )
        self.window_height = max(self.board_height + (2 * self.board_margin), 600)

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("ChuChu Rocket Simulation")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        self.mode = VisualizationMode.AUTO
        self.step_delay = 16  # 16ms = 60 steps per second

        self.auto_step_timer = 0

        # Event logging
        self.last_event_count = 0
        self.event_logging_enabled = False

    def set_mode(self, mode: VisualizationMode) -> None:
        self.mode = mode

    def _log_new_events(self) -> None:
        """Log any new events that occurred since last check"""
        if not self.event_logging_enabled:
            return

        current_event_count = len(self.engine.events)
        if current_event_count > self.last_event_count:
            # Print new events
            new_events = self.engine.events[self.last_event_count :]
            for event in new_events:
                if event.event_type != "sprites_moved":
                    print(f"[Step {event.step}] {event.event_type}: {event.data}")
            self.last_event_count = current_event_count

    def run(self) -> None:
        running = True

        while running:
            dt = self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keypress(event.key)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos, event.button)

            # Handle automatic stepping for different modes
            if self.engine.result.value == "ongoing":
                # Only auto-step if not in puzzle placement phase
                if (
                    not self.engine.puzzle_mode
                    or self.engine.phase == GamePhase.RUNNING
                ):
                    if self.mode == VisualizationMode.AUTO:
                        # Auto mode - run at game speed
                        self.auto_step_timer += dt
                        if self.auto_step_timer >= self.step_delay:
                            self.engine.step()
                            self.auto_step_timer = 0
                    elif self.mode == VisualizationMode.MANUAL:
                        # In manual mode, only advance when spacebar is pressed
                        pass

            # Log any new events
            self._log_new_events()

            self._draw()

        pygame.quit()

    def _handle_keypress(self, key: int) -> bool:
        if key == pygame.K_ESCAPE or key == pygame.K_q:
            return False

        elif key == pygame.K_SPACE:
            if self.engine.result == GameResult.ONGOING:
                # In puzzle mode placement phase, don't step
                if not (
                    self.engine.puzzle_mode and self.engine.phase == GamePhase.PLACEMENT
                ):
                    self.engine.step()

        elif key == pygame.K_r:
            self.engine.reset()

        elif key == pygame.K_1:
            self.mode = VisualizationMode.AUTO
            print("Mode: Auto (real-time)")

        elif key == pygame.K_2:
            self.mode = VisualizationMode.MANUAL
            print("Mode: Manual (SPACE to advance)")

        elif key == pygame.K_d:
            # Dump game engine state
            print("=== GAME ENGINE STATE DUMP ===")
            import json

            state_dict = self.engine.to_dict()
            print(json.dumps(state_dict, indent=2, default=str))
            print("=== END STATE DUMP ===")

        elif key == pygame.K_l:
            # Toggle event logging
            self.event_logging_enabled = not self.event_logging_enabled
            status = "enabled" if self.event_logging_enabled else "disabled"
            print(f"Event logging {status}")

        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            # Start game in puzzle mode
            if (
                self.engine.puzzle_mode
                and self.engine.phase == GamePhase.PLACEMENT
                and self.engine.result == GameResult.ONGOING
            ):
                print("Arrows:")
                for (x, y), direction in self.engine.board.arrows.items():
                    print(f" [({x},{y}),{direction}]")
                success = self.engine.start_game()
                if success:
                    print("Puzzle started")

        return True

    def _handle_mouse_click(self, pos: Tuple[int, int], button: int) -> None:

        x, y = pos
        # Adjust for board margin
        x -= self.board_margin
        y -= self.board_margin

        if x < 0 or y < 0 or x >= self.board_width or y >= self.board_height:
            return

        board_x = x // self.cell_size
        board_y = y // self.cell_size

        if not self.engine.board.is_valid_position(board_x, board_y):
            return

        if button == 1:  # Left click - Place/Rotate arrow
            # Define rotation order: LEFT, UP, RIGHT, DOWN
            rotation_order = [
                Direction.LEFT,
                Direction.UP,
                Direction.RIGHT,
                Direction.DOWN,
            ]

            if self.engine.board.has_arrow(board_x, board_y):
                # Rotate existing arrow
                current_direction = self.engine.board.get_arrow_direction(
                    board_x, board_y
                )
                current_index = rotation_order.index(current_direction)
                new_direction = rotation_order[
                    (current_index + 1) % len(rotation_order)
                ]

                self.engine.remove_arrow(board_x, board_y)
                self.engine.place_arrow(board_x, board_y, new_direction)
            else:
                # Place new arrow, starting with LEFT
                action = PlaceArrowAction(board_x, board_y, Direction.LEFT)
                action.execute(self.engine)

        elif button == 3:  # Right click - Remove arrow
            if self.engine.board.has_arrow(board_x, board_y):
                action = RemoveArrowAction(board_x, board_y)
                action.execute(self.engine)

    def _draw(self) -> None:
        self.screen.fill(Colors.WHITE)

        self._draw_board()
        self._draw_arrows()
        self._draw_sprites()
        self._draw_info_panel()
        self._draw_controls()

        pygame.display.flip()

    def _draw_board(self) -> None:
        # Draw border walls around the entire board
        self._draw_border_walls()

        for y in range(self.engine.board.height):
            for x in range(self.engine.board.width):
                rect = pygame.Rect(
                    x * self.cell_size + self.board_margin,
                    y * self.cell_size + self.board_margin,
                    self.cell_size,
                    self.cell_size,
                )
                cell_type = self.engine.board.get_cell_type(x, y)
                if cell_type == CellType.EMPTY:
                    color = Colors.WHITE
                elif cell_type == CellType.HOLE:
                    color = Colors.BLACK
                elif cell_type == CellType.ROCKET:
                    color = Colors.YELLOW
                elif cell_type == CellType.SPAWNER:
                    color = Colors.LIGHT_GRAY
                elif cell_type == CellType.WALL:
                    color = Colors.DARK_GRAY
                else:
                    color = Colors.LIGHT_GRAY
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, Colors.GRAY, rect, 1)

        # Draw walls
        for a, b in self.engine.board.walls:
            (x1, y1), (x2, y2) = a, b

            def in_board(x, y):
                return (
                    0 <= x < self.engine.board.width
                    and 0 <= y < self.engine.board.height
                )

            # Internal walls: draw between edges of adjacent tiles
            if in_board(x1, y1) and in_board(x2, y2):
                if x1 == x2 and abs(y1 - y2) == 1:
                    # Horizontal wall between (x, y1) and (x, y2)
                    x = x1
                    y_top = min(y1, y2)
                    x0 = x * self.cell_size + self.board_margin
                    y0 = (y_top + 1) * self.cell_size + self.board_margin
                    x1_ = (x + 1) * self.cell_size + self.board_margin
                    y1_ = (y_top + 1) * self.cell_size + self.board_margin
                    pygame.draw.line(
                        self.screen, Colors.DARK_GRAY, (x0, y0), (x1_, y1_), 6
                    )
                elif y1 == y2 and abs(x1 - x2) == 1:
                    # Vertical wall between (x1, y) and (x2, y)
                    y = y1
                    x_left = min(x1, x2)
                    x0 = (x_left + 1) * self.cell_size + self.board_margin
                    y0 = y * self.cell_size + self.board_margin
                    x1_ = (x_left + 1) * self.cell_size + self.board_margin
                    y1_ = (y + 1) * self.cell_size + self.board_margin
                    pygame.draw.line(
                        self.screen, Colors.DARK_GRAY, (x0, y0), (x1_, y1_), 6
                    )

    def _draw_border_walls(self) -> None:
        """Draw border walls around the entire board"""
        wall_thickness = 8

        # Top wall
        pygame.draw.rect(
            self.screen,
            Colors.DARK_GRAY,
            (
                self.board_margin - wall_thickness // 2,
                self.board_margin - wall_thickness // 2,
                self.board_width + wall_thickness,
                wall_thickness,
            ),
        )

        # Bottom wall
        pygame.draw.rect(
            self.screen,
            Colors.DARK_GRAY,
            (
                self.board_margin - wall_thickness // 2,
                self.board_margin + self.board_height - wall_thickness // 2,
                self.board_width + wall_thickness,
                wall_thickness,
            ),
        )

        # Left wall
        pygame.draw.rect(
            self.screen,
            Colors.DARK_GRAY,
            (
                self.board_margin - wall_thickness // 2,
                self.board_margin - wall_thickness // 2,
                wall_thickness,
                self.board_height + wall_thickness,
            ),
        )

        # Right wall
        pygame.draw.rect(
            self.screen,
            Colors.DARK_GRAY,
            (
                self.board_margin + self.board_width - wall_thickness // 2,
                self.board_margin - wall_thickness // 2,
                wall_thickness,
                self.board_height + wall_thickness,
            ),
        )

    def _draw_sprites(self) -> None:
        for sprite in self.engine.sprite_manager.get_active_sprites():
            center_x = int(
                sprite.x * self.cell_size + self.cell_size // 2 + self.board_margin
            )
            center_y = int(
                sprite.y * self.cell_size + self.cell_size // 2 + self.board_margin
            )
            radius = self.cell_size // 3

            if sprite.get_sprite_type() == SpriteType.MOUSE:
                color = Colors.BROWN
            elif sprite.get_sprite_type() == SpriteType.GOLD_MOUSE:
                color = Colors.YELLOW  # Gold color for 50 mice
            elif sprite.get_sprite_type() == SpriteType.BONUS_MOUSE:
                color = Colors.PINK  # Pink color for bonus mice
            elif sprite.get_sprite_type() == SpriteType.CAT:
                color = Colors.ORANGE
            elif sprite.get_sprite_type() == SpriteType.ROCKET:
                color = Colors.RED
            elif sprite.get_sprite_type() == SpriteType.SPAWNER:
                color = Colors.BLUE
            else:
                color = Colors.PURPLE

            pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
            self._draw_direction_indicator(center_x, center_y, sprite.direction, radius)

            # Draw text overlay for special mouse types
            if sprite.get_sprite_type() == SpriteType.GOLD_MOUSE:
                self._draw_mouse_text_overlay(center_x, center_y, "50")
            elif sprite.get_sprite_type() == SpriteType.BONUS_MOUSE:
                self._draw_mouse_text_overlay(center_x, center_y, "?")

            # Draw mouse count on rockets
            if sprite.get_sprite_type() == SpriteType.ROCKET:
                self._draw_rocket_mouse_count(sprite, center_x, center_y)

    def _draw_direction_indicator(
        self, center_x: int, center_y: int, direction: Direction, radius: int
    ) -> None:
        offset = radius // 2
        end_x = center_x + direction.dx * offset
        end_y = center_y + direction.dy * offset

        # Draw main direction line
        pygame.draw.line(
            self.screen, Colors.WHITE, (center_x, center_y), (end_x, end_y), 3
        )

        # Draw arrowhead
        arrow_size = 4
        if direction == Direction.UP:
            points = [
                (end_x, end_y),
                (end_x - arrow_size, end_y + arrow_size),
                (end_x + arrow_size, end_y + arrow_size),
            ]
        elif direction == Direction.DOWN:
            points = [
                (end_x, end_y),
                (end_x - arrow_size, end_y - arrow_size),
                (end_x + arrow_size, end_y - arrow_size),
            ]
        elif direction == Direction.LEFT:
            points = [
                (end_x, end_y),
                (end_x + arrow_size, end_y - arrow_size),
                (end_x + arrow_size, end_y + arrow_size),
            ]
        elif direction == Direction.RIGHT:
            points = [
                (end_x, end_y),
                (end_x - arrow_size, end_y - arrow_size),
                (end_x - arrow_size, end_y + arrow_size),
            ]

        pygame.draw.polygon(self.screen, Colors.WHITE, points)

    def _draw_rocket_mouse_count(
        self, rocket_sprite, center_x: int, center_y: int
    ) -> None:
        """Draw the number of mice collected on top of the rocket"""
        count_text = str(rocket_sprite.mice_collected)
        text_surface = self.small_font.render(count_text, True, Colors.BLACK)
        text_rect = text_surface.get_rect()

        # Position text above the rocket sprite
        text_x = center_x - text_rect.width // 2
        text_y = center_y - self.cell_size // 2 - text_rect.height - 2

        # Draw white background for better readability
        bg_rect = pygame.Rect(
            text_x - 2, text_y - 2, text_rect.width + 4, text_rect.height + 4
        )
        pygame.draw.rect(self.screen, Colors.WHITE, bg_rect)
        pygame.draw.rect(self.screen, Colors.BLACK, bg_rect, 1)

        # Draw the text
        self.screen.blit(text_surface, (text_x, text_y))

    def _draw_mouse_text_overlay(self, center_x: int, center_y: int, text: str) -> None:
        """Draw text above a mouse sprite (for '50' or '?' overlays)"""
        text_surface = self.small_font.render(text, True, Colors.BLACK)
        text_rect = text_surface.get_rect()

        # Position text above the mouse sprite
        text_x = center_x - text_rect.width // 2
        text_y = center_y - self.cell_size // 2 - text_rect.height - 2

        # Draw white background for better readability
        bg_rect = pygame.Rect(
            text_x - 2, text_y - 2, text_rect.width + 4, text_rect.height + 4
        )
        pygame.draw.rect(self.screen, Colors.WHITE, bg_rect)
        pygame.draw.rect(self.screen, Colors.BLACK, bg_rect, 1)

        # Draw the text
        self.screen.blit(text_surface, (text_x, text_y))

    def _draw_arrows(self) -> None:
        for (x, y), direction in self.engine.board.arrows.items():
            center_x = x * self.cell_size + self.cell_size // 2 + self.board_margin
            center_y = y * self.cell_size + self.cell_size // 2 + self.board_margin
            radius = self.cell_size // 3

            # Draw blue square background
            square_size = radius * 2
            square_rect = pygame.Rect(
                center_x - square_size // 2,
                center_y - square_size // 2,
                square_size,
                square_size,
            )
            pygame.draw.rect(self.screen, Colors.BLUE, square_rect)
            pygame.draw.rect(self.screen, Colors.BLACK, square_rect, 2)

            # Draw direction indicator (similar to sprite direction indicators)
            self._draw_direction_indicator(center_x, center_y, direction, radius)

    def _draw_info_panel(self) -> None:
        panel_x = self.board_width + self.board_margin + 10
        y_offset = 10

        # Calculate timer display (countdown from max_steps to 0)
        remaining_steps = max(0, self.engine.max_steps - self.engine.current_step)
        # Assuming 60 ticks per second, convert remaining steps to minutes:seconds
        remaining_seconds = remaining_steps // 60
        mins = remaining_seconds // 60
        secs = remaining_seconds % 60
        timer_display = f"{mins}:{secs:02d}"

        # Bonus mode display
        bonus_info = ""
        if self.engine.bonus_state.is_active():
            remaining_secs = self.engine.bonus_state.remaining_ticks / 60
            bonus_info = f"BONUS: {self.engine.bonus_state.mode.value.replace('_', ' ').title()} ({remaining_secs:.1f}s)"

        # Puzzle mode specific info
        puzzle_info = []
        if self.engine.puzzle_mode:
            phase_display = self.engine.phase.value.title()
            arrows_remaining = max(
                0, self.engine.board.max_arrows - len(self.engine.board.arrows)
            )
            puzzle_info = [
                f"Phase: {phase_display}",
                f"Arrows: {len(self.engine.board.arrows)}/{self.engine.board.max_arrows}",
                f"Remaining: {arrows_remaining}",
            ]
            if self.engine.phase == GamePhase.PLACEMENT:
                puzzle_info.append("Press ENTER to start!")

        info_lines = [
            f"Step: {self.engine.current_step}",
            f"Timer: {timer_display}",
            f"Result: {self.engine.result.value}",
            f"Mode: {self.mode.value}",
            bonus_info if bonus_info else "",
        ]

        # Add puzzle info if in puzzle mode
        if puzzle_info:
            info_lines.extend([""] + puzzle_info)

        info_lines.extend(["", "Game Stats:"])

        stats = self.engine.get_game_stats()
        info_lines.extend(
            [
                f"  Mice Active: {stats['mice']['active']}",
                f"  Mice Captured: {stats['mice']['captured']}",
                f"  Mice Escaped: {stats['mice']['escaped']}",
                f"  Cats Active: {stats['cats']['active']}",
                f"  Rockets: {stats['rockets']['total']}",
                f"  Mice Collected: {stats['rockets']['mice_collected']}",
                f"  Arrows Placed: {stats['arrows_placed']}",
            ]
        )

        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, Colors.BLACK)
            self.screen.blit(text, (panel_x, y_offset + i * 20))

    def _draw_controls(self) -> None:
        panel_x = self.board_width + self.board_margin + 10
        y_offset = self.window_height - 200

        controls = [
            "Controls:",
            "SPACE - Step (manual mode)",
            "R - Reset",
            "1 - Auto mode",
            "2 - Manual mode",
            "D - Dump game state",
            "L - Toggle event logging",
            "ESC/Q - Quit",
            "",
            "Mouse:",
            "Left Click - Place/Rotate Arrow",
            "Right Click - Remove Arrow",
        ]

        # Add puzzle mode controls
        if self.engine.puzzle_mode:
            controls.extend(
                [
                    "",
                    "Puzzle Mode:",
                    "ENTER - Start Game",
                ]
            )

        for i, line in enumerate(controls):
            text = self.small_font.render(line, True, Colors.DARK_GRAY)
            self.screen.blit(text, (panel_x, y_offset + i * 15))

    def take_screenshot(self, filename: str) -> None:
        pygame.image.save(self.screen, filename)

    def close(self) -> None:
        pygame.quit()


class HeadlessVisualizer:
    def __init__(self, engine: GameEngine):
        self.engine = engine

    def print_board(self) -> None:
        print(f"\nStep {self.engine.current_step} - Result: {self.engine.result.value}")
        print(str(self.engine.board))

        sprites_info = []
        for sprite in self.engine.sprite_manager.sprites.values():
            sprites_info.append(
                f"{sprite.sprite_id}: {sprite.position} ({sprite.state.value})"
            )

        if sprites_info:
            print("Sprites:", ", ".join(sprites_info))

        stats = self.engine.get_game_stats()
        print(
            f"Stats: {stats['mice']['active']} mice active, "
            f"{stats['mice']['escaped']} escaped, "
            f"{stats['arrows_placed']} arrows placed"
        )

    def run_simulation(
        self, max_steps: int = 100, verbose: bool = True
    ) -> Dict[str, Any]:
        for step in range(max_steps):
            if self.engine.result.value != "ongoing":
                break

            self.engine.step()

            if verbose and step % 10 == 0:
                self.print_board()

        if verbose:
            print(f"\nSimulation completed in {self.engine.current_step} steps")
            print(f"Final result: {self.engine.result.value}")
            self.print_board()

        return {
            "result": self.engine.result.value,
            "steps": self.engine.current_step,
            "final_stats": self.engine.get_game_stats(),
        }
