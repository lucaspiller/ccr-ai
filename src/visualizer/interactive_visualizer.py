"""
Interactive dual-pane visualizer combining game simulation with model activation analysis.

Left pane: Game board with manual stepping
Right pane: Real-time model activation visualizations
"""

import io
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch

from src.game.board import CellType, Direction
from src.game.engine import GameEngine, GamePhase, GameResult
from src.game.sprites import SpriteType
from src.game.visualization import Colors, VisualizationMode
from src.model.model_loader import ModelLoader
from src.puzzle_generator.puzzle_generator import PuzzleGenerator

from .visualizer import ActivationCapture


class InteractiveModelVisualizer:
    """Interactive visualizer combining game simulation with model analysis."""

    def __init__(
        self,
        model_path: str,
        difficulty: str = "easy",
        seed: int = 42,
        device: str = "auto",
        cell_size: int = 40,
    ):
        """Initialize the interactive visualizer.

        Args:
            model_path: Path to model checkpoint
            difficulty: Puzzle difficulty level
            seed: Random seed for puzzle generation
            device: Device for model inference
            cell_size: Size of each board cell in pixels
        """
        self.cell_size = cell_size
        self.seed = seed
        self.difficulty = difficulty

        # Initialize pygame
        pygame.init()

        # Load model
        print(f"Loading model: {model_path}")
        self.model_loader = ModelLoader(model_path, device=device)
        self.device = self.model_loader.get_device()

        # Set up activation capture
        self.activation_capture = ActivationCapture()
        self.activation_capture.register_hooks(self.model_loader)

        # Generate puzzle and create game engine
        self.puzzle_generator = PuzzleGenerator()
        self._generate_new_puzzle()

        # Calculate layout dimensions
        self._setup_layout()

        # Initialize pygame display
        self.screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption("Interactive Model Visualizer")

        # Initialize surfaces for different panels
        self.game_surface = pygame.Surface(
            (self.game_panel_width, self.game_panel_height)
        )
        self.model_surface = pygame.Surface(
            (self.model_panel_width, self.model_panel_height)
        )

        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)

        # Visualization state
        self.show_model_panel = True
        self.current_cnn_channel = 0
        self.current_cnn_block = 4  # Start with block4 (most semantic)
        self.cnn_blocks = ["cnn_block1", "cnn_block2", "cnn_block3", "cnn_block4"]
        self.last_activations = None
        self.model_viz_cache = {}

        # Run initial model inference
        self._update_model_visualizations()

        print("Interactive visualizer initialized")
        print("Controls: SPACE=step, R=reset, ENTER=start puzzle, M=toggle model panel")
        print("          C=cycle CNN channels, B=switch CNN blocks")

    def _generate_new_puzzle(self):
        """Generate a new puzzle and create game engine."""
        specs = self.puzzle_generator.get_puzzle_specs()
        spec = specs[self.difficulty]

        import random

        rng = random.Random(self.seed)
        level = self.puzzle_generator.generate_level(spec, rng)

        self.level = level
        self.engine = level.create_engine(max_steps=1000, puzzle_mode=True)
        self.engine.start_game()  # Start in placement phase

        # Cache board dimensions
        self.board_width = self.engine.board.width
        self.board_height = self.engine.board.height

    def _setup_layout(self):
        """Calculate layout dimensions for dual-pane display."""
        # Game panel (left side)
        self.board_margin = 20
        self.game_board_width = self.board_width * self.cell_size
        self.game_board_height = self.board_height * self.cell_size
        self.info_panel_width = 250

        self.game_panel_width = (
            self.game_board_width + self.info_panel_width + (2 * self.board_margin)
        )
        self.game_panel_height = max(
            self.game_board_height + (2 * self.board_margin), 600
        )

        # Model panel (right side) - make it bigger to show all visualizations
        self.model_panel_width = 1000
        self.model_panel_height = max(1000, self.game_panel_height)

        # Total window size
        self.panel_separator = 10
        self.total_width = (
            self.game_panel_width + self.model_panel_width + self.panel_separator
        )
        self.total_height = self.model_panel_height

    def _update_model_visualizations(self):
        """Run model inference and update visualization cache."""
        if self.engine.result != GameResult.ONGOING:
            return

        # Clear previous activations
        self.activation_capture.clear()

        # Get current game state
        game_state = self.engine.to_dict()

        # Process through perception
        perception_output = self.model_loader.perception_processor.process(game_state)

        # Get fused embedding
        fused_state_output = self.model_loader.state_fusion_processor.fuse(
            perception_output
        )
        fused_embedding = fused_state_output.fused_embedding

        # Run through policy head to capture activations
        with torch.no_grad():
            policy_logits = self.model_loader.policy_processor.policy_head(
                fused_embedding.unsqueeze(0)  # Add batch dim
            ).squeeze(
                0
            )  # Remove batch dim

        # Store results
        self.last_activations = {
            "game_state": game_state,
            "perception_output": perception_output,
            "fused_embedding": fused_embedding,
            "policy_logits": policy_logits,
            "activations": self.activation_capture.activations.copy(),
        }

        # Update visualization cache
        self._generate_model_visualizations()

    def _generate_model_visualizations(self):
        """Generate matplotlib visualizations and convert to pygame surfaces."""
        if not self.last_activations:
            return

        self.model_viz_cache.clear()

        # 1. Policy overlay
        self.model_viz_cache["policy"] = self._create_policy_overlay()

        # 2. CNN feature maps (current channel)
        if "cnn_block4" in self.last_activations["activations"]:
            self.model_viz_cache["cnn"] = self._create_cnn_visualization()

        # 3. Cat embeddings (if available)
        if "cat_embeddings_raw" in self.last_activations["activations"]:
            cat_embeddings = self.last_activations["activations"]["cat_embeddings_raw"]
            if cat_embeddings.size(0) > 0:
                self.model_viz_cache["cats"] = self._create_cat_visualization()

    def _create_policy_overlay(self):
        """Create policy overlay visualization as pygame surface."""
        policy_logits = self.last_activations["policy_logits"]
        prob = torch.sigmoid(policy_logits).cpu()

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))

        # Create heatmap: max probability across all 4 arrow directions for each tile
        from src.util.action_utils import (ACTION_TYPE_OFFSETS, BOARD_HEIGHT,
                                           BOARD_WIDTH, get_tile_index)

        heat = torch.zeros(BOARD_HEIGHT, BOARD_WIDTH)
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                tile_idx = get_tile_index(x, y)

                # Get probabilities for 4 arrow directions at this tile
                arrow_probs = [
                    prob[ACTION_TYPE_OFFSETS["place_up"] + tile_idx],  # up
                    prob[ACTION_TYPE_OFFSETS["place_down"] + tile_idx],  # down
                    prob[ACTION_TYPE_OFFSETS["place_left"] + tile_idx],  # left
                    prob[ACTION_TYPE_OFFSETS["place_right"] + tile_idx],  # right
                ]
                heat[y, x] = max(arrow_probs)

        # Show heatmap
        im = ax.imshow(
            heat.numpy(), cmap="hot", interpolation="nearest", vmin=0, vmax=1
        )

        # Draw boundary around actual puzzle area
        from matplotlib.patches import Rectangle

        puzzle_rect = Rectangle(
            (-0.5, -0.5),
            self.board_width,
            self.board_height,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(puzzle_rect)

        ax.set_title(f"Policy Overlay (Max: {prob.max():.3f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Convert to pygame surface
        surface = self._fig_to_surface(fig)
        plt.close(fig)
        return surface

    def _create_cnn_visualization(self):
        """Create CNN feature map visualization showing multiple channels as pygame surface."""
        current_block_name = self.cnn_blocks[self.current_cnn_block - 1]

        if current_block_name not in self.last_activations["activations"]:
            # Fallback to available block
            for block_name in self.cnn_blocks:
                if block_name in self.last_activations["activations"]:
                    current_block_name = block_name
                    self.current_cnn_block = self.cnn_blocks.index(block_name) + 1
                    break

        cnn_features = self.last_activations["activations"][current_block_name]
        if cnn_features.dim() == 4:  # [1, C, H, W]
            cnn_features = cnn_features.squeeze(0)

        num_channels = cnn_features.size(0)

        # Show 8 channels at once in a 2x4 grid
        channels_to_show = 8
        start_channel = (
            self.current_cnn_channel // channels_to_show
        ) * channels_to_show

        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(6, 4))
        fig.suptitle(
            f"CNN Block {self.current_cnn_block} - Channels {start_channel}-{start_channel+channels_to_show-1} of {num_channels-1}"
        )

        import torch.nn.functional as F

        for i, ax in enumerate(axes.flat):
            channel_idx = (start_channel + i) % num_channels

            # Get channel features
            channel_features = cnn_features[channel_idx].numpy()

            # Upsample to board size using nearest neighbor
            channel_upsampled = (
                F.interpolate(
                    torch.from_numpy(channel_features).unsqueeze(0).unsqueeze(0),
                    size=(self.board_height, self.board_width),
                    mode="nearest",
                )
                .squeeze()
                .numpy()
            )

            # Normalize 0-1
            if channel_upsampled.max() > channel_upsampled.min():
                channel_upsampled = (channel_upsampled - channel_upsampled.min()) / (
                    channel_upsampled.max() - channel_upsampled.min()
                )

            # Show heatmap
            im = ax.imshow(channel_upsampled, cmap="hot", alpha=0.8)
            ax.set_title(f"Ch {channel_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        # Convert to pygame surface
        surface = self._fig_to_surface(fig)
        plt.close(fig)
        return surface

    def _create_cat_visualization(self):
        """Create cat embeddings visualization as pygame surface."""
        cat_embeddings = self.last_activations["activations"]["cat_embeddings_raw"]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

        # Left plot: Raw embeddings heatmap
        im1 = ax1.imshow(cat_embeddings.T.numpy(), cmap="RdBu_r", aspect="auto")
        ax1.set_xlabel("Cat ID")
        ax1.set_ylabel("Embedding Dim")
        ax1.set_title("Cat Embeddings")

        # Right plot: Max-pooling winners
        max_values, max_indices = torch.max(cat_embeddings, dim=0)  # [32]

        colors = plt.cm.tab10(max_indices.numpy() % 10)
        ax2.bar(range(32), max_values.numpy(), color=colors)
        ax2.set_xlabel("Embedding Dim")
        ax2.set_ylabel("Max Value")
        ax2.set_title("Max-Pool Winners")

        plt.tight_layout()

        # Convert to pygame surface
        surface = self._fig_to_surface(fig)
        plt.close(fig)
        return surface

    def _fig_to_surface(self, fig):
        """Convert matplotlib figure to pygame surface."""
        # Save figure to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)

        # Load as pygame surface
        surface = pygame.image.load(buf)
        buf.close()

        return surface

    def _draw_game_panel(self):
        """Draw the game visualization panel."""
        self.game_surface.fill(Colors.WHITE)

        # Draw board
        self._draw_board()
        self._draw_arrows()
        self._draw_sprites()
        self._draw_game_info()

    def _draw_board(self):
        """Draw the game board on the game surface."""
        # Draw border walls
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
                pygame.draw.rect(self.game_surface, color, rect)
                pygame.draw.rect(self.game_surface, Colors.GRAY, rect, 1)

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
                    # Horizontal wall
                    x = x1
                    y_top = min(y1, y2)
                    x0 = x * self.cell_size + self.board_margin
                    y0 = (y_top + 1) * self.cell_size + self.board_margin
                    x1_ = (x + 1) * self.cell_size + self.board_margin
                    y1_ = (y_top + 1) * self.cell_size + self.board_margin
                    pygame.draw.line(
                        self.game_surface, Colors.DARK_GRAY, (x0, y0), (x1_, y1_), 6
                    )
                elif y1 == y2 and abs(x1 - x2) == 1:
                    # Vertical wall
                    y = y1
                    x_left = min(x1, x2)
                    x0 = (x_left + 1) * self.cell_size + self.board_margin
                    y0 = y * self.cell_size + self.board_margin
                    x1_ = (x_left + 1) * self.cell_size + self.board_margin
                    y1_ = (y + 1) * self.cell_size + self.board_margin
                    pygame.draw.line(
                        self.game_surface, Colors.DARK_GRAY, (x0, y0), (x1_, y1_), 6
                    )

    def _draw_border_walls(self):
        """Draw border walls around the board."""
        wall_thickness = 8

        # Top wall
        pygame.draw.rect(
            self.game_surface,
            Colors.DARK_GRAY,
            (
                self.board_margin - wall_thickness // 2,
                self.board_margin - wall_thickness // 2,
                self.game_board_width + wall_thickness,
                wall_thickness,
            ),
        )

        # Bottom wall
        pygame.draw.rect(
            self.game_surface,
            Colors.DARK_GRAY,
            (
                self.board_margin - wall_thickness // 2,
                self.board_margin + self.game_board_height - wall_thickness // 2,
                self.game_board_width + wall_thickness,
                wall_thickness,
            ),
        )

        # Left wall
        pygame.draw.rect(
            self.game_surface,
            Colors.DARK_GRAY,
            (
                self.board_margin - wall_thickness // 2,
                self.board_margin - wall_thickness // 2,
                wall_thickness,
                self.game_board_height + wall_thickness,
            ),
        )

        # Right wall
        pygame.draw.rect(
            self.game_surface,
            Colors.DARK_GRAY,
            (
                self.board_margin + self.game_board_width - wall_thickness // 2,
                self.board_margin - wall_thickness // 2,
                wall_thickness,
                self.game_board_height + wall_thickness,
            ),
        )

    def _draw_sprites(self):
        """Draw sprites on the game surface."""
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
                color = Colors.YELLOW
            elif sprite.get_sprite_type() == SpriteType.BONUS_MOUSE:
                color = Colors.PINK
            elif sprite.get_sprite_type() == SpriteType.CAT:
                color = Colors.ORANGE
            elif sprite.get_sprite_type() == SpriteType.ROCKET:
                color = Colors.RED
            elif sprite.get_sprite_type() == SpriteType.SPAWNER:
                color = Colors.BLUE
            else:
                color = Colors.PURPLE

            pygame.draw.circle(self.game_surface, color, (center_x, center_y), radius)
            self._draw_direction_indicator(center_x, center_y, sprite.direction, radius)

    def _draw_direction_indicator(
        self, center_x: int, center_y: int, direction: Direction, radius: int
    ):
        """Draw direction indicator for sprites."""
        offset = radius // 2
        end_x = center_x + direction.dx * offset
        end_y = center_y + direction.dy * offset

        # Draw main direction line
        pygame.draw.line(
            self.game_surface, Colors.WHITE, (center_x, center_y), (end_x, end_y), 3
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

        pygame.draw.polygon(self.game_surface, Colors.WHITE, points)

    def _draw_arrows(self):
        """Draw placed arrows on the game surface."""
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
            pygame.draw.rect(self.game_surface, Colors.BLUE, square_rect)
            pygame.draw.rect(self.game_surface, Colors.BLACK, square_rect, 2)

            # Draw direction indicator
            self._draw_direction_indicator(center_x, center_y, direction, radius)

    def _draw_game_info(self):
        """Draw game information panel."""
        panel_x = self.game_board_width + self.board_margin + 10
        y_offset = 10

        # Game status
        info_lines = [
            f"Step: {self.engine.current_step}",
            f"Result: {self.engine.result.value}",
            f"Phase: {self.engine.phase.value.title()}",
            "",
            f"Board: {self.board_width}x{self.board_height}",
            f"Difficulty: {self.difficulty.title()}",
            f"Seed: {self.seed}",
            "",
        ]

        # Puzzle info
        if self.engine.puzzle_mode:
            arrows_remaining = max(
                0, self.engine.board.max_arrows - len(self.engine.board.arrows)
            )
            info_lines.extend(
                [
                    f"Arrows: {len(self.engine.board.arrows)}/{self.engine.board.max_arrows}",
                    f"Remaining: {arrows_remaining}",
                    "",
                ]
            )
            if self.engine.phase == GamePhase.PLACEMENT:
                info_lines.append("Press ENTER to start!")

        # Game stats
        stats = self.engine.get_game_stats()
        info_lines.extend(
            [
                "Game Stats:",
                f"  Mice Active: {stats['mice']['active']}",
                f"  Mice Captured: {stats['mice']['captured']}",
                f"  Mice Escaped: {stats['mice']['escaped']}",
                f"  Cats Active: {stats['cats']['active']}",
                "",
                "Controls:",
                "SPACE - Step",
                "R - Reset",
                "ENTER - Start puzzle",
                "M - Toggle model panel",
                "C - Cycle CNN channels",
                "B - Switch CNN blocks",
                "Mouse - Place/remove arrows",
                "ESC/Q - Quit",
            ]
        )

        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, Colors.BLACK)
            self.game_surface.blit(text, (panel_x, y_offset + i * 18))

    def _draw_model_panel(self):
        """Draw the model visualization panel."""
        if not self.show_model_panel:
            self.model_surface.fill(Colors.LIGHT_GRAY)
            text = self.large_font.render(
                "Model Panel Hidden (M to show)", True, Colors.DARK_GRAY
            )
            text_rect = text.get_rect(
                center=(self.model_panel_width // 2, self.model_panel_height // 2)
            )
            self.model_surface.blit(text, text_rect)
            return

        self.model_surface.fill(Colors.WHITE)

        # Draw title and instructions
        title = self.font.render("Model Activations", True, Colors.BLACK)
        self.model_surface.blit(title, (10, 10))

        instructions1 = self.small_font.render(
            "Press C to cycle CNN channels, B to switch blocks", True, Colors.DARK_GRAY
        )
        self.model_surface.blit(instructions1, (10, 35))

        instructions2 = self.small_font.render(
            f"Current: Block {self.current_cnn_block}", True, Colors.DARK_GRAY
        )
        self.model_surface.blit(instructions2, (10, 50))

        if not self.last_activations:
            text = self.font.render("No activations available", True, Colors.DARK_GRAY)
            self.model_surface.blit(text, (10, 70))
            return

        # Draw visualizations in organized layout
        y_offset = 75
        spacing = 15

        # Policy overlay (top left)
        if "policy" in self.model_viz_cache:
            self.model_surface.blit(self.model_viz_cache["policy"], (10, y_offset))
            y_offset += self.model_viz_cache["policy"].get_height() + spacing

        # CNN features (takes up most space)
        if "cnn" in self.model_viz_cache:
            self.model_surface.blit(self.model_viz_cache["cnn"], (10, y_offset))
            y_offset += self.model_viz_cache["cnn"].get_height() + spacing

        # Cat embeddings (bottom if space allows)
        if "cats" in self.model_viz_cache:
            remaining_space = self.model_panel_height - y_offset
            if remaining_space > 250:  # Check if we have enough space
                self.model_surface.blit(self.model_viz_cache["cats"], (10, y_offset))

        # Add status text at bottom
        status_y = self.model_panel_height - 40
        if self.last_activations:
            max_prob = (
                torch.sigmoid(self.last_activations["policy_logits"]).max().item()
            )
            status_text = (
                f"Step: {self.engine.current_step} | Max Policy Prob: {max_prob:.3f}"
            )
            status_surface = self.small_font.render(status_text, True, Colors.DARK_GRAY)
            self.model_surface.blit(status_surface, (10, status_y))

    def _handle_keypress(self, key: int) -> bool:
        """Handle keyboard input."""
        if key == pygame.K_ESCAPE or key == pygame.K_q:
            return False

        elif key == pygame.K_SPACE:
            if self.engine.result == GameResult.ONGOING:
                if not (
                    self.engine.puzzle_mode and self.engine.phase == GamePhase.PLACEMENT
                ):
                    self.engine.step()
                    self._update_model_visualizations()

        elif key == pygame.K_r:
            self.engine.reset()
            self._update_model_visualizations()

        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            if (
                self.engine.puzzle_mode
                and self.engine.phase == GamePhase.PLACEMENT
                and self.engine.result == GameResult.ONGOING
            ):
                success = self.engine.start_game()
                if success:
                    print("Puzzle started")
                    self._update_model_visualizations()

        elif key == pygame.K_m:
            self.show_model_panel = not self.show_model_panel
            print(f"Model panel: {'shown' if self.show_model_panel else 'hidden'}")

        elif key == pygame.K_c:
            # Cycle CNN channels within current block
            current_block_name = self.cnn_blocks[self.current_cnn_block - 1]
            if current_block_name in self.last_activations.get("activations", {}):
                cnn_features = self.last_activations["activations"][current_block_name]
                if cnn_features.dim() == 4:
                    cnn_features = cnn_features.squeeze(0)
                num_channels = cnn_features.size(0)

                # Advance by 8 channels at a time (since we show 8 in grid)
                self.current_cnn_channel = (self.current_cnn_channel + 8) % num_channels
                self._generate_model_visualizations()  # Regenerate with new channel
                print(
                    f"CNN Block {self.current_cnn_block} channels: {self.current_cnn_channel}-{self.current_cnn_channel+7} of {num_channels-1}"
                )

        elif key == pygame.K_b:
            # Switch CNN blocks
            self.current_cnn_block = (self.current_cnn_block % 4) + 1
            self.current_cnn_channel = 0  # Reset channel when switching blocks
            self._generate_model_visualizations()  # Regenerate with new block
            print(f"Switched to CNN Block {self.current_cnn_block}")

        return True

    def _handle_mouse_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks for arrow placement."""
        x, y = pos

        # Check if click is in game panel
        if x > self.game_panel_width:
            return

        # Adjust for board margin
        x -= self.board_margin
        y -= self.board_margin

        if x < 0 or y < 0 or x >= self.game_board_width or y >= self.game_board_height:
            return

        board_x = x // self.cell_size
        board_y = y // self.cell_size

        if not self.engine.board.is_valid_position(board_x, board_y):
            return

        if button == 1:  # Left click - Place/Rotate arrow
            from src.game.actions import PlaceArrowAction

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
                # Place new arrow
                action = PlaceArrowAction(board_x, board_y, Direction.LEFT)
                action.execute(self.engine)

            # Update model after arrow placement
            self._update_model_visualizations()

        elif button == 3:  # Right click - Remove arrow
            if self.engine.board.has_arrow(board_x, board_y):
                from src.game.actions import RemoveArrowAction

                action = RemoveArrowAction(board_x, board_y)
                action.execute(self.engine)

                # Update model after arrow removal
                self._update_model_visualizations()

    def run(self):
        """Run the interactive visualizer."""
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keypress(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos, event.button)

            # Draw panels
            self._draw_game_panel()
            self._draw_model_panel()

            # Composite to main screen
            self.screen.blit(self.game_surface, (0, 0))
            if self.show_model_panel:
                self.screen.blit(
                    self.model_surface,
                    (self.game_panel_width + self.panel_separator, 0),
                )

            # Draw separator line
            separator_x = self.game_panel_width + self.panel_separator // 2
            pygame.draw.line(
                self.screen,
                Colors.DARK_GRAY,
                (separator_x, 0),
                (separator_x, self.total_height),
                2,
            )

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        # Cleanup
        self.activation_capture.remove_hooks()
        pygame.quit()
