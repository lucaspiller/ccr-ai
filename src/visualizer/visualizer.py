"""
Model activation visualizer for ChuChu Rocket neural network.

Implements the visualization strategy from visualizer_prd.md:
- CNN feature map heatmaps
- Policy logits as arrow overlays
- Cat encoder activations
- Combined multi-layer visualizations
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from src.game.board import CellType, Direction
from src.game.sprites import SpriteType
from src.model.model_loader import ModelLoader
from src.puzzle_generator.puzzle_generator import PuzzleGenerator
from src.util.action_utils import (ACTION_TYPE_OFFSETS, BOARD_HEIGHT,
                                   BOARD_WIDTH, decode_action, get_tile_index)


class ActivationCapture:
    """Captures and stores activations from forward hooks."""

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def save_tensor(self, name: str):
        """Create hook function to save tensor activations."""

        def _hook(module, input, output):
            self.activations[name] = output.detach().cpu()

        return _hook

    def register_hooks(self, model_loader: ModelLoader):
        """Register forward hooks on key model components."""
        # Get model components
        perception = model_loader.perception_processor
        fusion = model_loader.state_fusion_processor
        policy = model_loader.policy_processor

        # CNN encoder hooks (get conv blocks)
        cnn_encoder = perception.get_cnn_encoder()

        # Hook conv blocks: indices 2, 5, 8, 11 are after each ReLU
        self.hooks.append(
            cnn_encoder.conv_layers[2].register_forward_hook(
                self.save_tensor("cnn_block1")
            )
        )
        self.hooks.append(
            cnn_encoder.conv_layers[5].register_forward_hook(
                self.save_tensor("cnn_block2")
            )
        )
        self.hooks.append(
            cnn_encoder.conv_layers[8].register_forward_hook(
                self.save_tensor("cnn_block3")
            )
        )
        self.hooks.append(
            cnn_encoder.conv_layers[11].register_forward_hook(
                self.save_tensor("cnn_block4")
            )
        )

        # Cat encoder hook (before max pooling)
        cat_encoder = perception.cat_processor.encoder
        self.hooks.append(
            cat_encoder.cat_mlp.register_forward_hook(
                self.save_tensor("cat_embeddings_raw")
            )
        )

        # Fusion MLP hook
        self.hooks.append(
            fusion.fusion_mlp.register_forward_hook(self.save_tensor("fused_latent"))
        )

        # Policy head hook (pre-softmax logits)
        self.hooks.append(
            policy.policy_head.layer2.register_forward_hook(
                self.save_tensor("policy_logits")
            )
        )

    def clear(self):
        """Clear captured activations."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ModelVisualizer:
    """Main visualizer class for model activations."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize visualizer with model."""
        self.model_loader = ModelLoader(model_path, device=device)
        self.device = self.model_loader.get_device()

        # Initialize components
        self.puzzle_generator = PuzzleGenerator()
        self.activation_capture = ActivationCapture()

        # Register hooks
        self.activation_capture.register_hooks(self.model_loader)

        print(f"Visualizer initialized with model: {model_path}")
        print(f"Device: {self.device}")

    def generate_random_puzzle(self, difficulty: str, seed: int):
        """Generate a random puzzle for visualization."""
        rng = random.Random(seed)

        specs = self.puzzle_generator.get_puzzle_specs()
        spec = specs[difficulty]

        level = self.puzzle_generator.generate_level(spec, rng)
        return level

    def run_inference(self, level) -> Dict[str, torch.Tensor]:
        """Run model inference and capture activations."""
        # Clear previous activations
        self.activation_capture.clear()

        # Get game state
        engine = level.create_engine(max_steps=1000, puzzle_mode=True)
        engine.start_game()
        game_state = engine.to_dict()

        # Process through perception
        perception_output = self.model_loader.perception_processor.process(game_state)

        # Get fused embedding
        fused_state_output = self.model_loader.state_fusion_processor.fuse(
            perception_output
        )
        fused_embedding = fused_state_output.fused_embedding

        # Run through policy head
        with torch.no_grad():
            policy_logits = self.model_loader.policy_processor.policy_head(
                fused_embedding.unsqueeze(0)  # Add batch dim
            ).squeeze(
                0
            )  # Remove batch dim

        return {
            "game_state": game_state,
            "perception_output": perception_output,
            "fused_embedding": fused_embedding,
            "policy_logits": policy_logits,
            "activations": self.activation_capture.activations.copy(),
        }

    def visualize_cnn_features(
        self,
        activations: Dict[str, torch.Tensor],
        board_width: int,
        board_height: int,
        max_channels: int = 8,
        save_path: Optional[str] = None,
    ):
        """Visualize CNN feature maps as heatmaps."""
        cnn_blocks = ["cnn_block1", "cnn_block2", "cnn_block3", "cnn_block4"]

        fig, axes = plt.subplots(
            len(cnn_blocks),
            max_channels,
            figsize=(max_channels * 2, len(cnn_blocks) * 2),
        )

        for block_idx, block_name in enumerate(cnn_blocks):
            if block_name not in activations:
                continue

            # Get activations [C, H, W] (single sample)
            block_features = activations[block_name]
            if block_features.dim() == 4:  # [1, C, H, W]
                block_features = block_features.squeeze(0)

            num_channels = min(block_features.size(0), max_channels)

            for ch_idx in range(max_channels):
                ax = axes[block_idx, ch_idx]

                if ch_idx < num_channels:
                    # Get channel features
                    channel_features = block_features[ch_idx].numpy()

                    # Upsample to board size using nearest neighbor
                    channel_upsampled = (
                        F.interpolate(
                            torch.from_numpy(channel_features)
                            .unsqueeze(0)
                            .unsqueeze(0),
                            size=(board_height, board_width),
                            mode="nearest",
                        )
                        .squeeze()
                        .numpy()
                    )

                    # Normalize 0-1
                    if channel_upsampled.max() > channel_upsampled.min():
                        channel_upsampled = (
                            channel_upsampled - channel_upsampled.min()
                        ) / (channel_upsampled.max() - channel_upsampled.min())

                    # Show heatmap
                    im = ax.imshow(channel_upsampled, cmap="hot", alpha=0.8)
                    ax.set_title(f"{block_name}\nCh {ch_idx}")
                else:
                    ax.set_title(f"{block_name}\n(empty)")
                    ax.set_xticks([])
                    ax.set_yticks([])

                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_policy_overlay(
        self,
        policy_logits: torch.Tensor,
        board_width: int,
        board_height: int,
        k: int = 5,
        save_path: Optional[str] = None,
    ):
        """Visualize policy logits as arrow overlay on board."""
        # Convert to probabilities
        prob = torch.sigmoid(policy_logits).cpu()

        # Create heatmap: max probability across all 4 arrow directions for each tile
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

        fig, ax = plt.subplots(figsize=(BOARD_WIDTH / 2, BOARD_HEIGHT / 2))

        # Show heatmap
        im = ax.imshow(
            heat.numpy(), cmap="hot", interpolation="nearest", vmin=0, vmax=1
        )

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, BOARD_WIDTH, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, BOARD_HEIGHT, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

        # Draw boundary around actual puzzle area
        puzzle_rect = Rectangle(
            (-0.5, -0.5),
            board_width,
            board_height,
            linewidth=3,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(puzzle_rect)

        # Direction characters for visualization
        dir_chars = {
            "place_up": "↑",
            "place_down": "↓",
            "place_left": "←",
            "place_right": "→",
        }

        # Draw top-k predictions
        topk_indices = prob.topk(k).indices.tolist()
        for rank, a_id in enumerate(topk_indices):
            action_info = decode_action(a_id)
            if action_info.action_type in dir_chars:
                char = dir_chars[action_info.action_type]
                alpha = 1.0 - (rank * 0.15)
                ax.text(
                    action_info.x,
                    action_info.y,
                    char,
                    color="red",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    alpha=max(0.2, alpha),
                )

        ax.set_xticks(range(BOARD_WIDTH))
        ax.set_yticks(range(BOARD_HEIGHT))
        ax.set_xlabel("X (columns)")
        ax.set_ylabel("Y (rows)")
        ax.set_title(f"Policy Overlay\nPuzzle: {board_width}×{board_height}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Max Arrow Probability")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_cat_embeddings(
        self, activations: Dict[str, torch.Tensor], save_path: Optional[str] = None
    ):
        """Visualize cat encoder embeddings showing which cat dominated each dimension."""
        if "cat_embeddings_raw" not in activations:
            print("No cat embeddings found in activations")
            return

        cat_embeddings = activations["cat_embeddings_raw"]  # [num_cats, 32]

        if cat_embeddings.size(0) == 0:
            print("No cats found in this puzzle")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left plot: Raw embeddings heatmap
        im1 = ax1.imshow(cat_embeddings.T.numpy(), cmap="RdBu_r", aspect="auto")
        ax1.set_xlabel("Cat ID")
        ax1.set_ylabel("Embedding Dimension")
        ax1.set_title("Per-Cat Embeddings (32D)")
        plt.colorbar(im1, ax=ax1)

        # Right plot: Max-pooling winners
        max_values, max_indices = torch.max(cat_embeddings, dim=0)  # [32]

        # Create bar chart colored by winning cat
        colors = plt.cm.tab10(max_indices.numpy() % 10)
        bars = ax2.bar(range(32), max_values.numpy(), color=colors)
        ax2.set_xlabel("Embedding Dimension")
        ax2.set_ylabel("Max Value")
        ax2.set_title("Max-Pool Winners by Cat")

        # Add legend showing cat colors
        for cat_id in range(cat_embeddings.size(0)):
            ax2.bar([], [], color=plt.cm.tab10(cat_id % 10), label=f"Cat {cat_id}")
        if cat_embeddings.size(0) <= 10:
            ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_original_board(self, level, save_path: Optional[str] = None):
        """Visualize the original puzzle board with colors matching the game visualization."""
        board = level.board
        board_width, board_height = board.width, board.height

        # Create figure
        fig, ax = plt.subplots(figsize=(board_width * 0.8, board_height * 0.8))

        # Color mapping based on game visualization colors
        color_map = {
            CellType.EMPTY: "#FFFFFF",  # White
            CellType.HOLE: "#000000",  # Black
            CellType.ROCKET: "#FFFF00",  # Yellow
            CellType.SPAWNER: "#C8C8C8",  # Light gray
            CellType.WALL: "#404040",  # Dark gray
        }

        # Create board grid
        board_array = np.zeros((board_height, board_width, 3))

        # Fill in cell types
        for y in range(board_height):
            for x in range(board_width):
                cell_type = board.get_cell_type(x, y)
                color_hex = color_map.get(cell_type, "#C8C8C8")
                # Convert hex to RGB (0-1 range)
                color_rgb = [int(color_hex[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
                board_array[y, x] = color_rgb

        # Display board
        ax.imshow(board_array, interpolation="nearest")

        # Draw walls as thick lines
        for (x1, y1), (x2, y2) in board.walls:
            if (
                0 <= x1 < board_width
                and 0 <= y1 < board_height
                and 0 <= x2 < board_width
                and 0 <= y2 < board_height
            ):

                if x1 == x2 and abs(y1 - y2) == 1:
                    # Horizontal wall between (x, y1) and (x, y2)
                    y_wall = max(y1, y2) - 0.5
                    ax.plot(
                        [x1 - 0.5, x1 + 0.5],
                        [y_wall, y_wall],
                        color="#404040",
                        linewidth=6,
                    )
                elif y1 == y2 and abs(x1 - x2) == 1:
                    # Vertical wall between (x1, y) and (x2, y)
                    x_wall = max(x1, x2) - 0.5
                    ax.plot(
                        [x_wall, x_wall],
                        [y1 - 0.5, y1 + 0.5],
                        color="#404040",
                        linewidth=6,
                    )

        # Draw sprites
        sprite_colors = {
            SpriteType.MOUSE: "#8B4513",  # Brown
            SpriteType.GOLD_MOUSE: "#FFD700",  # Gold
            SpriteType.BONUS_MOUSE: "#FFC0CB",  # Pink
            SpriteType.CAT: "#FFA500",  # Orange
            SpriteType.ROCKET: "#FF0000",  # Red (override cell color)
            SpriteType.SPAWNER: "#0000FF",  # Blue
        }

        # Get sprites from level
        for sprite_id, sprite in level.sprite_manager.sprites.items():
            sprite_type = sprite.get_sprite_type()
            color = sprite_colors.get(sprite_type, "#800080")  # Purple default

            # Draw sprite as circle
            circle = plt.Circle((sprite.x, sprite.y), 0.3, color=color, zorder=10)
            ax.add_patch(circle)

            # Draw direction indicator
            if hasattr(sprite, "direction") and sprite.direction:
                dx, dy = sprite.direction.dx * 0.2, sprite.direction.dy * 0.2
                ax.arrow(
                    sprite.x,
                    sprite.y,
                    dx,
                    dy,
                    head_width=0.1,
                    head_length=0.1,
                    fc="white",
                    ec="white",
                    zorder=11,
                )

        # Draw placed arrows (if any)
        for (x, y), direction in board.arrows.items():
            # Blue square background
            square = Rectangle(
                (x - 0.3, y - 0.3),
                0.6,
                0.6,
                facecolor="blue",
                edgecolor="black",
                linewidth=2,
                zorder=8,
            )
            ax.add_patch(square)

            # White arrow
            dx, dy = direction.dx * 0.2, direction.dy * 0.2
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.1,
                head_length=0.1,
                fc="white",
                ec="white",
                zorder=9,
            )

        # Set up grid and labels
        ax.set_xlim(-0.5, board_width - 0.5)
        ax.set_ylim(-0.5, board_height - 0.5)
        ax.set_xticks(range(board_width))
        ax.set_yticks(range(board_height))
        ax.set_xlabel("X (columns)")
        ax.set_ylabel("Y (rows)")
        ax.set_title(f"Original Board: {board_width}×{board_height}")
        ax.grid(True, alpha=0.3)

        # Invert y-axis to match game coordinates
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_state(
        self, level, save_dir: Optional[str] = None, puzzle_id: str = "puzzle"
    ):
        """Complete visualization pipeline for a single game state."""
        # Run inference
        results = self.run_inference(level)

        # Get board dimensions
        board = level.board
        board_width, board_height = board.width, board.height

        print(f"Visualizing puzzle: {board_width}x{board_height}")
        print(f"Captured activations: {list(results['activations'].keys())}")

        # Original board
        board_path = f"{save_dir}/{puzzle_id}_original_board.png" if save_dir else None
        self.visualize_original_board(level, save_path=board_path)

        # CNN feature maps
        cnn_path = f"{save_dir}/{puzzle_id}_cnn_features.png" if save_dir else None
        self.visualize_cnn_features(
            results["activations"], board_width, board_height, save_path=cnn_path
        )

        # Policy overlay
        policy_path = f"{save_dir}/{puzzle_id}_policy_overlay.png" if save_dir else None
        self.visualize_policy_overlay(
            results["policy_logits"], board_width, board_height, save_path=policy_path
        )

        # Cat embeddings (if cats exist)
        cat_path = f"{save_dir}/{puzzle_id}_cat_embeddings.png" if save_dir else None
        self.visualize_cat_embeddings(results["activations"], save_path=cat_path)

        return results

    def __del__(self):
        """Cleanup hooks when visualizer is destroyed."""
        if hasattr(self, "activation_capture"):
            self.activation_capture.remove_hooks()
