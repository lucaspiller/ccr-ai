import numpy as np
import torch
from matplotlib import pyplot as plt

from src.util.action_utils import (ACTION_TYPE_OFFSETS, BOARD_HEIGHT,
                                   BOARD_WIDTH, decode_action, get_tile_index)


def viz_board(
    logits, target, action_mask, board_w, board_h, k=5, title="", save_path=None
):
    """Visualize board state with heatmap and arrows.

    Args:
        logits: Raw policy logits [700]
        target: Multi-hot target vector [700]
        action_mask: Action mask [700]
        board_w: Actual board width
        board_h: Actual board height
        k: Number of top predictions to show
        title: Plot title
        save_path: Path to save PNG file (if None, shows plot)
    """
    # Convert to probabilities
    prob = torch.sigmoid(logits).cpu()

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

    # Create mask overlay for invalid tiles
    # mask_overlay = torch.ones(BOARD_HEIGHT, BOARD_WIDTH)
    # for y in range(board_h, BOARD_HEIGHT):  # Rows beyond puzzle height
    #    mask_overlay[y, :] = 0.3
    # for x in range(board_w, BOARD_WIDTH):  # Columns beyond puzzle width
    #    mask_overlay[:, x] = 0.3
    #
    ## Apply mask to heatmap
    # heat_masked = heat * mask_overlay

    heat_masked = heat

    fig, ax = plt.subplots(figsize=(BOARD_WIDTH / 2, BOARD_HEIGHT / 2))

    # Show heatmap with masked regions dimmed
    im = ax.imshow(
        heat_masked.numpy(), cmap="hot", interpolation="nearest", vmin=0, vmax=1
    )

    # Add grid lines to show tile boundaries
    ax.set_xticks(np.arange(-0.5, BOARD_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_HEIGHT, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    # Draw boundary around actual puzzle area
    from matplotlib.patches import Rectangle

    puzzle_rect = Rectangle(
        (-0.5, -0.5), board_w, board_h, linewidth=3, edgecolor="blue", facecolor="none"
    )
    ax.add_patch(puzzle_rect)

    # Direction characters for visualization
    dir_chars = {
        "place_up": "↑",
        "place_down": "↓",
        "place_left": "←",
        "place_right": "→",
    }

    # Draw puzzle arrows (green)
    for a_id in target.nonzero().squeeze(-1).cpu().tolist():
        action_info = decode_action(a_id)
        if action_info.action_type in dir_chars:
            char = dir_chars[action_info.action_type]
            ax.text(
                action_info.x,
                action_info.y,
                char,
                color="lawngreen",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    # Draw mask visualization - show which tiles are masked in training
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            tile_idx = get_tile_index(x, y)
            # Check if any action for this tile is masked (use 'place_up' as representative)
            up_action_idx = ACTION_TYPE_OFFSETS["place_up"] + tile_idx
            if action_mask[up_action_idx] == 0:  # Masked tile
                ax.text(
                    x,
                    y,
                    "✕",
                    color="gray",
                    ha="center",
                    va="center",
                    fontsize=8,
                    alpha=0.6,
                )

    # Draw top-k predictions (red, fading)
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
    ax.set_title(
        f"{title}\nPuzzle: {board_w}×{board_h}, Full board: {BOARD_WIDTH}×{BOARD_HEIGHT}"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Max Arrow Probability")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()  # Close to free memory
    else:
        plt.show()
