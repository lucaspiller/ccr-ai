"""
PPO Environment wrapper for ChuChu Rocket puzzle mode.

Handles the two-phase puzzle structure:
1. Placement Phase: Agent places arrows within budget
2. Execution Phase: Game runs automatically until completion/failure
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


from ...game.actions import PlaceArrowAction, RemoveArrowAction
from ...game.board import Board, CellType, Direction
from ...game.engine import GameEngine, GamePhase, GameResult
from ...game.sprites import Cat, Mouse, Rocket, SpriteManager
from ...perception.processors import GameStateProcessor
from ...state_fusion.processors import StateFusionProcessor
from ...util.action_utils import create_action_mask
from .config import PPOConfig
from .puzzle_config import PuzzleConfig, SpriteConfig


@dataclass
class PPOStepResult:
    """Result of a single environment step."""

    observation: torch.Tensor
    reward: float
    done: bool
    info: Dict[str, Any]
    action_mask: torch.Tensor


class PPOEnvironment:
    """PPO Environment for ChuChu Rocket puzzle mode.

    This environment properly handles puzzle mode where:
    1. Agent places arrows during placement phase (limited by budget)
    2. Game runs automatically during execution phase
    3. Episode ends when all mice saved, any mice lost, or timeout
    """

    def __init__(
        self,
        config: PPOConfig,
        perception_processor: GameStateProcessor,
        state_fusion_processor: StateFusionProcessor,
        puzzle_config: Optional[PuzzleConfig] = None,
    ):
        self.config = config
        self.perception_processor = perception_processor
        self.state_fusion_processor = state_fusion_processor

        # Environment state
        self.game_engine: Optional[GameEngine] = None
        self.current_puzzle_config: PuzzleConfig = (
            puzzle_config or self._get_default_puzzle_config()
        )

        # Episode tracking
        self.episode_steps = 0
        self.placement_steps = 0
        self.arrows_placed = 0
        self.arrow_budget = 0
        self.initial_mice_count = 0
        self.target_mice = 0

        # Reward tracking
        self.prev_mice_in_rocket = 0
        self.prev_cats_in_rocket = 0
        self.prev_mice_in_holes = 0

        # Episode stats
        self.episode_stats = {
            "arrows_used": 0,
            "mice_saved": 0,
            "cats_fed": 0,
            "mice_lost_holes": 0,
            "placement_steps": 0,
            "execution_ticks": 0,
            "game_result": None,
        }

    def reset(
        self, puzzle_config: Optional[PuzzleConfig] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset environment and start new episode.

        Args:
            puzzle_config: Optional puzzle configuration

        Returns:
            Tuple of (observation, action_mask)
        """
        if puzzle_config:
            self.current_puzzle_config = puzzle_config

        # Create new game instance
        self._create_new_game()

        # Reset episode state
        self.episode_steps = 0
        self.placement_steps = 0
        self.arrows_placed = 0
        self.arrow_budget = self.current_puzzle_config.arrow_budget

        # Count initial mice for target
        stats = self.game_engine.get_game_stats()
        self.initial_mice_count = stats["mice"]["total"]
        self.target_mice = self.initial_mice_count

        # Initialize reward tracking
        self.prev_mice_in_rocket = 0
        self.prev_cats_in_rocket = 0
        self.prev_mice_in_holes = 0

        # Reset episode stats
        self.episode_stats = {
            "arrows_used": 0,
            "mice_saved": 0,
            "cats_fed": 0,
            "mice_lost_holes": 0,
            "placement_steps": 0,
            "execution_ticks": 0,
            "game_result": None,
        }

        # Debug info
        puzzle_id = self.current_puzzle_config.puzzle_id or "unknown"
        rockets_count = len(self.current_puzzle_config.rockets)
        cats_count = len(self.current_puzzle_config.cats)
        print(
            f"    Reset environment for {puzzle_id}: {self.game_engine.board.width}x{self.game_engine.board.height} "
            f"board, {self.initial_mice_count} mice, {cats_count} cats, {rockets_count} rockets, budget={self.arrow_budget}"
        )

        # Debug puzzle config
        if rockets_count == 0:
            print(f"    WARNING: No rockets in puzzle config!")
            print(f"    Puzzle config keys: {list(self.current_puzzle_config.keys())}")

        # Get initial observation and action mask
        observation = self._get_observation()
        action_mask = self._get_action_mask()

        return observation, action_mask

    def step(self, action: int) -> PPOStepResult:
        """Execute one environment step.

        Args:
            action: Action index to execute

        Returns:
            PPOStepResult with observation, reward, done, info, action_mask
        """
        self.episode_steps += 1

        # Check if we're still in placement phase
        if self.game_engine.phase == GamePhase.PLACEMENT:
            return self._handle_placement_step(action)
        else:
            # In execution phase, just run game until completion
            return self._handle_execution_phase()

    def _handle_placement_step(self, action: int) -> PPOStepResult:
        """Handle action during placement phase."""
        self.placement_steps += 1
        reward = 0.0
        done = False
        info = {}

        # Debug action
        from ...util.action_utils import decode_action

        try:
            action_info = decode_action(action)
            action_desc = f"{action_info.action_type}@({action_info.x},{action_info.y})"
        except:
            action_desc = f"action_{action}"

        # Execute placement action
        if self._is_valid_action(action):
            success = self._execute_action(action)
            if success:
                # Small negative reward for arrow placement (encourage efficiency)
                reward = self.config.reward_arrow_cost
                self.arrows_placed += 1
                self.episode_stats["arrows_used"] += 1
                print(
                    f"      ✅ Placed arrow: {action_desc} (total: {self.arrows_placed}/{self.arrow_budget})"
                )
            else:
                print(f"      Failed to execute: {action_desc}")
        else:
            print(f"      Invalid action: {action_desc}")

        # Check if placement phase should end
        placement_done = (
            self.arrows_placed >= self.arrow_budget
            or self.placement_steps >= self.config.placement_timeout
            or self.arrows_placed > 0  # For testing - start game after first arrow
        )

        # Get new observation and action mask
        observation = self._get_observation()
        action_mask = self._get_action_mask()

        if placement_done:
            # Start the game execution but mark this step as transitioning
            print(
                f"    Starting game execution with {self.arrows_placed} arrows placed"
            )

            self.game_engine.start_game()

            info.update(
                {
                    "phase": "transition_to_execution",
                    "arrows_placed": self.arrows_placed,
                    "arrow_budget": self.arrow_budget,
                    "placement_steps": self.placement_steps,
                }
            )
        else:
            info.update(
                {
                    "phase": "placement",
                    "arrows_placed": self.arrows_placed,
                    "arrow_budget": self.arrow_budget,
                    "placement_steps": self.placement_steps,
                }
            )

        return PPOStepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action_mask=action_mask,
        )

    def _handle_execution_phase(self) -> PPOStepResult:
        """Handle execution phase - run game until completion."""
        print(f"      Executing game simulation...")
        total_reward = 0.0
        execution_steps = 0

        # Run game until completion
        while self.game_engine.result == GameResult.ONGOING:
            # Step the game
            events = self.game_engine.step()
            execution_steps += 1

            # Calculate incremental reward
            step_reward = self._calculate_step_reward()
            total_reward += step_reward

            # Log significant events
            if step_reward != 0:
                print(
                    f"        Tick {execution_steps}: reward={step_reward:.2f} "
                    f"(mice in rocket: {self._count_mice_in_rocket()}, "
                    f"cats: {self._count_cats_in_rocket()})"
                )

            # Check for timeout
            if execution_steps >= self.config.execution_timeout:
                print(f"        Execution timeout at {execution_steps} ticks")
                break

        # Calculate final reward and determine if episode is done
        final_reward = self._calculate_final_reward()
        total_reward += final_reward

        # Episode is always done after execution phase
        done = True

        # Update episode stats
        self.episode_stats["execution_ticks"] = execution_steps
        self.episode_stats["game_result"] = self.game_engine.result.value

        if self.game_engine.result == GameResult.SUCCESS:
            print(f"      🎉 Game won after {execution_steps} ticks")
        else:
            print(
                f"      Game finished: {self.game_engine.result.value} after {execution_steps} ticks"
            )
        print(
            f"      Final stats: {self.episode_stats['mice_saved']} mice saved, "
            f"{self.episode_stats['cats_fed']} cats fed, total reward={total_reward:.2f}"
        )

        # Get final observation (though episode is done)
        observation = self._get_observation()
        action_mask = self._get_action_mask()  # All masked since episode is done

        info = {
            "phase": "execution",
            "execution_steps": execution_steps,
            "game_result": self.game_engine.result.value,
            "final_reward": final_reward,
            "episode_stats": self.episode_stats.copy(),
        }

        return PPOStepResult(
            observation=observation,
            reward=total_reward,
            done=done,
            info=info,
            action_mask=action_mask,
        )

    def _calculate_step_reward(self) -> float:
        """Calculate reward for a single game step during execution."""
        # Get current state
        current_mice_in_rocket = self._count_mice_in_rocket()
        current_cats_in_rocket = self._count_cats_in_rocket()
        current_mice_in_holes = self._count_mice_in_holes()

        # Calculate incremental changes
        mice_saved = current_mice_in_rocket - self.prev_mice_in_rocket
        cats_fed = current_cats_in_rocket - self.prev_cats_in_rocket
        mice_lost_holes = current_mice_in_holes - self.prev_mice_in_holes

        # Update tracking
        self.prev_mice_in_rocket = current_mice_in_rocket
        self.prev_cats_in_rocket = current_cats_in_rocket
        self.prev_mice_in_holes = current_mice_in_holes

        # Update episode stats
        self.episode_stats["mice_saved"] += mice_saved
        self.episode_stats["cats_fed"] += cats_fed
        self.episode_stats["mice_lost_holes"] += mice_lost_holes

        # Calculate step reward
        reward = (
            mice_saved * self.config.reward_mouse_saved
            + cats_fed * self.config.reward_cat_fed
            + mice_lost_holes * self.config.reward_mouse_lost_hole
        )

        return reward

    def _calculate_final_reward(self) -> float:
        """Calculate final reward bonus/penalty at episode end."""
        if self.game_engine.result == GameResult.SUCCESS:
            # Success: all mice saved
            return self.config.reward_success_bonus
        else:
            # Failure: timeout, mice lost, etc.
            return self.config.reward_failure_penalty

    def _get_observation(self) -> torch.Tensor:
        """Get current observation as state embedding."""
        # Get current game state
        game_state = self.game_engine.to_dict()

        # Process through perception
        perception_output = self.perception_processor.process(game_state)

        # Move perception output to the right device (for MPS compatibility)
        device = next(self.state_fusion_processor.fusion_mlp.parameters()).device
        if hasattr(perception_output, "grid_embedding"):
            perception_output.grid_embedding = perception_output.grid_embedding.to(
                device
            )
        if hasattr(perception_output, "global_features"):
            perception_output.global_features = perception_output.global_features.to(
                device
            )
        if hasattr(perception_output, "cat_embedding"):
            perception_output.cat_embedding = perception_output.cat_embedding.to(device)

        # Fuse into state embedding
        fused_output = self.state_fusion_processor.fuse(perception_output)

        # Check for NaN in the observation
        if torch.isnan(fused_output.fused_embedding).any():
            print(f"Warning: NaN detected in fused_embedding!")
            print(f"Game phase: {self.game_engine.phase}")
            print(f"Arrows placed: {self.arrows_placed}/{self.arrow_budget}")
            # Debug the pipeline
            print(
                f"Perception output shapes: grid={getattr(perception_output, 'grid_embedding', 'None')}, "
                f"global={getattr(perception_output, 'global_features', 'None')}, "
                f"cat={getattr(perception_output, 'cat_embedding', 'None')}"
            )
            # Return zeros instead of NaN
            return torch.zeros_like(fused_output.fused_embedding)

        # Also check for inf
        if torch.isinf(fused_output.fused_embedding).any():
            print(f"Warning: Inf detected in fused_embedding!")
            print(
                f"Min: {fused_output.fused_embedding.min()}, Max: {fused_output.fused_embedding.max()}"
            )
            # Clamp to reasonable range
            return torch.clamp(fused_output.fused_embedding, -10.0, 10.0)

        return fused_output.fused_embedding

    def _get_action_mask(self) -> torch.Tensor:
        """Get current action mask."""
        if self.game_engine.phase == GamePhase.PLACEMENT:
            # During placement, mask based on arrow budget and valid positions
            mask = self._get_placement_action_mask()
            print(
                f"      Action mask: {mask.sum()} valid actions (phase=placement, arrows={self.arrows_placed}/{self.arrow_budget})"
            )
            return mask
        else:
            # During execution, no actions allowed
            mask = torch.zeros(700, dtype=torch.bool)
            print(f"      Action mask: {mask.sum()} valid actions (phase=execution)")
            return mask

    def _get_placement_action_mask(self) -> torch.Tensor:
        """Get action mask for placement phase."""
        # Get basic action mask from game state
        basic_mask = create_action_mask(
            self.game_engine.board.width, self.game_engine.board.height
        )

        print(
            f"        Basic mask: {basic_mask.sum()} valid actions for {self.game_engine.board.width}x{self.game_engine.board.height} board"
        )

        # If arrow budget exhausted, only allow "start game" action or no-op
        if self.arrows_placed >= self.arrow_budget:
            # Mask all placement actions, allow only start game
            placement_mask = torch.zeros_like(basic_mask)
            print(f"        Arrow budget exhausted, no valid actions")
            # TODO: Define start game action index properly
            return placement_mask

        return basic_mask

    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid in current state."""
        action_mask = self._get_action_mask()
        return action_mask[action].item()

    def _execute_action(self, action: int) -> bool:
        """Execute the given action."""
        # Convert action index to game action
        game_action = self._action_index_to_game_action(action)

        if game_action is None:
            return False

        # Execute action through game engine
        try:
            if isinstance(game_action, PlaceArrowAction):
                success = self.game_engine.place_arrow(
                    game_action.x, game_action.y, game_action.direction
                )
            elif isinstance(game_action, RemoveArrowAction):
                success = self.game_engine.remove_arrow(game_action.x, game_action.y)
            else:
                return False
            return success
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False

    def _action_index_to_game_action(self, action_index: int):
        """Convert action index to game action using BC model's action encoding."""
        # Use the same action encoding as the BC training (from action_utils.py)
        from ...util.action_utils import decode_action

        # Debug action conversion
        print(f"        Converting action {action_index}")

        try:
            # Use the standardized action decoding
            action_info = decode_action(action_index)
            print(
                f"        -> {action_info.action_type} at ({action_info.x},{action_info.y})"
            )

            if action_info.action_type.startswith("place_"):
                # Map action type to direction
                direction_map = {
                    "place_up": Direction.UP,
                    "place_down": Direction.DOWN,
                    "place_left": Direction.LEFT,
                    "place_right": Direction.RIGHT,
                }
                direction = direction_map[action_info.action_type]
                return PlaceArrowAction(
                    x=action_info.x, y=action_info.y, direction=direction
                )

            elif action_info.action_type == "erase":
                return RemoveArrowAction(x=action_info.x, y=action_info.y)

            else:
                print(f"        -> Unknown action type: {action_info.action_type}")
                return None

        except Exception as e:
            print(f"        -> Failed to decode action: {e}")
            return None

    def _get_start_game_action(self) -> int:
        """Get action index for starting the game."""
        # This could be a special action or just any invalid action
        # For now, return an index beyond valid actions
        return 699  # Last valid action index + 1

    def _count_mice_in_rocket(self) -> int:
        """Count mice currently in rocket."""
        stats = self.game_engine.get_game_stats()
        return stats["mice"]["in_rocket"]

    def _count_cats_in_rocket(self) -> int:
        """Count cats currently in rocket."""
        stats = self.game_engine.get_game_stats()
        return stats["cats"]["in_rocket"]

    def _count_mice_in_holes(self) -> int:
        """Count mice that have fallen in holes."""
        stats = self.game_engine.get_game_stats()
        return stats["mice"]["in_holes"]

    def _create_new_game(self):
        """Create a new game instance from puzzle config."""
        # Create board from puzzle configuration
        board = Board(
            width=self.current_puzzle_config.width,
            height=self.current_puzzle_config.height,
        )

        # Add walls from config
        for wall in self.current_puzzle_config.walls:
            # Add wall as edge between adjacent tiles
            if wall.direction == "vertical":
                # Wall between (x,y) and (x+1,y)
                if wall.x + 1 < board.width:
                    board.walls.add(((wall.x, wall.y), (wall.x + 1, wall.y)))
            elif wall.direction == "horizontal":
                # Wall between (x,y) and (x,y+1)
                if wall.y + 1 < board.height:
                    board.walls.add(((wall.x, wall.y), (wall.x, wall.y + 1)))

        # Create sprite manager and add sprites
        sprite_manager = SpriteManager()

        # Add rockets
        for rocket_config in self.current_puzzle_config.rockets:
            # Set rocket cell on board
            board.set_cell_type(rocket_config.x, rocket_config.y, CellType.ROCKET)

            # Create rocket sprite
            rocket = Rocket(
                x=rocket_config.x,
                y=rocket_config.y,
                sprite_id=f"rocket_{rocket_config.player_id or 0}",
            )
            # Set player_id attribute for identification
            rocket.player_id = rocket_config.player_id or 0
            sprite_manager.add_sprite(rocket)

        # Add mice
        for i, mouse_config in enumerate(self.current_puzzle_config.mice):
            mouse = Mouse(x=mouse_config.x, y=mouse_config.y, sprite_id=f"mouse_{i}")
            sprite_manager.add_sprite(mouse)

        # Add cats
        for i, cat_config in enumerate(self.current_puzzle_config.cats):
            cat = Cat(x=cat_config.x, y=cat_config.y, sprite_id=f"cat_{i}")
            sprite_manager.add_sprite(cat)

        # Add holes
        for hole_config in self.current_puzzle_config.holes:
            # Set hole cell on board
            board.set_cell_type(hole_config.x, hole_config.y, CellType.HOLE)

        # Create game engine in puzzle mode
        self.game_engine = GameEngine(
            board=board,
            sprite_manager=sprite_manager,
            max_steps=1800,  # 30 seconds at 60fps
            puzzle_mode=True,
        )

    def _get_default_puzzle_config(self) -> PuzzleConfig:
        """Get default puzzle configuration for testing."""
        return PuzzleConfig(
            width=10,
            height=14,
            arrow_budget=3,
            walls=[],
            rockets=[SpriteConfig(x=5, y=7, player_id=0)],
            mice=[
                SpriteConfig(x=2, y=3, direction="RIGHT"),
                SpriteConfig(x=8, y=5, direction="LEFT"),
                SpriteConfig(x=4, y=9, direction="UP"),
            ],
            cats=[
                SpriteConfig(x=1, y=1, direction="DOWN"),
            ],
            holes=[],
        )


class PPOEnvironmentManager:
    """Manages multiple parallel PPO environments."""

    def __init__(
        self,
        config: PPOConfig,
        perception_processor: GameStateProcessor,
        state_fusion_processor: StateFusionProcessor,
        num_envs: Optional[int] = None,
    ):
        self.config = config
        self.num_envs = num_envs or config.num_parallel_envs

        # Create parallel environments
        self.envs = []
        for i in range(self.num_envs):
            env = PPOEnvironment(
                config=config,
                perception_processor=perception_processor,
                state_fusion_processor=state_fusion_processor,
            )
            self.envs.append(env)

    def reset_all(
        self, puzzle_configs: Optional[List[PuzzleConfig]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset all environments.

        Args:
            puzzle_configs: Optional list of puzzle configs (one per env)

        Returns:
            Tuple of (observations, action_masks) stacked across environments
        """
        observations = []
        action_masks = []

        for i, env in enumerate(self.envs):
            puzzle_config = puzzle_configs[i] if puzzle_configs else None
            obs, mask = env.reset(puzzle_config)
            observations.append(obs)
            action_masks.append(mask)

        return torch.stack(observations), torch.stack(action_masks)

    def step_all(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict], torch.Tensor]:
        """Step all environments.

        Args:
            actions: Tensor of actions for each environment

        Returns:
            Tuple of (observations, rewards, dones, infos, action_masks)
        """
        results = []
        for i, env in enumerate(self.envs):
            result = env.step(actions[i].item())
            results.append(result)

        observations = torch.stack([r.observation for r in results])
        rewards = torch.tensor([r.reward for r in results], dtype=torch.float32)
        dones = torch.tensor([r.done for r in results], dtype=torch.bool)
        infos = [r.info for r in results]
        action_masks = torch.stack([r.action_mask for r in results])

        return observations, rewards, dones, infos, action_masks
