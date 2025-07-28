# PPO Self-Play Training PRD

## Overview
Implement PPO self-play training to advance beyond BC baseline. This is Stage B of the training pipeline, using the BC-trained policy as initialization and optimizing through environment interaction with sparse rewards. Target: surpass BC performance by +10pp on medium puzzles.

## Prerequisites
- BC-trained model with ≥95% solve rate on Easy puzzles AND ≥0.35 Precision@k on Medium puzzles (ensures quality latent representations for RL)
- Existing perception, state_fusion, and policy layers
- Game engine with episode management and reward calculation
- Action masking for variable board sizes

## Training Components

### 1. Environment Wrapper (`ppo_env.py`)
- **Episode Management**: Handle puzzle initialization, step execution, and termination
- **Reward Calculation**: `mice_saved - cats_fed - arrow_cost * 0.1`
- **Action Masking**: Apply variable board size masks during action sampling
- **Observation Processing**: Convert game state to perception input format
- **Episode Timeout**: Maximum 200 steps per episode to prevent infinite loops
- **Success Tracking**: Track solve rate, steps-to-solution, and reward per episode

### 2. PPO Algorithm (`ppo_trainer.py`)
- **Actor-Critic Architecture**: Shared backbone with policy and value heads
- **Advantage Estimation**: GAE (λ=0.95) for variance reduction
- **Policy Updates**: Clipped PPO objective with KL penalty
- **Action Masking**: Policy logits for invalid actions set to -∞ before ratio calculation
- **Value Updates**: MSE loss with optional value clipping
- **Experience Collection**: Rollout buffer for on-policy learning
- **Entropy Regularization**: Encourage exploration (coefficient: 0.01)

### 3. Experience Collection (`rollout_buffer.py`)
- **Trajectory Storage**: (state, action, reward, value, log_prob, done, mask)
- **Batch Processing**: Collect N episodes before each update
- **Advantage Computation**: GAE calculation with proper bootstrapping
- **Buffer Management**: Clear after each update (on-policy requirement)
- **Variable Length**: Handle episodes of different lengths

### 4. Curriculum System (`curriculum.py`)
- **Difficulty Progression**: Start with BC validation set, gradually increase complexity
- **Dynamic Puzzle Generation**: Generate puzzles matching current skill level
- **Success Thresholds**: Advance difficulty when solve rate ≥80% over sliding window of 10k steps (≈300 episodes, weighted by puzzle length)
- **Curriculum Stages**:
  - Stage 1: Easy puzzles (5×5-8×6, 3-5 mice, 0-1 cats, arrow_budget: 5, holes: 0-1)
  - Stage 2: Medium puzzles (8×6-10×8, 5-7 mice, 1-2 cats, arrow_budget: 4, holes: 1-2) 
  - Stage 3: Hard puzzles (10×8-14×10, 7-10 mice, 2-4 cats, arrow_budget: 3, holes: 2-3)

### 5. Evaluation (`ppo_evaluator.py`)
- **Performance Metrics**: Solve rate by difficulty, average steps, reward per episode
- **Comparison Baselines**: BC model, random policy, BFS optimal
- **Statistical Testing**: Track performance over rolling windows
- **Checkpoint Evaluation**: Test best model on fixed held-out test set (500 puzzles across difficulties)
- **Visualization**: Learning curves, action distributions, failure analysis

## PPO Hyperparameters

### Core Algorithm
- **Learning Rate**: 3e-4 with cosine decay (5% warmup over 10k steps)
- **Discount Factor (γ)**: 0.99
- **GAE Lambda (λ)**: 0.95
- **Clip Epsilon**: 0.2 → 0.1 (linear decay over training)
- **Value Clip Epsilon**: 0.2 → 0.1 (linear decay over training)
- **Entropy Coefficient**: 0.01 → 0.001 (linear decay after 80% solve rate)
- **KL Target**: 0.01 with adaptive β (if KL > 2×target, stop epoch early)

### Training Schedule
- **Rollout Length**: 2048 steps per update
- **Batch Size**: 64 (minibatch size during updates)
- **PPO Epochs**: 4 (reuse each batch 4 times)
- **Max Gradient Norm**: 1.0
- **Total Environment Steps**: 10M
- **Evaluation Frequency**: Every 100k steps on fixed held-out test set (500 puzzles)

### Environment
- **Parallel Environments**: 16 (for faster data collection)
- **Episode Timeout**: 200 steps
- **Reward Scaling**: No additional scaling (raw reward)
- **Action Masking**: Applied during action sampling

## Reward Function Design

### Primary Reward Components (Normalized for PPO Stability)
```python
def calculate_reward(prev_state, action, new_state):
    mice_saved = new_state.mice_in_own_rocket - prev_state.mice_in_own_rocket
    cats_fed = new_state.cats_in_own_rocket - prev_state.cats_in_own_rocket
    mice_lost_holes = new_state.mice_in_holes - prev_state.mice_in_holes
    arrow_cost = 0.05 if action_places_arrow(action) else 0.0
    
    # Immediate reward (normalized to 1-digit range)
    reward = mice_saved * 1.0 - cats_fed * 1.5 - mice_lost_holes * 1.0 - arrow_cost
    
    # Terminal bonus/penalty
    if episode_done:
        if mice_saved >= target_mice:
            reward += 5.0  # Success bonus
        else:
            reward -= 1.0  # Failure penalty
            
    return reward
```

### Reward Shaping (Optional)
- **Progress Reward**: Small positive reward for mice moving toward rocket
- **Efficiency Penalty**: Penalize excessive arrow usage
- **Cat Avoidance**: Negative reward for cats approaching rocket

## Success Criteria

### Primary Objectives
- **Performance**: ≥85% solve rate on Medium puzzles (vs 70% BC baseline)
- **Efficiency**: Average steps ≤ 1.3× BFS optimal (vs 1.5× BC baseline)
- **Generalization**: ≥60% solve rate on Hard puzzles (vs 50% BC baseline)

### Secondary Objectives
- **Sample Efficiency**: Achieve targets within 10M environment steps
- **Stability**: Consistent performance over multiple random seeds
- **Robustness**: Handle novel puzzle configurations not seen during BC

## Implementation Phases

### Phase 1: Core PPO Implementation
1. Environment wrapper with reward function
2. Basic PPO trainer with action masking
3. Experience collection and advantage computation
4. Model loading from BC checkpoint

### Phase 2: Training Infrastructure
1. Parallel environment collection
2. Evaluation and logging systems
3. Model checkpointing and best model tracking
4. Learning curve visualization

### Phase 3: Curriculum and Optimization
1. Curriculum system implementation
2. Hyperparameter tuning
3. Advanced reward shaping (if needed)
4. Performance optimization

## File Structure
```
src/training/ppo/
├── PRD.md
├── __init__.py
├── ppo_env.py           # Environment wrapper and reward calculation
├── ppo_trainer.py       # Main PPO algorithm implementation
├── rollout_buffer.py    # Experience collection and storage
├── curriculum.py        # Difficulty progression system
├── ppo_evaluator.py     # Performance evaluation and metrics
├── config.py            # PPO hyperparameters and settings
└── utils.py             # Helper functions

model/
├── ppo_checkpoint_latest.pth
├── ppo_checkpoint_best.pth
└── ppo_final.pth

logs/ppo/
├── training_curves.png
├── performance_metrics.json
└── tensorboard_logs/
```

## Dependencies
- BC-trained model from `src/training/bc/`
- Existing perception, state_fusion, and policy layers
- Game engine with episode management
- PyTorch for PPO implementation
- TensorBoard for logging and visualization
- Optional: Ray/multiprocessing for parallel environments

## Critical Implementation Details

### Value Head Initialization
- **Small Weight Initialization**: Initialize value head weights as N(0, 1e-4) to avoid large early advantages
- **Separate Learning Rate**: Consider separate LR schedule for value head (higher decay)

### Episode Management
- **Early Termination**: Terminate episode immediately when all mice are scored (saves compute)
- **Hole Penalty**: Include mice_lost_holes in reward calculation (-1.0 per mouse lost)
- **Buffer Clearing**: Ensure rollout buffer is cleared after each update (on-policy requirement)

### Action Masking in PPO Loss
- **Masked Ratio Calculation**: Set invalid action logits to -∞ before computing policy ratios
- **Log Prob Consistency**: Ensure stored log_probs match model version used for action selection

## Risk Mitigation

### Common PPO Issues
- **Policy Collapse**: Monitor KL divergence and use conservative updates with adaptive β
- **Value Function Accuracy**: Ensure value head learns meaningful state values with proper initialization
- **Exploration**: Maintain sufficient entropy throughout training with scheduled decay
- **Reward Hacking**: Design robust reward function and monitor for gaming

### Debugging Strategy
- **Sanity Checks**: Verify environment returns, reward calculation, action masking
- **Ablation Studies**: Test different hyperparameters and reward components
- **Baseline Comparisons**: Always compare against BC performance
- **Failure Analysis**: Investigate specific puzzle types where agent fails