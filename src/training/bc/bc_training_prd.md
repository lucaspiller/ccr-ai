# Behaviour Cloning Training PRD

## Overview
Implement 3-phase supervised learning to train the policy head on BFS-optimal solutions with **variable board sizes**. This is Stage A of the training pipeline, targeting 95% solve rate on simple puzzles before advancing to PPO self-play.

## 3-Phase Training Strategy

### Phase 1: BC-Set (Multi-Hot Labels)
- **Input**: Final board state with all BFS arrows already placed
- **Output**: Multi-hot vector (700-dim) indicating which arrows should be placed
- **Goal**: Learn global arrow placement strategy without sequential overfitting
- **Target**: ≥70% Jaccard accuracy on arrow set prediction

### Phase 2: BC-Sequence (Fine-tuning) 
- **Input**: Initial board states (first action only)
- **Output**: Single action (700-dim softmax)
- **Goal**: Add ordering bias and first-action selection
- **Duration**: 5-10 epochs fine-tuning from BC-Set weights

### Phase 3: PPO Self-Play
- **Initialization**: Hybrid policy from BC-Set + BC-Sequence
- **Goal**: Learn timing, waiting, and budget optimization through RL
- **Advantage**: Only needs to learn sequential logic, not basic routing

## Data Format
- **Input**: CSV with puzzle configurations and BFS solutions
- **Columns**: seed, board_w, board_h, num_walls, num_mice, num_rockets, num_cats, num_holes, arrow_budget, bfs_solution, difficulty_label
- **Solution format**: `[[x,y,direction], ...]` where direction is LEFT/RIGHT/UP/DOWN
- **Board sizes**: Variable (5×5 to 14×10) requiring action masking
- **Distribution**: 16k Easy, 7k Medium, 3 Hard puzzles

## Training Components

### 1. BC-Set Data Pipeline (`bc_set_data_loader.py`)
- Parse CSV and convert BFS solutions to multi-hot target vectors
- Generate final board states with **all BFS arrows placed**
- **Generate action masks** for variable board sizes (mark invalid tiles as 0)
- Create state-multihot pairs: (final_perception_output, arrow_set_target, action_mask)
- **Target format**: 700-dim binary vector with 1s for each BFS arrow location+direction
- Split: 80% train, 10% val, 10% test (puzzle-level, not episode-level)
- Batch processing with configurable batch size

### 2. Action Masking (`action_masking.py`)
- **Dynamic mask generation**: For board size W×H, mask actions where tile_index ≥ W×H
- **5×5 board example**: Mask actions for tile indices 25-139 (set probability to 0)
- **Probability renormalization**: After masking, renormalize valid action probabilities
- **Loss masking**: Cross-entropy loss ignores masked logits (no gradient flow)

### 3. BC-Set Training Loop (`bc_set_trainer.py`)
- **Masked binary cross-entropy loss**: Multi-label classification with action masking
- **Loss function**: `BCE(sigmoid(logits[valid_actions]), targets[valid_actions])`
- **Metrics**: Jaccard accuracy, precision, recall on arrow set prediction
- Adam optimizer with learning rate scheduling
- Validation every N steps with set-prediction metrics
- Model checkpointing and early stopping at 70% Jaccard accuracy

### 4. Evaluation (`evaluator.py`)
- Load test puzzles and run full episodes
- **Apply action masks** during inference for each board size
- Measure solve rate by difficulty level and board size
- Track steps-to-solution vs BFS optimal
- Generate training curves and performance reports

### 5. Model Management (`model_manager.py`)
- Save/load model weights to `model/bc_checkpoint_*.pth`
- Handle model versioning and best model tracking
- Export final trained model for PPO phase

## Phase-Specific Success Criteria

### Phase 1: BC-Set
- **Primary**: ≥70% Jaccard accuracy on arrow set prediction
- **Secondary**: ≥80% precision, ≥60% recall on individual arrows
- **Class balance**: Handle sparse targets (≈3 positives out of 700)

### Phase 2: BC-Sequence Fine-tuning  
- **Primary**: ≥60% first-action accuracy on initial states
- **Secondary**: Maintain BC-Set arrow knowledge while adding ordering

### Phase 3: PPO (Future)
- **Primary**: ≥95% solve rate on Easy puzzles
- **Secondary**: ≥70% solve rate on Medium puzzles  
- **Efficiency**: Average steps ≤ 1.5× BFS optimal

## Configuration
- Batch size: 64
- Learning rate: 1e-3 with cosine annealing
- Max epochs: 100 with early stopping
- Validation frequency: Every 500 steps
- Gradient clipping: 1.0

## Action Masking Implementation

### Mask Generation
For a board of size W×H:
- **Valid tiles**: 0 to (W×H - 1)  
- **Invalid tiles**: W×H to 139 (for 14×10 max)
- **Action types**: Place arrows (4 types) + erase = 5 actions per tile
- **Total actions**: 700 (140 tiles × 5 actions)

### Masking Strategy
```python
def create_action_mask(board_w: int, board_h: int) -> torch.Tensor:
    """Create action mask for given board size."""
    mask = torch.zeros(700)  # All invalid by default
    max_tile = board_w * board_h
    
    # Enable valid tiles for all 5 action types
    for action_type_offset in [0, 140, 280, 420, 560]:  # up, down, left, right, erase
        mask[action_type_offset:action_type_offset + max_tile] = 1.0
    
    return mask
```

### Training Integration
- **Data loader**: Generate masks alongside state-action pairs
- **Loss function**: Use masked cross-entropy (ignore invalid actions)
- **Inference**: Apply mask before action sampling/selection

## File Structure
```
src/training/bc/
├── PRD.md
├── __init__.py
├── action_masking.py        # Action mask generation utilities
├── bc_set_data_loader.py    # BC-Set: Multi-hot target generation
├── bc_set_trainer.py        # BC-Set: Multi-label training loop
├── bc_sequence_trainer.py   # BC-Sequence: Fine-tuning from BC-Set
├── data_loader.py           # Legacy sequential data loader
├── trainer.py               # Legacy sequential trainer
├── evaluator.py             # Performance evaluation with masking
├── model_manager.py         # Save/load utilities
└── config.py                # Training hyperparameters

model/
├── bc_set_best.pth          # Phase 1: Best BC-Set model
├── bc_sequence_best.pth     # Phase 2: Fine-tuned sequence model
└── bc_final.pth             # Phase 3: Ready for PPO initialization
```

## Dependencies
- Uses existing perception, state_fusion, and policy layers
- Requires game engine for state generation from CSV parameters
- PyTorch for training infrastructure