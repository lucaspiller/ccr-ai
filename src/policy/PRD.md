# Policy Head PRD

## 1. Overview

The Policy Head is the decision-making component of our ChuChu Rocket AI architecture. It takes the 128-dimensional fused state embedding and outputs a probability distribution over all possible game actions, enabling the agent to decide where and what to place on each tick.

**Input**: 128-dimensional fused embedding from state-fusion layer  
**Output**: 700-dimensional action probability distribution (softmax)  
**Role**: Action selection and decision-making for optimal tile placement

## 2. Action Space Design

### 2.1 Action Categories
- **Arrow Placement**: Place directional arrow (↑↓←→) at specific tile
- **Erase Action**: Remove existing arrow from specific tile

### 2.2 Action Encoding
```
Total Actions: 700
├── Arrow Placement: 560 actions
│   ├── Up arrows: 140 actions (tiles 0-139)
│   ├── Down arrows: 140 actions (tiles 140-279) 
│   ├── Left arrows: 140 actions (tiles 280-419)
│   └── Right arrows: 140 actions (tiles 420-559)
└── Erase Actions: 140 actions (tiles 560-699)
```

### 2.3 Tile Indexing
- Board: 10 height × 14 width = 140 tiles
- Tile index = row * 14 + col
- Action index calculation:
  - Place Up: tile_idx
  - Place Down: tile_idx + 140
  - Place Left: tile_idx + 280
  - Place Right: tile_idx + 420
  - Erase: tile_idx + 560

## 3. Architecture Requirements

### 3.1 Network Design
According to AI overview specification:
- **Layer 1**: 128 → 256 (Linear + ReLU)
- **Layer 2**: 256 → 700 (Linear + Softmax)
- **Parameter Count**: ~40K parameters
- **Activation**: ReLU for hidden layers, Softmax for output

### 3.2 Output Specifications
- **Shape**: (700,) probability distribution
- **Range**: [0, 1] with sum = 1.0
- **Semantics**: P(action | state) for each possible action

## 4. Technical Design

### 4.1 Core Components

#### PolicyHead (nn.Module)
```python
class PolicyHead(nn.Module):
    def __init__(self, config: PolicyConfig):
        # 128 → 256 → 700 MLP
        # ReLU activations
        # Optional dropout during training
    
    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        # Returns action logits before softmax
    
    def get_action_probabilities(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        # Returns softmax probabilities
```

#### PolicyConfig
```python
@dataclass
class PolicyConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 700
    dropout_rate: float = 0.1
    use_bias: bool = True
    weight_init: str = "xavier"
```

#### PolicyOutput
```python
@dataclass
class PolicyOutput:
    action_logits: torch.Tensor      # Raw logits [700]
    action_probs: torch.Tensor       # Softmax probabilities [700]
    selected_action: Optional[int]   # Sampled/selected action index
    action_type: Optional[str]       # "place_up", "place_down", "place_left", "place_right", "erase"
    target_tile: Optional[Tuple[int, int]]  # (row, col) for action
    confidence: Optional[float]      # Max probability in distribution
```

### 4.2 Action Decoding
- **decode_action(action_idx: int) → ActionInfo**: Convert action index to game action
- **encode_action(action_type: str, tile: Tuple[int, int]) → int**: Convert game action to index
- **get_valid_actions(game_state) → List[int]**: Get legal actions for current state

### 4.3 Sampling Strategies
- **Deterministic**: Select highest probability action
- **Categorical**: Sample from probability distribution
- **Top-k**: Sample from top-k highest probability actions
- **Temperature**: Scale logits before softmax for exploration control

## 5. Integration Points

### 5.1 Input Interface
- Receives `FusedStateOutput.fused_embedding` from state-fusion layer
- Compatible with batch processing for training
- Handles both single inference and batch training modes

### 5.2 Output Interface
- Provides action probabilities for action selection
- Supplies logits for policy gradient training (PPO)
- Supports action masking via planner integration

### 5.3 Training Integration
- **Behavior Cloning**: Supervised learning from optimal action labels
- **PPO Training**: Policy gradient updates using advantage estimates
- **Action Masking**: Integration with rule-based planner for safety

## 6. Performance Requirements

### 6.1 Latency Targets
- **Forward Pass**: < 0.05ms on target hardware
- **Batch Processing**: > 1000 decisions/second
- **Memory Usage**: < 10MB for inference

### 6.2 Quality Metrics
- **Action Accuracy**: Match expert demonstrations (>95% on simple puzzles)
- **Exploration Balance**: Maintain entropy for learning
- **Convergence Speed**: Fast policy gradient updates

## 7. Configuration Options

### 7.1 Architecture Parameters
- `hidden_dim`: Hidden layer size (default: 256)
- `dropout_rate`: Training dropout (default: 0.1)
- `use_bias`: Enable bias terms (default: True)
- `weight_init`: Initialization strategy (default: "xavier")

### 7.2 Training Parameters
- `temperature`: Softmax temperature for exploration (default: 1.0)
- `action_mask_enabled`: Enable planner action masking (default: False)
- `gradient_clipping`: Max gradient norm (default: 1.0)

### 7.3 Inference Parameters
- `sampling_strategy`: "deterministic", "categorical", "top_k"
- `top_k`: Number of top actions for top-k sampling (default: 10)
- `exploration_noise`: Add noise for exploration (default: 0.0)

## 8. Validation & Monitoring

### 8.1 Output Validation
- Probability distribution sums to 1.0
- No NaN or infinite values in outputs
- Action indices within valid range [0, 699]

### 8.2 Performance Monitoring
- Action entropy tracking (exploration measure)
- Confidence distribution analysis
- Action type frequency monitoring
- Invalid action rate tracking

### 8.3 Training Monitoring
- Policy loss convergence
- Gradient norms and flow
- Action prediction accuracy
- KL divergence from previous policy

## 9. Action Space Utilities

### 9.1 Action Conversion Functions
```python
def decode_action(action_idx: int) -> ActionInfo:
    """Convert action index to structured action information."""

def encode_action(action_type: str, row: int, col: int) -> int:
    """Convert structured action to index."""

def get_tile_index(row: int, col: int) -> int:
    """Convert board coordinates to tile index."""

def get_tile_coords(tile_idx: int) -> Tuple[int, int]:
    """Convert tile index to board coordinates."""
```

### 9.2 Action Filtering
```python
def get_placement_actions(tile_idx: int) -> List[int]:
    """Get all arrow placement actions for a tile."""

def get_erase_action(tile_idx: int) -> int:
    """Get erase action for a tile."""

def filter_valid_actions(action_probs: torch.Tensor, game_state: Dict) -> torch.Tensor:
    """Apply game rules to mask invalid actions."""
```

## 10. Testing Strategy

### 10.1 Unit Tests
- Action encoding/decoding correctness
- Probability distribution validation
- Network architecture verification
- Batch processing functionality

### 10.2 Integration Tests
- End-to-end pipeline with state-fusion layer
- Action execution in game engine
- Training loop integration
- Planner masking compatibility

### 10.3 Performance Tests
- Latency benchmarking
- Memory usage profiling
- Batch throughput measurement
- Action quality evaluation

## 11. Success Criteria

### 11.1 Functional Requirements
- ✅ Outputs valid 700-dimensional probability distributions
- ✅ Correctly encodes/decodes all action types
- ✅ Integrates with state-fusion layer inputs
- ✅ Supports both training and inference modes

### 11.2 Performance Requirements
- ✅ Meets latency targets (< 0.05ms forward pass)
- ✅ Achieves target parameter count (~40K)
- ✅ Maintains stable training dynamics

### 11.3 Quality Requirements
- ✅ >95% accuracy on behavior cloning
- ✅ Effective exploration during RL training
- ✅ Sensible action selection on game scenarios

## 12. Dependencies

### 12.1 Input Dependencies
- `src/state_fusion/data_types.py`: FusedStateOutput
- `src/state_fusion/processors.py`: State-fusion layer

### 12.2 Output Dependencies
- Training loop: Requires action logits for PPO
- Game engine: Needs decoded actions for execution
- Planner: May need action masking integration

## 13. Future Extensions

### 13.1 Advanced Architectures
- Multi-head attention over action space
- Hierarchical action selection (tile → direction)
- Convolutional layers for spatial action reasoning

### 13.2 Enhanced Training
- Curriculum learning with action difficulty
- Imitation learning from human players
- Self-play improvement mechanisms

### 13.3 Action Space Extensions
- Multi-tile actions (arrow sequences)
- Conditional actions (if-then placements)
- Resource-aware action selection