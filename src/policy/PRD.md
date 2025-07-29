# Policy & Value Head PRD

## 1. Overview

The Policy and Value Heads form the dual-output decision-making component of our ChuChu Rocket AI architecture. Both heads consume the same 128-dimensional fused state embedding to produce complementary outputs for reinforcement learning.

**Shared Input**: 128-dimensional fused embedding from state-fusion layer  
**Policy Output**: 700-dimensional action probability distribution (softmax)  
**Value Output**: Single scalar value estimate for current state  
**Role**: Action selection (policy) and state evaluation (value) for PPO training

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

### 3.1 Policy Head Network Design
According to AI overview specification:
- **Layer 1**: 128 → 256 (Linear + ReLU)
- **Layer 2**: 256 → 700 (Linear + Softmax)
- **Parameter Count**: ~40K parameters
- **Activation**: ReLU for hidden layers, Softmax for output

### 3.2 Value Head Network Design
According to AI overview specification:
- **Layer 1**: 128 → 64 (Linear + ReLU)
- **Layer 2**: 64 → 1 (Linear)
- **Parameter Count**: ~8K parameters
- **Activation**: ReLU for hidden layer, no activation for scalar output

### 3.3 Output Specifications
#### Policy Head
- **Shape**: (700,) probability distribution
- **Range**: [0, 1] with sum = 1.0
- **Semantics**: P(action | state) for each possible action

#### Value Head
- **Shape**: (1,) scalar value
- **Range**: Unbounded real number
- **Semantics**: Expected cumulative reward from current state

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
```

#### ValueHead (nn.Module)
```python
class ValueHead(nn.Module):
    def __init__(self, config: ValueConfig):
        # 128 → 64 → 1 MLP
        # ReLU activation for hidden layer
        # No activation for output
    
    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        # Returns scalar value estimate
```

#### PolicyProcessor (nn.Module)
```python
class PolicyProcessor(nn.Module):
    def __init__(self, policy_config: PolicyConfig, value_config: ValueConfig):
        self.policy_head = PolicyHead(policy_config)
        self.value_head = ValueHead(value_config)
    
    def forward(self, fused_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns (action_logits, value_estimate)
```

#### PolicyConfig & ValueConfig
```python
@dataclass
class PolicyConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 700
    dropout_rate: float = 0.1
    use_bias: bool = True
    weight_init: str = "xavier"

@dataclass
class ValueConfig:
    input_dim: int = 128
    hidden_dim: int = 64
    output_dim: int = 1
    use_bias: bool = True
    weight_init: str = "xavier"
```

#### PolicyValueOutput
```python
@dataclass
class PolicyValueOutput:
    action_logits: torch.Tensor      # Raw logits [700]
    action_probs: torch.Tensor       # Softmax probabilities [700]
    value_estimate: torch.Tensor     # State value [1]
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
- **PolicyProcessor**: Returns unified (logits, value) tuple
- **Policy Head**: Provides action probabilities for action selection and logits for PPO
- **Value Head**: Supplies state value estimates for advantage calculation
- Supports action masking via planner integration

### 5.3 Training Integration
- **Behavior Cloning**: Policy head trained on optimal action labels (value head optional)
- **PPO Training**: Both heads trained jointly using policy gradients and value function loss
- **Advantage Calculation**: Value head predictions used for GAE computation
- **Parameter Groups**: Both heads can share optimizer settings or use different learning rates

## 6. Performance Requirements

### 6.1 Latency Targets
- **Combined Forward Pass**: < 0.06ms on target hardware (both heads)
- **Batch Processing**: > 1000 decisions/second
- **Memory Usage**: < 15MB for inference (includes value head)

### 6.2 Quality Metrics
- **Policy Accuracy**: Match expert demonstrations (>95% on simple puzzles)
- **Value Accuracy**: Low MSE on return predictions during PPO
- **Exploration Balance**: Maintain entropy for learning
- **Convergence Speed**: Fast joint training of both heads

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