# Perception Layer PRD

## 1. Overview

The Perception layer converts raw ChuChu Rocket game state (from GameEngine.to_dict()) into neural network-ready tensor representations. It implements the state representation v3 from the AI overview, producing a 28-channel grid tensor and 16-dimensional global feature vector.

## 2. Input Format

### 2.1 GameEngine.to_dict() Structure
```python
{
    "board": {
        "width": 14, "height": 10,
        "grid": [[cell_values...]],  # CellType enum values
        "arrows": {"x,y": "direction_name"},
        "walls": [((x1,y1), (x2,y2))],
        "max_arrows": 3
    },
    "sprite_manager": {
        "sprites": {
            "sprite_id": {
                "type": "mouse|cat|rocket|spawner|gold_mouse|bonus_mouse",
                "x": float, "y": float,
                "state": "active|captured|escaped", 
                "direction": "UP|DOWN|LEFT|RIGHT",
                "move_interval_ticks": int,
                "ticks_since_last_move": int,
                # Type-specific fields:
                "mice_collected": int,  # rockets only
                "spawn_direction": str  # spawners only
            }
        }
    },
    "current_step": int,
    "current_tick": int,
    "max_steps": int,
    "result": "ongoing|success|failure|timeout",
    "bonus_state": {
        "mode": "none|mouse_mania|cat_mania|speed_up|slow_down|...",
        "remaining_ticks": int,
        "duration_ticks": int
    }
}
```

## 3. Output Format

### 3.1 Grid Tensor: `torch.Tensor[28, height, width]`

| Channel | Name | Range | Description |
|---------|------|-------|-------------|
| 0 | Wall Vertical | {0,1} | Wall between tile and right neighbor |
| 1 | Wall Horizontal | {0,1} | Wall between tile and tile below |
| 2-9 | Rocket P0-P7 | {0,1} | Player rocket masks (8 players max) |
| 10 | Spawner | {0,1} | Tile has spawner |
| 11-14 | Arrow ↑↓←→ | {0,1} | Arrow direction geometry |
| 15 | Arrow Owner ID | 0-8 | Player owning arrow (0=none, 1-8=player+1) |
| 16 | Arrow Health | 0-2 | 0=none, 1=healthy, 2=shrunk |
| 17-20 | Mouse Flow ↑↓←→ | [0,1] | Directional mouse flow (normalized) |
| 21 | Mouse Flow Confidence | [0,1] | Total flow magnitude (normalized) |
| 22-25 | Cat Orientation ↑↓←→ | {0,1} | Cat positions by facing direction |
| 26 | Gold Mouse | {0,1} | Gold mouse present |
| 27 | Bonus Mouse | {0,1} | Bonus mouse present |

### 3.2 Global Features: `torch.Tensor[16]`

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | Remaining Ticks | float | (max_steps - current_step) / 10000 |
| 1 | Arrow Budget | float | (max_arrows - placed_arrows) / 3 |
| 2 | Live Cat Count | float | active_cats / 16 |
| 3-7 | Bonus State | one-hot | [None, Mouse Mania, Cat Mania, Speed Up, Slow Down] |
| 8-15 | Player Scores | float | mice_collected[p0-p7] / 100 |

### 3.3 Cat Set Encoding: `torch.Tensor[32]`

Fixed-size embedding from variable cat list via set encoder:
- Per-cat features: [x_norm, y_norm, dir_onehot(4), threat_features(2)]
- Shared MLP: 8 → 32 → 32 with ReLU
- Max pooling over all cats
- LayerNorm on result

## 4. Core Components

### 4.1 GameStateProcessor
```python
class GameStateProcessor:
    def __init__(self, board_width: int = 14, board_height: int = 10):
        self.width = board_width  
        self.height = board_height
        
    def process(self, game_state: Dict[str, Any]) -> PerceptionOutput:
        """Convert GameEngine.to_dict() to tensor representation"""
```

### 4.2 GridEncoder
```python
class GridEncoder:
    def encode_walls(self, walls: List, grid_shape: Tuple) -> torch.Tensor
    def encode_rockets(self, sprites: Dict, grid_shape: Tuple) -> torch.Tensor  
    def encode_arrows(self, arrows: Dict, grid_shape: Tuple) -> torch.Tensor
    def encode_mouse_flow(self, sprites: Dict, grid_shape: Tuple) -> torch.Tensor
    def encode_cats(self, sprites: Dict, grid_shape: Tuple) -> torch.Tensor
    def encode_special_mice(self, sprites: Dict, grid_shape: Tuple) -> torch.Tensor
```

### 4.3 GlobalFeatureExtractor
```python
class GlobalFeatureExtractor:
    def extract_timing_features(self, game_state: Dict) -> torch.Tensor
    def extract_bonus_features(self, bonus_state: Dict) -> torch.Tensor  
    def extract_score_features(self, sprites: Dict) -> torch.Tensor
```

### 4.4 CatSetEncoder  
```python
class CatSetEncoder(nn.Module):
    def __init__(self):
        self.cat_mlp = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.layer_norm = nn.LayerNorm(32)
        
    def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
        # cat_features: [num_cats, 8]
        # Returns: [32] via max pooling
```

### 4.5 PerceptionOutput
```python
@dataclass  
class PerceptionOutput:
    grid_tensor: torch.Tensor      # [28, height, width]
    global_features: torch.Tensor  # [16]  
    cat_embedding: torch.Tensor    # [32]
    
    def get_combined_embedding(self) -> torch.Tensor:
        """Flatten grid, concatenate with global + cat features"""
        # Returns: [28*H*W + 16 + 32] for fusion MLP
```

## 5. Implementation Requirements

### 5.1 Core Modules
- `src/perception/processors.py` - GameStateProcessor
- `src/perception/encoders.py` - GridEncoder, GlobalFeatureExtractor  
- `src/perception/cat_encoder.py` - CatSetEncoder neural module
- `src/perception/data_types.py` - PerceptionOutput, constants
- `src/perception/__init__.py` - Public API

### 5.2 Key Algorithms

#### Mouse Flow Computation
1. Group mice by tile position
2. Calculate net movement direction over next 4 ticks using current velocity
3. Compute majority direction and confidence per tile
4. Normalize by maximum expected flow

#### Wall Encoding
1. Parse wall edge list: `[((x1,y1), (x2,y2))]`
2. For vertical walls (x1==x2): set bit in left tile's channel 0
3. For horizontal walls (y1==y2): set bit in upper tile's channel 1
4. Handle board wrapping correctly

#### Cat Threat Features
1. `shrunk_arrow_ahead`: Check if tile in movement direction has damaged arrow
2. `dist_to_enemy_rocket`: L1 distance to nearest non-player rocket / 20

### 5.3 Performance Requirements
- Process game state in <1ms on CPU
- Support batch processing for training
- Memory efficient: reuse tensors where possible
- Deterministic output for same input

### 5.4 Error Handling
- Validate input game state structure
- Handle missing/malformed sprite data gracefully  
- Clamp values to expected ranges
- Log warnings for unexpected conditions

## 6. Testing Strategy

### 6.1 Unit Tests
- Test each encoder component independently
- Verify tensor shapes and value ranges
- Test edge cases (empty board, max sprites, etc.)
- Property-based testing for consistency

### 6.2 Integration Tests  
- End-to-end processing of real game states
- Performance benchmarking
- Memory usage validation
- Determinism verification

### 6.3 Visual Validation
- Render grid tensors as images for debugging
- Compare mouse flow with actual sprite movements
- Validate wall placement visually

## 7. Success Criteria

1. **Correctness**: All 28 grid channels encode expected information
2. **Performance**: <1ms processing time for single game state
3. **Completeness**: Handle all sprite types and game states
4. **Robustness**: No crashes on malformed input
5. **Testability**: >95% test coverage
6. **Integration**: Ready for State-Fusion layer consumption

## 8. Future Extensions

- Multi-tick mouse flow prediction (beyond 4 ticks)
- Player-aware arrow ownership in multiplayer
- Dynamic board size support
- Compressed state representation for memory efficiency