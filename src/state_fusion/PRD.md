# State-Fusion Layer PRD

## 1. Overview

The State-Fusion layer is the second layer in our ChuChu Rocket AI architecture. It takes the multi-modal perception outputs (grid tensor, global features, cat embedding) and fuses them into a single 128-dimensional latent representation that summarizes "what's going on this tick."

**Input**: `PerceptionOutput` from the perception layer  
**Output**: 128-dimensional fused embedding tensor  
**Role**: Feature fusion and dimensionality compression for decision-making

## 2. Architecture Requirements

### 2.1 Input Processing
- **Grid tensor**: 28 × 9 × 12 spatial features from CNN encoder
- **Global features**: 16-dimensional normalized game state vector  
- **Cat embedding**: 32-dimensional set-encoded cat representation

### 2.2 Fusion Strategy
According to the AI overview, this layer should:
1. Concatenate all perception outputs
2. Pass through a small MLP to compress into 128-d latent
3. Provide stable, information-dense representation for policy/value heads

### 2.3 Output Specification
- **Shape**: (128,) tensor per game state
- **Range**: Normalized activations suitable for downstream heads
- **Semantics**: Dense encoding of current game situation

## 3. Technical Design

### 3.1 Concatenation Strategy
```
Total input dims = (28 × 9 × 12) + 16 + 32 = 3072 + 16 + 32 = 3120
```

### 3.2 MLP Architecture
- **Input**: 3120-dimensional concatenated features
- **Hidden**: 512 → 256 → 128 (2-3 layer MLP with ReLU)
- **Output**: 128-dimensional fused embedding
- **Normalization**: LayerNorm on output for stability
- **Dropout**: Optional 0.1-0.2 dropout during training

### 3.3 Parameter Count
Estimated ~1.8M parameters for the fusion MLP:
- Layer 1: 3120 × 512 = ~1.6M params
- Layer 2: 512 × 256 = ~130K params  
- Layer 3: 256 × 128 = ~33K params
- Total: ~1.8M params

## 4. Implementation Components

### 4.1 Core Classes
- `StateFusionProcessor`: Main fusion logic
- `FusionConfig`: Configuration for MLP architecture
- `FusedStateOutput`: Output data structure

### 4.2 Key Methods
- `fuse(perception_output: PerceptionOutput) -> torch.Tensor`
- `get_output_shape() -> tuple`
- `enable_dropout(enabled: bool)`

### 4.3 Validation & Monitoring
- Input shape validation
- Output range checking
- Gradient flow monitoring
- Activation statistics tracking

## 5. Performance Requirements

### 5.1 Latency Targets
- **Forward pass**: < 0.1ms on target hardware
- **Memory usage**: < 50MB for batch processing
- **Throughput**: > 1000 fusions/second

### 5.2 Quality Metrics
- Stable gradients during training
- No activation saturation
- Preserves information from all input modalities

## 6. Data Flow

```
PerceptionOutput
├── grid_tensor (28×9×12)
├── global_features (16,)  
└── cat_embedding (32,)
           ↓
    Flatten & Concat
           ↓
   [3120-d vector]
           ↓
      Fusion MLP
           ↓
   [128-d embedding] → Policy/Value Heads
```

## 7. Configuration Options

### 7.1 Architecture Parameters
- `hidden_dims`: List of hidden layer sizes (default: [512, 256])
- `dropout_rate`: Training dropout rate (default: 0.1)
- `use_layer_norm`: Enable output normalization (default: True)
- `activation`: Activation function (default: ReLU)

### 7.2 Training Parameters
- `gradient_clipping`: Max gradient norm (default: 1.0)
- `weight_init`: Initialization strategy (default: Xavier)

## 8. Testing Strategy

### 8.1 Unit Tests
- Shape compatibility with various input sizes
- Gradient computation correctness
- Configuration parameter validation

### 8.2 Integration Tests  
- End-to-end pipeline with perception layer
- Batch processing functionality
- Memory usage under load

### 8.3 Performance Tests
- Latency benchmarking
- Memory profiling
- Throughput measurement

## 9. Success Criteria

### 9.1 Functional
- ✅ Processes all perception outputs correctly
- ✅ Outputs stable 128-d embeddings
- ✅ Integrates with policy/value heads

### 9.2 Performance
- ✅ Meets latency requirements (< 0.1ms)
- ✅ Stable training dynamics
- ✅ Preserves information from inputs

### 9.3 Quality
- ✅ No gradient explosion/vanishing
- ✅ Consistent output distributions
- ✅ Good downstream task performance

## 10. Dependencies

### 10.1 Input Dependencies
- `src/perception/data_types.py`: PerceptionOutput definition
- `src/perception/processors.py`: Perception layer output

### 10.2 Output Dependencies  
- Policy head: Consumes 128-d embedding
- Value head: Consumes 128-d embedding
- Training loop: Requires stable gradients

## 11. Future Extensions

### 11.1 Attention Mechanisms
- Self-attention over spatial grid features
- Cross-attention between modalities

### 11.2 Residual Connections
- Skip connections for gradient flow
- Multi-scale feature fusion

### 11.3 Dynamic Architecture
- Adaptive hidden sizes based on game complexity
- Learnable fusion weights