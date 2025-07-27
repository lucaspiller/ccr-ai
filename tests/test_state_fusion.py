"""
Tests for the state-fusion layer.
"""

import pytest
import torch

from src.perception.data_types import PerceptionOutput
from src.state_fusion.data_types import FusionConfig, FusedStateOutput
from src.state_fusion.processors import StateFusionProcessor, fuse_perception_output


class TestFusionConfig:
    """Test FusionConfig data class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FusionConfig()
        assert config.hidden_dims == [512, 256]
        assert config.dropout_rate == 0.1
        assert config.use_layer_norm is True
        assert config.activation == "relu"
        assert config.gradient_clipping == 1.0
        assert config.weight_init == "xavier"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FusionConfig(
            hidden_dims=[256, 128],
            dropout_rate=0.2,
            use_layer_norm=False,
            activation="gelu",
            gradient_clipping=None,
            weight_init="kaiming"
        )
        assert config.hidden_dims == [256, 128]
        assert config.dropout_rate == 0.2
        assert config.use_layer_norm is False
        assert config.activation == "gelu"
        assert config.gradient_clipping is None
        assert config.weight_init == "kaiming"
    
    def test_invalid_dropout_rate(self):
        """Test validation of dropout rate."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            FusionConfig(dropout_rate=-0.1)
        
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            FusionConfig(dropout_rate=1.5)
    
    def test_invalid_activation(self):
        """Test validation of activation function."""
        with pytest.raises(ValueError, match="activation must be one of"):
            FusionConfig(activation="invalid")
    
    def test_invalid_weight_init(self):
        """Test validation of weight initialization."""
        with pytest.raises(ValueError, match="weight_init must be one of"):
            FusionConfig(weight_init="invalid")
    
    def test_empty_hidden_dims(self):
        """Test validation of hidden dimensions."""
        with pytest.raises(ValueError, match="hidden_dims must contain at least one layer size"):
            FusionConfig(hidden_dims=[])


class TestFusedStateOutput:
    """Test FusedStateOutput data class."""
    
    def test_valid_output(self):
        """Test creating valid fused state output."""
        embedding = torch.randn(128)
        output = FusedStateOutput(
            fused_embedding=embedding,
            source_step=10,
            source_tick=20,
            fusion_time_ms=1.5
        )
        assert torch.equal(output.fused_embedding, embedding)
        assert output.source_step == 10
        assert output.source_tick == 20
        assert output.fusion_time_ms == 1.5
    
    def test_invalid_embedding_shape(self):
        """Test validation of embedding shape."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="fused_embedding must be 1D"):
            FusedStateOutput(fused_embedding=torch.randn(1, 128))
        
        # Wrong size
        with pytest.raises(ValueError, match="fused_embedding must have 128 dimensions"):
            FusedStateOutput(fused_embedding=torch.randn(64))
    
    def test_to_device(self):
        """Test moving output to different device."""
        embedding = torch.randn(128)
        output = FusedStateOutput(fused_embedding=embedding)
        
        # Test moving to CPU (should work regardless of current device)
        output_cpu = output.to_device(torch.device("cpu"))
        assert output_cpu.fused_embedding.device == torch.device("cpu")
        assert output_cpu.source_step == output.source_step
    
    def test_detach(self):
        """Test detaching tensors from computation graph."""
        embedding = torch.randn(128, requires_grad=True)
        output = FusedStateOutput(fused_embedding=embedding)
        
        detached_output = output.detach()
        assert not detached_output.fused_embedding.requires_grad
        assert detached_output.source_step == output.source_step


class TestStateFusionProcessor:
    """Test StateFusionProcessor class."""
    
    def create_mock_perception_output(self) -> PerceptionOutput:
        """Create a mock perception output for testing."""
        return PerceptionOutput(
            grid_tensor=torch.randn(28, 10, 14),
            global_features=torch.randn(16),
            cat_embedding=torch.randn(32),
            source_step=5,
            source_tick=10,
            cat_count=3,
            encoding_time_ms=2.0
        )
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        # Default config
        processor = StateFusionProcessor()
        assert processor.config is not None
        assert isinstance(processor.fusion_mlp, torch.nn.Module)
        
        # Custom config
        config = FusionConfig(hidden_dims=[256])
        processor = StateFusionProcessor(config)
        assert processor.config == config
    
    def test_fuse_single_output(self):
        """Test fusing a single perception output."""
        processor = StateFusionProcessor()
        perception_output = self.create_mock_perception_output()
        
        fused_output = processor.fuse(perception_output)
        
        assert isinstance(fused_output, FusedStateOutput)
        assert fused_output.fused_embedding.shape == (128,)
        assert fused_output.source_step == 5
        assert fused_output.source_tick == 10
        assert fused_output.fusion_time_ms is None  # Profiling disabled by default
    
    def test_fuse_with_profiling(self):
        """Test fusing with profiling enabled."""
        processor = StateFusionProcessor()
        processor.enable_profiling()
        
        perception_output = self.create_mock_perception_output()
        fused_output = processor.fuse(perception_output)
        
        assert fused_output.fusion_time_ms is not None
        assert fused_output.fusion_time_ms > 0
    
    def test_fuse_batch(self):
        """Test fusing batch of perception outputs."""
        processor = StateFusionProcessor()
        perception_outputs = [self.create_mock_perception_output() for _ in range(3)]
        
        fused_outputs = processor.fuse_batch(perception_outputs)
        
        assert len(fused_outputs) == 3
        for output in fused_outputs:
            assert isinstance(output, FusedStateOutput)
            assert output.fused_embedding.shape == (128,)
    
    def test_fuse_batch_tensor(self):
        """Test fusing batch into single tensor."""
        processor = StateFusionProcessor()
        perception_outputs = [self.create_mock_perception_output() for _ in range(3)]
        
        batch_tensor = processor.fuse_batch_tensor(perception_outputs)
        
        assert batch_tensor.shape == (3, 128)
    
    def test_input_validation(self):
        """Test input validation."""
        processor = StateFusionProcessor()
        
        # Wrong grid tensor shape
        invalid_output = PerceptionOutput(
            grid_tensor=torch.randn(28, 9, 14),  # Wrong height
            global_features=torch.randn(16),
            cat_embedding=torch.randn(32),
            _validate_shapes=False
        )
        
        with pytest.raises(ValueError, match="Grid tensor shape mismatch"):
            processor.fuse(invalid_output)
        
        # Wrong global features shape
        invalid_output = PerceptionOutput(
            grid_tensor=torch.randn(28, 10, 14),
            global_features=torch.randn(15),  # Wrong size
            cat_embedding=torch.randn(32),
            _validate_shapes=False
        )
        
        with pytest.raises(ValueError, match="Global features shape mismatch"):
            processor.fuse(invalid_output)
        
        # Wrong cat embedding shape
        invalid_output = PerceptionOutput(
            grid_tensor=torch.randn(28, 10, 14),
            global_features=torch.randn(16),
            cat_embedding=torch.randn(31),  # Wrong size
            _validate_shapes=False
        )
        
        with pytest.raises(ValueError, match="Cat embedding shape mismatch"):
            processor.fuse(invalid_output)
    
    def test_metrics_tracking(self):
        """Test metrics tracking."""
        processor = StateFusionProcessor()
        processor.enable_profiling()
        
        # Process some outputs
        for _ in range(3):
            perception_output = self.create_mock_perception_output()
            processor.fuse(perception_output)
        
        metrics = processor.get_metrics()
        assert metrics["profiling_enabled"] is True
        assert metrics["fusion_times"]["count"] == 3
        assert "mean_ms" in metrics["fusion_times"]
        
        # Reset metrics
        processor.reset_metrics()
        metrics = processor.get_metrics()
        assert "fusion_times" not in metrics
    
    def test_error_tracking(self):
        """Test error tracking."""
        processor = StateFusionProcessor()
        
        # Cause a shape error
        invalid_output = PerceptionOutput(
            grid_tensor=torch.randn(28, 9, 14),
            global_features=torch.randn(16),
            cat_embedding=torch.randn(32),
            _validate_shapes=False
        )
        
        with pytest.raises(ValueError):
            processor.fuse(invalid_output)
        
        metrics = processor.get_metrics()
        assert metrics["error_counts"]["general"] == 1
        assert metrics["error_counts"]["shape"] == 1
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        config = FusionConfig(hidden_dims=[512, 256])
        processor = StateFusionProcessor(config)
        
        param_count = processor.get_parameter_count()
        assert param_count > 0
        
        # Rough calculation: 3968->512, 512->256, 256->128
        # Plus biases and layer norm if enabled
        expected_min = (3968 * 512) + (512 * 256) + (256 * 128)
        assert param_count >= expected_min
    
    def test_train_eval_modes(self):
        """Test training and evaluation modes."""
        processor = StateFusionProcessor()
        
        # Set to training mode
        processor.train()
        assert processor.fusion_mlp.training is True
        
        # Set to eval mode
        processor.eval()
        assert processor.fusion_mlp.training is False
    
    def test_convenience_function(self):
        """Test convenience function."""
        perception_output = self.create_mock_perception_output()
        fused_output = fuse_perception_output(perception_output)
        
        assert isinstance(fused_output, FusedStateOutput)
        assert fused_output.fused_embedding.shape == (128,)


class TestDifferentConfigurations:
    """Test processor with different configurations."""
    
    def create_mock_perception_output(self) -> PerceptionOutput:
        """Create a mock perception output for testing."""
        return PerceptionOutput(
            grid_tensor=torch.randn(28, 10, 14),
            global_features=torch.randn(16),
            cat_embedding=torch.randn(32)
        )
    
    def test_different_hidden_sizes(self):
        """Test with different hidden layer sizes."""
        configs = [
            FusionConfig(hidden_dims=[256]),
            FusionConfig(hidden_dims=[1024, 512, 256]),
            FusionConfig(hidden_dims=[128, 64])
        ]
        
        perception_output = self.create_mock_perception_output()
        
        for config in configs:
            processor = StateFusionProcessor(config)
            fused_output = processor.fuse(perception_output)
            assert fused_output.fused_embedding.shape == (128,)
    
    def test_different_activations(self):
        """Test with different activation functions."""
        activations = ["relu", "gelu", "tanh"]
        perception_output = self.create_mock_perception_output()
        
        for activation in activations:
            config = FusionConfig(activation=activation)
            processor = StateFusionProcessor(config)
            fused_output = processor.fuse(perception_output)
            assert fused_output.fused_embedding.shape == (128,)
    
    def test_with_without_layer_norm(self):
        """Test with and without layer normalization."""
        perception_output = self.create_mock_perception_output()
        
        # With layer norm
        config_with_ln = FusionConfig(use_layer_norm=True)
        processor_with_ln = StateFusionProcessor(config_with_ln)
        output_with_ln = processor_with_ln.fuse(perception_output)
        
        # Without layer norm
        config_without_ln = FusionConfig(use_layer_norm=False)
        processor_without_ln = StateFusionProcessor(config_without_ln)
        output_without_ln = processor_without_ln.fuse(perception_output)
        
        # Both should produce valid outputs
        assert output_with_ln.fused_embedding.shape == (128,)
        assert output_without_ln.fused_embedding.shape == (128,)
        
        # Parameter counts should be different
        assert (processor_with_ln.get_parameter_count() != 
                processor_without_ln.get_parameter_count())
    
    def test_no_dropout_in_eval(self):
        """Test that dropout is disabled in eval mode."""
        config = FusionConfig(dropout_rate=0.5)
        processor = StateFusionProcessor(config)
        perception_output = self.create_mock_perception_output()
        
        # In training mode, outputs might be different due to dropout
        processor.train()
        output1 = processor.fuse(perception_output)
        output2 = processor.fuse(perception_output)
        
        # In eval mode, outputs should be deterministic
        processor.eval()
        torch.manual_seed(42)
        output3 = processor.fuse(perception_output)
        torch.manual_seed(42)
        output4 = processor.fuse(perception_output)
        
        assert torch.allclose(output3.fused_embedding, output4.fused_embedding)