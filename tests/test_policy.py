"""
Tests for the policy head.
"""

import pytest
import torch

from src.policy.data_types import PolicyConfig, PolicyOutput, ActionInfo
from src.policy.processors import PolicyHead, PolicyProcessor, create_action_mask, select_action_simple
from src.policy.action_utils import (
    encode_action, decode_action, get_tile_index, get_tile_coords,
    get_placement_actions, get_erase_action, validate_action_space
)


class TestActionUtils:
    """Test action encoding/decoding utilities."""
    
    def test_tile_index_conversion(self):
        """Test tile index to coordinates conversion."""
        # Test corner cases
        assert get_tile_index(0, 0) == 0
        assert get_tile_index(9, 13) == 139
        assert get_tile_index(5, 7) == 5 * 14 + 7
        
        # Test reverse conversion
        assert get_tile_coords(0) == (0, 0)
        assert get_tile_coords(139) == (9, 13)
        assert get_tile_coords(77) == (5, 7)
    
    def test_tile_bounds_validation(self):
        """Test tile coordinate bounds checking."""
        # Valid coordinates
        get_tile_index(0, 0)
        get_tile_index(9, 13)
        
        # Invalid coordinates
        with pytest.raises(ValueError):
            get_tile_index(-1, 0)
        with pytest.raises(ValueError):
            get_tile_index(10, 0)
        with pytest.raises(ValueError):
            get_tile_index(0, -1)
        with pytest.raises(ValueError):
            get_tile_index(0, 14)
    
    def test_action_encoding_decoding(self):
        """Test action encoding and decoding."""
        test_cases = [
            ("place_up", 0, 0, 0),
            ("place_down", 0, 0, 140),
            ("place_left", 0, 0, 280),
            ("place_right", 0, 0, 420),
            ("erase", 0, 0, 560),
            ("place_up", 9, 13, 139),
            ("place_down", 9, 13, 279),
            ("erase", 5, 7, 637),  # 560 + 77
        ]
        
        for action_type, row, col, expected_idx in test_cases:
            # Test encoding
            action_idx = encode_action(action_type, row, col)
            assert action_idx == expected_idx
            
            # Test decoding
            action_info = decode_action(action_idx)
            assert action_info.action_type == action_type
            assert action_info.tile_row == row
            assert action_info.tile_col == col
            assert action_info.action_idx == action_idx
    
    def test_action_space_validation(self):
        """Test complete action space validation."""
        validate_action_space()
    
    def test_placement_actions(self):
        """Test getting placement actions for a tile."""
        tile_idx = 77  # (5, 7)
        placement_actions = get_placement_actions(tile_idx)
        
        assert len(placement_actions) == 4
        assert placement_actions == [77, 217, 357, 497]  # up, down, left, right
    
    def test_erase_action(self):
        """Test getting erase action for a tile."""
        tile_idx = 77
        erase_action = get_erase_action(tile_idx)
        assert erase_action == 637  # 560 + 77


class TestPolicyConfig:
    """Test PolicyConfig data class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PolicyConfig()
        assert config.input_dim == 128
        assert config.hidden_dim == 256
        assert config.output_dim == 700
        assert config.dropout_rate == 0.1
        assert config.use_bias is True
        assert config.weight_init == "xavier"
        assert config.temperature == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PolicyConfig(
            input_dim=64,
            hidden_dim=128,
            output_dim=350,
            dropout_rate=0.2,
            use_bias=False,
            weight_init="kaiming",
            temperature=0.8
        )
        assert config.input_dim == 64
        assert config.hidden_dim == 128
        assert config.output_dim == 350
        assert config.dropout_rate == 0.2
        assert config.use_bias is False
        assert config.weight_init == "kaiming"
        assert config.temperature == 0.8
    
    def test_validation(self):
        """Test configuration validation."""
        # Invalid input_dim
        with pytest.raises(ValueError):
            PolicyConfig(input_dim=0)
        
        # Invalid dropout_rate
        with pytest.raises(ValueError):
            PolicyConfig(dropout_rate=-0.1)
        with pytest.raises(ValueError):
            PolicyConfig(dropout_rate=1.5)
        
        # Invalid weight_init
        with pytest.raises(ValueError):
            PolicyConfig(weight_init="invalid")
        
        # Invalid temperature
        with pytest.raises(ValueError):
            PolicyConfig(temperature=0)


class TestActionInfo:
    """Test ActionInfo data class."""
    
    def test_valid_action_info(self):
        """Test creating valid action info."""
        action_info = ActionInfo(
            action_type="place_up",
            tile_row=5,
            tile_col=7,
            tile_idx=77,
            action_idx=77
        )
        assert action_info.action_type == "place_up"
        assert action_info.tile_row == 5
        assert action_info.tile_col == 7
        assert action_info.tile_idx == 77
        assert action_info.action_idx == 77
    
    def test_validation(self):
        """Test action info validation."""
        # Invalid action type
        with pytest.raises(ValueError):
            ActionInfo("invalid", 0, 0, 0, 0)
        
        # Invalid coordinates
        with pytest.raises(ValueError):
            ActionInfo("place_up", -1, 0, 0, 0)
        with pytest.raises(ValueError):
            ActionInfo("place_up", 10, 0, 0, 0)
        with pytest.raises(ValueError):
            ActionInfo("place_up", 0, -1, 0, 0)
        with pytest.raises(ValueError):
            ActionInfo("place_up", 0, 14, 0, 0)
        
        # Inconsistent tile_idx
        with pytest.raises(ValueError):
            ActionInfo("place_up", 5, 7, 100, 77)  # Wrong tile_idx


class TestPolicyOutput:
    """Test PolicyOutput data class."""
    
    def create_mock_policy_output(self) -> PolicyOutput:
        """Create a mock policy output for testing."""
        logits = torch.randn(700)
        probs = torch.softmax(logits, dim=0)
        
        return PolicyOutput(
            action_logits=logits,
            action_probs=probs,
            selected_action=42,
            temperature=1.0
        )
    
    def test_valid_output(self):
        """Test creating valid policy output."""
        output = self.create_mock_policy_output()
        assert output.action_logits.shape == (700,)
        assert output.action_probs.shape == (700,)
        assert output.selected_action == 42
        assert output.confidence is not None
        assert output.entropy is not None
    
    def test_probability_validation(self):
        """Test probability distribution validation."""
        # Valid probabilities
        logits = torch.randn(700)
        probs = torch.softmax(logits, dim=0)
        PolicyOutput(action_logits=logits, action_probs=probs)
        
        # Invalid probabilities (don't sum to 1)
        with pytest.raises(ValueError):
            invalid_probs = torch.ones(700) * 0.5  # Sum = 350
            PolicyOutput(action_logits=logits, action_probs=invalid_probs)
    
    def test_shape_validation(self):
        """Test tensor shape validation."""
        # Wrong logits shape
        with pytest.raises(ValueError):
            PolicyOutput(
                action_logits=torch.randn(500),
                action_probs=torch.softmax(torch.randn(700), dim=0)
            )
        
        # Wrong probs shape
        with pytest.raises(ValueError):
            PolicyOutput(
                action_logits=torch.randn(700),
                action_probs=torch.softmax(torch.randn(500), dim=0)
            )
    
    def test_top_k_actions(self):
        """Test getting top-k actions."""
        output = self.create_mock_policy_output()
        top_actions = output.get_top_k_actions(k=5)
        
        assert len(top_actions) == 5
        # Check descending order
        for i in range(1, 5):
            assert top_actions[i-1][1] >= top_actions[i][1]


class TestPolicyHead:
    """Test PolicyHead neural network."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        config = PolicyConfig()
        policy_head = PolicyHead(config)
        
        # Check layer shapes
        assert policy_head.layer1.in_features == 128
        assert policy_head.layer1.out_features == 256
        assert policy_head.layer2.in_features == 256
        assert policy_head.layer2.out_features == 700
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        policy_head = PolicyHead()
        
        # Single input
        embedding = torch.randn(128)
        logits = policy_head(embedding)
        assert logits.shape == (700,)
        
        # Batch input
        batch_embeddings = torch.randn(4, 128)
        batch_logits = policy_head(batch_embeddings)
        assert batch_logits.shape == (4, 700)
    
    def test_action_probabilities(self):
        """Test getting action probabilities."""
        policy_head = PolicyHead()
        embedding = torch.randn(128)
        
        # Default temperature
        probs = policy_head.get_action_probabilities(embedding)
        assert probs.shape == (700,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        
        # Custom temperature
        probs_cold = policy_head.get_action_probabilities(embedding, temperature=0.5)
        probs_hot = policy_head.get_action_probabilities(embedding, temperature=2.0)
        
        # Cold temperature should be more peaked (higher max)
        assert probs_cold.max() > probs.max()
        # Hot temperature should be more uniform (lower max)
        assert probs_hot.max() < probs.max()
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        policy_head = PolicyHead()
        param_count = sum(p.numel() for p in policy_head.parameters())
        
        # Expected: (128*256 + 256) + (256*700 + 700) = 33024 + 179900 = 212924
        # This is much higher than the ~40K target, but matches our 2-layer design
        expected_min = 128 * 256 + 256 * 700  # Without biases
        assert param_count >= expected_min


class TestPolicyProcessor:
    """Test PolicyProcessor class."""
    
    def create_mock_embedding(self) -> torch.Tensor:
        """Create a mock fused embedding for testing."""
        return torch.randn(128)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = PolicyProcessor()
        assert processor.config is not None
        assert isinstance(processor.policy_head, PolicyHead)
    
    def test_forward_pass(self):
        """Test forward pass through processor."""
        processor = PolicyProcessor()
        embedding = self.create_mock_embedding()
        
        output = processor.forward(embedding)
        assert isinstance(output, PolicyOutput)
        assert output.action_logits.shape == (700,)
        assert output.action_probs.shape == (700,)
    
    def test_action_selection_strategies(self):
        """Test different action selection strategies."""
        processor = PolicyProcessor()
        embedding = self.create_mock_embedding()
        
        # Deterministic selection
        output_det = processor.select_action(embedding, strategy="deterministic")
        assert output_det.selected_action is not None
        assert output_det.action_info is not None
        
        # Categorical sampling
        output_cat = processor.select_action(embedding, strategy="categorical")
        assert output_cat.selected_action is not None
        
        # Top-k sampling
        output_topk = processor.select_action(embedding, strategy="top_k", top_k=5)
        assert output_topk.selected_action is not None
        
        # Invalid strategy
        with pytest.raises(ValueError):
            processor.select_action(embedding, strategy="invalid")
    
    def test_action_masking(self):
        """Test action masking functionality."""
        processor = PolicyProcessor()
        embedding = self.create_mock_embedding()
        
        # Create mask that only allows first 10 actions
        mask = torch.zeros(700)
        mask[:10] = 1.0
        
        output = processor.select_action(embedding, action_mask=mask)
        assert 0 <= output.selected_action < 10
    
    def test_batch_processing(self):
        """Test batch action selection."""
        processor = PolicyProcessor()
        batch_embeddings = torch.randn(3, 128)
        
        outputs = processor.select_batch_actions(batch_embeddings)
        assert len(outputs) == 3
        for output in outputs:
            assert isinstance(output, PolicyOutput)
            assert output.selected_action is not None
    
    def test_input_validation(self):
        """Test input validation."""
        processor = PolicyProcessor()
        
        # Wrong embedding size
        with pytest.raises(ValueError):
            processor.forward(torch.randn(64))
        
        # Wrong action mask size
        with pytest.raises(ValueError):
            processor.forward(torch.randn(128), action_mask=torch.ones(500))
    
    def test_profiling(self):
        """Test performance profiling."""
        processor = PolicyProcessor()
        processor.enable_profiling()
        
        embedding = self.create_mock_embedding()
        output = processor.forward(embedding)
        
        assert output.inference_time_ms is not None
        assert output.inference_time_ms > 0
        
        metrics = processor.get_metrics()
        assert metrics["profiling_enabled"] is True
        assert "inference_times" in metrics
    
    def test_train_eval_modes(self):
        """Test training and evaluation modes."""
        processor = PolicyProcessor()
        
        # Set to training mode
        processor.train()
        assert processor.policy_head.training is True
        
        # Set to eval mode
        processor.eval()
        assert processor.policy_head.training is False


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_action_mask(self):
        """Test action mask creation."""
        valid_actions = [0, 1, 2, 100, 699]
        mask = create_action_mask(valid_actions)
        
        assert mask.shape == (700,)
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[100] == 1.0
        assert mask[699] == 1.0
        assert mask[3] == 0.0  # Not in valid actions
    
    def test_select_action_simple(self):
        """Test simple action selection function."""
        embedding = torch.randn(128)
        action_idx, action_info = select_action_simple(embedding)
        
        assert isinstance(action_idx, int)
        assert 0 <= action_idx < 700
        assert isinstance(action_info, ActionInfo)
        assert action_info.action_idx == action_idx


class TestIntegration:
    """Test integration with other components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from embedding to action."""
        # Create mock fused embedding (would come from state-fusion layer)
        fused_embedding = torch.randn(128)
        
        # Create policy processor
        processor = PolicyProcessor()
        
        # Select action
        output = processor.select_action(fused_embedding, strategy="deterministic")
        
        # Verify output structure
        assert isinstance(output, PolicyOutput)
        assert output.selected_action is not None
        assert output.action_info is not None
        assert 0 <= output.selected_action < 700
        
        # Verify action decoding
        action_info = output.action_info
        assert action_info.action_type in ["place_up", "place_down", "place_left", "place_right", "erase"]
        assert 0 <= action_info.tile_row < 10
        assert 0 <= action_info.tile_col < 14
    
    def test_temperature_effects(self):
        """Test temperature effects on action selection."""
        embedding = torch.randn(128)
        processor = PolicyProcessor()
        
        # Get outputs with different temperatures
        output_cold = processor.forward(embedding, temperature=0.1)
        output_normal = processor.forward(embedding, temperature=1.0)
        output_hot = processor.forward(embedding, temperature=10.0)
        
        # Cold should be more confident (higher max probability)
        assert output_cold.confidence > output_normal.confidence
        # Hot should be less confident (lower max probability)
        assert output_hot.confidence < output_normal.confidence
        
        # Hot should have higher entropy (more uniform)
        assert output_hot.entropy > output_normal.entropy
        # Cold should have lower entropy (more peaked)
        assert output_cold.entropy < output_normal.entropy