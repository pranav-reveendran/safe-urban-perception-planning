"""
Testing Utilities for Autonomous Vehicle Perception
==================================================

This module provides testing utilities for validating model performance,
data integrity, and system functionality.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import unittest
from pathlib import Path

def test_model_output_shape(model: torch.nn.Module, 
                           input_shape: Tuple[int, ...] = (4, 512, 512)) -> bool:
    """Test if model produces expected output shape."""
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        output = model(dummy_input)
        
        expected_shape = (1, 5, 512, 512)  # 5 classes
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        return True

def test_data_loader_consistency(data_loader) -> Dict[str, bool]:
    """Test data loader for consistency and expected formats."""
    results = {}
    
    # Test batch consistency
    batch1 = next(iter(data_loader))
    batch2 = next(iter(data_loader))
    
    results['batch_keys_consistent'] = set(batch1.keys()) == set(batch2.keys())
    results['batch_size_consistent'] = batch1['bev_features'].shape[0] == batch2['bev_features'].shape[0]
    
    # Test data types
    results['bev_features_float'] = batch1['bev_features'].dtype == torch.float32
    results['segmentation_long'] = batch1['segmentation'].dtype == torch.long
    
    # Test value ranges
    bev_data = batch1['bev_features']
    seg_data = batch1['segmentation']
    
    results['bev_normalized'] = (bev_data.min() >= -3.0) and (bev_data.max() <= 10.0)
    results['seg_valid_classes'] = (seg_data >= 0).all() and (seg_data <= 4).all()
    
    return results

def test_attention_mechanism(attention_module: torch.nn.Module,
                           feature_shape: Tuple[int, ...] = (2, 256, 64, 64)) -> bool:
    """Test attention mechanism functionality."""
    attention_module.eval()
    
    with torch.no_grad():
        dummy_features = torch.randn(*feature_shape)
        
        if hasattr(attention_module, 'forward'):
            output = attention_module(dummy_features)
            
            # Check output shape matches input
            assert output.shape == dummy_features.shape, "Attention output shape mismatch"
            
            # Check attention weights are in valid range
            if hasattr(attention_module, 'attention_weights'):
                weights = attention_module.attention_weights
                assert (weights >= 0).all() and (weights <= 1).all(), "Invalid attention weights"
            
            return True
        
    return False

def validate_bev_encoding(bev_tensor: torch.Tensor) -> Dict[str, bool]:
    """Validate BEV tensor encoding."""
    results = {}
    
    # Check shape
    results['correct_shape'] = bev_tensor.shape[-2:] == (512, 512)
    results['correct_channels'] = bev_tensor.shape[-3] == 4
    
    # Check channel properties
    height_max = bev_tensor[..., 0, :, :]
    height_mean = bev_tensor[..., 1, :, :]
    intensity_max = bev_tensor[..., 2, :, :]
    density = bev_tensor[..., 3, :, :]
    
    results['height_range_valid'] = (height_max >= -3.0).all() and (height_max <= 5.0).all()
    results['intensity_normalized'] = (intensity_max >= 0.0).all() and (intensity_max <= 1.0).all()
    results['density_normalized'] = (density >= 0.0).all() and (density <= 1.0).all()
    
    return results

def performance_benchmark(model: torch.nn.Module, 
                         input_tensor: torch.Tensor,
                         target_fps: float = 30.0) -> Dict[str, float]:
    """Benchmark model performance."""
    import time
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'meets_target': fps >= target_fps,
        'target_fps': target_fps
    }

class ModelTests(unittest.TestCase):
    """Unit tests for model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_unet_baseline(self):
        """Test baseline U-Net functionality."""
        from models.unet import UNet
        
        model = UNet(in_channels=4, out_channels=5).to(self.device)
        self.assertTrue(test_model_output_shape(model))
        
    def test_attention_unet(self):
        """Test attention-enhanced U-Net."""
        from models.attention_unet import AttentionEnhancedUNet
        
        model = AttentionEnhancedUNet(
            in_channels=4, 
            out_channels=5,
            attention_type='mixed'
        ).to(self.device)
        
        self.assertTrue(test_model_output_shape(model))

def run_all_tests():
    """Run comprehensive test suite."""
    print("🧪 Running Autonomous Vehicle Perception Tests...")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("✅ All tests completed!")

if __name__ == '__main__':
    run_all_tests() 