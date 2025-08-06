#!/usr/bin/env python3
"""
Test script to verify VACE implementation

This script performs basic tests to ensure the VACE implementation works correctly.
"""

import sys
import os
import torch
import timm
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_vace_loss():
    """Test VACE loss function"""
    print("Testing VACE loss function...")
    
    from src.loss import VACE
    
    # Test parameters
    batch_size = 8
    num_classes = 5
    
    # Create random data
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create VACE loss
    loss_fn = VACE(num_classes=num_classes, a=1.0, b=1.0, tau_min=0.5, tau_max=2.0)
    
    # Forward pass
    loss_fn.train()  # Training mode
    loss = loss_fn(logits, targets)
    
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Tau values: {loss_fn.get_tau().tolist()}")
    
    # Backward pass
    loss.backward()
    print(f"  Gradients computed successfully: {logits.grad is not None}")
    
    # Test eval mode
    loss_fn.eval()
    with torch.no_grad():
        eval_loss = loss_fn(logits, targets)
    print(f"  Eval loss: {eval_loss.item():.4f}")
    
    print("✓ VACE loss test passed")
    return True


def test_dataset():
    """Test dataset loading"""
    print("Testing dataset loading...")
    
    from src.dataset.ufgvc import UFGVCDataset
    
    try:
        # Test dataset creation (don't download, just test structure)
        available_datasets = UFGVCDataset.list_available_datasets()
        print(f"  Available datasets: {list(available_datasets.keys())}")
        
        print("✓ Dataset test passed")
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_timm_integration():
    """Test timm integration"""
    print("Testing timm integration...")
    
    try:
        # Test model creation
        model = timm.create_model('resnet18', pretrained=False, num_classes=0)
        
        # Test forward_features
        x = torch.randn(2, 3, 224, 224)
        features = model.forward_features(x)
        
        print(f"  Model: resnet18")
        print(f"  Input shape: {x.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature dim: {model.num_features}")
        
        # Test with classifier head
        head = torch.nn.Linear(model.num_features, 10)
        
        # Global pooling if needed
        if features.dim() > 2:
            features = features.mean(dim=(2, 3))
        
        logits = head(features)
        print(f"  Logits shape: {logits.shape}")
        
        # Test data config and transforms
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg, is_training=True)
        print(f"  Data config: {data_cfg}")
        print(f"  Transform created successfully")
        
        print("✓ Timm integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Timm integration test failed: {e}")
        return False


def test_config_system():
    """Test configuration system"""
    print("Testing configuration system...")
    
    try:
        from src.utils.config import Config
        
        # Create default config
        config = Config()
        
        # Test validation
        is_valid = config.validate()
        print(f"  Default config valid: {is_valid}")
        
        # Test dict conversion
        config_dict = config.to_dict()
        print(f"  Config keys: {list(config_dict.keys())}")
        
        # Test config from dict
        new_config = Config(config_dict)
        print(f"  Config creation from dict successful")
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_metrics():
    """Test metrics calculation"""
    print("Testing metrics calculation...")
    
    try:
        from src.utils.metrics import MetricsCalculator
        
        # Generate sample data
        np.random.seed(42)
        num_samples = 100
        num_classes = 5
        
        targets = np.random.randint(0, num_classes, num_samples)
        predictions = np.random.randint(0, num_classes, num_samples)
        probabilities = np.random.random((num_samples, num_classes))
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_metrics(predictions, targets, probabilities)
        
        print(f"  Calculated metrics: {list(metrics.keys())}")
        print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        
        print("✓ Metrics test passed")
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        # Core dependencies
        import torch
        import torchvision
        import timm
        import numpy as np
        import pandas as pd
        
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Timm: {timm.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Pandas: {pd.__version__}")
        
        # Project modules
        from src.loss import VACE, EMAStats
        from src.dataset.ufgvc import UFGVCDataset
        from src.utils.metrics import MetricsCalculator
        from src.utils.config import Config
        from src.utils.logger import Logger
        
        print("  All project modules imported successfully")
        
        print("✓ Import test passed")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("VACE IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_vace_loss,
        test_dataset,
        test_timm_integration,
        test_config_system,
        test_metrics,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VACE implementation is ready.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
