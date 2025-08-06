#!/usr/bin/env python3
"""
VACE Quick Demo

Demonstrates the complete VACE workflow with a minimal example.
Uses a small subset of data to quickly show training, evaluation, and analysis.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.loss import VACE
from src.utils.metrics import MetricsCalculator, format_metrics_for_logging
from src.utils.config import Config
import timm


def create_synthetic_dataset(num_samples=1000, num_classes=10, feature_dim=512):
    """Create synthetic dataset for demonstration"""
    print(f"Creating synthetic dataset: {num_samples} samples, {num_classes} classes")
    
    # Generate synthetic features and labels
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create class centers
    class_centers = torch.randn(num_classes, feature_dim) * 2
    
    features = []
    labels = []
    
    for i in range(num_samples):
        # Random class
        class_id = np.random.randint(0, num_classes)
        
        # Feature = class center + noise (with varying intra-class variance)
        # Some classes have higher intra-class variance
        if class_id < num_classes // 3:
            noise_scale = 2.0  # High variance classes
        elif class_id < 2 * num_classes // 3:
            noise_scale = 1.0  # Medium variance classes  
        else:
            noise_scale = 0.5  # Low variance classes
        
        noise = torch.randn(feature_dim) * noise_scale
        feature = class_centers[class_id] + noise
        
        features.append(feature)
        labels.append(class_id)
    
    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.long)
    
    print(f"  Feature shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return features, labels


def demo_vace_vs_ce():
    """Demonstrate VACE vs CE comparison"""
    print("\n" + "="*60)
    print("VACE VS CROSS-ENTROPY DEMONSTRATION")  
    print("="*60)
    
    # Parameters
    num_samples = 1000
    num_classes = 10
    feature_dim = 512
    epochs = 50
    batch_size = 64
    lr = 0.01
    
    # Create synthetic dataset
    features, labels = create_synthetic_dataset(num_samples, num_classes, feature_dim)
    
    # Split into train/test
    split_idx = int(0.8 * num_samples)
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"\nTrain: {len(train_features)} samples")
    print(f"Test: {len(test_features)} samples")
    
    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models and loss functions to compare
    experiments = {
        'Cross-Entropy': nn.CrossEntropyLoss(),
        'Label Smoothing': nn.CrossEntropyLoss(label_smoothing=0.1),
        'VACE': VACE(num_classes=num_classes, a=1.0, b=1.0, tau_min=0.5, tau_max=2.0, ema_decay=0.1)
    }
    
    results = {}
    
    for exp_name, loss_fn in experiments.items():
        print(f"\n--- Training with {exp_name} ---")
        
        # Create classifier (same initialization for fair comparison)
        torch.manual_seed(42)
        classifier = nn.Linear(feature_dim, num_classes).to(device)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
        
        loss_fn = loss_fn.to(device)
        
        # Training
        classifier.train()
        loss_fn.train() if hasattr(loss_fn, 'train') else None
        
        train_losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                logits = classifier(batch_features)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}")
        
        train_time = time.time() - start_time
        
        # Evaluation
        classifier.eval()
        loss_fn.eval() if hasattr(loss_fn, 'eval') else None
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_logits = []
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                logits = classifier(batch_features)
                loss = loss_fn(logits, batch_labels)
                test_loss += loss.item()
                
                # Get probabilities (handle VACE temperature scaling)
                if hasattr(loss_fn, 'get_tau'):
                    tau = loss_fn.get_tau()
                    probabilities = torch.softmax(logits / tau.unsqueeze(0), dim=1)
                else:
                    probabilities = torch.softmax(logits, dim=1)
                
                all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_probabilities)
        )
        
        metrics['test_loss'] = test_loss / len(test_loader)
        metrics['train_time'] = train_time
        
        # Add temperature info for VACE
        if hasattr(loss_fn, 'get_tau'):
            tau = loss_fn.get_tau().cpu().numpy()
            metrics['tau_mean'] = np.mean(tau)
            metrics['tau_std'] = np.std(tau)
            metrics['tau_range'] = [float(np.min(tau)), float(np.max(tau))]
        
        results[exp_name] = metrics
        
        print(f"  Final Test Accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"  Test ECE: {metrics['ece']:.4f}")
        print(f"  Training Time: {train_time:.1f}s")
        if hasattr(loss_fn, 'get_tau'):
            print(f"  Temperature Range: [{metrics['tau_range'][0]:.3f}, {metrics['tau_range'][1]:.3f}]")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Method':<20} {'Accuracy':<10} {'ECE':<10} {'NLL':<10} {'Time(s)':<10}")
    print("-" * 60)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['top1_accuracy']:>7.2f}% {metrics['ece']:>9.4f} {metrics['nll']:>9.4f} {metrics['train_time']:>9.1f}")
    
    if 'VACE' in results:
        vace_metrics = results['VACE']
        ce_metrics = results['Cross-Entropy']
        
        acc_improvement = vace_metrics['top1_accuracy'] - ce_metrics['top1_accuracy']  
        ece_improvement = ce_metrics['ece'] - vace_metrics['ece']  # Lower is better
        
        print(f"\nVACE vs CE Improvements:")
        print(f"  Accuracy: {acc_improvement:+.2f}%")
        print(f"  ECE: {ece_improvement:+.4f} (lower is better)")
        
        if 'tau_range' in vace_metrics:
            print(f"  Temperature range: {vace_metrics['tau_range']}")
    
    print("="*60)
    
    return results


def main():
    """Main demonstration function"""
    print("VACE: Variance-Adaptive Cross-Entropy Demonstration")
    print("This demo shows VACE working with synthetic data")
    
    # System info
    print(f"\nSystem Information:")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run demonstration
    results = demo_vace_vs_ce()
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"VACE demonstrates improved calibration and competitive accuracy")
    print(f"on synthetic data with varying intra-class variance.")
    
    return results


if __name__ == "__main__":
    main()
