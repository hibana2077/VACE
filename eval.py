#!/usr/bin/env python3
"""
VACE Evaluation Script

Evaluate trained models on test sets and generate comprehensive reports.
Supports loading checkpoints and evaluating with various metrics.

Usage:
    python eval.py --checkpoint ./outputs/checkpoints/best_model.pth --dataset cotton80
    python eval.py --config configs/cotton_r50_224.yaml --checkpoint ./outputs/checkpoints/best_model.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.loss import VACE
from src.dataset.ufgvc import UFGVCDataset
from src.utils.metrics import MetricsCalculator, format_metrics_for_logging
from src.utils.config import Config


class VAEEvaluator:
    """Model evaluator for VACE"""
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        
        # Model components (loaded from checkpoint)
        self.model = None
        self.head = None
        self.loss_fn = None
        
        # Dataset info
        self.num_classes = None
        self.class_names = None
        
        # Load checkpoint and setup
        self.load_checkpoint()
        self.setup_data()
    
    def load_checkpoint(self):
        """Load model from checkpoint"""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model configuration from checkpoint
        if 'config' in checkpoint:
            # Update config with checkpoint config (model architecture, etc.)
            checkpoint_config = checkpoint['config']
            if 'model' in checkpoint_config:
                self.config['model'] = checkpoint_config['model']
            if 'loss' in checkpoint_config:
                self.config['loss'] = checkpoint_config['loss']
        
        # Setup model architecture
        self.setup_model()
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.head.load_state_dict(checkpoint['head_state_dict'])
        
        if 'loss_state_dict' in checkpoint and hasattr(self.loss_fn, 'load_state_dict'):
            self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_acc' in checkpoint:
            print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    def setup_model(self):
        """Setup model architecture"""
        backbone_name = self.config['model']['backbone']
        
        print(f"Setting up model: {backbone_name}")
        
        # Create backbone
        self.model = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        # Get number of classes from dataset info or config
        if self.num_classes is None:
            # Try to infer from loss config if available
            if 'loss' in self.config and 'num_classes' in self.config['loss']:
                self.num_classes = self.config['loss']['num_classes']
            else:
                # Default fallback - will be updated when data is loaded
                self.num_classes = 10
        
        # Create classifier head
        feature_dim = self.model.num_features
        self.head = nn.Linear(feature_dim, self.num_classes, bias=True)
        
        # Setup loss function (for potential tau information)
        loss_config = self.config.get('loss', {})
        if loss_config.get('type') == 'vace':
            self.loss_fn = VACE(
                num_classes=self.num_classes,
                a=loss_config.get('a', 1.0),
                b=loss_config.get('b', 1.0),
                tau_min=loss_config.get('tau_min', 0.5),
                tau_max=loss_config.get('tau_max', 2.0),
                ema_decay=loss_config.get('ema_decay', 0.1),
                label_smoothing=loss_config.get('label_smoothing', 0.0),
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        # Move to device
        self.model = self.model.to(self.device)
        self.head = self.head.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        
        # Set to eval mode
        self.model.eval()
        self.head.eval()
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
    
    def setup_data(self):
        """Setup data loaders"""
        dataset_config = self.config.get('dataset', {})
        dataset_name = dataset_config.get('name', 'cotton80')
        root = dataset_config.get('root', './data')
        batch_size = self.config.get('training', {}).get('batch_size', 64)
        
        print(f"Setting up dataset: {dataset_name}")
        
        # Get transform
        transform = self.get_transform()
        
        # Create test dataset
        try:
            self.test_dataset = UFGVCDataset(
                dataset_name=dataset_name,
                root=root,
                split='test',
                transform=transform,
                download=False  # Assume data is already downloaded
            )
            
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Update class information
            self.num_classes = len(self.test_dataset.classes)
            self.class_names = self.test_dataset.classes
            
            print(f"Test dataset: {len(self.test_dataset)} samples, {self.num_classes} classes")
            
        except Exception as e:
            print(f"Error setting up test dataset: {e}")
            self.test_dataset = None
            self.test_loader = None
    
    def get_transform(self):
        """Get evaluation transform"""
        # Create temporary model to get data config
        backbone_name = self.config['model']['backbone']
        temp_model = timm.create_model(backbone_name, pretrained=True)
        data_cfg = timm.data.resolve_data_config(temp_model.pretrained_cfg)
        
        # Evaluation transform (no augmentation)
        transform = timm.data.create_transform(
            **data_cfg,
            is_training=False,
            interpolation='bicubic'
        )
        
        return transform
    
    def evaluate(self) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        if self.test_loader is None:
            raise ValueError("Test dataset not available")
        
        print("Running evaluation...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_logits = []
        total_loss = 0.0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.test_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                features = self.model.forward_features(images)
                
                if features.dim() > 2:
                    features = features.mean(dim=(2, 3))
                
                logits = self.head(features)
                
                # Compute loss
                loss = self.loss_fn(logits, targets)
                total_loss += loss.item()
                
                # Get probabilities (considering temperature if VACE)
                if hasattr(self.loss_fn, 'get_tau'):
                    tau = self.loss_fn.get_tau()
                    probabilities = torch.softmax(logits / tau.unsqueeze(0), dim=1)
                else:
                    probabilities = torch.softmax(logits, dim=1)
                
                # Collect results
                all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(self.test_loader)} batches")
        
        eval_time = time.time() - start_time
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        logits = np.array(all_logits)
        
        print(f"Evaluation completed in {eval_time:.1f}s")
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_metrics(
            predictions, targets, probabilities, self.class_names
        )
        
        metrics['loss'] = total_loss / len(self.test_loader)
        
        # Calculate margin statistics if using VACE
        if hasattr(self.loss_fn, 'get_tau'):
            tau = self.loss_fn.get_tau().cpu().numpy()
            margin_stats = self.metrics_calculator.calculate_margin_statistics(
                logits, targets, tau
            )
            metrics.update(margin_stats)
            
            # Add tau statistics
            metrics['tau_mean'] = np.mean(tau)
            metrics['tau_std'] = np.std(tau)
            metrics['tau_min'] = np.min(tau)
            metrics['tau_max'] = np.max(tau)
        
        # Per-class accuracy
        per_class_acc = self.metrics_calculator.calculate_per_class_accuracy(
            predictions, targets, self.num_classes
        )
        
        return {
            'metrics': metrics,
            'per_class_accuracy': per_class_acc,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'logits': logits
        }
    
    def generate_report(self, results: Dict[str, Any], output_dir: str):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = results['metrics']
        per_class_acc = results['per_class_accuracy']
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Main metrics
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        if 'top5_accuracy' in metrics:
            print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print(f"Loss: {metrics['loss']:.6f}")
        
        # Calibration metrics
        print(f"\nCalibration Metrics:")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  MCE: {metrics.get('mce', 0.0):.4f}")
        print(f"  NLL: {metrics['nll']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        # Per-class metrics
        print(f"\nPer-class Metrics:")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        
        # Margin statistics (if available)
        if 'margin_mean' in metrics:
            print(f"\nMargin Statistics:")
            print(f"  Mean: {metrics['margin_mean']:.4f}")
            print(f"  Std: {metrics['margin_std']:.4f}")
            print(f"  Negative ratio: {metrics['negative_margin_ratio']:.4f}")
        
        # Temperature statistics (if available)
        if 'tau_mean' in metrics:
            print(f"\nTemperature Statistics:")
            print(f"  Mean: {metrics['tau_mean']:.4f}")
            print(f"  Std: {metrics['tau_std']:.4f}")
            print(f"  Range: [{metrics['tau_min']:.4f}, {metrics['tau_max']:.4f}]")
        
        # Generate plots
        self.generate_plots(results, output_path)
        
        # Save detailed results
        self.save_detailed_results(results, output_path)
        
        print(f"\nDetailed results saved to: {output_path}")
        print("="*80)
    
    def generate_plots(self, results: Dict[str, Any], output_path: Path):
        """Generate evaluation plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            metrics = results['metrics']
            predictions = results['predictions']
            targets = results['targets']
            probabilities = results['probabilities']
            
            # 1. Confusion Matrix
            plt.figure(figsize=(12, 10))
            cm = metrics['confusion_matrix']
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            if len(cm) <= 20:  # Only show class names if not too many
                sns.heatmap(cm_norm, annot=True, fmt='.2f', 
                           xticklabels=self.class_names[:len(cm)],
                           yticklabels=self.class_names[:len(cm)],
                           cmap='Blues')
            else:
                sns.heatmap(cm_norm, cmap='Blues')
            
            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Reliability Diagram
            bin_confidences, bin_accuracies, bin_counts = self.metrics_calculator.get_reliability_diagram_data(
                predictions, targets, probabilities
            )
            
            plt.figure(figsize=(8, 6))
            
            # Perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            # Actual calibration
            mask = bin_counts > 0
            if np.any(mask):
                plt.plot(bin_confidences[mask], bin_accuracies[mask], 'ro-', 
                        label='Model calibration', linewidth=2, markersize=8)
                
                # Bar chart for sample counts
                plt.bar(bin_confidences[mask], bin_counts[mask] / np.sum(bin_counts), 
                       alpha=0.3, width=0.08, label='Sample density')
            
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title(f'Reliability Diagram (ECE = {metrics["ece"]:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(output_path / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Per-class Accuracy
            per_class_acc = results['per_class_accuracy']
            
            plt.figure(figsize=(max(8, len(per_class_acc) * 0.4), 6))
            bars = plt.bar(range(len(per_class_acc)), per_class_acc * 100)
            plt.xlabel('Class Index')
            plt.ylabel('Accuracy (%)')
            plt.title('Per-class Accuracy')
            plt.xticks(range(len(per_class_acc)))
            
            # Color bars by accuracy
            for i, bar in enumerate(bars):
                if per_class_acc[i] < 0.5:
                    bar.set_color('red')
                elif per_class_acc[i] < 0.8:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            plt.tight_layout()
            plt.savefig(output_path / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Temperature Distribution (if VACE)
            if hasattr(self.loss_fn, 'get_tau'):
                tau = self.loss_fn.get_tau().cpu().numpy()
                
                plt.figure(figsize=(10, 6))
                
                # Subplot 1: Histogram
                plt.subplot(1, 2, 1)
                plt.hist(tau, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Temperature (τ)')
                plt.ylabel('Frequency')
                plt.title('Temperature Distribution')
                plt.axvline(tau.mean(), color='red', linestyle='--', label=f'Mean: {tau.mean():.3f}')
                plt.legend()
                
                # Subplot 2: Per-class temperature
                plt.subplot(1, 2, 2)
                bars = plt.bar(range(len(tau)), tau)
                plt.xlabel('Class Index')
                plt.ylabel('Temperature (τ)')
                plt.title('Per-class Temperature')
                plt.axhline(1.0, color='red', linestyle='--', label='τ = 1.0')
                
                # Color by temperature value
                for i, bar in enumerate(bars):
                    if tau[i] < 1.0:
                        bar.set_color('blue')
                    elif tau[i] > 1.5:
                        bar.set_color('red')
                    else:
                        bar.set_color('green')
                
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_path / 'temperature_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Generated plots in {output_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    def save_detailed_results(self, results: Dict[str, Any], output_path: Path):
        """Save detailed results to files"""
        metrics = results['metrics']
        
        # Save metrics as JSON
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, np.floating):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.integer):
                serializable_metrics[key] = int(value)
            else:
                serializable_metrics[key] = value
        
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Save per-class results
        per_class_results = {
            'class_names': self.class_names,
            'per_class_accuracy': results['per_class_accuracy'].tolist()
        }
        
        with open(output_path / 'per_class_results.json', 'w') as f:
            json.dump(per_class_results, f, indent=2)
        
        # Save predictions
        np.savez(
            output_path / 'predictions.npz',
            predictions=results['predictions'],
            targets=results['targets'],
            probabilities=results['probabilities']
        )
        
        print(f"Saved detailed results to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VACE Model Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       help='Path to config file (optional, uses checkpoint config if available)')
    parser.add_argument('--dataset', type=str,
                       help='Dataset name (overrides config)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for evaluation')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}")
    
    # Override config with command line arguments
    if args.dataset:
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['name'] = args.dataset
    
    if args.batch_size:
        if 'training' not in config:
            config['training'] = {}
        config['training']['batch_size'] = args.batch_size
    
    # Ensure checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Create evaluator and run evaluation
    try:
        evaluator = VAEEvaluator(config, args.checkpoint)
        results = evaluator.evaluate()
        evaluator.generate_report(results, args.output_dir)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
