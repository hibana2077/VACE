#!/usr/bin/env python3
"""
VACE Training Script

A comprehensive training script for VACE (Variance-Adaptive Cross-Entropy) experiments.
Supports multiple datasets, backbones, and baseline comparisons as specified in the documentation.

Usage:
    python train.py --config configs/cotton_r50_224.yaml
    python train.py --dataset cotton80 --backbone resnet50 --epochs 100
"""

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import timm
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.loss import VACE
from src.dataset.ufgvc import UFGVCDataset
from src.utils.metrics import MetricsCalculator
from src.utils.logger import Logger
from src.utils.config import Config


class VACETrainer:
    """Main trainer class for VACE experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        self.set_seed(config.get('seed', 42))
        
        # Initialize components
        self.logger = Logger(config)
        self.metrics_calculator = MetricsCalculator()
        
        # Model and training components (initialized later)
        self.model = None
        self.head = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler('cuda') if config.get('use_amp', True) else None
        
        # Data loaders (initialized later)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_model_path = None
        
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def setup_data(self):
        """Setup data loaders according to timm requirements"""
        dataset_name = self.config['dataset']['name']
        root = self.config['dataset']['root']
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['dataset'].get('num_workers', 4)
        
        self.logger.info(f"Setting up dataset: {dataset_name}")
        
        # Create datasets for each split
        datasets = {}
        for split in ['train', 'val', 'test']:
            try:
                # Get transform for the split
                transform = self.get_transform(split)
                
                datasets[split] = UFGVCDataset(
                    dataset_name=dataset_name,
                    root=root,
                    split=split,
                    transform=transform,
                    download=True
                )
                
                self.logger.info(f"{split.capitalize()} dataset: {len(datasets[split])} samples")
                
            except ValueError as e:
                self.logger.warning(f"Could not create {split} dataset: {e}")
                datasets[split] = None
        
        # Create data loaders
        if datasets['train'] is not None:
            self.train_loader = DataLoader(
                datasets['train'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            
        if datasets['val'] is not None:
            self.val_loader = DataLoader(
                datasets['val'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
        if datasets['test'] is not None:
            self.test_loader = DataLoader(
                datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        # Store dataset info
        if datasets['train'] is not None:
            self.num_classes = len(datasets['train'].classes)
            self.class_names = datasets['train'].classes
            self.logger.info(f"Number of classes: {self.num_classes}")
        else:
            raise RuntimeError("Training dataset is required")
    
    def get_transform(self, split: str):
        """Get transforms according to timm requirements"""
        # First, create a temporary model to get data config
        backbone_name = self.config['model']['backbone']
        temp_model = timm.create_model(backbone_name, pretrained=True)
        
        # Get timm data config
        data_cfg = timm.data.resolve_data_config(temp_model.pretrained_cfg)
        
        # Override interpolation if specified in config to avoid duplicate argument
        if 'interpolation' in self.config['dataset']:
            data_cfg['interpolation'] = self.config['dataset']['interpolation']
        
        if split == 'train':
            # Training transforms with augmentation
            transform = timm.data.create_transform(
                **data_cfg,
                is_training=True,
                auto_augment=self.config['dataset'].get('auto_augment', 'rand-m9-mstd0.5-inc1'),
                re_prob=self.config['dataset'].get('re_prob', 0.25),
                re_mode=self.config['dataset'].get('re_mode', 'pixel'),
                re_count=self.config['dataset'].get('re_count', 1),
            )
        else:
            # Validation/test transforms without augmentation
            transform = timm.data.create_transform(
                **data_cfg,
                is_training=False,
            )
        
        return transform
    
    def setup_model(self):
        """Setup model architecture"""
        backbone_name = self.config['model']['backbone']
        use_pretrained = self.config['model'].get('pretrained', True)
        
        self.logger.info(f"Creating model: {backbone_name} (pretrained: {use_pretrained})")
        
        # Create backbone without classifier head
        self.model = timm.create_model(
            backbone_name, 
            pretrained=use_pretrained, 
            num_classes=0  # Remove classifier head
        )
        
        # Create custom classifier head
        feature_dim = self.model.num_features
        self.head = nn.Linear(feature_dim, self.num_classes, bias=True)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.head = self.head.to(self.device)
        
        self.logger.info(f"Model feature dimension: {feature_dim}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Head parameters: {sum(p.numel() for p in self.head.parameters()):,}")
    
    def setup_loss(self):
        """Setup loss function"""
        loss_config = self.config['loss']
        loss_type = loss_config['type']
        
        if loss_type == 'vace':
            self.loss_fn = VACE(
                num_classes=self.num_classes,
                a=loss_config.get('a', 1.0),
                b=loss_config.get('b', 1.0),
                tau_min=loss_config.get('tau_min', 0.5),
                tau_max=loss_config.get('tau_max', 2.0),
                ema_decay=loss_config.get('ema_decay', 0.1),
                label_smoothing=loss_config.get('label_smoothing', 0.0),
            )
            self.logger.info(f"Using VACE loss: {self.loss_fn}")
            
        elif loss_type == 'ce':
            label_smoothing = loss_config.get('label_smoothing', 0.0)
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.logger.info(f"Using CrossEntropy loss (label_smoothing: {label_smoothing})")
            
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_fn = self.loss_fn.to(self.device)
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        opt_config = self.config['optimizer']
        opt_type = opt_config['type']
        
        # Collect parameters
        params = list(self.model.parameters()) + list(self.head.parameters())
        if hasattr(self.loss_fn, 'parameters'):
            params.extend(list(self.loss_fn.parameters()))
        
        # Create optimizer
        if opt_type == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4),
                nesterov=opt_config.get('nesterov', True)
            )
        elif opt_type == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        # Create scheduler
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine')
        
        if sched_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('eta_min', 0.0)
            )
        elif sched_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'multistep':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sched_config.get('milestones', [60, 80]),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {opt_type}, Scheduler: {sched_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.head.train()
        if hasattr(self.loss_fn, 'train'):
            self.loss_fn.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast('cuda'):
                    # Extract features using timm
                    features = self.model.forward_features(images)
                    
                    # Global average pooling if needed
                    if features.dim() > 2:
                        features = features.mean(dim=(2, 3))
                    
                    # Classifier head
                    logits = self.head(features)
                    
                    # Compute loss
                    loss = self.loss_fn(logits, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                features = self.model.forward_features(images)
                
                if features.dim() > 2:
                    features = features.mean(dim=(2, 3))
                
                logits = self.head(features)
                loss = self.loss_fn(logits, targets)
                
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += pred.eq(targets).sum().item()
            total_samples += targets.size(0)
            
            # Log progress
            if batch_idx % self.config['logging'].get('log_interval', 100) == 0:
                acc = 100.0 * total_correct / total_samples
                self.logger.info(
                    f'Epoch: {self.current_epoch} [{batch_idx:3d}/{len(self.train_loader):3d}] '
                    f'Loss: {loss.item():.6f} Acc: {acc:.2f}%'
                )
        
        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, data_loader: DataLoader, split: str = 'val') -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.head.eval()
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                features = self.model.forward_features(images)
                
                if features.dim() > 2:
                    features = features.mean(dim=(2, 3))
                
                logits = self.head(features)
                
                # Compute loss
                loss = self.loss_fn(logits, targets)
                
                # Get probabilities (considering temperature if VACE)
                if hasattr(self.loss_fn, 'get_tau'):
                    # For VACE, scale logits by temperature
                    tau = self.loss_fn.get_tau()
                    probabilities = torch.softmax(logits / tau.unsqueeze(0), dim=1)
                else:
                    probabilities = torch.softmax(logits, dim=1)
                
                # Collect results
                total_loss += loss.item()
                all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_probabilities),
            self.class_names
        )
        
        metrics['loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
        }
        
        # Save loss function state if it has one (for VACE)
        if hasattr(self.loss_fn, 'state_dict'):
            checkpoint['loss_state_dict'] = self.loss_fn.state_dict()
        
        # Save scheduler state
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save paths
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            self.logger.info(f'New best model saved: {best_path}')
        
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        epochs = self.config['training']['epochs']
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate(self.val_loader, 'val')
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - start_time
            
            log_str = f'Epoch {epoch:3d}/{epochs:3d} | Time: {epoch_time:.1f}s | '
            log_str += f'Train Loss: {train_metrics["loss"]:.6f} Acc: {train_metrics["accuracy"]:.2f}% | '
            
            if val_metrics:
                log_str += f'Val Loss: {val_metrics["loss"]:.6f} Acc: {val_metrics["top1_accuracy"]:.2f}% '
                log_str += f'ECE: {val_metrics.get("ece", 0.0):.4f} | '
            
            log_str += f'LR: {train_metrics["lr"]:.2e}'
            
            # Log tau if using VACE
            if hasattr(self.loss_fn, 'get_tau'):
                tau = self.loss_fn.get_tau().cpu().numpy()
                log_str += f' | τ: [{tau.min():.3f}, {tau.max():.3f}]'
            
            self.logger.info(log_str)
            
            # Check if best model
            is_best = False
            if val_metrics and val_metrics['top1_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['top1_accuracy']
                is_best = True
            
            # Save checkpoint
            all_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            
            # Save checkpoint every N epochs or if best
            save_freq = self.config['output'].get('save_frequency', 10)
            if epoch % save_freq == 0 or is_best or epoch == epochs:
                self.save_checkpoint(all_metrics, is_best)
        
        self.logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def evaluate_test(self):
        """Evaluate on test set"""
        if self.test_loader is None:
            self.logger.warning("No test dataset available")
            return
        
        self.logger.info("Evaluating on test set...")
        
        # Load best model if available
        if self.best_model_path and self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.head.load_state_dict(checkpoint['head_state_dict'])
            if 'loss_state_dict' in checkpoint and hasattr(self.loss_fn, 'load_state_dict'):
                self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            self.logger.info(f"Loaded best model from {self.best_model_path}")
        
        # Test evaluation
        test_metrics = self.validate(self.test_loader, 'test')
        
        # Log results
        self.logger.info("=== Test Results ===")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.6f}")
            else:
                self.logger.info(f"{key}: {value}")
    
    def run(self):
        """Run the complete training pipeline"""
        try:
            self.setup_data()
            self.setup_model()
            self.setup_loss()
            self.setup_optimizer()
            
            self.train()
            self.evaluate_test()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VACE Training Script')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Quick configuration options (override config file)
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--backbone', type=str, help='Backbone model name')
    parser.add_argument('--loss', type=str, choices=['vace', 'ce'], help='Loss function type')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Paths
    parser.add_argument('--data-root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config(args) -> Dict[str, Any]:
    """Create default configuration from command line arguments"""
    config = {
        'seed': args.seed,
        'dataset': {
            'name': args.dataset or 'cotton80',
            'root': args.data_root,
            'num_workers': 4,
        },
        'model': {
            'backbone': args.backbone or 'resnet50',
            'pretrained': True,
        },
        'loss': {
            'type': args.loss or 'vace',
            'a': 1.0,
            'b': 1.0,
            'tau_min': 0.5,
            'tau_max': 2.0,
            'ema_decay': 0.1,
            'label_smoothing': 0.0,
        },
        'training': {
            'epochs': args.epochs or 100,
            'batch_size': args.batch_size or 64,
            'use_amp': True,
        },
        'optimizer': {
            'type': 'sgd',
            'lr': args.lr or 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'nesterov': True,
        },
        'scheduler': {
            'type': 'cosine',
            'eta_min': 0.0,
        },
        'output': {
            'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints'),
            'log_dir': os.path.join(args.output_dir, 'logs'),
            'save_frequency': 10,
        },
        'logging': {
            'log_interval': 100,
            'use_wandb': False,
        }
    }
    
    return config


def main():
    """Main function"""
    args = parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = create_default_config(args)
        print("Using default configuration")
    
    # Create output directories
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    
    # Create and run trainer
    trainer = VACETrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
