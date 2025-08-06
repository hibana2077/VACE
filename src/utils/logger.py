"""
Logging utilities for VACE training

Provides comprehensive logging including:
- Console logging with timestamps
- File logging
- TensorBoard integration
- Weights & Biases integration (optional)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import numpy as np


class Logger:
    """Comprehensive logger for training experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config['output']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name
        self.experiment_name = self._create_experiment_name()
        
        # Setup console and file logging
        self._setup_logging()
        
        # Setup TensorBoard (optional)
        self.tb_writer = None
        if config['logging'].get('use_tensorboard', False):
            self._setup_tensorboard()
        
        # Setup Weights & Biases (optional)
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            self._setup_wandb()
        
        # Log experiment configuration
        self._log_config()
    
    def _create_experiment_name(self) -> str:
        """Create unique experiment name"""
        dataset = self.config['dataset']['name']
        model = self.config['model']['backbone']
        loss = self.config['loss']['type']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return f"{dataset}_{model}_{loss}_{timestamp}"
    
    def _setup_logging(self):
        """Setup console and file logging"""
        # Create logger
        self.logger = logging.getLogger('VACE')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f'{self.experiment_name}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tb_dir = self.log_dir / 'tensorboard' / self.experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            
            self.tb_writer = SummaryWriter(tb_dir)
            self.logger.info(f"TensorBoard logging enabled. Log dir: {tb_dir}")
            
        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tb_writer = None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            import wandb
            
            # Initialize wandb
            wandb.init(
                project=self.config.get('project_name', 'vace'),
                name=self.experiment_name,
                config=self.config,
                dir=str(self.log_dir)
            )
            
            self.logger.info("Weights & Biases logging enabled")
            
        except ImportError:
            self.logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _log_config(self):
        """Log experiment configuration"""
        self.logger.info("=== Experiment Configuration ===")
        
        # Save config to file
        config_file = self.log_dir / f'{self.experiment_name}_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Log key configuration items
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Dataset: {self.config['dataset']['name']}")
        self.logger.info(f"Model: {self.config['model']['backbone']}")
        self.logger.info(f"Loss: {self.config['loss']['type']}")
        self.logger.info(f"Epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"Learning rate: {self.config['optimizer']['lr']}")
        self.logger.info(f"Optimizer: {self.config['optimizer']['type']}")
        
        if self.config['loss']['type'] == 'vace':
            self.logger.info(f"VACE params: a={self.config['loss']['a']}, b={self.config['loss']['b']}, "
                           f"tau_range=[{self.config['loss']['tau_min']}, {self.config['loss']['tau_max']}], "
                           f"ema_decay={self.config['loss']['ema_decay']}")
        
        self.logger.info("=" * 50)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all configured loggers"""
        # Console/file logging (already handled by individual log calls)
        
        # TensorBoard logging
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tag = f"{prefix}/{key}" if prefix else key
                    self.tb_writer.add_scalar(tag, value, step)
        
        # Weights & Biases logging
        if self.use_wandb:
            try:
                import wandb
                
                wandb_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tag = f"{prefix}/{key}" if prefix else key
                        wandb_metrics[tag] = value
                
                wandb_metrics['step'] = step
                wandb.log(wandb_metrics)
                
            except Exception as e:
                self.logger.warning(f"Failed to log to wandb: {e}")
    
    def log_histogram(self, name: str, data, step: int):
        """Log histogram data"""
        if self.tb_writer is not None:
            import torch
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            self.tb_writer.add_histogram(name, data, step)
    
    def log_image(self, name: str, image, step: int):
        """Log image"""
        if self.tb_writer is not None:
            self.tb_writer.add_image(name, image, step)
    
    def log_text(self, name: str, text: str, step: int):
        """Log text"""
        if self.tb_writer is not None:
            self.tb_writer.add_text(name, text, step)
    
    def log_confusion_matrix(self, cm, class_names, step: int, normalize: bool = True):
        """Log confusion matrix as image"""
        if self.tb_writer is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import torch
            
            # Normalize if requested
            if normalize:
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            else:
                cm_norm = cm
            
            # Create figure
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=True, fmt='.2f' if normalize else 'd',
                       xticklabels=class_names[:len(cm)] if class_names else range(len(cm)),
                       yticklabels=class_names[:len(cm)] if class_names else range(len(cm)),
                       cmap='Blues')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Convert to tensor and log
            fig = plt.gcf()
            fig.canvas.draw()
            img = torch.from_numpy(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8))
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = img.permute(2, 0, 1)  # HWC to CHW
            
            self.tb_writer.add_image('confusion_matrix', img, step)
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to log confusion matrix: {e}")
    
    def log_reliability_diagram(self, bin_confidences, bin_accuracies, bin_counts, step: int):
        """Log reliability diagram"""
        if self.tb_writer is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            import torch
            
            # Create reliability diagram
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            # Plot actual calibration
            mask = bin_counts > 0
            if np.any(mask):
                ax.plot(bin_confidences[mask], bin_accuracies[mask], 'ro-', 
                       label='Model calibration')
                
                # Add bar chart showing sample counts
                ax2 = ax.twinx()
                ax2.bar(bin_confidences[mask], bin_counts[mask], alpha=0.3, width=0.1)
                ax2.set_ylabel('Sample count')
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title('Reliability Diagram')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            
            # Convert to tensor and log
            fig.canvas.draw()
            img = torch.from_numpy(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8))
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = img.permute(2, 0, 1)  # HWC to CHW
            
            self.tb_writer.add_image('reliability_diagram', img, step)
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to log reliability diagram: {e}")
    
    def close(self):
        """Close all loggers"""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# Utility functions
def setup_logging_directory(base_dir: str) -> Path:
    """Setup logging directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(base_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def log_system_info(logger: Logger):
    """Log system information"""
    import torch
    import platform
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info("=" * 40)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'output': {
            'log_dir': './test_logs',
        },
        'logging': {
            'use_tensorboard': True,
            'use_wandb': False,
        },
        'dataset': {'name': 'test'},
        'model': {'backbone': 'resnet50'},
        'loss': {'type': 'vace', 'a': 1.0, 'b': 1.0},
        'training': {'epochs': 100, 'batch_size': 32},
        'optimizer': {'type': 'sgd', 'lr': 0.1},
    }
    
    # Test logger
    logger = Logger(config)
    
    # Test logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    
    # Test metrics logging
    metrics = {
        'loss': 0.5,
        'accuracy': 85.2,
        'ece': 0.05
    }
    logger.log_metrics(metrics, step=1, prefix='train')
    
    # Test system info
    log_system_info(logger)
    
    logger.close()
