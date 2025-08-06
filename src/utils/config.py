"""
Configuration utilities for VACE

Provides configuration management with validation and defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "cotton80"
    root: str = "./data"
    num_workers: int = 4
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    interpolation: str = "bicubic"
    re_prob: float = 0.25
    re_mode: str = "pixel"
    re_count: int = 1


@dataclass
class ModelConfig:
    """Model configuration"""
    backbone: str = "resnet50"
    pretrained: bool = True


@dataclass
class LossConfig:
    """Loss function configuration"""
    type: str = "vace"  # "vace" or "ce"
    a: float = 1.0
    b: float = 1.0
    tau_min: float = 0.5
    tau_max: float = 2.0
    ema_decay: float = 0.1
    label_smoothing: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 64
    use_amp: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    type: str = "sgd"  # "sgd" or "adamw"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    betas: tuple = (0.9, 0.999)  # For AdamW


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    type: str = "cosine"  # "cosine", "step", "multistep", "none"
    eta_min: float = 0.0
    step_size: int = 30
    gamma: float = 0.1
    milestones: list = None


@dataclass
class OutputConfig:
    """Output configuration"""
    checkpoint_dir: str = "./outputs/checkpoints"
    log_dir: str = "./outputs/logs"
    save_frequency: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_interval: int = 100
    use_tensorboard: bool = False
    use_wandb: bool = False


class Config:
    """Main configuration class"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.loss = LossConfig()
        self.training = TrainingConfig()
        self.optimizer = OptimizerConfig()
        self.scheduler = SchedulerConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()
        
        # General settings
        self.seed = 42
        self.project_name = "vace"
        
        if config_dict:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if hasattr(getattr(self, key), '__dict__'):
                    # Update dataclass attributes
                    config_obj = getattr(self, key)
                    if isinstance(value, dict):
                        for attr, attr_value in value.items():
                            if hasattr(config_obj, attr):
                                setattr(config_obj, attr, attr_value)
                else:
                    # Update simple attributes
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        # Convert dataclass attributes
        for attr_name in ['dataset', 'model', 'loss', 'training', 'optimizer', 'scheduler', 'output', 'logging']:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if hasattr(attr, '__dict__'):
                    result[attr_name] = attr.__dict__
                else:
                    result[attr_name] = attr
        
        # Add simple attributes
        result['seed'] = self.seed
        result['project_name'] = self.project_name
        
        return result
    
    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate dataset
        if self.dataset.name not in ['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 
                                     'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6']:
            errors.append(f"Unknown dataset: {self.dataset.name}")
        
        # Validate model
        # Note: Actual timm model validation would require importing timm
        
        # Validate loss
        if self.loss.type not in ['vace', 'ce']:
            errors.append(f"Unknown loss type: {self.loss.type}")
        
        if self.loss.type == 'vace':
            if self.loss.a <= 0:
                errors.append("VACE parameter 'a' must be positive")
            if self.loss.b < 0:
                errors.append("VACE parameter 'b' must be non-negative")
            if self.loss.tau_min <= 0:
                errors.append("VACE parameter 'tau_min' must be positive")
            if self.loss.tau_max <= self.loss.tau_min:
                errors.append("VACE parameter 'tau_max' must be greater than tau_min")
            if not 0 < self.loss.ema_decay <= 1:
                errors.append("VACE parameter 'ema_decay' must be in (0, 1]")
        
        if not 0 <= self.loss.label_smoothing < 1:
            errors.append("Label smoothing must be in [0, 1)")
        
        # Validate training
        if self.training.epochs <= 0:
            errors.append("Training epochs must be positive")
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Validate optimizer
        if self.optimizer.type not in ['sgd', 'adamw']:
            errors.append(f"Unknown optimizer type: {self.optimizer.type}")
        if self.optimizer.lr <= 0:
            errors.append("Learning rate must be positive")
        if self.optimizer.weight_decay < 0:
            errors.append("Weight decay must be non-negative")
        
        # Validate scheduler
        if self.scheduler.type not in ['cosine', 'step', 'multistep', 'none']:
            errors.append(f"Unknown scheduler type: {self.scheduler.type}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output.log_dir, exist_ok=True)
        os.makedirs(self.dataset.root, exist_ok=True)


def create_config_templates():
    """Create example configuration templates"""
    templates_dir = Path("configs")
    templates_dir.mkdir(exist_ok=True)
    
    # Template 1: Cotton80 with ResNet50 and VACE
    config1 = Config()
    config1.dataset.name = "cotton80"
    config1.model.backbone = "resnet50"
    config1.loss.type = "vace"
    config1.training.epochs = 100
    config1.training.batch_size = 64
    config1.optimizer.lr = 0.1
    
    config1.save(templates_dir / "cotton_r50_224.yaml")
    
    # Template 2: Cotton80 with ResNet18 and baseline CE
    config2 = Config()
    config2.dataset.name = "cotton80"
    config2.model.backbone = "resnet18"
    config2.loss.type = "ce"
    config2.loss.label_smoothing = 0.1
    config2.training.epochs = 100
    config2.training.batch_size = 64
    config2.optimizer.lr = 0.1
    
    config2.save(templates_dir / "cotton_r18_224.yaml")
    
    # Template 3: Soybean with ConvNeXt and VACE
    config3 = Config()
    config3.dataset.name = "soybean"
    config3.model.backbone = "convnext_tiny.fb_in1k"
    config3.loss.type = "vace"
    config3.loss.b = 0.5
    config3.training.epochs = 200
    config3.training.batch_size = 32
    config3.optimizer.type = "adamw"
    config3.optimizer.lr = 1e-4
    config3.scheduler.type = "cosine"
    
    config3.save(templates_dir / "soybean_convnext_224.yaml")
    
    # Template 4: EfficientNet for quick testing
    config4 = Config()
    config4.dataset.name = "cotton80"
    config4.model.backbone = "efficientnet_b0.ra_in1k"
    config4.loss.type = "vace"
    config4.training.epochs = 50
    config4.training.batch_size = 128
    config4.optimizer.lr = 0.05
    
    config4.save(templates_dir / "cotton_effnet_224.yaml")
    
    print(f"Created configuration templates in {templates_dir}")


def load_config_from_args(args) -> Config:
    """Load configuration from command line arguments"""
    config = Config()
    
    # Update from args if provided
    if hasattr(args, 'dataset') and args.dataset:
        config.dataset.name = args.dataset
    if hasattr(args, 'backbone') and args.backbone:
        config.model.backbone = args.backbone
    if hasattr(args, 'loss') and args.loss:
        config.loss.type = args.loss
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'lr') and args.lr:
        config.optimizer.lr = args.lr
    if hasattr(args, 'seed') and args.seed:
        config.seed = args.seed
    if hasattr(args, 'data_root') and args.data_root:
        config.dataset.root = args.data_root
    if hasattr(args, 'output_dir') and args.output_dir:
        config.output.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        config.output.log_dir = os.path.join(args.output_dir, 'logs')
    
    return config


# Example usage and testing
if __name__ == "__main__":
    # Create and test configuration
    config = Config()
    
    # Print default configuration
    print("Default configuration:")
    config_dict = config.to_dict()
    print(yaml.dump(config_dict, default_flow_style=False, indent=2))
    
    # Validate configuration
    is_valid = config.validate()
    print(f"Configuration is valid: {is_valid}")
    
    # Test saving and loading
    config.save("test_config.yaml")
    loaded_config = Config.load("test_config.yaml")
    
    print("\nLoaded configuration matches original:", 
          config.to_dict() == loaded_config.to_dict())
    
    # Create templates
    create_config_templates()
    
    # Clean up test file
    os.remove("test_config.yaml")
