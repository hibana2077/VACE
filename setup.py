#!/usr/bin/env python3
"""
Setup script for VACE

Installs required dependencies and sets up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    return True


def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        cmd = f"{sys.executable} -m pip install -r requirements.txt"
        return run_command(cmd)
    else:
        print("requirements.txt not found")
        return False


def setup_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "data",
        "outputs",
        "outputs/checkpoints", 
        "outputs/logs",
        "configs",
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_name}")
    
    return True


def create_configs():
    """Create configuration templates"""
    print("Creating configuration templates...")
    
    try:
        # Import after installation
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from src.utils.config import create_config_templates
        
        create_config_templates()
        return True
        
    except Exception as e:
        print(f"Failed to create config templates: {e}")
        return False


def test_installation():
    """Test the installation"""
    print("Testing installation...")
    
    try:
        # Test basic imports
        import torch
        import timm
        import numpy as np
        import pandas as pd
        
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ timm {timm.__version__}")
        print(f"  ✓ NumPy {np.__version__}")
        print(f"  ✓ Pandas {pd.__version__}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Installation test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("VACE SETUP SCRIPT")
    print("=" * 60)
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Setting up directories", setup_directories),
        ("Creating config templates", create_configs),
        ("Testing installation", test_installation),
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if step_func():
                print(f"✓ {step_name} completed successfully")
                success_count += 1
            else:
                print(f"✗ {step_name} failed")
        except Exception as e:
            print(f"✗ {step_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"SETUP RESULTS: {success_count}/{len(steps)} steps completed")
    
    if success_count == len(steps):
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the implementation: python test_implementation.py")
        print("2. Download a dataset: python download_dataset.py --dataset cotton80")
        print("3. Start training: python train.py --config configs/cotton_r50_224.yaml")
    else:
        print("❌ Setup incomplete. Please check the errors above.")
    
    print("=" * 60)
    
    return success_count == len(steps)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
