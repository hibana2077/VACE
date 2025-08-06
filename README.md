# VACE: Variance-Adaptive Cross-Entropy for Ultra-Fine-Grained Visual Categorization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A minimally invasive modification to Cross-Entropy loss that introduces **class-wise temperature scaling** based on **intra-class variance estimation**. VACE is specifically designed for Ultra-Fine-Grained Visual Categorization (Ultra-FGVC) scenarios where classes exhibit **large intra-class variation** and **small inter-class differences**.

## 🚀 Key Features

- **📊 Minimal Overhead**: Only O(K) additional parameters per class, nearly identical speed to standard CE
- **🔧 Drop-in Replacement**: Can directly replace `nn.CrossEntropyLoss` in existing training pipelines  
- **🎯 Improved Calibration**: Better calibration compared to standard CE and label smoothing
- **⚙️ timm Compatible**: Works seamlessly with all timm CNN backbones
- **🎨 No Contrastive Learning**: Pure supervised learning approach, no additional complexity

## 📖 Method Overview

VACE introduces a **class-wise temperature** τc for each class, automatically adjusted based on **intra-class scatter**:

```
τc = clip(a + b·σc², τmin, τmax)
```

Where σc² is the EMA-smoothed variance of logits for class c. The loss becomes:

```
L_VACE = -log(exp(z_yi/τyi) / Σ_k exp(z_k/τk))
```

**Intuition**: Classes with high intra-class variation get higher τ (smoother decisions), while classes with low variation get lower τ (sharper decisions).

## 🛠️ Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/VACE.git
cd VACE

# Install dependencies and setup
python setup.py
```

### Manual Installation

```bash
pip3 install uv
uv pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Test Installation
```bash
python test_implementation.py
```

### 2. Download Dataset
```bash
# List available datasets
python download_dataset.py --list

# Download Cotton80 dataset
python download_dataset.py --dataset cotton80
```

### 3. Train Model
```bash
# Train with VACE on Cotton80 using ResNet50
python train.py --config configs/cotton_r50.yaml

# Or with command line arguments
python train.py --dataset cotton80 --backbone resnet50 --loss vace --epochs 100
```

### 4. Evaluate Model
```bash
python eval.py --checkpoint outputs/checkpoints/best_model.pth --dataset cotton80
```

## 🔧 Usage Examples

### Basic VACE Loss
```python
from src.loss import VACE

# Create VACE loss
loss_fn = VACE(
    num_classes=80,
    a=1.0,           # Base temperature
    b=1.0,           # Variance scaling
    tau_min=0.5,     # Min temperature  
    tau_max=2.0,     # Max temperature
    ema_decay=0.1    # EMA smoothing
)

# Use like standard CrossEntropy
logits = model(x)
loss = loss_fn(logits, targets)
```

### Integration with timm Models
```python
import timm
from src.loss import VACE

# Create model without classification head
model = timm.create_model('resnet50', pretrained=True, num_classes=0)
head = nn.Linear(model.num_features, num_classes)

# Training loop
for x, y in dataloader:
    # Extract features
    features = model.forward_features(x)
    if features.dim() > 2:
        features = features.mean(dim=(2, 3))  # Global average pooling
    
    # Classification
    logits = head(features)
    loss = loss_fn(logits, y)  # VACE automatically updates τ in training mode
```

### Configuration-based Training
```python
# Load configuration
from src.utils.config import Config

config = Config.load('configs/cotton_r50_224.yaml')
config.training.epochs = 200  # Modify as needed
config.loss.b = 0.5          # Adjust VACE parameters

# Save modified config
config.save('my_experiment.yaml')
```

## 📊 Available Datasets

VACE supports several Ultra-FGVC datasets:

- **cotton80**: Cotton classification with 80 classes  
- **soybean**: Soybean variety classification
- **soy_ageing_r1-r6**: Soybean aging datasets (multiple rounds)

```bash
# Download any dataset
python download_dataset.py --dataset cotton80
python download_dataset.py --dataset soybean
```

## ⚙️ Configuration

### Example Configuration (YAML)
```yaml
dataset:
  name: "cotton80"
  root: "./data"
  
model:
  backbone: "resnet50"
  pretrained: true

loss:
  type: "vace"
  a: 1.0
  b: 1.0  
  tau_min: 0.5
  tau_max: 2.0
  ema_decay: 0.1

training:
  epochs: 100
  batch_size: 64
  use_amp: true

optimizer:
  type: "sgd"
  lr: 0.1
  momentum: 0.9
  weight_decay: 1e-4
```

### Pre-built Configs
- `configs/cotton_r50_224.yaml`: Cotton80 + ResNet50 + VACE
- `configs/cotton_r18_224.yaml`: Cotton80 + ResNet18 + CE baseline  
- `configs/soybean_convnext_224.yaml`: Soybean + ConvNeXt + VACE

## 📈 Experiment Results

### Metrics Collected
- **Accuracy**: Top-1, Top-5 (when applicable)
- **Calibration**: ECE, MCE, NLL, Brier Score  
- **Per-class**: Macro F1, Recall, Precision
- **Efficiency**: Training time, memory usage, throughput

### Baseline Comparisons
- Cross-Entropy (CE)
- Label Smoothing  
- LDAM (Label-Distribution-Aware Margin)

## 📁 Project Structure

```
VACE/
├── src/
│   ├── loss.py                 # VACE loss implementation
│   ├── dataset/
│   │   └── ufgvc.py           # Dataset loading utilities
│   └── utils/
│       ├── config.py          # Configuration management
│       ├── logger.py          # Logging utilities  
│       └── metrics.py         # Metrics calculation
├── configs/                   # Configuration templates
├── docs/                      # Documentation
├── scripts/                   # Training scripts for HPC
├── train.py                   # Main training script
├── eval.py                    # Evaluation script
├── download_dataset.py        # Dataset download utility
└── setup.py                   # Setup script
```

## 🔬 Advanced Usage

### Custom Datasets
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    # Implement your dataset
    pass

# Use with VACE
dataloader = DataLoader(MyDataset(...), batch_size=64)
# Training loop as shown above
```

### Experiment Tracking
```python
# Enable TensorBoard logging
config['logging']['use_tensorboard'] = True

# Enable Weights & Biases
config['logging']['use_wandb'] = True
```

### Multi-GPU Training
```python
# The training script supports DataParallel automatically
# Just run on multi-GPU machine:
python train.py --config configs/cotton_r50_224.yaml
```

## 🚀 High-Performance Computing

Pre-configured PBS scripts for HPC environments:

```bash
# Submit to A100 queue
qsub scripts/train_a100.sh

# Submit to V100 queue  
qsub scripts/train_v100.sh
```

## 📋 Requirements

### Core Dependencies
- Python ≥ 3.8
- PyTorch ≥ 1.9
- timm ≥ 0.6.0
- NumPy, Pandas, PIL

### Optional Dependencies  
- TensorBoard (logging)
- Weights & Biases (experiment tracking)
- Matplotlib, Seaborn (visualization)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## 📚 Citation

If you use VACE in your research, please cite:

```bibtex
@article{vace2024,
  title={VACE: Variance-Adaptive Cross-Entropy for Ultra-Fine-Grained Visual Categorization},
  author={Your Name},
  journal={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024}
}
``` -->

## 🆘 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Dataset download fails**: Check internet connection and disk space
3. **Import errors**: Run `python test_implementation.py` to diagnose

### Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Open an [issue](https://github.com/your-username/VACE/issues)
- 💬 Start a [discussion](https://github.com/your-username/VACE/discussions)

## 🙏 Acknowledgments

- [timm](https://github.com/rwightman/pytorch-image-models) for the excellent model library
- [Ultra-FGVC](https://arxiv.org/abs/2203.03619) for the benchmark datasets
- PyTorch team for the deep learning framework