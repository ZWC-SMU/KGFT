# KGFT
# KGFT: Kernel-Guided Feature Transform \[[Paper](https://arxiv.org/abs/2608.12737)]

## 🚀 What is KGFT?
KGFT (Kernel-Guided Feature Transform) is a lightweight, plug-and-play module that explicitly transfers geometric structure from network parameters to feature representations — for the first time, unifying the Kernel Manifold (weights) and the Data Manifold (features) in a single mathematical framework.

## 💡 The Core Insights
In any convolutional or linear layer, weights and features share the same channel space. Their covariance structures — the Gram matrix of weights and the covariance of features — are geometrically coupled. KGFT leverages this coupling to guide representation learning in a principled way:
- Kernel Geometry: $\mathbf{G} = \mathbf{W}\mathbf{W}^\top$ captures filter correlations.
- Data Geometry: $\mathbf{K} = \mathbf{X}_f^\top \mathbf{X}_f / \sqrt{N}$ captures feature covariances.
- Geometric Guidance: $\mathbf{K}' = \mathbf{M} \cdot \mathbf{K}$, where $\mathbf{M}$ is derived from $\mathbf{G}$.
- Adaptive Fusion: $\mathbf{Y} = \mathbf{X} + s \cdot \text{Proj}(\mathbf{X}_f \mathbf{K}')$ with learnable strength.

### 💡 Two Modes: Exploit & Explore
| Mode    | Applied to     | Effect                                                                 |
|---------|----------------|------------------------------------------------------------------------|
| Exploit | Shallow layers | Preserve and reinforce dominant kernel directions                      |
| Explore | Deep layers    | Suppress over-correlation and encourage semantic diversity             |

## 🛠️ Installation
bash
pip install git+https://github.com/ZWC-SMU/KGFT.git

## 🚀 Basic Usage in ResNet
## 🚀 Quick Start

### Basic Usage

```python
import torch
from kgft import KernelGuidedFeatureTransform

# Get the weight of previous convolutional layer
prev_conv = model.layer1[0].conv1  # Example: get a conv layer
prev_weight = prev_conv.weight

# Initialize KGFT module
kgft = KernelGuidedFeatureTransform(
    dim=64,                      # Number of channels
    prev_conv_weight=prev_weight,# Weight of previous conv layer
    epsilon=1e-5,                # For numerical stability
    init_strength=0.5            # Initial strength (0~1)
)

# Set operating mode
kgft.set_config("exploit")       # "exploit" for shallow, "explore" for deep

# Forward pass
x = torch.randn(32, 64, 32, 32)  # (batch, channels, height, width)
y = kgft(x)                      # Same shape as input

# Get current strength value
current_strength = kgft.strength

### ✨ KGFT vs. Conventional Attention
| Property              | Conventional Attention (SENet, CBAM) | KGFT (Ours)                                 |
|-----------------------|--------------------------------------|---------------------------------------------|
| Mechanism             | Re-weighting channels (scalar)       | Reshaping channel geometry (full matrix)    |
| Information Source    | Feature responses only               | Parameter + Feature manifolds               |
| Layer Adaptation      | Same across all layers               | Exploit ↔ Explore depth-aware scheduling    |
| Architecture Support  | Typically CNN-only                   | CNNs, ViTs, LLMs unified                    |

##### 📊 Performance Highlights
| Model       | Dataset      | Baseline | + KGFT  | Gain   |
|-------------|--------------|----------|---------|--------|
| ResNet-18   | CIFAR-100    | 74.60%   | 76.84%  | +2.24% |
| ResNet-20   | CIFAR-100    | 67.61%   | 69.39%  | +1.78% |
| ViT-Tiny    | CIFAR-100    | 54.33%   | 54.96%  | +0.63% |
| ResNet-50   | ImageNet-1K  | 75.43%   | 76.58%  | +1.15% |
| LLaMA-7B    | GSM8K        | 37.50%   | 38.32%  | +0.82% |
| LLaMA-7B    | MAWPS        | 79.00%   | 82.46%  | +3.46% |
