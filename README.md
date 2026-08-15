# KGFT
# KGFT: Kernel-Guided Feature Transform \[[Paper](https://arxiv.org/abs/2608.12737)]

## 🚀 What is KGFT?
KGFT (Kernel-Guided Feature Transform) is a lightweight, plug-and-play module that explicitly transfers geometric structure from network parameters to feature representations — for the first time, unifying the Kernel Manifold (weights) and the Data Manifold (features) in a single mathematical framework.

## The Core Insights
In any convolutional or linear layer, weights and features share the same channel space. Their covariance structures — the Gram matrix of weights and the covariance of features — are geometrically coupled. KGFT leverages this coupling to guide representation learning in a principled way:
- Kernel Geometry: $\mathbf{G} = \mathbf{W}\mathbf{W}^\top$ captures filter correlations.
- Data Geometry: $\mathbf{K} = \mathbf{X}_f^\top \mathbf{X}_f / \sqrt{N}$ captures feature covariances.
- Geometric Guidance: $\mathbf{K}' = \mathbf{M} \cdot \mathbf{K}$, where $\mathbf{M}$ is derived from $\mathbf{G}$.
- Adaptive Fusion: $\mathbf{Y} = \mathbf{X} + s \cdot \text{Proj}(\mathbf{X}_f \mathbf{K}')$ with learnable strength.

## Two Modes: Exploit & Explore


## Performance Highlights
Model	Dataset	Baseline	+ KGFT	Gain
ResNet-50	ImageNet-1K	75.43%	76.58%	+1.15%
ResNet-18	CIFAR-100	74.60%	76.84%	+2.24%
ViT-Tiny	CIFAR-100	54.33%	54.96%	+0.63%
LLaMA-7B	GSM8K	37.50%	38.32%	+0.82%
LLaMA-7B	MAWPS	79.00%	82.46%	+3.46%
