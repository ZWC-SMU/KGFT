# KGFT
# KGFT: Kernel-Guided Feature Transform \[[Paper](https://arxiv.org/abs/2608.12737)]

## 🚀 What is KGFT?
KGFT (Kernel-Guided Feature Transform) is a lightweight, plug-and-play module that explicitly transfers geometric structure from network parameters to feature representations — for the first time, unifying the Kernel Manifold (weights) and the Data Manifold (features) in a single mathematical framework.

## The Core Insights
In any convolutional or linear layer, weights and features share the same channel space. Their covariance structures — the Gram matrix of weights and the covariance of features — are geometrically coupled. KGFT leverages this coupling to guide representation learning in a principled way:
Kernel Geometry: G=WW^T captures filter correlations
