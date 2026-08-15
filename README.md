<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KGFT: Kernel-Guided Feature Transform</title>
    <style>
        /* GitHub Markdown 样式 */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
            color: #24292e;
            background-color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }
        h1 { font-size: 2em; border-bottom: 2px solid #eaecef; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.25em; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
        code {
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        }
        pre {
            padding: 16px;
            overflow: auto;
            font-size: 85%;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 3px;
            margin-bottom: 16px;
        }
        pre code {
            padding: 0;
            background-color: transparent;
            font-size: 100%;
        }
        blockquote {
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0 0 16px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        table th, table td {
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
        }
        table th {
            font-weight: 600;
            background-color: #f6f8fa;
        }
        table tr:nth-child(2n) {
            background-color: #f6f8fa;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        .badge {
            display: inline-block;
            padding: 0.2em 0.6em;
            font-size: 85%;
            font-weight: 500;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
            background-color: #f1f8ff;
            color: #0366d6;
            border: 1px solid #c8e1ff;
        }
        .badge-success {
            background-color: #dcffe4;
            color: #22863a;
            border-color: #b3e6c9;
        }
        .badge-warning {
            background-color: #fffbdd;
            color: #735c0f;
            border-color: #f5e0b3;
        }
        .highlight-box {
            padding: 16px;
            background-color: #f6f8fa;
            border-left: 4px solid #0366d6;
            margin: 16px 0;
            border-radius: 3px;
        }
        .highlight-box.warning {
            border-left-color: #d73a49;
        }
        .highlight-box.success {
            border-left-color: #28a745;
        }
        hr {
            border: 0;
            height: 1px;
            background: #dfe2e5;
            margin: 24px 0;
        }
        .container {
            display: flex;
            flex-direction: column;
        }
        @media (max-width: 768px) {
            body { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 标题和简介 -->
        <h1>KGFT: Kernel-Guided Feature Transform</h1>
        <p><em>Bridging the Geometry of Parameters and Features for Better Representation Learning</em></p>
        
        <p>
            <span class="badge">arXiv:2608.12345</span>
            <span class="badge badge-success">MIT License</span>
            <span class="badge">Python 3.8+</span>
            <span class="badge">PyTorch 2.0+</span>
            <span class="badge">⭐ GitHub Stars</span>
        </p>
        
        <hr>
        
        <!-- What is KGFT -->
        <h2>🚀 What is KGFT?</h2>
        <p><strong>KGFT (Kernel-Guided Feature Transform)</strong> is a lightweight, plug-and-play module that <strong>explicitly transfers geometric structure from network parameters to feature representations</strong> — for the first time, unifying the <em>Kernel Manifold</em> (weights) and the <em>Data Manifold</em> (features) in a single mathematical framework.</p>
        
        <div class="highlight-box">
            <h3>The Core Insight</h3>
            <p>In any convolutional or linear layer, <strong>weights and features share the same channel space</strong>. Their covariance structures — the Gram matrix of weights and the covariance of features — are <strong>geometrically coupled</strong>. KGFT leverages this coupling to guide representation learning in a principled way:</p>
            <ol>
                <li><strong>Kernel Geometry</strong>: $\mathbf{G} = \mathbf{W}\mathbf{W}^\top$ captures filter correlations</li>
                <li><strong>Data Geometry</strong>: $\mathbf{K} = \mathbf{X}_f^\top \mathbf{X}_f / \sqrt{N}$ captures feature covariances</li>
                <li><strong>Geometric Guidance</strong>: $\mathbf{K}' = \mathbf{M} \cdot \mathbf{K}$, where $\mathbf{M}$ is derived from $\mathbf{G}$</li>
                <li><strong>Adaptive Fusion</strong>: $\mathbf{Y} = \mathbf{X} + s \cdot \text{Proj}(\mathbf{X}_f \mathbf{K}')$ with learnable strength $s$</li>
            </ol>
        </div>
        
        <!-- Two Modes -->
        <h3>Two Modes: Exploit & Explore</h3>
        <table>
            <thead>
                <tr>
                    <th>Mode</th>
                    <th>Application</th>
                    <th>Effect</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Exploit</strong></td>
                    <td>Shallow layers</td>
                    <td>Preserve and reinforce dominant kernel directions</td>
                </tr>
                <tr>
                    <td><strong>Explore</strong></td>
                    <td>Deep layers</td>
                    <td>Suppress over-correlation and encourage semantic diversity</td>
                </tr>
            </tbody>
        </table>
        <blockquote>
            <p>📖 Full derivation in our <a href="#">paper</a>.</p>
        </blockquote>
        
        <hr>
        
        <!-- Why KGFT -->
        <h2>✨ Why KGFT?</h2>
        <table>
            <thead>
                <tr>
                    <th>Property</th>
                    <th>Conventional Attention (SENet, CBAM)</th>
                    <th><strong>KGFT (Ours)</strong></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Mechanism</strong></td>
                    <td>Re-weighting channels (scalar)</td>
                    <td>Reshaping channel geometry (full matrix)</td>
                </tr>
                <tr>
                    <td><strong>Information Source</strong></td>
                    <td>Feature responses only</td>
                    <td><strong>Parameter + Feature manifolds</strong></td>
                </tr>
                <tr>
                    <td><strong>Layer Adaptation</strong></td>
                    <td>Same across all layers</td>
                    <td><strong>Exploit ↔ Explore</strong> depth-aware scheduling</td>
                </tr>
                <tr>
                    <td><strong>Architecture Support</strong></td>
                    <td>Typically CNN-only</td>
                    <td><strong>CNNs, ViTs, LLMs</strong> unified</td>
                </tr>
            </tbody>
        </table>
        
        <hr>
        
        <!-- Performance -->
        <h2>📊 Performance Highlights</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Dataset</th>
                    <th>Baseline</th>
                    <th><strong>+ KGFT</strong></th>
                    <th>Gain</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ResNet-50</td>
                    <td>ImageNet-1K</td>
                    <td>75.43%</td>
                    <td><strong>76.58%</strong></td>
                    <td><strong>+1.15%</strong></td>
                </tr>
                <tr>
                    <td>ResNet-18</td>
                    <td>CIFAR-100</td>
                    <td>74.60%</td>
                    <td><strong>76.84%</strong></td>
                    <td><strong>+2.24%</strong></td>
                </tr>
                <tr>
                    <td>ResNet-20</td>
                    <td>CIFAR-100</td>
                    <td>67.61%</td>
                    <td><strong>69.39%</strong></td>
                    <td><strong>+1.78%</strong></td>
                </tr>
                <tr>
                    <td>ViT-Tiny</td>
                    <td>CIFAR-100</td>
                    <td>54.33%</td>
                    <td><strong>54.96%</strong></td>
                    <td><strong>+0.63%</strong></td>
                </tr>
                <tr>
                    <td>LLaMA-7B</td>
                    <td>GSM8K</td>
                    <td>37.50%</td>
                    <td><strong>38.32%</strong></td>
                    <td><strong>+0.82%</strong></td>
                </tr>
                <tr>
                    <td>LLaMA-7B</td>
                    <td>MAWPS</td>
                    <td>79.00%</td>
                    <td><strong>82.46%</strong></td>
                    <td><strong>+3.46%</strong></td>
                </tr>
            </tbody>
        </table>
        <blockquote>
            <p>📊 Full results across 8 random seeds, including standard deviations, available in the paper.</p>
        </blockquote>
        
        <hr>
        
        <!-- Installation -->
        <h2>🛠️ Installation</h2>
        <pre><code>pip install git+https://github.com/ZWC-SMU/KGFT.git</code></pre>
        <p>Or clone and install locally:</p>
        <pre><code>git clone https://github.com/ZWC-SMU/KGFT.git
        cd KGFT
        pip install -e .</code></pre>
        
        <hr>
        
        <!-- Quick Start -->
        <h2>🚀 Quick Start</h2>
        <h3>Basic Usage in ResNet</h3>
        <pre><code>import torch
        from kgft import KGFT
        
        # Insert after any convolutional block
        kgft = KGFT(
            channels=64,                # Number of channels
            mode="exploit",             # "exploit" for shallow, "explore" for deep
            learnable_strength=True,    # Learnable guidance strength
            epsilon=1e-5                # For numerical stability
        )
        
        # Forward pass
        x = torch.randn(32, 64, 32, 32)  # (batch, channels, height, width)
        y = kgft(x)                       # Same shape as input</code></pre>
        
        <h3>Insert into Your ResNet Architecture</h3>
        <pre><code>from kgft import KGFT
        
        # Insert KGFT after a specific block in ResNet
        model.layer1[2].add_module("kgft", KGFT(channels=64, mode="exploit"))
        model.layer2[3].add_module("kgft", KGFT(channels=128, mode="exploit"))
        model.layer3[5].add_module("kgft", KGFT(channels=256, mode="exploit"))
        model.layer4[2].add_module("kgft", KGFT(channels=512, mode="explore"))</code></pre>
        
        <h3>Usage in Transformer (ViT / LLaMA)</h3>
        <pre><code>from kgft import KGFT
        
        # Use the down-projection weight matrix
        kgft = KGFT(
            channels=768,               # Hidden dimension
            mode="exploit",
            use_transformer=True,       # Automatically adapts to MLP weights
        )
        
        # Insert after specific transformer blocks
        model.blocks[9].add_module("kgft", kgft)</code></pre>
        
        <hr>
        
        <!-- Repository Structure -->
        <h2>📁 Repository Structure</h2>
        <pre><code>KGFT/
        ├── kgft/
        │   ├── __init__.py
        │   ├── kgft.py              # Core KGFT module
        │   ├── scheduling.py        # Depth-aware Exploit/Explore scheduler
        │   └── utils.py             # Gram matrix, covariance helpers
        ├── examples/
        │   ├── train_cifar.py       # CIFAR-100 training script
        │   ├── train_imagenet.py    # ImageNet-1K training script
        │   └── finetune_llama.py    # LLaMA-7B fine-tuning with LoRA
        ├── configs/                  # YAML configurations for different architectures
        ├── tests/                    # Unit tests
        ├── README.md
        └── LICENSE</code></pre>
        
        <hr>
        
        <!-- Key Results -->
        <h2>📊 Key Results (Detailed)</h2>
        <h3>CIFAR-100 (8 runs, mean ± std)</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Baseline</th>
                    <th><strong>+ KGFT</strong></th>
                    <th>Gain</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ResNet-20</td>
                    <td>67.61 ± 0.28</td>
                    <td><strong>69.39 ± 0.16</strong></td>
                    <td><strong>+1.78</strong></td>
                </tr>
                <tr>
                    <td>ResNet-32</td>
                    <td>69.69 ± 0.36</td>
                    <td><strong>70.85 ± 0.55</strong></td>
                    <td><strong>+1.16</strong></td>
                </tr>
                <tr>
                    <td>ResNet-18</td>
                    <td>74.60 ± 0.79</td>
                    <td><strong>76.84 ± 0.26</strong></td>
                    <td><strong>+2.24</strong></td>
                </tr>
                <tr>
                    <td>ViT-Tiny</td>
                    <td>54.33 ± 0.18</td>
                    <td><strong>54.96 ± 0.48</strong></td>
                    <td><strong>+0.63</strong></td>
                </tr>
            </tbody>
        </table>
        
        <h3>ImageNet-1K (4 runs)</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Baseline</th>
                    <th><strong>+ KGFT</strong></th>
                    <th>Gain</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ResNet-34</td>
                    <td>73.57 ± 0.18</td>
                    <td><strong>74.07 ± 0.48</strong></td>
                    <td><strong>+0.50</strong></td>
                </tr>
                <tr>
                    <td>ResNet-50</td>
                    <td>75.43 ± 0.31</td>
                    <td><strong>76.58 ± 0.16</strong></td>
                    <td><strong>+1.15</strong></td>
                </tr>
            </tbody>
        </table>
        
        <hr>
        
        <!-- Configuration -->
        <h2>🔧 Configuration Options</h2>
        <pre><code>KGFT(
            channels: int,                    # Number of channels
            mode: str = "exploit",            # "exploit" or "explore"
            learnable_strength: bool = True,  # Learnable guidance strength
            epsilon: float = 1e-5,            # For numerical stability
            use_transformer: bool = False,    # Set True for Transformer layers
            warmup_epochs: int = 50,          # Warmup steps for learnable strength
        )</code></pre>
        
        <h3>Recommended Scheduling</h3>
        <table>
            <thead>
                <tr>
                    <th>Architecture</th>
                    <th>Layers</th>
                    <th>Mode</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ResNet-18/34/50</td>
                    <td>Stage 1-3</td>
                    <td>Exploit</td>
                </tr>
                <tr>
                    <td>ResNet-18/34/50</td>
                    <td>Stage 4</td>
                    <td>Explore</td>
                </tr>
                <tr>
                    <td>ViT-Tiny</td>
                    <td>Layers 4, 8</td>
                    <td>Exploit</td>
                </tr>
                <tr>
                    <td>ViT-Tiny</td>
                    <td>Layer 12</td>
                    <td>Explore</td>
                </tr>
                <tr>
                    <td>LLaMA-7B</td>
                    <td>Layers 9, 21, 32</td>
                    <td>Exploit</td>
                </tr>
            </tbody>
        </table>
        
        <hr>
        
        <!-- Citation -->
        <h2>📝 Citation</h2>
        <p>If you find KGFT useful for your research, please cite our work:</p>
        <pre><code>@article{feng2026kgft,
            title={Dual-Manifold Geometry Guided Representation Learning: Adaptive Coupling Between Kernel and Data Spaces},
            author={Feng, Qianjin and ...},
            journal={arXiv preprint arXiv:2608.12345},
            year={2026}
        }</code></pre>
        
        <hr>
        
        <!-- Contact -->
        <h2>📬 Contact & Discussion</h2>
        <ul>
            <li><strong>Paper:</strong> <a href="#">arXiv:2608.12345</a></li>
            <li><strong>Issues:</strong> <a href="#">GitHub Issues</a></li>
            <li><strong>Email:</strong> fengqj99@smu.edu.cn</li>
        </ul>
        <p>We welcome discussions, suggestions, and contributions! 🚀</p>
        
        <hr>
        
        <!-- Acknowledgments -->
        <h2>🙏 Acknowledgments</h2>
        <p>This work was supported by the National Key R&D Program of China (2024YFA1012002, 2018YFC2001203) and the National Natural Science Foundation of China (62471214). Special thanks to the AI Computing Platform at the School of Biomedical Engineering, Southern Medical University.</p>
        
        <hr>
        
        <p><strong>If KGFT inspires your work, please ⭐ star this repo — it helps others discover the dual-manifold perspective!</strong><br>
        <strong>Let's build the geometry-aware AI together.</strong> 🚀</p>
    </div>
</body>
</html>
