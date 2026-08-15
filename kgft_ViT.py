import torch
import torch.nn as nn
from timm import create_model


DATASET = 'cifar100'
MODEL_NAME = 'vit_tiny_patch16_224'

ENABLE_KGFT = True
M_EPSILON = 1e-3

# ===== 可学习强度配置 =====
LEARN_STRENGTH = True
STRENGTH_LR_FACTOR = 0.5   # 强度学习率 = 主学习率 * 因子

# ===== 强度预热配置 =====
WARMUP_EPOCHS = 10          # 短预热

# ========== ViT KGFT 配置（稀疏插入，仅 L3, L7, L11）==========
VIT_KGFT_CONFIG = {
    3: {'mode': 'exploit', 'insert': True, 'init_strength': 0.1},
    7: {'mode': 'exploit', 'insert': True, 'init_strength': 0.1},
    11: {'mode': 'explore', 'insert': True, 'init_strength': 0.1},
}

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.01

# ========================= ViT 模型配置 =========================
VIT_MODEL_CONFIGS = {
    'vit_tiny_patch16_224': {'embed_dim': 192, 'num_layers': 12, 'num_heads': 3},
    'vit_small_patch16_224': {'embed_dim': 384, 'num_layers': 12, 'num_heads': 6},
    'vit_base_patch16_224': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12},
}

vit_cfg = VIT_MODEL_CONFIGS[MODEL_NAME]
NUM_LAYERS = vit_cfg['num_layers']
EMBED_DIM = vit_cfg['embed_dim']

# ========================= KGFT 核心模块（ViT 极简版）=========================
class KernelGuidedFeatureTransform(nn.Module):
    """
    核引导特征变换（ViT 极简版）
    移除 out_proj 和 LayerNorm，仅保留几何变换 + 强度门控。
    """
    def __init__(self, dim, init_strength=0.5, eps=1e-3, mode='exploit'):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.mode = mode

        # 只有强度参数（标量）
        s_clamped = max(1e-7, min(1 - 1e-7, init_strength))
        raw_init = torch.logit(torch.tensor(s_clamped, dtype=torch.float32))
        if LEARN_STRENGTH:
            self.raw_strength = nn.Parameter(raw_init)
        else:
            self.register_buffer('raw_strength', raw_init)

    @property
    def strength(self):
        return torch.sigmoid(self.raw_strength).item()

    def set_config(self, mode, strength=None):
        if mode not in ['exploit', 'explore', 'none']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        if strength is not None:
            with torch.no_grad():
                s_clamped = max(1e-7, min(1 - 1e-7, strength))
                raw_val = torch.logit(torch.tensor(s_clamped, dtype=torch.float32))
                self.raw_strength.data.fill_(raw_val)

    def forward(self, x, weight):
        """
        x: (B, N, D) – FFN 的输出
        weight: (D_in, D) – FFN 第一层权重 (intermediate_dim, D)
        """
        if self.mode == 'none':
            return x

        s = torch.sigmoid(self.raw_strength)
        B, N, D = x.shape

        # 1. 核流形度量 G = W^T W
        G = torch.mm(weight.t(), weight)   # (D, D)

        # 2. 构建引导矩阵 M
        if self.mode == 'exploit':
            M = G + self.eps * torch.eye(D, device=x.device)
        elif self.mode == 'explore':
            diag = torch.diag(G)
            diag_sqrt = torch.sqrt(diag + 1e-8)
            D_inv = torch.diag(1.0 / diag_sqrt)
            C = D_inv @ G @ D_inv
            s_mean = torch.mean(diag)
            M = s_mean * (torch.eye(D, device=x.device) - C) + self.eps * torch.eye(D, device=x.device)
        else:
            M = torch.eye(D, device=x.device)

        # 3. 数据流形度量 K = X^T X
        K = torch.bmm(x.transpose(1, 2), x) / (N ** 0.5)   # (B, D, D)

        # 4. 变换度量 K' = M K
        K_prime = torch.matmul(M.unsqueeze(0), K)          # (B, D, D)

        # 5. 特征变换 Y = X K'
        y = torch.bmm(x, K_prime)                          # (B, N, D)

        # 6. 强度门控 + 残差（无额外投影和 LN）
        return y * s + x

# ========================= 构建 ViT + KGFT =========================
def build_vit_with_kgft():
    model = create_model(MODEL_NAME, pretrained=False, num_classes=1000)

    if not ENABLE_KGFT:
        return model

    for idx, block in enumerate(model.blocks):
        if idx not in VIT_KGFT_CONFIG:
            continue
        cfg = VIT_KGFT_CONFIG[idx]
        if not cfg.get('insert', False):
            continue

        mode = cfg['mode']
        init_strength = cfg.get('init_strength', 0.5)

        fc1_weight = block.mlp.fc1.weight
        kgft = KernelGuidedFeatureTransform(
            dim=EMBED_DIM,
            init_strength=init_strength,
            eps=M_EPSILON,
            mode=mode
        )

        block.mlp.kgft = kgft
        original_mlp_forward = block.mlp.forward

        def new_mlp_forward(self, x):
            x = original_mlp_forward(x)
            x = self.kgft(x, fc1_weight)
            return x

        import types
        block.mlp.forward = types.MethodType(new_mlp_forward, block.mlp)

        print(f"  Layer {idx}: KGFT inserted (mode={mode}, init_strength={init_strength:.2f})")

    return model

