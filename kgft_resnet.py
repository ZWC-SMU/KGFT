import torch
import torch.nn as nn
import torch.nn.functional as F

M_EPSILON = 1e-3

# ========================= KGLA core module =========================
class KernelGuidedLinearAttention(nn.Module):
    def __init__(self, dim, prev_conv_weight, epsilon=None, init_strength=0.5):
        super().__init__()
        if epsilon is None:
            epsilon = M_EPSILON
        self.dim = dim
        self.epsilon = epsilon
        self.prev_conv_weight = prev_conv_weight
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.mode = 'exploit'

        # ===== 将强度变为可训练参数 =====
        s_clamped = max(1e-7, min(1 - 1e-7, init_strength))
        raw_init = torch.logit(torch.tensor(s_clamped, dtype=torch.float32))
        self.raw_strength = nn.Parameter(raw_init)

    @property
    def strength(self):
        """返回当前实际强度 (0~1)"""
        return torch.sigmoid(self.raw_strength).item()

    def set_config(self, mode):
        """Set the KGLA operating mode."""
        if mode not in ['exploit', 'explore', 'none']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def _compute_M(self, kernel_weight):
        C_out = kernel_weight.size(0)
        flat = kernel_weight.view(C_out, -1)
        G = flat @ flat.T

        diag = torch.diag(G)
        diag_safe = diag + 1e-8
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(diag_safe))
        C = D_inv_sqrt @ G @ D_inv_sqrt
        scale = torch.mean(diag)

        if self.mode == 'exploit':
            M_mode = G + self.epsilon * torch.eye(C_out, device=G.device)
        elif self.mode == 'explore':
            M_mode = scale * (1 - C) + self.epsilon * torch.eye(C_out, device=G.device)
        else:  # 'none'
            M_mode = torch.eye(C_out, device=G.device)

        I_mat = torch.eye(C_out, device=G.device)
        M = I_mat + (M_mode - I_mat)
        return M

    def forward(self, x):
        if self.mode == 'none':
            return x
        s = torch.sigmoid(self.raw_strength)  # 实际强度值 (0~1)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        N = H * W
        M = self._compute_M(self.prev_conv_weight)
        KTV = torch.einsum('bni,bnj->bij', x_flat, x_flat) / (N ** 0.5)
        T = M @ KTV
        Y_flat = torch.einsum('bni,bij->bnj', x_flat, T)
        Y = Y_flat.transpose(1, 2).reshape(B, C, H, W)
        Y = self.out_proj(Y)
        Y = Y.permute(0, 2, 3, 1).contiguous()
        Y = self.ln(Y)
        Y = Y.permute(0, 3, 1, 2)
        return Y * s + x

# ========================= 基础模块 =========================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kgla_module=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.kgla = kgla_module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.kgla is not None: out = self.kgla(out)   # BasicBlock 插在 conv2 后（保持不变）
        out = self.bn2(out)
        # out = F.relu(out)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, kgla_module=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.kgla = kgla_module
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x); out = self.bn1(out); out = F.relu(out)
        out = self.conv2(out)
        # ===== 修改点：KGLA 插入在 conv2 之后、bn2 之前 =====
        if self.kgla is not None:
            out = self.kgla(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # out = F.relu(out)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = F.relu(out)
        return out

# ========================= ResNet 主网络（接收 stage_schedule）=========================
class ResNet_KGLA(nn.Module):
    def __init__(self, num_classes=100, blocks_per_stage=[3,3,3],
                 base_channels=16, block_type='basic', kgla_enable=None,
                 stage_schedule=None):
        super().__init__()
        self.in_planes = base_channels
        self.num_stages = len(blocks_per_stage)
        self.block_type = block_type
        if kgla_enable is None:
            kgla_enable = ['none'] * self.num_stages
        self.kgla_enable = kgla_enable
        self.stage_schedule = stage_schedule if stage_schedule is not None else {}

        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_layers = nn.ModuleList()
        for stage_idx, num_blocks in enumerate(blocks_per_stage):
            stride = 1 if stage_idx == 0 else 2
            planes = base_channels * (2 ** stage_idx)
            stage = self._make_stage(planes, num_blocks, stride, stage_idx)
            self.stage_layers.append(stage)
            self.in_planes = planes * (4 if block_type == 'bottleneck' else 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_channels = base_channels * (2 ** (self.num_stages - 1))
        if block_type == 'bottleneck':
            final_channels *= 4
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_stage(self, planes, num_blocks, stride, stage_idx):
        BlockClass = Bottleneck if self.block_type == 'bottleneck' else BasicBlock
        expansion = 4 if self.block_type == 'bottleneck' else 1
        blocks = []
        enable = self.kgla_enable[stage_idx] if stage_idx < len(self.kgla_enable) else 'none'

        stage_cfg = self.stage_schedule.get(stage_idx, {})
        insert_positions = stage_cfg.get('insert_positions', None)
        if insert_positions is None:
            insert_positions = [num_blocks - 1]

        for i in range(num_blocks):
            s = stride if i == 0 else 1
            block = BlockClass(self.in_planes, planes, stride=s, kgla_module=None)
            blocks.append(block)
            self.in_planes = planes * expansion

            # ===== 检查是否在当前位置插入 KGLA =====
            if enable != 'none' and i in insert_positions:
                # ===== 修改点：Bottleneck 使用 conv2 权重，维度为 planes =====
                if self.block_type == 'bottleneck':
                    conv_weight = block.conv2.weight
                    kgla_dim = planes
                else:  # BasicBlock
                    conv_weight = block.conv2.weight
                    kgla_dim = planes

                kgla = KernelGuidedLinearAttention(
                    dim=kgla_dim,
                    prev_conv_weight=conv_weight,
                    epsilon=M_EPSILON,
                    init_strength=0.5,
                )
                block.kgla = kgla

        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        for stage in self.stage_layers:
            for block in stage:
                out = block(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def set_stage_configs(self, configs):
        """Apply stage-wise KGLA modes."""
        for stage_idx, config in configs.items():
            if stage_idx >= len(self.stage_layers):
                continue

            mode = config.get('mode', 'none')
            for block in self.stage_layers[stage_idx]:
                if block.kgla is not None:
                    block.kgla.set_config(mode)

    def get_stage_strengths(self):
        """
        返回每个 Stage 中所有 KGLA 的实际强度列表
        返回格式: {stage_idx: [strength1, strength2, ...]}
        """
        strengths = {}
        for i in range(self.num_stages):
            stage_vals = []
            for block in self.stage_layers[i]:
                if hasattr(block, 'kgla') and block.kgla is not None:
                    stage_vals.append(block.kgla.strength)
            if stage_vals:
                strengths[i] = stage_vals
            else:
                strengths[i] = None
        return strengths
