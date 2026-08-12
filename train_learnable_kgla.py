"""
ResNet with Kernel-Guided Linear Attention (KGLA).

This open-source version keeps only the learnable-strength formulation.
KGLA strength is represented by a trainable scalar parameter and optimized
with a dedicated learning-rate group and warm-up schedule.

Key design:
    - Stage-wise Exploit / Explore modes.
    - Optional multiple KGLA insertion positions per stage.
    - Learnable KGLA strength in (0, 1) via sigmoid parameterization.
    - Dedicated learning rate for strength parameters.
    - Warm-up for stable strength optimization.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.models as models

warnings.filterwarnings("ignore")

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ========================= 用户配置区域 =========================
CONFIGS = [
    {'gpu': 0, 'seed': 19},
    # {'gpu': 1, 'seed': 74},
    # {'gpu': 2, 'seed': 37},
    # {'gpu': 3, 'seed': 42},
    #{'gpu': 4, 'seed': 19},
    #{'gpu': 5, 'seed': 74},
    #{'gpu': 6, 'seed': 37},
    #{'gpu': 7, 'seed': 42},
    # {'gpu': 4, 'seed': 7},
    # {'gpu': 5, 'seed': 8},
    # {'gpu': 6, 'seed': 2008},
    # {'gpu': 7, 'seed': 25},
    # {'gpu': 4, 'seed': 7},
    #{'gpu': 5, 'seed': 8},
    #{'gpu': 6, 'seed': 2008},
    #{'gpu': 7, 'seed': 25},
]
CONFIG_INDEX = 0  # 选择第几个配置
GPU_ID = CONFIGS[CONFIG_INDEX]['gpu']
SEED = CONFIGS[CONFIG_INDEX]['seed']

DATASET = 'imagenet'          # 'cifar100' 或 'tinyimagenet' 或 'imagenet'
MODEL_NAME = 'resnet34'       # 可选: resnet20/32/44/56/110/18/34/50

ENABLE_KGLA = True            # 总开关，设为 False 则所有 KGLA 失效
M_EPSILON = 1e-3

# ===== 可学习强度配置 =====
STRENGTH_LR_FACTOR = 0.5     # 强度学习率 = 主学习率 * 该因子

# ===== 强度预热配置 =====
WARMUP_EPOCHS = 50            # 预热周期，建议 30~50

# ---------- 统一配置：模型 -> {stage_index: {'mode': str}} ----------
# mode: 'exploit' / 'explore'
#    'none' 表示该 Stage 不插入 KGLA 模块
#    其他非 'none' 值表示插入 KGLA 模块，并在训练时按指定模式运行
# insert_positions: list of block indices (0-based) 指定在该 Stage 的哪些 Block 后插入 KGLA
#   
MODEL_KGLA_CONFIGS = {
    'resnet18': {
        0: {'mode': 'exploit'},
        1: {'mode': 'exploit'},
        2: {'mode': 'exploit'},
        3: {'mode': 'explore'},
    },
    # ========== ResNet34：多点插入（ImageNet 测试主力）==========
    'resnet34': {
        0: {
            'mode': 'exploit',
            'insert_positions': [2],        # 3 blocks: Block1(中间) + Block2(出口)
        },
        1: {
            'mode': 'exploit',
            'insert_positions': [1, 3],     # 4 blocks: Block1,2(中间) + Block3(出口)
        },
        2: {
            'mode': 'exploit',
            'insert_positions': [2, 5],     # 6 blocks: Block2,4(中间) + Block5(出口)
        },
        3: {
            'mode': 'explore',
            'insert_positions': [2],        # 3 blocks: Block1(中间) + Block2(出口)
        },
    },
    # ========== ResNet50：仅 Stage 出口单点插入（默认行为）==========
    'resnet50': {
        0: {
            'mode': 'exploit',
            'insert_positions': [2],  # 3 blocks: Block1(中间) + Block2(出口)
        },
        1: {
            'mode': 'exploit',
            'insert_positions': [1, 3],  # 4 blocks: Block1,2(中间) + Block3(出口)
        },
        2: {
            'mode': 'exploit',
            'insert_positions': [2, 5],  # 6 blocks: Block2,4(中间) + Block5(出口)
        },
        3: {
            'mode': 'explore',
            'insert_positions': [2],  # 3 blocks: Block1(中间) + Block2(出口)
        },
    },
    # ==========================================================
    'resnet20': {
        0: {'mode': 'exploit'},
        1: {'mode': 'exploit'},
        2: {'mode': 'explore'},
    },
    # ========== ResNet32：每 Stage 5 个 Block，双点插入 ==========
    'resnet32': {
        0: {
            'mode': 'exploit',
            'insert_positions': [2, 4],   # Block2 (中间) + Block4 (出口)
        },
        1: {
            'mode': 'exploit',
            'insert_positions': [2, 4],
        },
        2: {
            'mode': 'explore',
            'insert_positions': [2, 4],
        },
    },
    'resnet44': {
        0: {'mode': 'exploit'},
        1: {'mode': 'exploit'},
        2: {'mode': 'explore'},
    },
    'resnet56': {
        0: {'mode': 'exploit'},
        1: {'mode': 'exploit'},
        2: {'mode': 'explore'},
    },
    'resnet110': {
        0: {'mode': 'exploit'},
        1: {'mode': 'exploit'},
        2: {'mode': 'explore'},
    },
}
# ================================================================

if GPU_ID is not None and GPU_ID >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if device.type == 'cuda':
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU")

# ========================= 超参数 =========================
if DATASET in ['cifar10', 'cifar100']:
    BATCH_SIZE, EPOCHS, SGD_LR, CUTOUT_SIZE, DATA_ROOT = 512, 200, 0.1, 16, '../data'
elif DATASET == 'tinyimagenet':
    BATCH_SIZE, EPOCHS, SGD_LR, CUTOUT_SIZE, DATA_ROOT = 256, 200, 0.1, 16, './data/tiny-imagenet-200'
elif DATASET == 'imagenet':
    BATCH_SIZE, EPOCHS, SGD_LR, CUTOUT_SIZE, DATA_ROOT = 256, 100, 0.1, 0, './data/imagenet'
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")

WEIGHT_DECAY = 1e-4
GAMMA = 0.1
LABEL_SMOOTHING = 0.01
CUTOUT_ENABLE = True if DATASET in ['cifar10', 'cifar100', 'tinyimagenet'] else False
LR_MILESTONE_RATIOS = [0.3, 0.6, 0.8]

milestones = sorted(set([int(EPOCHS * r) for r in LR_MILESTONE_RATIOS if 0 < int(EPOCHS * r) < EPOCHS]))
if not milestones:
    milestones = [int(EPOCHS * 0.5)]

# ========================= ResNet 配置 =========================
RESNET_CONFIGS = {
    'resnet20':  {'blocks': [3, 3, 3], 'base_channels': 16, 'stages': 3, 'block_type': 'basic'},
    'resnet32':  {'blocks': [5, 5, 5], 'base_channels': 16, 'stages': 3, 'block_type': 'basic'},
    'resnet44':  {'blocks': [7, 7, 7], 'base_channels': 16, 'stages': 3, 'block_type': 'basic'},
    'resnet56':  {'blocks': [9, 9, 9], 'base_channels': 16, 'stages': 3, 'block_type': 'basic'},
    'resnet110': {'blocks': [18, 18, 18], 'base_channels': 16, 'stages': 3, 'block_type': 'basic'},
    'resnet18':  {'blocks': [2, 2, 2, 2], 'base_channels': 64, 'stages': 4, 'block_type': 'basic'},
    'resnet34':  {'blocks': [3, 4, 6, 3], 'base_channels': 64, 'stages': 4, 'block_type': 'basic'},
    'resnet50':  {'blocks': [3, 4, 6, 3], 'base_channels': 64, 'stages': 4, 'block_type': 'bottleneck'},
}

model_cfg = RESNET_CONFIGS[MODEL_NAME]
blocks_per_stage = model_cfg['blocks']
base_channels = model_cfg['base_channels']
num_stages = model_cfg['stages']
block_type = model_cfg['block_type']

# ---------- 从统一配置生成 kgla_enable 和 stage_schedule ----------
model_config = MODEL_KGLA_CONFIGS.get(MODEL_NAME, {})

kgla_enable = ['none'] * num_stages
stage_schedule = {}
for stage_idx, cfg in model_config.items():
    if stage_idx >= num_stages:
        continue
    mode = cfg.get('mode', 'none')
    if mode != 'none':
        kgla_enable[stage_idx] = 'kgla'
    # 保存完整配置，包括 insert_positions
    stage_schedule[stage_idx] = {
        'mode': mode,
        'range': cfg.get('range', (0.0, 0.0)),
        'insert_positions': cfg.get('insert_positions', None)  # 默认为 None，表示只在最后一块插入
    }

if not ENABLE_KGLA:
    kgla_enable = ['none'] * num_stages
    stage_schedule = {}

# ========================= 数据集配置 =========================
DATASET_CONFIGS = {
    'cifar10' : {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010], 'size': 32, 'classes': 10, 'loader': torchvision.datasets.CIFAR10},
    'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761], 'size': 32, 'classes': 100, 'loader': torchvision.datasets.CIFAR100},
    'tinyimagenet': {'mean': [0.4802, 0.4481, 0.3975], 'std': [0.2302, 0.2265, 0.2262], 'size': 64, 'classes': 200, 'loader': None},
    'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'size': 224, 'classes': 1000, 'loader': None},
}
num_classes = DATASET_CONFIGS[DATASET]['classes']

# ========================= Cutout =========================
class Cutout:
    def __init__(self, size=16):
        self.size = size
    def __call__(self, img):
        if self.size <= 0:
            return img
        h, w = img.size(1), img.size(2)
        y = np.random.randint(h); x = np.random.randint(w)
        y1 = max(0, y - self.size // 2); y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2); x2 = min(w, x + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

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


# # ========================= ImageNet dataset =========================
class ImageNetDataset(ImageFolder):
    """ImageNet folder dataset wrapper."""

    pass


# ========================= Data loading =========================
def get_data_loaders():
    ds_info = DATASET_CONFIGS[DATASET]
    mean, std, image_size = ds_info['mean'], ds_info['std'], ds_info['size']
    padding = 4 if image_size <= 32 else 8 if image_size <= 64 else 16

    train_transforms = []
    if DATASET == 'imagenet':
        train_transforms.append(transforms.RandomResizedCrop(image_size))
    else:
        train_transforms.append(transforms.RandomCrop(image_size, padding=padding))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    if CUTOUT_ENABLE and CUTOUT_SIZE > 0:
        train_transforms.append(Cutout(size=CUTOUT_SIZE))
    train_transforms.append(transforms.Normalize(mean, std))

    transform_train = transforms.Compose(train_transforms)
    transform_val = transforms.Compose([
        transforms.Resize(image_size + 32) if DATASET == 'imagenet' else transforms.Lambda(lambda x: x),
        transforms.CenterCrop(image_size) if DATASET == 'imagenet' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # if DATASET in ['cifar10', 'cifar100']:
    #     loader_class = ds_info['loader']
    #     trainset = loader_class(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    #     valset = loader_class(root=DATA_ROOT, train=False, download=True, transform=transform_val)
    # else:
    #     train_dir, val_dir = os.path.join(DATA_ROOT, 'train'), os.path.join(DATA_ROOT, 'val')
    #     trainset = ImageFolder(train_dir, transform=transform_train)
    #     valset = ImageFolder(val_dir, transform=transform_val)

    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')
    trainset = ImageNetDataset(train_dir, transform=transform_train)
    valset = ImageNetDataset(val_dir, transform=transform_val)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    print(f"{DATASET} loaded: train {len(trainset)}, val {len(valset)}, classes={num_classes}")
    return train_loader, val_loader

# ========================= 训练函数 =========================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def train_model(model, model_name, train_loader, val_loader, device,
                epochs, seed=None, milestones=milestones, save_prefix=None,
                learn_strength=False, strength_lr_factor=0.1,
                warmup_epochs=WARMUP_EPOCHS):
    print(f"\n========== Training {model_name} ==========")
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ----- Separate backbone and learnable-strength parameters -----
    base_params = []
    strength_params = []

    for name, param in model.named_parameters():
        if 'raw_strength' in name:
            strength_params.append(param)
        else:
            base_params.append(param)

    if not strength_params:
        raise RuntimeError("No learnable KGLA strength parameters were found.")

    optimizer = optim.SGD(
        [
            {
                'params': base_params,
                'lr': SGD_LR,
                'momentum': 0.9,
                'weight_decay': WEIGHT_DECAY,
            },
            {
                'params': strength_params,
                'lr': 1e-8,
                'momentum': 0.9,
                'weight_decay': WEIGHT_DECAY,
            },
        ]
    )

    print(
        f"Strength LR factor: {strength_lr_factor:.3f} | "
        f"Warm-up epochs: {warmup_epochs}"
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=GAMMA)
    print(f"Scheduler: MultiStepLR (milestones={milestones}, gamma={GAMMA})")

    model_dir = os.path.join('.', 'result', 'model', SCRIPT_NAME)
    os.makedirs(model_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # ----- Warm up the learnable-strength learning rate -----
        model_lr = optimizer.param_groups[0]['lr']
        warmup_factor = min(1.0, epoch / max(1, warmup_epochs))
        strength_lr = model_lr * strength_lr_factor * warmup_factor
        optimizer.param_groups[1]['lr'] = strength_lr

        # ----- Apply stage-wise modes -----
        configs = {}
        stage_info = []
        strengths_dict = model.get_stage_strengths()

        for i in range(model.num_stages):
            if i in stage_schedule:
                mode = stage_schedule[i]['mode']
                configs[i] = {'mode': mode}

                stage_vals = strengths_dict.get(i)
                if mode == 'none':
                    stage_info.append(f"S{i}:off")
                elif stage_vals:
                    values = ','.join(f"{v:.3f}" for v in stage_vals)
                    stage_info.append(f"S{i}:{mode}({values})")
                else:
                    stage_info.append(f"S{i}:{mode}")
            else:
                configs[i] = {'mode': 'none'}
                stage_info.append(f"S{i}:off")

        model.set_stage_configs(configs)

        current_strength_lr = optimizer.param_groups[1]['lr']
        print(
            f"Epoch {epoch}/{epochs} | Strength LR: {current_strength_lr:.6f} | "
            + ", ".join(stage_info)
        )

        # print(f"Epoch {epoch}/{epochs} | ")

        # ----- 训练与验证 -----
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {best_acc:.2f}% | LR: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_filename = f"{save_prefix}_seed{seed}.pth" if seed is not None else f"{save_prefix}.pth"
            save_path = os.path.join(model_dir, save_filename)
            torch.save(model.state_dict(), save_path)
            print(f"*** New best accuracy: {best_acc:.2f}% | Model saved to {save_path}")

    return best_acc

# ========================= 主函数 =========================
class Tee:
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, message):
        self.stdout.write(message)
        if '\r' not in message:
            self.file.write(message)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout

def main():
    log_dir = os.path.join('.', 'result', 'log', SCRIPT_NAME)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- Experiment naming ----------
    learn_str = "_learnable"
    warmup_str = f"_warmup{WARMUP_EPOCHS}"
    lrf_str = f"_lrf{STRENGTH_LR_FACTOR:.1f}"

    kgla_suffix = "_decoupled" if ENABLE_KGLA else "_standard"
    name_core = f"{MODEL_NAME}_{DATASET}{kgla_suffix}{learn_str}{warmup_str}{lrf_str}_bs{BATCH_SIZE}_ep{EPOCHS}_lr{SGD_LR}_seed{SEED}{range_str}"
    log_filename = os.path.join(log_dir, f"log_{name_core}.txt")
    model_save_prefix = f"best_{name_core}"

    tee = Tee(log_filename)
    try:
        print("=" * 60)
        print(f"Experiment: {name_core}")
        print(f"Model: {MODEL_NAME} (stages={num_stages})")
        if ENABLE_KGLA:
            print("KGLA enabled: True (Learnable Strength)")
            print(f"  Strength LR factor: {STRENGTH_LR_FACTOR}")
            print(f"  Strength warm-up epochs: {WARMUP_EPOCHS}")
            print("Stage Schedule:")

            for i in range(num_stages):
                cfg = stage_schedule.get(i)
                if cfg is None or cfg.get('mode', 'none') == 'none':
                    print(f"  S{i}: off")
                else:
                    pos_info = ""
                    if cfg.get('insert_positions') is not None:
                        pos_info = f" insert_positions={cfg['insert_positions']}"
                    print(
                        f"  S{i}: mode={cfg['mode']}{pos_info}"
                    )
        else:
            print("KGLA enabled: False")
        print(f"Log file: {log_filename}")
        print("=" * 60)

        train_loader, val_loader = get_data_loaders()

        model = ResNet_KGLA(
            num_classes=num_classes,
            blocks_per_stage=blocks_per_stage,
            base_channels=base_channels,
            block_type=block_type,
            kgla_enable=kgla_enable,
            stage_schedule=stage_schedule
        )
        #
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"总参数量: {total_params:,}")
        # print(f"可训练参数量: {trainable_params:,}")
        # total_size = total_params * 4 / (1024 ** 2)  # 假设float32
        # print(f"模型大小 (FP32): {total_size:.2f} MB")

        # model = models.resnet34()
        # model.fc = nn.Linear(model.fc.in_features, 1000)  # 想输出为9个类别时

        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"总参数量: {total_params:,}")
        # print(f"可训练参数量: {trainable_params:,}")
        # total_size = total_params * 4 / (1024 ** 2)  # 假设float32
        # print(f"模型大小 (FP32): {total_size:.2f} MB")
        #
        # exit()

        acc = train_model(
            model, name_core, train_loader, val_loader, device,
            epochs=EPOCHS, seed=SEED, milestones=milestones,
            save_prefix=model_save_prefix,
            strength_lr_factor=STRENGTH_LR_FACTOR,
            warmup_epochs=WARMUP_EPOCHS
        )

        print("\n" + "=" * 60)
        print(f"Final Result: {name_core}: {acc:.2f}%")
        print("=" * 60)
    finally:
        tee.close()

if __name__ == '__main__':
    main()