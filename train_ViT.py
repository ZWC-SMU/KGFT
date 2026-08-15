"""
ViT (Vision Transformer) with KGFT - Decoupled Mode & Strength (Minimal Version)
========================================================================
- 移除 out_proj 和 LayerNorm，只保留核心几何变换 + 强度门控
- 大幅降低参数量（每模块仅 1 个可学习参数），避免过拟合
- 学习率调回 1e-4，与基线一致
- 其他配置与 CNN 版本保持一致
========================================================================
"""

import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from timm import create_model
from tqdm import tqdm
import random
import numpy as np

from kgft_ViT import build_vit_with_kgft

warnings.filterwarnings("ignore")

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ========================= 用户配置区域 =========================
CONFIGS = [
    {'gpu': 0, 'seed': 19},
    {'gpu': 1, 'seed': 74},
    {'gpu': 2, 'seed': 37},
    {'gpu': 3, 'seed': 42},
    {'gpu': 4, 'seed': 19},
    {'gpu': 5, 'seed': 74},
    {'gpu': 6, 'seed': 37},
    {'gpu': 7, 'seed': 42},
]
CONFIG_INDEX = 3   # 选择第几个配置
GPU_ID = CONFIGS[CONFIG_INDEX]['gpu']
SEED = CONFIGS[CONFIG_INDEX]['seed']

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
# 其他层默认不插入（insert=False 或未定义）
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
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# ========================= 超参数 =========================
if DATASET == 'cifar100':
    BATCH_SIZE, EPOCHS, LR, DATA_ROOT = 256, 200, 1e-4, '../data'   # LR 与基线一致
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.01

# ========================= ViT 模型配置 =========================
VIT_MODEL_CONFIGS = {
    'vit_tiny_patch16_224': {'embed_dim': 192, 'num_layers': 12, 'num_heads': 3},
    'vit_small_patch16_224': {'embed_dim': 384, 'num_layers': 12, 'num_heads': 6},
    'vit_base_patch16_224': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12},
}

if MODEL_NAME not in VIT_MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

vit_cfg = VIT_MODEL_CONFIGS[MODEL_NAME]
NUM_LAYERS = vit_cfg['num_layers']
EMBED_DIM = vit_cfg['embed_dim']

# ========================= 数据集配置 =========================
DATASET_CONFIGS = {
    'cifar100': {
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'size': 224,
        'classes': 100,
        'loader': CIFAR100
    },
}
num_classes = DATASET_CONFIGS[DATASET]['classes']

# ========================= 数据加载 =========================
def get_data_loaders():
    ds_info = DATASET_CONFIGS[DATASET]
    mean, std, image_size = ds_info['mean'], ds_info['std'], ds_info['size']

    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    valset = CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"{DATASET} loaded: train {len(trainset)}, val {len(valset)}, classes={num_classes}")
    return train_loader, val_loader


# ========================= 训练与验证函数 =========================
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
                epochs, seed=None, save_prefix=None,
                learn_strength=False, strength_lr_factor=0.1,
                warmup_epochs=0):
    print(f"\n========== Training {model_name} ==========")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ----- 分离参数组 -----
    base_params = []
    strength_params = []
    for name, param in model.named_parameters():
        if 'raw_strength' in name:
            strength_params.append(param)
        else:
            base_params.append(param)

    if strength_params and learn_strength:
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': LR, 'weight_decay': WEIGHT_DECAY},
            {'params': strength_params, 'lr': 1e-8, 'weight_decay': WEIGHT_DECAY}
        ])
        print(f"Strength parameters will use LR = {LR * strength_lr_factor:.6f} (factor={strength_lr_factor})")
        if warmup_epochs > 0:
            print(f"Strength warm-up enabled for {warmup_epochs} epochs")
        else:
            print("Strength warm-up disabled")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print(f"Scheduler: CosineAnnealingLR (T_max={epochs})")

    model_dir = os.path.join('.', 'result', 'model', SCRIPT_NAME)
    os.makedirs(model_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # ===== 预热逻辑 =====
        if strength_params and learn_strength:
            model_lr = optimizer.param_groups[0]['lr']
            if warmup_epochs > 0:
                warmup_factor = min(1.0, epoch / warmup_epochs)
            else:
                warmup_factor = 1.0
            strength_lr = model_lr * strength_lr_factor * warmup_factor
            optimizer.param_groups[1]['lr'] = strength_lr

        # ----- 收集强度信息 -----
        block_info = []
        for idx, block in enumerate(model.blocks):
            if hasattr(block.mlp, 'kgft') and block.mlp.kgft is not None:
                kgft = block.mlp.kgft
                if learn_strength:
                    kgft.set_config(kgft.mode)
                else:
                    progress = (epoch - 1) / (epochs - 1)
                    strength = 0.5 + 0.5 * progress
                    kgft.set_config(kgft.mode, strength)
                block_info.append(f"L{idx}:{kgft.mode}({kgft.strength:.3f})")

        if block_info:
            current_strength_lr = optimizer.param_groups[1]['lr'] if strength_params and learn_strength else 0
            print(f"Epoch {epoch}/{epochs} | Strength LR: {current_strength_lr:.6f} | " + ", ".join(block_info))
        else:
            print(f"Epoch {epoch}/{epochs}")

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


# ========================= 日志输出 =========================
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


# ========================= 主函数 =========================
def main():
    log_dir = os.path.join('.', 'result', 'log', SCRIPT_NAME)
    os.makedirs(log_dir, exist_ok=True)

    learn_str = "_learn" if LEARN_STRENGTH else "_anneal"
    warmup_str = f"_warmup{WARMUP_EPOCHS}" if (LEARN_STRENGTH and WARMUP_EPOCHS > 0) else ""
    lrf_str = f"_lrf{STRENGTH_LR_FACTOR:.4f}" if LEARN_STRENGTH else ""

    kgft_suffix = "_kgft" if ENABLE_KGFT else "_standard"
    name_core = f"{MODEL_NAME}_{DATASET}{kgft_suffix}{learn_str}{warmup_str}{lrf_str}_bs{BATCH_SIZE}_ep{EPOCHS}_lr{LR}_seed{SEED}"
    log_filename = os.path.join(log_dir, f"log_{name_core}.txt")
    model_save_prefix = f"best_{name_core}"

    tee = Tee(log_filename)
    try:
        print("=" * 60)
        print(f"Experiment: {name_core}")
        print(f"Model: {MODEL_NAME} (layers={NUM_LAYERS}, embed_dim={EMBED_DIM})")
        if ENABLE_KGFT:
            print("KGFT enabled: True (ViT Decoupled Mode & Strength - Minimal)")
            print(f"Learnable Strength: {LEARN_STRENGTH}")
            if LEARN_STRENGTH:
                print(f"  Strength LR factor: {STRENGTH_LR_FACTOR}")
                if WARMUP_EPOCHS > 0:
                    print(f"  Strength Warm-up epochs: {WARMUP_EPOCHS}")
                else:
                    print("  Strength Warm-up: disabled")
            else:
                print("  (Using manual annealing schedule)")
            print("Layer Schedule:")
            for i in range(NUM_LAYERS):
                if i in VIT_KGFT_CONFIG:
                    cfg = VIT_KGFT_CONFIG[i]
                    if cfg.get('insert', False):
                        init_str = cfg.get('init_strength', 0.5)
                        print(f"  L{i}: mode={cfg['mode']} (inserted, init_strength={init_str:.2f})")
                    else:
                        print(f"  L{i}: mode={cfg['mode']} (skipped)")
                else:
                    print(f"  L{i}: off")
        else:
            print("KGFT enabled: False")
        print(f"Log file: {log_filename}")
        print("=" * 60)

        train_loader, val_loader = get_data_loaders()
        model = build_vit_with_kgft()
        acc = train_model(
            model, name_core, train_loader, val_loader, device,
            epochs=EPOCHS, seed=SEED,
            save_prefix=model_save_prefix,
            learn_strength=LEARN_STRENGTH,
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