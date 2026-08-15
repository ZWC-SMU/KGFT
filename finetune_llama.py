#!/usr/bin/env python3
"""LoRA fine-tuning for LLaMA-1 7B with KGFT on selected MLP blocks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from kgft_llama import (
    KGFT_CONFIG_NAME,
    KGFT_STATE_NAME,
    attach_attention_mask_hook,
    build_kgft_config,
    inject_kgft,
    load_kgft_state,
    parse_int_csv,
    parse_str_csv,
    save_kgft_artifacts,
    set_runtime_scale,
    set_strength_trainable,
    strength_summary,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]


class CastOutputToFloat(nn.Sequential):
    """Keep the LM-head input in its original dtype and return FP32 logits."""

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.input_dtype = module.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.to(self.input_dtype)).to(torch.float32)


class KGFTTrainer(Trainer):
    """Trainer with a separate learning-rate multiplier for KGFT strengths."""

    def __init__(self, *args: Any, kgft_strength_lr_factor: float, **kwargs: Any) -> None:
        self.kgft_strength_lr_factor = float(kgft_strength_lr_factor)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        regular_params: list[nn.Parameter] = []
        strength_params: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("kgft.raw_strength"):
                strength_params.append(param)
            else:
                regular_params.append(param)

        if not regular_params:
            raise RuntimeError("No trainable LoRA parameters were found")
        if not strength_params:
            raise RuntimeError("No trainable KGFT strength parameters were found")

        base_lr = float(self.args.learning_rate)
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": regular_params,
                    "lr": base_lr,
                    "weight_decay": float(self.args.weight_decay),
                    "group_name": "lora",
                },
                {
                    "params": strength_params,
                    "lr": base_lr * self.kgft_strength_lr_factor,
                    "weight_decay": 0.0,
                    "group_name": "kgft_strength",
                },
            ],
            lr=base_lr,
            betas=(
                float(getattr(self.args, "adam_beta1", 0.9)),
                float(getattr(self.args, "adam_beta2", 0.999)),
            ),
            eps=float(getattr(self.args, "adam_epsilon", 1e-8)),
        )
        return self.optimizer


class KGFTLifecycleCallback(TrainerCallback):
    """Warm up KGFT contribution and save its non-PEFT state at checkpoints."""

    def __init__(self, *, config: dict[str, Any], warmup_steps: int) -> None:
        self.config = config
        self.warmup_steps = max(0, int(warmup_steps))

    def _scale(self, global_step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, (global_step + 1) / self.warmup_steps))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            set_runtime_scale(model, self._scale(int(state.global_step)))
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            set_runtime_scale(model, self._scale(int(state.global_step)))
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            save_kgft_artifacts(model, checkpoint_dir, self.config)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            set_runtime_scale(model, 1.0)
            save_kgft_artifacts(model, args.output_dir, self.config)
        return control


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="Local LLaMA-1 7B HF directory")
    p.add_argument("--train_json", required=True, help="Path to the training JSON file")
    p.add_argument("--output_dir", default="./outputs/llama1-7b-lora-kgft-math10k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_shuffle_seed", type=int, default=42)
    p.add_argument("--num_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--global_batch_size", type=int, default=16)
    p.add_argument("--cutoff_len", type=int, default=256)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=80)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument(
        "--torch_compile",
        action="store_true",
        help="Reserved for future torch.compile support",
    )
    p.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16"],
        default="auto",
        help="auto uses BF16 when supported by the current CUDA device",
    )
    p.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection",
    )

    p.add_argument("--kgft_layers", default="8,20,31", help="Zero-based decoder layer indices")
    p.add_argument("--kgft_modes", default="exploit,exploit,explore")
    p.add_argument("--kgft_init_strength", type=float, default=0.1)
    p.add_argument("--kgft_eps", type=float, default=1e-3)
    p.add_argument(
        "--kgft_kernel_source",
        choices=["base", "effective", "lora"],
        default="effective",
        help="effective = frozen base down_proj plus current LoRA delta",
    )
    p.add_argument(
        "--kgft_covariance_norm",
        choices=["sqrt_tokens", "tokens", "tokens_hidden"],
        default="tokens",
    )
    p.add_argument(
        "--kgft_output_norm",
        choices=["none", "match_rms"],
        default="match_rms",
        help="match_rms controls LLM-scale numerical growth while preserving KGFT direction",
    )
    p.add_argument("--kgft_strength_lr_factor", type=float, default=0.5)
    p.add_argument(
        "--kgft_strength_warmup_steps",
        type=int,
        default=-1,
        help="-1 uses --warmup_steps; 0 disables KGFT contribution warm-up",
    )
    return p.parse_args()


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_kgft_strength_grad_guards(model: nn.Module) -> None:
    """Fail immediately if a KGFT strength gradient becomes non-finite."""

    def make_hook(name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not torch.isfinite(grad).all():
                raise FloatingPointError(
                    f"Non-finite KGFT strength gradient detected in {name}: {grad}"
                )
            return grad
        return hook

    found = 0
    for name, param in model.named_parameters():
        if name.endswith("kgft.raw_strength") and param.requires_grad:
            param.register_hook(make_hook(name))
            found += 1
    if found == 0:
        raise RuntimeError("No trainable kgft.raw_strength parameter found for gradient guard")


def build_train_prompt(example: dict) -> str:
    instruction = str(example["instruction"])
    input_text = str(example.get("input", "") or "")
    output = str(example["output"])
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed_everywhere(args.seed)

    if args.global_batch_size % args.micro_batch_size != 0:
        raise ValueError("global_batch_size must be divisible by micro_batch_size")
    grad_accum = args.global_batch_size // args.micro_batch_size

    layer_indices = parse_int_csv(args.kgft_layers)
    modes = parse_str_csv(args.kgft_modes)
    if len(layer_indices) != len(modes):
        raise ValueError("--kgft_layers and --kgft_modes must contain the same number of entries")
    if any(mode not in {"exploit", "explore"} for mode in modes):
        raise ValueError("--kgft_modes entries must be exploit or explore")

    strength_warmup_steps = (
        args.warmup_steps
        if args.kgft_strength_warmup_steps < 0
        else args.kgft_strength_warmup_steps
    )
    kgft_config = build_kgft_config(
        layer_indices=layer_indices,
        modes=modes,
        init_strength=args.kgft_init_strength,
        eps=args.kgft_eps,
        kernel_source=args.kgft_kernel_source,
        covariance_norm=args.kgft_covariance_norm,
        output_norm=args.kgft_output_norm,
        strength_lr_factor=args.kgft_strength_lr_factor,
        strength_warmup_steps=strength_warmup_steps,
        vit_reference_layers=[3, 7, 11],
    )

    base_model = Path(args.base_model)
    train_json = Path(args.train_json)
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    if not train_json.exists():
        raise FileNotFoundError(train_json)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        use_fast=False,
        trust_remote_code=False,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")

    if args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--precision bf16 requested, but this GPU does not support BF16")
        use_bf16 = True
    elif args.precision == "fp16":
        use_bf16 = False
    else:
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Training precision: {'bf16' if use_bf16 else 'fp16'}")

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=model_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager",
        trust_remote_code=False,
    )

    for param in model.parameters():
        param.requires_grad = False
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Inject before PEFT to keep adapter key paths stable.
    wrappers = inject_kgft(model, kgft_config)
    print("KGFT insertion schedule:")
    for idx, wrapper in wrappers.items():
        print(
            f"  L{idx}: {wrapper.kgft.mode}, init_strength={args.kgft_init_strength:.4f}, "
            f"kernel={args.kgft_kernel_source}"
        )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    set_strength_trainable(model, True)
    install_kgft_strength_grad_guards(model)
    attach_attention_mask_hook(model)
    model.config.use_cache = False

    lora_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.endswith("kgft.raw_strength")
    )
    kgft_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and name.endswith("kgft.raw_strength")
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable params: {lora_trainable:,}")
    print(f"KGFT trainable params: {kgft_trainable:,}")
    print(f"Total trainable: {lora_trainable + kgft_trainable:,} / {total:,}")
    print(f"LoRA target modules: {TARGET_MODULES}")
    if not (50_000_000 <= lora_trainable <= 60_000_000):
        raise RuntimeError(
            "Unexpected LoRA parameter count. Check that this is LLaMA-1 7B "
            "and that all five target-module types were found."
        )
    if kgft_trainable != len(layer_indices):
        raise RuntimeError(
            f"Expected {len(layer_indices)} KGFT scalars, found {kgft_trainable} trainable values"
        )

    raw = load_dataset("json", data_files=str(train_json), split="train")
    required = {"instruction", "output"}
    missing = required - set(raw.column_names)
    if missing:
        raise ValueError(f"Training JSON is missing columns: {sorted(missing)}")
    raw = raw.shuffle(seed=args.dataset_shuffle_seed)

    cutoff_len = args.cutoff_len
    eos_id = tokenizer.eos_token_id

    def tokenize_example(example: dict) -> dict:
        prompt = build_train_prompt(example)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if result["input_ids"][-1] != eos_id and len(result["input_ids"]) < cutoff_len:
            result["input_ids"].append(eos_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = raw.map(
        tokenize_example,
        remove_columns=raw.column_names,
        desc="Tokenizing training data",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_kgft_artifacts(model, output_dir, kgft_config)

    if args.resume_from_checkpoint:
        resume_dir = Path(args.resume_from_checkpoint)
        if not resume_dir.exists():
            raise FileNotFoundError(resume_dir)
        load_kgft_state(model, resume_dir, strict=True)
        print(f"Loaded KGFT state from {resume_dir / KGFT_STATE_NAME}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.0,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        group_by_length=False,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=True,
        save_safetensors=True,
        logging_nan_inf_filter=False,
        max_grad_norm=1.0,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    callback = KGFTLifecycleCallback(
        config=kgft_config,
        warmup_steps=strength_warmup_steps,
    )
    trainer = KGFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[callback],
        kgft_strength_lr_factor=args.kgft_strength_lr_factor,
    )

    if args.torch_compile:
        raise RuntimeError(
            "--torch_compile is currently unsupported because KGFT hooks and callbacks "
            "can cause graph breaks."
        )
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    set_runtime_scale(model, 1.0)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    save_kgft_artifacts(model, output_dir, kgft_config)
    tokenizer.save_pretrained(str(output_dir / "tokenizer_snapshot"))

    recipe = {
        "base_model": str(base_model.resolve()),
        "train_json": str(train_json.resolve()),
        "seed": args.seed,
        "dataset_shuffle_seed": args.dataset_shuffle_seed,
        "target_modules": TARGET_MODULES,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "cutoff_len": args.cutoff_len,
        "warmup_steps": args.warmup_steps,
        "train_on_inputs": True,
        "dtype": "bf16" if use_bf16 else "fp16",
        "kgft": kgft_config,
        "kgft_learned_strength": strength_summary(model),
        "kgft_files": [KGFT_CONFIG_NAME, KGFT_STATE_NAME],
    }
    with (output_dir / "reproduction_recipe.json").open("w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

    print(f"Final learned KGFT strengths: {strength_summary(model)}")


if __name__ == "__main__":
    main()
