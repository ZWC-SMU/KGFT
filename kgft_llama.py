from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

KGFT_CONFIG_NAME = "kgft_config.json"
KGFT_STATE_NAME = "kgft_state.pt"


def _base_linear(module: nn.Module) -> nn.Module:
    """Return the underlying nn.Linear from a PEFT LoRA layer or plain linear."""
    getter = getattr(module, "get_base_layer", None)
    if callable(getter):
        base = getter()
    else:
        base = getattr(module, "base_layer", module)
    if not hasattr(base, "weight"):
        raise TypeError(f"Expected a linear-like module with .weight, got {type(module)!r}")
    return base


def _active_adapter_names(module: nn.Module) -> list[str]:
    if bool(getattr(module, "disable_adapters", False)):
        return []
    if bool(getattr(module, "merged", False)):
        return []

    names = getattr(module, "active_adapters", None)
    if names is None:
        names = getattr(module, "active_adapter", None)
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    else:
        names = list(names)

    lora_a = getattr(module, "lora_A", {})
    lora_b = getattr(module, "lora_B", {})
    return [name for name in names if name in lora_a and name in lora_b]


def _validate_no_dora(module: nn.Module, adapter_names: Iterable[str]) -> None:
    magnitude = getattr(module, "lora_magnitude_vector", None)
    if magnitude is None:
        return
    for name in adapter_names:
        try:
            present = name in magnitude
        except TypeError:
            present = False
        if present:
            raise NotImplementedError(
                "KGFT's implicit effective-weight operator currently supports standard LoRA, "
                "not DoRA. Set use_dora=False in LoraConfig."
            )


def _lora_factors(module: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Return active (A, B, scaling) factors where delta_W = scaling * B @ A."""
    names = _active_adapter_names(module)
    _validate_no_dora(module, names)
    factors: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for name in names:
        a = module.lora_A[name].weight
        b = module.lora_B[name].weight
        scale = module.scaling[name]
        if isinstance(scale, torch.Tensor):
            scale = float(scale.detach().cpu())
        factors.append((a, b, float(scale)))
    return factors


def apply_weight(
    module: nn.Module,
    x: torch.Tensor,
    *,
    transpose: bool,
    source: str,
) -> torch.Tensor:
    """Apply W or W^T without materializing the effective LoRA weight.

    For a down projection with W shaped (hidden, intermediate):
      transpose=False: input (..., intermediate) -> (..., hidden), x @ W^T
      transpose=True:  input (..., hidden)       -> (..., intermediate), x @ W
    """
    if source not in {"base", "effective", "lora"}:
        raise ValueError(f"Unknown kernel source: {source}")

    base = _base_linear(module)
    factors = _lora_factors(module) if source in {"effective", "lora"} else []

    out: torch.Tensor | None = None
    if source in {"base", "effective"}:
        weight = base.weight.transpose(0, 1) if transpose else base.weight
        out = F.linear(x, weight, bias=None)

    for a, b, scale in factors:
        if transpose:
            # x @ (B A) = (x @ B) @ A
            delta_input = x.to(dtype=b.dtype)
            hidden = F.linear(delta_input, b.transpose(0, 1), bias=None)
            hidden = hidden.to(dtype=a.dtype)
            delta = F.linear(hidden, a.transpose(0, 1), bias=None)
        else:
            delta_input = x.to(dtype=a.dtype)
            hidden = F.linear(delta_input, a, bias=None)
            hidden = hidden.to(dtype=b.dtype)
            delta = F.linear(hidden, b, bias=None)
        target_dtype = out.dtype if out is not None else x.dtype
        delta = (delta * scale).to(dtype=target_dtype)
        out = delta if out is None else out + delta

    if out is None:
        output_dim = base.weight.shape[1] if transpose else base.weight.shape[0]
        out = x.new_zeros(*x.shape[:-1], output_dim)
    return out


def effective_row_norm_sq(
    module: nn.Module,
    *,
    source: str,
    base_row_norm_sq: torch.Tensor,
) -> torch.Tensor:
    """Compute diag(W W^T) for base/effective/LoRA W without forming W."""
    if source == "base":
        return base_row_norm_sq

    factors = _lora_factors(module)
    if not factors:
        if source == "effective":
            return base_row_norm_sq
        return torch.zeros_like(base_row_norm_sq)

    a_cat = torch.cat([a for a, _, _ in factors], dim=0)
    b_cat = torch.cat([b * scale for _, b, scale in factors], dim=1)

    a32 = a_cat.float()
    b32 = b_cat.float()
    aa_t = a32 @ a32.transpose(0, 1)
    delta_norm = ((b32 @ aa_t) * b32).sum(dim=1)

    if source == "lora":
        return delta_norm.to(base_row_norm_sq.device)

    base = _base_linear(module)
    # Row-wise <W0, delta_W> without materializing delta_W.
    a_for_base = a_cat.to(dtype=base.weight.dtype)
    base_a_t = F.linear(base.weight, a_for_base, bias=None).float()
    cross = 2.0 * (base_a_t * b32).sum(dim=1)
    result = base_row_norm_sq.float() + cross + delta_norm
    return result.clamp_min(0.0).to(base_row_norm_sq.device)


class KernelGuidedFeatureTransform(nn.Module):
    """KGFT applied to the output of a LLaMA MLP block."""

    def __init__(
        self,
        *,
        mode: str,
        init_strength: float = 0.1,
        eps: float = 1e-3,
        kernel_source: str = "effective",
        covariance_norm: str = "tokens",
        output_norm: str = "match_rms",
        causal: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"exploit", "explore"}:
            raise ValueError(f"mode must be exploit/explore, got {mode!r}")
        if kernel_source not in {"base", "effective", "lora"}:
            raise ValueError(f"kernel_source must be base/effective/lora, got {kernel_source!r}")
        if covariance_norm not in {"sqrt_tokens", "tokens", "tokens_hidden"}:
            raise ValueError(f"Unsupported covariance_norm: {covariance_norm}")
        if output_norm not in {"none", "match_rms"}:
            raise ValueError(f"Unsupported output_norm: {output_norm}")

        self.mode = mode
        self.eps = float(eps)
        self.kernel_source = kernel_source
        self.covariance_norm = covariance_norm
        self.output_norm = output_norm
        self.causal = bool(causal)

        init_strength = min(max(float(init_strength), 1e-7), 1.0 - 1e-7)
        raw = torch.logit(torch.tensor(init_strength, dtype=torch.float32))
        self.raw_strength = nn.Parameter(raw)
        self.register_buffer("runtime_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    def strength_tensor(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_strength) * self.runtime_scale

    def set_runtime_scale(self, value: float) -> None:
        self.runtime_scale.fill_(min(max(float(value), 0.0), 1.0))

    def _mask_for_x(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if attention_mask is None:
            return x.new_ones((bsz, seq_len, 1))
        mask = attention_mask
        if mask.ndim != 2:
            return x.new_ones((bsz, seq_len, 1))
        if mask.shape[0] != bsz:
            return x.new_ones((bsz, seq_len, 1))
        if mask.shape[1] < seq_len:
            return x.new_ones((bsz, seq_len, 1))
        mask = mask[:, -seq_len:]
        return mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

    @staticmethod
    def _raise_if_nonfinite(name: str, tensor: torch.Tensor) -> None:
        """Raise at the first non-finite KGFT forward tensor."""
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return
        finite_values = tensor.detach()[finite_mask]
        if finite_values.numel() > 0:
            finite_min = float(finite_values.min().cpu())
            finite_max = float(finite_values.max().cpu())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
        raise FloatingPointError(
            f"{name} contains NaN/Inf: dtype={tensor.dtype}, "
            f"shape={tuple(tensor.shape)}, finite_min={finite_min}, "
            f"finite_max={finite_max}"
        )

    def _projected_gram_affinity(
        self,
        x32: torch.Tensor,
        down_proj: nn.Module,
        *,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Compute (X W)(X W)^T exactly in FP32 without forming W W^T."""
        base = _base_linear(down_proj)
        intermediate_size = int(base.weight.shape[1])

        factors = (
            _lora_factors(down_proj)
            if self.kernel_source in {"effective", "lora"}
            else []
        )

        projected_lora_inputs: list[
            tuple[torch.Tensor, torch.Tensor, float]
        ] = []

        with torch.autocast(device_type=x32.device.type, enabled=False):
            for a, b, scale in factors:
                xb = F.linear(
                    x32,
                    b.float().transpose(0, 1),
                    bias=None,
                )
                projected_lora_inputs.append((xb, a, scale))

            affinity: torch.Tensor | None = None

            for begin in range(0, intermediate_size, chunk_size):
                finish = min(begin + chunk_size, intermediate_size)
                projected: torch.Tensor | None = None

                if self.kernel_source in {"base", "effective"}:
                    weight_chunk = (
                        base.weight[:, begin:finish]
                        .transpose(0, 1)
                        .float()
                    )
                    projected = F.linear(x32, weight_chunk, bias=None)

                if self.kernel_source in {"effective", "lora"}:
                    delta_projected: torch.Tensor | None = None
                    for xb, a, scale in projected_lora_inputs:
                        a_chunk = (
                            a[:, begin:finish]
                            .transpose(0, 1)
                            .float()
                        )
                        part = F.linear(xb, a_chunk, bias=None) * float(scale)
                        delta_projected = (
                            part
                            if delta_projected is None
                            else delta_projected + part
                        )

                    if delta_projected is None:
                        delta_projected = x32.new_zeros(
                            *x32.shape[:-1],
                            finish - begin,
                        )

                    projected = (
                        delta_projected
                        if projected is None
                        else projected + delta_projected
                    )

                if projected is None:
                    raise RuntimeError("No kernel projection was constructed")

                chunk_affinity = torch.bmm(
                    projected,
                    projected.transpose(1, 2),
                )
                affinity = (
                    chunk_affinity
                    if affinity is None
                    else affinity + chunk_affinity
                )

        if affinity is None:
            raise RuntimeError("Projected Gram affinity is empty")
        return affinity

    def forward(
        self,
        x: torch.Tensor,
        down_proj: nn.Module,
        base_row_norm_sq: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"KGFT expects (B, N, H), got shape {tuple(x.shape)}")

        mask = self._mask_for_x(x, attention_mask)
        mask32 = mask.float()
        x32 = x.float() * mask32

        with torch.autocast(device_type=x.device.type, enabled=False):
            xx_affinity = torch.bmm(x32, x32.transpose(1, 2))

            if self.mode == "exploit":
                kernel_affinity = self._projected_gram_affinity(
                    x32,
                    down_proj,
                )
                affinity = kernel_affinity + self.eps * xx_affinity
            else:
                row_norm_sq = effective_row_norm_sq(
                    down_proj,
                    source=self.kernel_source,
                    base_row_norm_sq=base_row_norm_sq,
                ).to(device=x.device, dtype=torch.float32)

                inv_row_norm = torch.rsqrt(
                    row_norm_sq.clamp_min(1e-8)
                ).view(1, 1, -1)

                normalized_kernel_affinity = self._projected_gram_affinity(
                    x32 * inv_row_norm,
                    down_proj,
                )
                mean_stretch = row_norm_sq.mean()
                affinity = (
                    mean_stretch
                    * (xx_affinity - normalized_kernel_affinity)
                    + self.eps * xx_affinity
                )

            token_mask = mask32.squeeze(-1)
            affinity = affinity * token_mask.unsqueeze(1)

            if self.causal:
                seq_len = x.shape[1]
                causal_mask = torch.ones(
                    (seq_len, seq_len),
                    device=x.device,
                    dtype=torch.float32,
                ).tril_()
                affinity = affinity * causal_mask.unsqueeze(0)
                valid_tokens = (
                    token_mask.cumsum(dim=1)
                    .unsqueeze(-1)
                    .clamp_min(1.0)
                )
            else:
                valid_tokens = (
                    token_mask.sum(dim=1, keepdim=True)
                    .unsqueeze(-1)
                    .clamp_min(1.0)
                )

            y = torch.bmm(affinity, x32)

            if self.covariance_norm == "sqrt_tokens":
                denom = valid_tokens.sqrt()
            elif self.covariance_norm == "tokens":
                denom = valid_tokens
            else:
                denom = valid_tokens * float(x.shape[-1])
            y = y / denom

            if self.output_norm == "match_rms":
                target_rms = (
                    x32.square()
                    .mean(dim=-1, keepdim=True)
                    .add(1e-6)
                    .sqrt()
                )
                y_rms = (
                    y.square()
                    .mean(dim=-1, keepdim=True)
                    .add(1e-6)
                    .sqrt()
                )
                rms_ratio = (
                    target_rms / y_rms.clamp_min(1e-3)
                ).clamp(max=10.0).detach()
                y = y * rms_ratio

            update32 = y * mask32
            strength32 = self.strength_tensor().to(
                device=x.device,
                dtype=torch.float32,
            )
            output32 = x.float() + strength32 * update32

        checks_left = int(getattr(self, "_finite_checks_left", 4))
        if checks_left > 0:
            self._raise_if_nonfinite("KGFT input", x32)
            self._raise_if_nonfinite("KGFT affinity", affinity)
            self._raise_if_nonfinite("KGFT update", update32)
            self._raise_if_nonfinite("KGFT FP32 output", output32)
            self._finite_checks_left = checks_left - 1

        output = output32.to(dtype=x.dtype)
        if checks_left > 0:
            self._raise_if_nonfinite("KGFT cast output", output)
        return output


class KGFTMLPWrapper(nn.Module):
    """Wrap a LLaMA MLP and apply KGFT immediately after its output."""

    def __init__(
        self,
        base_mlp: nn.Module,
        *,
        layer_idx: int,
        mode: str,
        init_strength: float,
        eps: float,
        kernel_source: str,
        covariance_norm: str,
        output_norm: str,
        causal: bool,
    ) -> None:
        super().__init__()
        if not hasattr(base_mlp, "down_proj"):
            raise TypeError(f"Layer {layer_idx} MLP has no down_proj; got {type(base_mlp)!r}")
        self.base_mlp = base_mlp
        self.layer_idx = int(layer_idx)
        self.kgft = KernelGuidedFeatureTransform(
            mode=mode,
            init_strength=init_strength,
            eps=eps,
            kernel_source=kernel_source,
            covariance_norm=covariance_norm,
            output_norm=output_norm,
            causal=causal,
        )
        base_weight = _base_linear(base_mlp.down_proj).weight.detach()
        self.kgft.to(device=base_weight.device)
        base_row_norm_sq = base_weight.float().square().sum(dim=1)
        self.register_buffer("base_row_norm_sq", base_row_norm_sq, persistent=False)
        self._attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self._attention_mask = attention_mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mlp_output = self.base_mlp(hidden_states)
        return self.kgft(
            mlp_output,
            self.base_mlp.down_proj,
            self.base_row_norm_sq,
            self._attention_mask,
        )


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        getattr(base, "model", None),
        getattr(getattr(base, "base_model", None), "model", None),
        base,
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            layers = candidate.layers
            if isinstance(layers, (nn.ModuleList, list, tuple)):
                return layers
    raise AttributeError("Could not locate LLaMA decoder layers (expected model.model.layers)")


def build_kgft_config(
    *,
    layer_indices: list[int],
    modes: list[str],
    init_strength: float,
    eps: float,
    kernel_source: str,
    covariance_norm: str,
    output_norm: str,
    strength_lr_factor: float,
    strength_warmup_steps: int,
    vit_reference_layers: list[int] | None = None,
    causal: bool = True,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "architecture": "llama_mlp_output_kgft",
        "layer_indices_zero_based": layer_indices,
        "modes": modes,
        "init_strength": float(init_strength),
        "eps": float(eps),
        "kernel_source": kernel_source,
        "covariance_norm": covariance_norm,
        "output_norm": output_norm,
        "causal": bool(causal),
        "strength_lr_factor": float(strength_lr_factor),
        "strength_warmup_steps": int(strength_warmup_steps),
        "vit_reference_layers_zero_based": vit_reference_layers or [3, 7, 11],
    }


def inject_kgft(model: nn.Module, config: dict[str, Any]) -> dict[int, KGFTMLPWrapper]:
    layers = get_decoder_layers(model)
    indices = [int(x) for x in config["layer_indices_zero_based"]]
    modes = [str(x) for x in config["modes"]]
    if len(indices) != len(modes):
        raise ValueError("KGFT layer_indices and modes must have the same length")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate KGFT layer index: {indices}")

    wrappers: dict[int, KGFTMLPWrapper] = {}
    for idx, mode in zip(indices, modes):
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"KGFT layer {idx} is outside [0, {len(layers) - 1}]")
        existing = layers[idx].mlp
        if isinstance(existing, KGFTMLPWrapper):
            wrappers[idx] = existing
            continue
        wrapper = KGFTMLPWrapper(
            existing,
            layer_idx=idx,
            mode=mode,
            init_strength=float(config.get("init_strength", 0.1)),
            eps=float(config.get("eps", 1e-3)),
            kernel_source=str(config.get("kernel_source", "effective")),
            covariance_norm=str(config.get("covariance_norm", "tokens")),
            output_norm=str(config.get("output_norm", "match_rms")),
            causal=bool(config.get("causal", True)),
        )
        layers[idx].mlp = wrapper
        wrappers[idx] = wrapper
    return wrappers


def kgft_wrappers(model: nn.Module) -> dict[int, KGFTMLPWrapper]:
    wrappers: dict[int, KGFTMLPWrapper] = {}
    for idx, layer in enumerate(get_decoder_layers(model)):
        if isinstance(layer.mlp, KGFTMLPWrapper):
            wrappers[idx] = layer.mlp
    return wrappers


def attach_attention_mask_hook(model: nn.Module) -> None:
    old_handles = getattr(model, "_kgft_attention_mask_hook_handles", [])
    for old_handle in old_handles:
        try:
            old_handle.remove()
        except Exception:
            pass

    def pre_hook(_module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        mask = kwargs.get("attention_mask")
        for wrapper in kgft_wrappers(model).values():
            wrapper.set_attention_mask(mask)

    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    setattr(model, "_kgft_attention_mask_hook_handles", [handle])


def set_strength_trainable(model: nn.Module, trainable: bool = True) -> None:
    for wrapper in kgft_wrappers(model).values():
        wrapper.kgft.raw_strength.requires_grad_(trainable)


def set_runtime_scale(model: nn.Module, value: float) -> None:
    for wrapper in kgft_wrappers(model).values():
        wrapper.kgft.set_runtime_scale(value)


def strength_summary(model: nn.Module) -> dict[str, float]:
    return {
        str(idx): float(torch.sigmoid(wrapper.kgft.raw_strength.detach()).cpu())
        for idx, wrapper in kgft_wrappers(model).items()
    }


def save_kgft_artifacts(model: nn.Module, directory: str | Path, config: dict[str, Any]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "raw_strength": {
            str(idx): wrapper.kgft.raw_strength.detach().float().cpu()
            for idx, wrapper in kgft_wrappers(model).items()
        },
        "learned_strength": strength_summary(model),
    }
    torch.save(state, directory / KGFT_STATE_NAME)
    config_to_write = dict(config)
    config_to_write["learned_strength"] = state["learned_strength"]
    with (directory / KGFT_CONFIG_NAME).open("w", encoding="utf-8") as f:
        json.dump(config_to_write, f, ensure_ascii=False, indent=2)


def load_kgft_config(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / KGFT_CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing KGFT config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_kgft_state(model: nn.Module, directory: str | Path, *, strict: bool = True) -> None:
    path = Path(directory) / KGFT_STATE_NAME
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Missing KGFT state: {path}")
        return
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    raw_state = state.get("raw_strength", {})
    wrappers = kgft_wrappers(model)
    missing: list[int] = []
    for idx, wrapper in wrappers.items():
        key = str(idx)
        if key not in raw_state:
            missing.append(idx)
            continue
        value = raw_state[key].to(
            device=wrapper.kgft.raw_strength.device,
            dtype=wrapper.kgft.raw_strength.dtype,
        )
        wrapper.kgft.raw_strength.data.copy_(value)
    if strict and missing:
        raise KeyError(f"KGFT state is missing layers: {missing}")


def parse_int_csv(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("Expected at least one comma-separated integer")
    return result


def parse_str_csv(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("Expected at least one comma-separated value")
    return result


def map_vit_layers_to_target(vit_layers: list[int], *, vit_depth: int, target_depth: int) -> list[int]:
    if vit_depth < 2 or target_depth < 2:
        raise ValueError("Both depths must be at least 2")
    return [round(i * (target_depth - 1) / (vit_depth - 1)) for i in vit_layers]
