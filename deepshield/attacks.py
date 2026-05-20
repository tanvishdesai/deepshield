from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _attack_forward_logits(adapter, inputs):
    context = getattr(adapter, "attack_autograd_context", None)
    if callable(context):
        with context():
            return adapter.forward_attack_logits(inputs)
    return adapter.forward_attack_logits(inputs)


def _clone_inputs(inputs):
    if isinstance(inputs, dict):
        return {key: value.detach().clone() for key, value in inputs.items()}
    return inputs.detach().clone()


def _requires_grad(inputs):
    if isinstance(inputs, dict):
        for value in inputs.values():
            value.requires_grad_(True)
        return inputs
    inputs.requires_grad_(True)
    return inputs


def _zero_grad(inputs):
    if isinstance(inputs, dict):
        for value in inputs.values():
            if value.grad is not None:
                value.grad.zero_()
        return
    if inputs.grad is not None:
        inputs.grad.zero_()


def _apply_step(clean_inputs, adv_inputs, grads, *, eps: float, step_size: float, attack_keys: set[str] | None = None):
    if isinstance(adv_inputs, dict):
        updated = {}
        for key, value in adv_inputs.items():
            grad = grads[key]
            if attack_keys is not None and key not in attack_keys:
                updated[key] = value.detach()
                continue
            delta = (value + step_size * grad.sign()) - clean_inputs[key]
            delta = delta.clamp(-eps, eps)
            updated[key] = _clamp_tensor(clean_inputs[key] + delta, key)
        return updated
    delta = (adv_inputs + step_size * grads.sign()) - clean_inputs
    delta = delta.clamp(-eps, eps)
    return _clamp_tensor(clean_inputs + delta, None)


def _clamp_tensor(tensor: torch.Tensor, key: str | None) -> torch.Tensor:
    if key == "audio" or tensor.ndim <= 2:
        return tensor.clamp(-1.0, 1.0).detach()
    return tensor.clamp(0.0, 1.0).detach()


def _collect_grads(inputs):
    if isinstance(inputs, dict):
        return {key: value.grad.detach() for key, value in inputs.items()}
    return inputs.grad.detach()


def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, grad.ndim))
    denom = grad.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-12) if reduce_dims else grad.abs().mean().clamp_min(1e-12)
    return grad / denom


def evasion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels > 0.5
    if not torch.any(mask):
        return logits.sum() * 0.0
    target = torch.zeros_like(logits[mask])
    return F.binary_cross_entropy_with_logits(logits[mask], target)


def fgsm_attack(adapter, inputs, labels, *, eps: float, attack_keys: set[str] | None = None):
    adv_inputs = _requires_grad(_clone_inputs(inputs))
    logits = _attack_forward_logits(adapter, adv_inputs)
    loss = evasion_loss(logits, labels)
    loss.backward()
    grads = _collect_grads(adv_inputs)
    return _apply_step(_clone_inputs(inputs), adv_inputs, grads, eps=eps, step_size=eps, attack_keys=attack_keys)


def pgd_attack(
    adapter,
    inputs,
    labels,
    *,
    eps: float,
    steps: int,
    step_size: float | None = None,
    attack_keys: set[str] | None = None,
):
    clean_inputs = _clone_inputs(inputs)
    adv_inputs = _clone_inputs(inputs)
    step_size = step_size or eps / max(steps // 2, 1)
    for _ in range(steps):
        adv_inputs = _requires_grad(adv_inputs)
        logits = _attack_forward_logits(adapter, adv_inputs)
        loss = evasion_loss(logits, labels)
        loss.backward()
        grads = _collect_grads(adv_inputs)
        adv_inputs = _apply_step(clean_inputs, adv_inputs, grads, eps=eps, step_size=step_size, attack_keys=attack_keys)
    return adv_inputs


def bim_attack(adapter, inputs, labels, *, eps: float, steps: int, attack_keys: set[str] | None = None):
    return pgd_attack(adapter, inputs, labels, eps=eps, steps=steps, step_size=eps / max(steps, 1), attack_keys=attack_keys)


def mifgsm_attack(
    adapter,
    inputs,
    labels,
    *,
    eps: float,
    steps: int,
    decay: float = 1.0,
    step_size: float | None = None,
    attack_keys: set[str] | None = None,
):
    clean_inputs = _clone_inputs(inputs)
    adv_inputs = _clone_inputs(inputs)
    step_size = step_size or eps / max(steps, 1)
    momentum = {key: torch.zeros_like(value) for key, value in adv_inputs.items()} if isinstance(adv_inputs, dict) else torch.zeros_like(adv_inputs)

    for _ in range(steps):
        adv_inputs = _requires_grad(adv_inputs)
        logits = _attack_forward_logits(adapter, adv_inputs)
        loss = evasion_loss(logits, labels)
        loss.backward()
        grads = _collect_grads(adv_inputs)
        if isinstance(grads, dict):
            for key, grad in grads.items():
                momentum[key] = decay * momentum[key] + _normalize_grad(grad)
            adv_inputs = _apply_step(clean_inputs, adv_inputs, momentum, eps=eps, step_size=step_size, attack_keys=attack_keys)
        else:
            momentum = decay * momentum + _normalize_grad(grads)
            adv_inputs = _apply_step(clean_inputs, adv_inputs, momentum, eps=eps, step_size=step_size, attack_keys=attack_keys)
    return adv_inputs


@lru_cache(maxsize=32)
def _dct_matrix(size: int, device: str) -> torch.Tensor:
    import math

    matrix = [
        [
            (math.sqrt(1.0 / size) if i == 0 else math.sqrt(2.0 / size)) * math.cos((j + 0.5) * math.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def _dct2(x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    dct_h = _dct_matrix(height, str(x.device)).to(dtype=x.dtype)
    dct_w = _dct_matrix(width, str(x.device)).to(dtype=x.dtype)
    return torch.matmul(torch.matmul(dct_h, x), dct_w.t())


def _idct2(x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    dct_h = _dct_matrix(height, str(x.device)).to(dtype=x.dtype)
    dct_w = _dct_matrix(width, str(x.device)).to(dtype=x.dtype)
    return torch.matmul(torch.matmul(dct_h.t(), x), dct_w)


def frequency_targeted_pgd(adapter, inputs: torch.Tensor, labels: torch.Tensor, *, eps: float, steps: int, high_freq_ratio: float = 0.5):
    clean_inputs = inputs.detach().clone()
    adv_inputs = inputs.detach().clone()
    step_size = eps / max(steps // 2, 1)
    channels = adv_inputs.shape[1]
    height, width = adv_inputs.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=adv_inputs.device), torch.arange(width, device=adv_inputs.device), indexing="ij")
    mask = ((yy + xx) >= int((height + width) * (1 - high_freq_ratio))).float()

    for _ in range(steps):
        adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
        logits = _attack_forward_logits(adapter, adv_inputs)
        loss = evasion_loss(logits, labels)
        loss.backward()
        grad = adv_inputs.grad.detach()
        filtered_grad = []
        for channel_idx in range(channels):
            grad_freq = _dct2(grad[:, channel_idx])
            grad_freq = grad_freq * mask
            filtered_grad.append(_idct2(grad_freq))
        filtered_grad = torch.stack(filtered_grad, dim=1)
        delta = (adv_inputs + step_size * filtered_grad.sign()) - clean_inputs
        delta = delta.clamp(-eps, eps)
        adv_inputs = (clean_inputs + delta).clamp(0.0, 1.0)
    return adv_inputs.detach()


def patch_pgd_attack(adapter, inputs: torch.Tensor, labels: torch.Tensor, *, eps: float, steps: int, patch_ratio: float = 0.2):
    clean_inputs = inputs.detach().clone()
    adv_inputs = inputs.detach().clone()
    step_size = eps / max(steps // 2, 1)
    _, _, height, width = inputs.shape
    patch_h = max(int(height * patch_ratio), 1)
    patch_w = max(int(width * patch_ratio), 1)
    top = (height - patch_h) // 2
    left = (width - patch_w) // 2
    mask = torch.zeros_like(inputs)
    mask[:, :, top:top + patch_h, left:left + patch_w] = 1.0

    for _ in range(steps):
        adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
        logits = _attack_forward_logits(adapter, adv_inputs)
        loss = evasion_loss(logits, labels)
        loss.backward()
        grad = adv_inputs.grad.detach() * mask
        delta = (adv_inputs + step_size * grad.sign()) - clean_inputs
        delta = delta.clamp(-eps, eps) * mask
        adv_inputs = (clean_inputs + delta).clamp(0.0, 1.0)
    return adv_inputs.detach()


def sparse_frame_pgd(adapter, inputs: torch.Tensor, labels: torch.Tensor, *, eps: float, steps: int, frame_fraction: float = 0.2):
    clean_inputs = inputs.detach().clone()
    adv_inputs = inputs.detach().clone()
    step_size = eps / max(steps // 2, 1)
    num_frames = inputs.shape[2]
    keep = max(int(num_frames * frame_fraction), 1)
    frame_ids = torch.linspace(0, num_frames - 1, keep).round().long()
    mask = torch.zeros_like(inputs)
    mask[:, :, frame_ids] = 1.0

    for _ in range(steps):
        adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
        logits = _attack_forward_logits(adapter, adv_inputs)
        loss = evasion_loss(logits, labels)
        loss.backward()
        grad = adv_inputs.grad.detach() * mask
        delta = (adv_inputs + step_size * grad.sign()) - clean_inputs
        delta = delta.clamp(-eps, eps) * mask
        adv_inputs = (clean_inputs + delta).clamp(0.0, 1.0)
    return adv_inputs.detach()


def temporally_consistent_pgd(
    adapter,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float,
    steps: int,
    smoothness_weight: float = 0.2,
):
    clean_inputs = inputs.detach().clone()
    adv_inputs = inputs.detach().clone()
    step_size = eps / max(steps // 2, 1)

    for _ in range(steps):
        adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
        logits = _attack_forward_logits(adapter, adv_inputs)
        delta = adv_inputs - clean_inputs
        temporal_tv = torch.mean(torch.abs(delta[:, :, 1:] - delta[:, :, :-1]))
        loss = evasion_loss(logits, labels) + smoothness_weight * temporal_tv
        loss.backward()
        grad = adv_inputs.grad.detach()
        delta = (adv_inputs + step_size * grad.sign()) - clean_inputs
        delta = delta.clamp(-eps, eps)
        adv_inputs = (clean_inputs + delta).clamp(0.0, 1.0)
    return adv_inputs.detach()


def psychoacoustic_pgd(adapter, inputs: torch.Tensor, labels: torch.Tensor, *, eps: float, steps: int):
    clean_inputs = inputs.detach().clone()
    adv_inputs = inputs.detach().clone()
    step_size = eps / max(steps // 2, 1)

    for _ in range(steps):
        adv_inputs = adv_inputs.detach().clone().requires_grad_(True)
        logits = _attack_forward_logits(adapter, adv_inputs)
        loss = evasion_loss(logits, labels)
        loss.backward()
        grad = adv_inputs.grad.detach()
        spec = torch.stft(clean_inputs, n_fft=512, hop_length=160, return_complex=True)
        mask = spec.abs() / (spec.abs().amax(dim=(-2, -1), keepdim=True) + 1e-6)
        grad_spec = torch.stft(grad, n_fft=512, hop_length=160, return_complex=True)
        weighted = grad_spec * mask
        grad = torch.istft(weighted, n_fft=512, hop_length=160, length=clean_inputs.shape[-1])
        delta = (adv_inputs + step_size * grad.sign()) - clean_inputs
        delta = delta.clamp(-eps, eps)
        adv_inputs = (clean_inputs + delta).clamp(-1.0, 1.0)
    return adv_inputs.detach()


def multimodal_asymmetric_pgd(adapter, inputs: dict[str, torch.Tensor], labels: torch.Tensor, *, eps: float, steps: int, mode: str):
    if mode == "audio_only":
        return pgd_attack(adapter, inputs, labels, eps=eps, steps=steps, attack_keys={"audio"})
    if mode == "video_only":
        return pgd_attack(adapter, inputs, labels, eps=eps, steps=steps, attack_keys={"video"})
    if mode == "both":
        return pgd_attack(adapter, inputs, labels, eps=eps, steps=steps)
    raise ValueError(f"Unknown asymmetric mode: {mode}")


class _TwoClassImageWrapper(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        # torchattacks infers device from model parameters during construction.
        # The adapter itself is not an nn.Module, so expose a tiny parameter on
        # the adapter device to avoid repeated "set_device() manual" warnings.
        self._attack_device_ref = nn.Parameter(torch.empty(0, device=adapter.device), requires_grad=False)

    def forward(self, x):
        fake_logit = _attack_forward_logits(self.adapter, x)
        return torch.stack([-fake_logit, fake_logit], dim=1)


def cw_image_attack(adapter, inputs: torch.Tensor, labels: torch.Tensor, *, c: float = 1.0, steps: int = 100):
    try:
        import torchattacks
    except Exception as exc:
        raise RuntimeError("torchattacks is required for the C&W attack.") from exc
    attack = torchattacks.CW(_TwoClassImageWrapper(adapter), c=c, steps=steps)
    if hasattr(attack, "set_device"):
        attack.set_device(inputs.device)
    fake_mask = labels > 0.5
    if not torch.any(fake_mask):
        return inputs
    attacked = inputs.detach().clone()
    target_labels = torch.zeros_like(labels[fake_mask], dtype=torch.long, device=inputs.device)
    attacked[fake_mask] = attack(inputs[fake_mask], target_labels)
    return attacked.detach()


def autoattack_image(adapter, inputs: torch.Tensor, labels: torch.Tensor, *, eps: float):
    try:
        from autoattack import AutoAttack
    except Exception as exc:
        raise RuntimeError("autoattack is required for the AutoAttack runner.") from exc
    fake_mask = labels > 0.5
    if not torch.any(fake_mask):
        return inputs
    wrapper = _TwoClassImageWrapper(adapter)
    adversary = AutoAttack(wrapper, norm="Linf", eps=eps, version="standard")
    attacked = inputs.detach().clone()
    attacked[fake_mask] = adversary.run_standard_evaluation(inputs[fake_mask], torch.ones(fake_mask.sum(), dtype=torch.long, device=inputs.device), bs=min(8, fake_mask.sum().item()))
    return attacked.detach()


ATTACKS: dict[str, Any] = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
    "bim": bim_attack,
    "mifgsm": mifgsm_attack,
    "frequency_targeted_pgd": frequency_targeted_pgd,
    "patch_pgd": patch_pgd_attack,
    "sparse_frame_pgd": sparse_frame_pgd,
    "temporally_consistent_pgd": temporally_consistent_pgd,
    "psychoacoustic_pgd": psychoacoustic_pgd,
    "cw": cw_image_attack,
    "autoattack": autoattack_image,
}
