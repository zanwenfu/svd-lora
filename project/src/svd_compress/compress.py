"""Core SVD compression routines.

The code operates on a PEFT LoRA model in memory: it walks the model, finds every
`lora_A` / `lora_B` pair, and replaces them with lower-rank factors whose product
is the rank-k truncated SVD of the original product.

We preserve the LoRA `(alpha, r)` interface by absorbing the new rank into the
`scaling` attribute that PEFT uses internally, so downstream code (inference,
saving, merging) works unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AdapterSpec:
    """Per-module compression record."""

    name: str
    original_rank: int
    adaptive_rank: int
    retained_energy: float  # fraction in [0, 1]

    @property
    def param_ratio(self) -> float:
        return self.adaptive_rank / max(self.original_rank, 1)


@dataclass
class CompressionResult:
    """Aggregate result from a compression pass."""

    energy_threshold: float
    adapters: List[AdapterSpec]

    @property
    def average_rank(self) -> float:
        if not self.adapters:
            return 0.0
        return sum(a.adaptive_rank for a in self.adapters) / len(self.adapters)

    @property
    def parameter_ratio(self) -> float:
        """Sum(adaptive_rank) / sum(original_rank) across all adapters."""
        total_orig = sum(a.original_rank for a in self.adapters)
        total_new = sum(a.adaptive_rank for a in self.adapters)
        return total_new / total_orig if total_orig > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "energy_threshold": self.energy_threshold,
            "num_adapters": len(self.adapters),
            "average_rank": self.average_rank,
            "parameter_ratio": self.parameter_ratio,
            "adapters": [asdict(a) for a in self.adapters],
        }


def select_rank_by_energy(sigma: torch.Tensor, tau: float) -> Tuple[int, float]:
    """Return the smallest k such that cumulative sigma_i^2 / total >= tau.

    `sigma` is expected to be a 1-D tensor of non-negative singular values,
    sorted in descending order (as returned by :func:`torch.linalg.svd`).
    Returns the chosen rank and the actual retained energy at that rank.
    """
    if sigma.numel() == 0:
        return 0, 0.0
    energy = sigma.pow(2)
    total = energy.sum()
    if total <= 0:
        return 1, 0.0
    cumulative = torch.cumsum(energy, dim=0) / total
    # smallest k (1-indexed) with cumulative[k-1] >= tau
    mask = cumulative >= tau
    k = int(torch.nonzero(mask, as_tuple=False)[0].item()) + 1 if mask.any() else sigma.numel()
    return k, float(cumulative[k - 1].item())


def _iter_lora_modules(model: nn.Module):
    """Yield (qualified_name, lora_A_linear, lora_B_linear, scaling) tuples."""
    for name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue
        if not isinstance(lora_A, nn.ModuleDict) or not isinstance(lora_B, nn.ModuleDict):
            continue
        # PEFT stores per-adapter entries keyed by adapter name (default: "default")
        for adapter_key in lora_A.keys():
            a_lin = lora_A[adapter_key]
            b_lin = lora_B[adapter_key]
            if not (isinstance(a_lin, nn.Linear) and isinstance(b_lin, nn.Linear)):
                continue
            scaling_map = getattr(module, "scaling", None)
            if isinstance(scaling_map, dict):
                scaling = float(scaling_map.get(adapter_key, 1.0))
            else:
                scaling = 1.0
            yield f"{name}::{adapter_key}", module, adapter_key, a_lin, b_lin, scaling


def _replace_linear(
    old: nn.Linear,
    new_weight: torch.Tensor,
    new_bias: Optional[torch.Tensor] = None,
) -> nn.Linear:
    """Build a new nn.Linear with the given weight / bias, preserving dtype & device."""
    out_features, in_features = new_weight.shape
    has_bias = old.bias is not None if new_bias is None else True
    new = nn.Linear(in_features, out_features, bias=has_bias)
    new.to(dtype=old.weight.dtype, device=old.weight.device)
    with torch.no_grad():
        new.weight.copy_(new_weight)
        if has_bias and new_bias is not None:
            new.bias.copy_(new_bias)
        elif has_bias and old.bias is not None:
            new.bias.zero_()
    return new


def compress_peft_model(
    peft_model: nn.Module,
    energy_threshold: float = 0.9,
    adapter_name: str = "default",
    verbose: bool = True,
) -> CompressionResult:
    """Compress every LoRA adapter in `peft_model` in-place.

    For each adapter module with factors (A, B) and scaling s = alpha / r_max,
    we compute dW = s * B @ A, take its SVD, and pick k = smallest rank
    retaining at least `energy_threshold` of the Frobenius energy. We then
    replace lora_A / lora_B with new rank-k linears A', B' whose product is
    the truncated SVD of dW, and reset `scaling` to 1.0 so the effective
    update is preserved exactly.
    """
    records: List[AdapterSpec] = []

    for qname, parent, key, a_lin, b_lin, scaling in _iter_lora_modules(peft_model):
        if key != adapter_name:
            continue

        A = a_lin.weight.detach().to(torch.float32)  # (r, in)
        B = b_lin.weight.detach().to(torch.float32)  # (out, r)
        r_orig = A.shape[0]

        delta = scaling * (B @ A)  # (out, in), the effective LoRA update

        # Full SVD (economy size): U (out, r_orig), S (r_orig,), Vh (r_orig, in)
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        k, retained = select_rank_by_energy(S, energy_threshold)

        # New LoRA factors: A' = V_k^T (k, in), B' = U_k @ diag(sigma_1..k) (out, k)
        A_new = Vh[:k, :].contiguous()
        B_new = (U[:, :k] * S[:k].unsqueeze(0)).contiguous()

        # Replace factor linears
        parent.lora_A[key] = _replace_linear(a_lin, A_new)
        parent.lora_B[key] = _replace_linear(b_lin, B_new)
        if hasattr(parent, "r") and isinstance(parent.r, dict):
            parent.r[key] = k
        # Scaling has been absorbed into B_new; keep interface consistent.
        if hasattr(parent, "scaling") and isinstance(parent.scaling, dict):
            parent.scaling[key] = 1.0
        # Keep alpha consistent with the new r so (alpha / r) = 1.
        if hasattr(parent, "lora_alpha") and isinstance(parent.lora_alpha, dict):
            parent.lora_alpha[key] = k

        records.append(
            AdapterSpec(
                name=qname,
                original_rank=r_orig,
                adaptive_rank=k,
                retained_energy=retained,
            )
        )

        if verbose:
            print(
                f"[svd] {qname}: rank {r_orig} -> {k}  "
                f"(retained {retained * 100:.2f}% energy)"
            )

    result = CompressionResult(energy_threshold=energy_threshold, adapters=records)
    if verbose:
        print(
            f"[svd] done: {len(records)} adapters, avg rank {result.average_rank:.2f}, "
            f"param ratio {result.parameter_ratio * 100:.1f}%"
        )
    return result


def compress_and_save(
    peft_model: nn.Module,
    output_dir: str,
    energy_threshold: float = 0.9,
    tokenizer=None,
    adapter_name: str = "default",
    verbose: bool = True,
) -> CompressionResult:
    """Compress `peft_model` and write the compressed adapter + metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)
    result = compress_peft_model(
        peft_model,
        energy_threshold=energy_threshold,
        adapter_name=adapter_name,
        verbose=verbose,
    )
    peft_model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "svd_compression_report.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return result
