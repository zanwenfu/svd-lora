"""SVD-guided adaptive rank compression for LoRA adapters.

Implements the procedure from Section 4 of the project report:

1. Train a standard LoRA model at a sufficiently large rank r_max.
2. For each adapter module, compute the effective update
       dW* = (alpha / r_max) * B @ A
   and its SVD  dW* = U Sigma V^T.
3. Select the smallest k such that the cumulative squared-singular-value mass
   (Frobenius energy) reaches a fraction tau (e.g. 0.9).
4. Re-parameterise the adapter with new low-rank factors
       B' = U[:, :k] @ diag(sigma_1..k),   A' = V[:, :k]^T
   so that B' @ A' exactly equals the rank-k truncation of dW*.
5. Optionally fine-tune a few additional epochs in the reduced subspace.

The public entry points are :func:`compress_peft_model` and :func:`compress_and_save`.
"""

from .compress import (
    AdapterSpec,
    CompressionResult,
    compress_peft_model,
    compress_and_save,
    select_rank_by_energy,
)

__all__ = [
    "AdapterSpec",
    "CompressionResult",
    "compress_peft_model",
    "compress_and_save",
    "select_rank_by_energy",
]
