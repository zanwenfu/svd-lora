# HiRA, Sparse LoRA — Pushing the Efficiency Frontier of LoRA

**SVD-Guided Adaptive Rank Compression + A Controlled Study of LoRA, Sparse LoRA, and HiRA**

Report: [report/ECE685D_Project_Report.pdf](report/ECE685D_Project_Report.pdf)

This document is the technical deep-dive. For the one-minute pitch and quick-start, see the [top-level README](../README.md).

---

## Contents

- [1. Motivation](#1-motivation)
- [2. Methods](#2-methods)
  - [2.1 LoRA](#21-lora)
  - [2.2 Sparse LoRA (our simplified variant)](#22-sparse-lora-our-simplified-variant)
  - [2.3 HiRA](#23-hira)
  - [2.4 SVD-Guided Adaptive Rank Compression (ours)](#24-svd-guided-adaptive-rank-compression-ours)
- [3. Experimental setup](#3-experimental-setup)
- [4. Results](#4-results)
- [5. Reproducing everything](#5-reproducing-everything)
- [6. Repository layout](#6-repository-layout)
- [7. Extending this codebase](#7-extending-this-codebase)

---

## 1. Motivation

LoRA and its descendants dominate the PEFT landscape. But three questions remain practically important and surprisingly underexplored in a controlled setting:

1. **Which LoRA variant wins under a matched budget?** Published comparisons often differ in backbones, sequence lengths, optimizers, or seeds. We hold all of those fixed and vary only the adaptation mechanism.
2. **How much of LoRA's rank is actually needed?** The global rank `r` is a blunt instrument — some modules need more capacity, others need less.
3. **Can we compress a trained LoRA adapter post-hoc, without retraining from scratch, without hurting accuracy?**

The answers we arrived at are: (1) standard LoRA is the most robust, (2) for DistilBERT on sentiment tasks, an average effective rank of **≈ 2.4** captures ≥ 90% of the signal, and (3) yes — cleanly, via SVD.

## 2. Methods

### 2.1 LoRA

Standard low-rank adaptation. Given a frozen pretrained weight `W₀ ∈ R^(d_out × d_in)`, LoRA parameterizes the update as

```
ΔW = (α / r) · B A,   A ∈ R^(r × d_in),   B ∈ R^(d_out × r)
```

We train only `A`, `B`, and the task head. Implemented via HuggingFace PEFT.

- Entry point: [src/train_lora.py](src/train_lora.py)
- Script: [scripts/train_lora.sh](scripts/train_lora.sh)

### 2.2 Sparse LoRA (our simplified variant)

Our Sparse LoRA keeps the LoRA factorization but adds **L1 regularization** on the adapter parameters and optionally applies **magnitude pruning** after training. This does not reproduce every detail of the official SparseLoRA work (contextual/structured sparsity, custom CUDA kernels) — we intentionally implement a lightweight variant that slots into the standard HuggingFace LoRA pipeline so that accuracy comparisons are apples-to-apples.

Consequence: parameter-count reductions are real, but we do **not** obtain wall-clock inference speedups on modern GPUs without sparse-aware kernels. This is discussed at length in Appendix A of the report.

- Entry point: [src/sparse_lora/train_distilbert_sparse_lora.py](src/sparse_lora/train_distilbert_sparse_lora.py)
- MLM variant: [src/sparse_lora/train_distilbert_sparse_lora_wikitext2.py](src/sparse_lora/train_distilbert_sparse_lora_wikitext2.py)
- Evaluation & plots: [src/sparse_lora/analyze_sparse_lora_results.py](src/sparse_lora/analyze_sparse_lora_results.py)

### 2.3 HiRA

HiRA (Hadamard high-Rank Adaptation, Huang et al. 2025) replaces the additive low-rank update with a *multiplicative* modulation of `W₀`:

```
ΔW = W₀ ⊙ (L₁ L₂)
```

`L₁ L₂` is itself low-rank, but the element-wise interaction with `W₀` induces updates of a much higher effective rank under the same parameter budget. We use the authors' reference implementation, vendored as [src/hira/](src/hira/).

- Entry point: [src/train_hira.py](src/train_hira.py)
- Script: [scripts/train_hira.sh](scripts/train_hira.sh)

### 2.4 SVD-Guided Adaptive Rank Compression (ours)

The novel contribution. Three stages:

**Stage I — high-rank warm-up.** Train a standard LoRA model at `r_max = 8` for `T₁ = 3` epochs. Yields learned updates
`ΔW⋆_ℓ = (α / r_max) · B_ℓ A_ℓ` for each adapter module `ℓ`.

**Stage II — per-module SVD rank selection.** For each module, compute the SVD

```
ΔW⋆_ℓ = U_ℓ Σ_ℓ V_ℓ^⊤
```

and pick the smallest `k_ℓ` such that the cumulative squared-singular-value mass
`Σ_{i≤k} σ²_{ℓ,i} / Σ_i σ²_{ℓ,i}` reaches a threshold `τ` (we use 0.9). Rebuild the adapter factors as

```
B'_ℓ = U_{ℓ,k} · diag(σ_{ℓ,1..k}),   A'_ℓ = V_{ℓ,k}^⊤
```

so that `B'_ℓ A'_ℓ` is *exactly* the rank-`k_ℓ` Eckart-Young-Mirsky optimal truncation of `ΔW⋆_ℓ`. We absorb the old scaling into `B'`, set PEFT's `scaling = 1.0`, and update `r` per module — so downstream PEFT machinery (saving, loading, merging, inference) keeps working.

**Stage III — post-compression fine-tuning.** Resume training for `T₂` (typically 2) additional epochs in the reduced-rank subspace. This absorbs the (small) approximation error and, empirically, frequently *improves* accuracy on IMDB — removing low-variance directions seems to regularize.

**Core API** (30 lines of code, zero non-torch dependencies):

```python
from peft import AutoPeftModelForSequenceClassification
from svd_compress import compress_peft_model, compress_and_save

model = AutoPeftModelForSequenceClassification.from_pretrained("results/lora_sst2/final_model")

# In-memory compression — the model is modified in place.
result = compress_peft_model(model, energy_threshold=0.9)
print(result.to_dict())
# {'energy_threshold': 0.9, 'num_adapters': 12,
#  'average_rank': 2.42, 'parameter_ratio': 0.302, 'adapters': [...]}

# Or: compress + save + emit a JSON report in one call.
compress_and_save(model, "results/lora_sst2_svd", energy_threshold=0.9)
```

Everything is in [src/svd_compress/](src/svd_compress/):

- [compress.py](src/svd_compress/compress.py): core routines (`select_rank_by_energy`, `compress_peft_model`, `compress_and_save`)
- [compress_lora.py](src/svd_compress/compress_lora.py): CLI wrapper with optional evaluation + Stage III
- CLI: `./scripts/compress_svd.sh {sst2|imdb} [tau] [post_epochs]`

## 3. Experimental setup

Kept deliberately boring so that performance differences can be attributed to the adaptation mechanism, not to the training recipe.

| Hyperparameter | Value |
| --- | --- |
| Base model | `distilbert-base-uncased` |
| Datasets | SST-2, IMDB, WikiText-2 |
| Adapter target modules | attention `q_lin`, `v_lin` |
| LoRA rank `r` | 8 |
| LoRA `α` | 16 |
| LoRA dropout | 0.1 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Max sequence length | 128 |
| Seed | 42 |
| Optimizer | AdamW (default `Trainer`) |
| Weight decay | 0.01 |

Every method uses the same backbone, same tokenizer, same splits, same collator, same seed. IMDB has no validation split, so we deterministically carve 10% off the training set.

## 4. Results

All numbers are reproduced from JSON outputs in [results/](results/) and tables / figures in the report.

### 4.1 SVD-compressed LoRA vs. standard LoRA

| Dataset | Method | Accuracy | F1 | Adapter params vs. LoRA |
| --- | --- | --- | --- | --- |
| SST-2 | LoRA  | 0.8922 | 0.8967 | 100% |
| SST-2 | **SVD-compressed (τ=0.9, Stage III 2 epochs)** | **0.8933** | 0.8961 | **~30%** |
| IMDB | LoRA | 0.8681 | 0.8715 | 100% |
| IMDB | **SVD-compressed (τ=0.9, Stage III 2 epochs)** | **0.8812** | **0.8924** | **~30%** |

Per-module ranks selected on SST-2 (all 12 attention projections):

| Module | Original r | Adaptive k | Energy retained |
| --- | ---: | ---: | ---: |
| layer 1 `q_lin` | 8 | 2 | 92.0% |
| layer 1 `v_lin` | 8 | 2 | 94.0% |
| layer 2 `q_lin` | 8 | 3 | 94.9% |
| layer 2 `v_lin` | 8 | 2 | 90.3% |
| layer 3 `q_lin` | 8 | 2 | 91.4% |
| layer 3 `v_lin` | 8 | 3 | 93.2% |
| layer 4 `q_lin` | 8 | 2 | 98.1% |
| layer 4 `v_lin` | 8 | 3 | 97.8% |
| layer 5 `q_lin` | 8 | 2 | 99.0% |
| layer 5 `v_lin` | 8 | 2 | 96.8% |
| layer 6 `q_lin` | 8 | 2 | 99.98% |
| layer 6 `v_lin` | 8 | 4 | 93.3% |
| **Average** | **8** | **2.42** | — |

### 4.2 Three-way PEFT comparison

| Method | SST-2 Acc / F1 | IMDB Acc / F1 |
| --- | --- | --- |
| LoRA | 0.9037 / 0.9058 | 0.8679 / 0.8667 |
| Sparse LoRA | 0.8899 / 0.8909 | 0.8047 / 0.7879 |
| HiRA | 0.8704 / 0.8723 | 0.8375 / 0.8400 |

### 4.3 Inference latency

Measured on the same batch size (32) and tokenizer. Full JSON for sparse LoRA at [results/benchmark_sparse_lora/sparse_lora_inference_benchmark.json](results/benchmark_sparse_lora/sparse_lora_inference_benchmark.json).

- LoRA and Sparse LoRA are essentially tied — PyTorch/HF do not exploit LoRA parameter sparsity at inference time.
- HiRA adds a consistent overhead (element-wise Hadamard modulation in the forward pass).
- Absolute latency scales with sequence length, as expected from attention's quadratic cost.

### 4.4 Figures

All plots used in the report are in [figures/analysis_plots/](figures/analysis_plots/):

- `accuracy_by_dataset.png`, `f1_by_dataset.png` — headline accuracy / F1 bars
- `sparsity_by_dataset_cls.png`, `sparsity_wikitext2_mlm.png` — adapter sparsity
- `confusion_{imdb,sst2}_Sparse_LoRA.png` — per-class confusion
- `perplexity_mlm_datasets.png` — WikiText-2 MLM perplexity

Regenerate with:

```bash
make figures
# or: cd project/src && python -m sparse_lora.analyze_sparse_lora_results
```

## 5. Reproducing everything

```bash
pip install -r project/requirements.txt

# --- Three-way comparison ---------------------------------------------
./project/scripts/train_lora.sh sst2
./project/scripts/train_lora.sh imdb
./project/scripts/train_hira.sh sst2
./project/scripts/train_hira.sh imdb
# Sparse LoRA (classification)
cd project/src && python -m sparse_lora.train_distilbert_sparse_lora --dataset sst2
cd project/src && python -m sparse_lora.train_distilbert_sparse_lora --dataset imdb
# Sparse LoRA (WikiText-2 MLM)
cd project/src && python -m sparse_lora.train_distilbert_sparse_lora_wikitext2

# --- Novel contribution -----------------------------------------------
./project/scripts/compress_svd.sh sst2 0.9 2
./project/scripts/compress_svd.sh imdb 0.9 2

# --- Evaluation -------------------------------------------------------
./project/scripts/test_baseline_acc.sh sst2
./project/scripts/test_lora_acc.sh    sst2
./project/scripts/test_hira_acc.sh    sst2

# --- Latency benchmarks -----------------------------------------------
./project/scripts/test_lora_lat.sh         sst2
./project/scripts/test_sparse_lora_lat.sh  sst2
./project/scripts/test_hira_lat.sh         sst2
```

### Device support

Scripts run unchanged on CUDA, Apple MPS, and CPU. Selection is automatic in the Python entry points. Override device selection via `CUDA_VISIBLE_DEVICES`.

### Conda env support

If you need a specific conda env:

```bash
CONDA_ENV=myenv ./project/scripts/train_lora.sh sst2
```

The scripts will `conda activate` it for you. Nothing is hard-coded.

## 6. Repository layout

```
project/
├── README.md                      ← this file
├── requirements.txt
├── report/
│   └── ECE685D_Project_Report.pdf ← full technical report
│
├── src/
│   ├── train_lora.py              ← HuggingFace LoRA training
│   ├── train_hira.py              ← HiRA training
│   ├── test_baseline.py           ← untuned DistilBERT eval
│   ├── test_model.py              ← LoRA / SparseLoRA eval
│   ├── test_model_hira.py         ← HiRA eval
│   ├── benchmark_inference.py     ← latency measurement
│   ├── benchmark_inference_hira.py
│   ├── visualize_latency.py
│   ├── sparse_lora/
│   │   ├── train_distilbert_sparse_lora.py           (SST-2 / IMDB)
│   │   ├── train_distilbert_sparse_lora_wikitext2.py (MLM)
│   │   ├── test_sparse_lora.py
│   │   ├── test_sparse_lora_wikitext2.py
│   │   └── analyze_sparse_lora_results.py            (figures)
│   ├── hira/
│   │   ├── tuners/lora.py, adaption_prompt.py, prompt_tuning.py, ...
│   │   └── utils/config.py, save_and_load.py, ...
│   └── svd_compress/              ← the novel contribution
│       ├── __init__.py
│       ├── compress.py            ← core SVD compression
│       └── compress_lora.py       ← CLI wrapper
│
├── scripts/
│   ├── _common.sh                 ← shared env setup (no hardcoded paths)
│   ├── train_lora.sh, train_hira.sh
│   ├── compress_svd.sh            ← SVD-guided compression CLI
│   ├── test_baseline_{acc,lat}.sh
│   ├── test_lora_{acc,lat}.sh
│   ├── test_hira_{acc,lat}.sh
│   └── test_sparse_lora_lat.sh
│
├── figures/analysis_plots/        ← plots used in the paper
└── results/                       ← final adapters + metrics JSONs
    ├── sparse_lora_sst2/final_sparse_lora_model/
    ├── sparse_lora_imdb/final_sparse_lora_model/
    ├── sparse_lora_wikitext2/final_sparse_lora_model/
    └── benchmark_sparse_lora/sparse_lora_inference_benchmark.json
```

## 7. Extending this codebase

- **Different energy threshold τ.** Pass it to the script or the Python API. `τ = 0.95` keeps a bit more mass at higher rank; `τ = 0.85` is more aggressive.
- **Compress an existing adapter you already have on disk** — just point `compress_lora.py` at the directory. Output is a drop-in PEFT adapter.
- **Different backbone.** The SVD pass walks the model generically (any `lora_A`/`lora_B` `ModuleDict` pair); it is not DistilBERT-specific. Change the `target_modules` in the training LoRA config to apply to other architectures (e.g., `["q_proj", "v_proj"]` for Llama).
- **Different rank-selection criterion.** `select_rank_by_energy` is a drop-in point. Swap in nuclear-norm-based or BIC-based selection by replacing that one function.
- **Verify the math.** The compression is exact to Eckart-Young-Mirsky: `||ΔW⋆ - ΔW̃||_F / ||ΔW⋆||_F = sqrt(1 - retained_energy)`. A sanity check is in the README at the repo root.

---

Questions, issues, or ideas: open a GitHub issue. PRs are welcome.
