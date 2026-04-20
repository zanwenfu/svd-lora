# Pushing the Efficiency Frontier of LoRA

**SVD-Guided Adaptive Rank Compression + A Controlled Study of LoRA, Sparse LoRA, and HiRA**

> Compressed adapters use **~30% of LoRA's parameters** while **matching or surpassing** baseline performance — including **+2.1 F1 on IMDB** over standard LoRA.

This repository contains the implementation, experiments, results, and final report for a research project by Qinsi Wang, Yuhan Cheng, Zanwen Fu, and Hancheng Ye (Duke ECE). It has two contributions:

1. **A fair, apples-to-apples comparison** of three representative PEFT methods — standard LoRA, Sparse LoRA, and HiRA — fine-tuning a shared `distilbert-base-uncased` backbone on SST-2, IMDB, and WikiText-2 under a strictly matched training protocol.
2. **A novel SVD-guided adaptive rank compression scheme** for LoRA: train at rank 8, truncate each adapter to the smallest rank retaining ≥ 90% of its Frobenius-energy spectrum, then fine-tune briefly in the reduced subspace. Average effective rank drops to **2.42** across attention projections.

📄 **Report:** [project/report/ECE685D_Project_Report.pdf](project/report/ECE685D_Project_Report.pdf)
🔬 **Project README** (methods, setup, reproduction): [project/README.md](project/README.md)

---

## Headline results

**SVD-compressed LoRA vs. standard LoRA** (DistilBERT, rank 8 → adaptive rank 2.42, ~30% params):

| Dataset | Method | Accuracy | F1 |
| --- | --- | --- | --- |
| SST-2 | LoRA  | 0.8922 | 0.8967 |
| SST-2 | **SVD-compressed (ours)** | **0.8933** | 0.8961 |
| IMDB | LoRA | 0.8681 | 0.8715 |
| IMDB | **SVD-compressed (ours)** | **0.8812** | **0.8924** |

**Three-way PEFT comparison** under a matched budget (r=8, α=16, 3 epochs, lr=3e-4):

| Method | SST-2 Acc / F1 | IMDB Acc / F1 |
| --- | --- | --- |
| LoRA | 0.9037 / 0.9058 | 0.8679 / 0.8667 |
| Sparse LoRA | 0.8899 / 0.8909 | 0.8047 / 0.7879 |
| HiRA | 0.8704 / 0.8723 | 0.8375 / 0.8400 |

Full tables, latency benchmarks, confusion matrices, and discussion are in the [report](project/report/ECE685D_Project_Report.pdf) and in [project/README.md](project/README.md).

---

## Quick start

```bash
# 1. Install deps (CUDA / MPS / CPU all supported)
pip install -r project/requirements.txt

# 2. Train standard LoRA on SST-2
./project/scripts/train_lora.sh sst2

# 3. Compress it with SVD-guided adaptive rank selection
./project/scripts/compress_svd.sh sst2 0.9 2
#                                   dataset  tau  post-fine-tune epochs

# 4. Evaluate the compressed adapter
./project/scripts/test_lora_acc.sh sst2 project/results/lora_sst2_svd
```

Or via `make`:

```bash
make install
make train-lora D=sst2
make compress   D=sst2 TAU=0.9 POST=2
make eval-lora  D=sst2
```

All scripts honor `CONDA_ENV` (activate a named conda env) and `GPU_ID` (pick a device). Nothing is hard-coded.

---

## The idea in 60 seconds

Standard LoRA uses a fixed global rank `r` for every adapter, but the intrinsic dimensionality of each module's learned update varies — some are over-parameterized, others under-parameterized.

We let the data decide:

1. **Stage I — high-rank warm-up.** Train LoRA normally at `r = 8`.
2. **Stage II — per-module SVD.** For each adapter module `ℓ`, compute
   ΔW*ℓ = (α/r) · BA, take its truncated SVD ΔW*ℓ = UΣV⊤, and pick the
   smallest `k_ℓ` such that the leading `k_ℓ` singular values retain
   ≥ τ (= 0.9) of the Frobenius energy. Rebuild factors as
   B'ℓ = U·diag(σ), A'ℓ = V⊤ of rank `k_ℓ`.
3. **Stage III — post-compression fine-tuning.** Resume training for a
   few epochs in the reduced subspace to absorb the approximation error.

The core compression pass is ~30 lines of code, exact to the Eckart-Young-Mirsky bound, and preserves the standard PEFT adapter interface:

```python
from svd_compress import compress_peft_model
from peft import AutoPeftModelForSequenceClassification

model = AutoPeftModelForSequenceClassification.from_pretrained("results/lora_sst2/final_model")
result = compress_peft_model(model, energy_threshold=0.9)
print(f"avg_rank={result.average_rank:.2f}  param_ratio={result.parameter_ratio:.1%}")
```

---

## Repository layout

```
.
├── project/                      ← main research project
│   ├── README.md                 ← methods, setup, reproduction, results
│   ├── report/                   ← the full technical report (PDF)
│   ├── requirements.txt
│   ├── src/
│   │   ├── train_lora.py         ← LoRA training entry point
│   │   ├── train_hira.py         ← HiRA training entry point
│   │   ├── test_baseline.py      ← untuned DistilBERT evaluation
│   │   ├── test_model.py         ← LoRA / Sparse LoRA evaluation
│   │   ├── test_model_hira.py    ← HiRA evaluation
│   │   ├── benchmark_inference.py
│   │   ├── benchmark_inference_hira.py
│   │   ├── visualize_latency.py
│   │   ├── sparse_lora/          ← Sparse LoRA (L1 + magnitude pruning) + analysis
│   │   ├── hira/                 ← HiRA PEFT implementation (tuners, utils)
│   │   └── svd_compress/         ← our SVD-guided adaptive rank compression
│   ├── scripts/                  ← portable .sh entry points (no hardcoded envs)
│   ├── results/                  ← final adapters + metrics JSONs (small, reviewable)
│   └── figures/analysis_plots/   ← plots used in the paper
├── course/                       ← supporting ECE685D course materials (low-key)
├── LICENSE                       ← MIT
├── CITATION.cff                  ← citation metadata
├── Makefile                      ← repo-level make targets
└── README.md                     ← this file
```

---

## Reproducing the paper

Full step-by-step reproduction — datasets, seeds, checkpoint naming, figure regeneration — lives in [project/README.md](project/README.md). The short version:

```bash
# Three-way comparison
for D in sst2 imdb; do ./project/scripts/train_lora.sh $D; done
for D in sst2 imdb; do ./project/scripts/train_hira.sh $D; done
# Sparse LoRA: see project/src/sparse_lora/train_distilbert_sparse_lora.py

# Novel contribution
./project/scripts/compress_svd.sh sst2 0.9 2
./project/scripts/compress_svd.sh imdb 0.9 2
```

All runs use seed 42, batch size 16, learning rate 3e-4, 3 epochs, r=8, α=16, dropout=0.1.

---

## Citation

```bibtex
@techreport{wang2025pushinglora,
  author      = {Zanwen Fu and Qinsi Wang and Yuhan Cheng and Hancheng Ye},
  title       = {Pushing the Efficiency Frontier of LoRA: A Systematic Study
                 and SVD-Guided Adaptive Rank Compression},
  institution = {Duke University, Department of Electrical and Computer Engineering},
  year        = {2025},
  note        = {ECE685D Project Report}
}
```

See also [CITATION.cff](CITATION.cff).

---

## License

MIT — see [LICENSE](LICENSE).

---

## About the course materials

The [`course/`](course/) directory contains lecture notes, assignments, discussion materials, and past-year papers from **Duke COMPSCI 675D / ECE685D — Introduction to Deep Learning (Fall 2025)**. They are kept in-tree for reference but are not part of the research artifact. The scientific work lives in [`project/`](project/).
