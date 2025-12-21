#!/usr/bin/env python3
"""
Analyze Sparse LoRA results on:
- SST-2
- IMDB
- WikiText-2 (MLM)

For SST-2 / IMDB (classification):
    - Loads JSON metrics saved by test_sparse_lora.py
    - Computes LoRA sparsity from the checkpoint
    - Produces:
        * Metrics summary table (printed)
        * Accuracy / F1 barplots
        * Confusion matrix heatmaps

For WikiText-2 (MLM):
    - Loads JSON metrics saved by test_sparse_lora_wikitext2.py
    - Computes LoRA sparsity from the checkpoint
    - Produces:
        * Perplexity summary table (printed)
        * Perplexity barplot
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from peft import AutoPeftModelForSequenceClassification, PeftModel
from transformers import AutoModelForMaskedLM


# -----------------------------
# 1) Config for all runs
# -----------------------------

@dataclass
class RunConfig:
    name: str           # label to show in tables/plots, e.g. "Sparse LoRA"
    dataset: str        # e.g. "sst2", "imdb", "wikitext2"
    model_path: str     # directory that contains sparse_lora_validation_results.json
    base_model: str     # base HF model name (e.g. "distilbert-base-uncased")
    task_type: str      # "cls" for classification, "mlm" for language modeling


RUNS: List[RunConfig] = [
    RunConfig(
        name="Sparse LoRA",
        dataset="sst2",
        model_path="./results_sparse_lora/final_sparse_lora_model",
        base_model="distilbert-base-uncased",
        task_type="cls",
    ),
    RunConfig(
        name="Sparse LoRA",
        dataset="imdb",
        model_path="./results_sparse_lora_imdb/final_sparse_lora_model",
        base_model="distilbert-base-uncased",
        task_type="cls",
    ),
    RunConfig(
        name="Sparse LoRA",
        dataset="wikitext2",
        model_path="./results_sparse_lora_wikitext2/final_sparse_lora_model",
        base_model="distilbert-base-uncased",
        task_type="mlm",
    ),
    # Add more runs (e.g. dense baselines) by appending here.
]


# -----------------------------
# 2) Utilities
# -----------------------------

def load_json_metrics(run: RunConfig) -> Dict[str, Any]:
    # Classification runs use the validation JSON;
    # WikiText-2 MLM run uses the *_test_results.json we saved.
    if run.task_type == "cls":
        filename = "sparse_lora_validation_results.json"
    elif run.task_type == "mlm":
        filename = "sparse_lora_wikitext2_test_results.json"
    else:
        raise ValueError(f"Unknown task_type: {run.task_type}")

    json_path = os.path.join(run.model_path, filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON results not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



def compute_lora_sparsity(run: RunConfig) -> float:
    """
    Compute fraction of zero parameters among all LoRA weights in this model.
    Returns sparsity in [0, 1].

    For classification runs:
        use AutoPeftModelForSequenceClassification.from_pretrained(...)
    For MLM runs:
        load a base AutoModelForMaskedLM and wrap with PeftModel.from_pretrained(...)
    """
    print(f"[Sparsity] Loading PEFT model from {run.model_path} ...")

    if run.task_type == "cls":
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            run.model_path,
            device_map=None,  # keep on CPU for analysis
        )
    elif run.task_type == "mlm":
        # Base masked LM model + LoRA adapters
        base = AutoModelForMaskedLM.from_pretrained(run.base_model)
        model = PeftModel.from_pretrained(base, run.model_path)
    else:
        raise ValueError(f"Unknown task_type: {run.task_type}")

    total = 0
    zeros = 0

    with torch.no_grad():
        for n, p in model.named_parameters():
            # Only count LoRA parameters (linear lora_A / lora_B matrices)
            if "lora_" in n:
                tensor = p.detach().cpu()
                total += tensor.numel()
                zeros += (tensor == 0).sum().item()

    if total == 0:
        print(f"[Sparsity] WARNING: found no LoRA params for run {run.name} ({run.dataset})")
        return 0.0

    sparsity = zeros / total
    print(
        f"[Sparsity] Sparse LoRA on {run.dataset} -> {sparsity * 100:.2f}% zeros "
        f"({zeros}/{total})"
    )
    return sparsity


# -----------------------------
# 3) Build metrics tables
# -----------------------------

def build_cls_metrics_dataframe(
    runs: List[RunConfig],
    compute_sparsity_flag: bool = True,
) -> pd.DataFrame:
    """
    Build a dataframe for classification runs (SST-2 / IMDB).
    """
    rows = []
    for run in runs:
        if run.task_type != "cls":
            continue

        data = load_json_metrics(run)
        metrics = data["eval_metrics"]

        # HF Trainer prefix: eval_accuracy, eval_precision, etc.
        acc = metrics.get("eval_accuracy", None)
        prec = metrics.get("eval_precision", None)
        rec = metrics.get("eval_recall", None)
        f1 = metrics.get("eval_f1", None)
        f1_pos = metrics.get("eval_f1_positive", None)
        f1_neg = metrics.get("eval_f1_negative", None)

        # confusion matrix and counts
        cm = np.array(data["confusion_matrix"])
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]

        sparsity = compute_lora_sparsity(run) if compute_sparsity_flag else None

        rows.append({
            "run_name": run.name,
            "dataset": run.dataset,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_negative": f1_neg,
            "f1_positive": f1_pos,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sparsity": sparsity,
            "eval_split": data.get("eval_split"),
            "eval_samples": data.get("eval_samples"),
        })

    df = pd.DataFrame(rows)
    return df


def build_mlm_metrics_dataframe(
    runs: List[RunConfig],
    compute_sparsity_flag: bool = True,
) -> pd.DataFrame:
    """
    Build a dataframe for MLM runs (WikiText-2).
    Only uses eval_loss + perplexity.
    """
    rows = []
    for run in runs:
        if run.task_type != "mlm":
            continue

        data = load_json_metrics(run)
        metrics = data["eval_metrics"]

        eval_loss = metrics.get("eval_loss", None)
        ppl = metrics.get("perplexity", None)

        sparsity = compute_lora_sparsity(run) if compute_sparsity_flag else None

        rows.append({
            "run_name": run.name,
            "dataset": run.dataset,
            "eval_loss": eval_loss,
            "perplexity": ppl,
            "sparsity": sparsity,
            "eval_split": data.get("eval_split"),
            "eval_samples": data.get("eval_samples"),
        })

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# 4) Plotting helpers
# -----------------------------

def plot_metric_bars(
    df: pd.DataFrame,
    metric: str,
    save_path: Optional[str] = None,
):
    """
    Bar plot of a given metric for each classification run, grouped by dataset.
    metric: one of "accuracy", "f1", "precision", "recall"
    """
    assert metric in {"accuracy", "f1", "precision", "recall"}, f"Unexpected metric: {metric}"

    if df.empty:
        print(f"[Plot] No classification runs to plot for metric {metric}.")
        return

    # sort for a stable order
    df_sorted = df.sort_values(by=["dataset", "run_name"])

    datasets = df_sorted["dataset"].unique()
    x_labels = []
    values = []

    for ds in datasets:
        sub = df_sorted[df_sorted["dataset"] == ds]
        for _, row in sub.iterrows():
            x_labels.append(f"{row['run_name']} ({ds})")
            values.append(row[metric])

    x = np.arange(len(x_labels))

    plt.figure(figsize=(10, 4))
    plt.bar(x, values)
    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric.upper()} by run and dataset")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved {metric} bar plot to {save_path}")
    else:
        plt.show()

def plot_sparsity_bars(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Bar plot of LoRA sparsity for each run (1 = 100% zeros).
    Uses the 'sparsity' column already present in the dataframe.
    """
    # Filter out any rows without sparsity (just in case)
    df = df.dropna(subset=["sparsity"])

    df_sorted = df.sort_values(by=["dataset", "run_name"])
    labels = [f"{row['run_name']} ({row['dataset']})" for _, row in df_sorted.iterrows()]
    values = df_sorted["sparsity"].tolist()

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("LoRA sparsity (fraction of zeros)")
    plt.title("LoRA sparsity by run and dataset")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved sparsity bar plot to {save_path}")
    else:
        plt.show()


def plot_perplexity_bars(df_mlm: pd.DataFrame, save_path: Optional[str] = None):
    """
    Bar plot of perplexity for MLM runs (e.g. WikiText-2).
    """
    if df_mlm.empty:
        print("[Plot] No MLM runs to plot perplexity.")
        return

    df_sorted = df_mlm.sort_values(by=["dataset", "run_name"])

    x_labels = []
    values = []

    for _, row in df_sorted.iterrows():
        x_labels.append(f"{row['run_name']} ({row['dataset']})")
        values.append(row["perplexity"])

    x = np.arange(len(x_labels))

    plt.figure(figsize=(8, 4))
    plt.bar(x, values)
    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylabel("Perplexity")
    plt.title("Perplexity by run and dataset (MLM)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved perplexity bar plot to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    dataset: str,
    run_name: str,
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    """
    Simple 2x2 confusion matrix heatmap for classification runs.
    """
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {run_name} on {dataset}")
    fig.colorbar(im, ax=ax)

    classes = ["Neg", "Pos"]
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Print values inside cells
    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved confusion matrix to {save_path}")
    else:
        plt.show()


# -----------------------------
# 5) Main entry
# -----------------------------

def main():
    os.makedirs("./analysis_plots", exist_ok=True)

    # ---- Classification runs (SST-2 / IMDB) ----
    df_cls = build_cls_metrics_dataframe(RUNS, compute_sparsity_flag=True)
    if not df_cls.empty:
        print("\n=== Classification summary table (SST-2 / IMDB) ===")
        print(
            df_cls.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
            )
        )

        # Bar plots
        plot_metric_bars(
            df_cls,
            "accuracy",
            save_path="./analysis_plots/accuracy_by_dataset.png",
        )
        plot_metric_bars(
            df_cls,
            "f1",
            save_path="./analysis_plots/f1_by_dataset.png",
        )
        # Sparsity (efficiency) for classification runs
        plot_sparsity_bars(
            df_cls,
            save_path="./analysis_plots/sparsity_by_dataset_cls.png",
        )


        # Confusion matrices per classification run
        for run in RUNS:
            if run.task_type != "cls":
                continue
            data = load_json_metrics(run)
            cm = np.array(data["confusion_matrix"])
            plot_confusion_matrix(
                cm,
                dataset=run.dataset,
                run_name=run.name,
                normalize=False,
                save_path=f"./analysis_plots/confusion_{run.dataset}_{run.name.replace(' ', '_')}.png",
            )
    else:
        print("\n[Info] No classification runs found.")

    # ---- MLM runs (WikiText-2) ----
    df_mlm = build_mlm_metrics_dataframe(RUNS, compute_sparsity_flag=True)
    if not df_mlm.empty:
        print("\n=== MLM summary table (WikiText-2, etc.) ===")
        print(
            df_mlm.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
            )
        )
        # Sparsity (efficiency) for MLM (WikiText-2)
        plot_sparsity_bars(
            df_mlm,
            save_path="./analysis_plots/sparsity_wikitext2_mlm.png",
        )


        # Perplexity bar plot
        plot_perplexity_bars(
            df_mlm,
            save_path="./analysis_plots/perplexity_mlm_datasets.png",
        )
    else:
        print("\n[Info] No MLM runs found.")

    print("\nAnalysis complete.\n")


if __name__ == "__main__":
    main()
