"""CLI: compress a trained LoRA adapter via SVD-guided rank reduction.

Example:
    python -m svd_compress.compress_lora \
        --model_path  results/lora_sst2/final_model \
        --output_dir  results/lora_sst2_svd \
        --energy_threshold 0.9

Optionally continues fine-tuning the compressed adapter (`--post_epochs > 0`),
which is Stage III of the procedure described in the report.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .compress import compress_and_save


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True,
                   help="Path to a trained PEFT LoRA adapter directory.")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the SVD-compressed adapter.")
    p.add_argument("--energy_threshold", type=float, default=0.9,
                   help="Cumulative Frobenius-energy fraction to retain per module.")
    p.add_argument("--dataset", choices=["sst2", "imdb"], default=None,
                   help="If set, evaluate before and after compression.")
    p.add_argument("--post_epochs", type=int, default=0,
                   help="Stage-III fine-tuning epochs in the reduced subspace.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _build_dataset(tokenizer, dataset_name: str, max_length: int):
    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2")
        text_col = "sentence"
        remove_cols = ["sentence", "idx"]
    elif dataset_name == "imdb":
        ds = load_dataset("stanfordnlp/imdb")
        text_col = "text"
        remove_cols = ["text"]
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]
    else:
        raise ValueError(dataset_name)

    def _tok(ex):
        return tokenizer(ex[text_col], truncation=True, max_length=max_length, padding=False)

    ds = ds.map(_tok, batched=True, remove_columns=remove_cols)
    ds = ds.rename_column("label", "labels")
    return ds


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading PEFT model from {args.model_path}")
    model = AutoPeftModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tokenized = None
    if args.dataset is not None:
        tokenized = _build_dataset(tokenizer, args.dataset, args.max_length)

    # Optional pre-compression evaluation
    def _evaluate(m, tag: str):
        if tokenized is None:
            return None
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        eval_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"_tmp_eval_{tag}"),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        t = Trainer(
            model=m,
            args=eval_args,
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=_compute_metrics,
        )
        metrics = t.evaluate()
        print(f"[eval:{tag}] {metrics}")
        return metrics

    _evaluate(model, "before")

    # Stage II: SVD compression
    result = compress_and_save(
        model,
        output_dir=args.output_dir,
        energy_threshold=args.energy_threshold,
        tokenizer=tokenizer,
    )
    print(f"Compression report saved to {args.output_dir}/svd_compression_report.json")

    # Stage III: optional post-compression fine-tuning
    if args.post_epochs > 0:
        if tokenized is None:
            raise SystemExit("--post_epochs > 0 requires --dataset so we have training data")
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.post_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none",
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        trainer.save_model(args.output_dir)

    _evaluate(model, "after")

    print(
        f"\nDone. avg_rank={result.average_rank:.2f}  "
        f"param_ratio={result.parameter_ratio * 100:.1f}%  "
        f"tau={result.energy_threshold}"
    )


if __name__ == "__main__":
    main()
