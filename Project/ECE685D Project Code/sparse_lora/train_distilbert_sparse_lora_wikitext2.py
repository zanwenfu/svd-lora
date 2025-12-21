#!/usr/bin/env python
"""
DistilBERT + Sparse LoRA fine-tuning on WikiText-2 (masked language modeling).

- Uses DistilBertForMaskedLM instead of sequence classification.
- Applies LoRA on attention (q_lin, v_lin).
- Adds L1 regularization on LoRA params.
- Optionally prunes small LoRA weights after training.
- Reports eval loss and perplexity.

MacBook/CPU/MPS friendly.
"""

import argparse
import os
import json
import math
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT with Sparse LoRA on WikiText-2 (MLM)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base pretrained model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_sparse_lora_wikitext2",
        help="Output directory (base)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device train/eval batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length (for chunking text)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Checkpoint save steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Eval steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    # Sparse LoRA specific
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=1e-4,
        help="L1 regularization strength on LoRA parameters",
    )
    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=0.01,
        help=(
            "Magnitude pruning threshold for LoRA params AFTER training. "
            "Weights with |w| < threshold are set to 0. 0 = no pruning."
        ),
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Masking probability for MLM",
    )
    return parser.parse_args()


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_and_preprocess_wikitext2(
    tokenizer,
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Load WikiText-2 and convert text into contiguous token blocks of length max_length.
    """
    print("Loading WikiText-2 dataset (wikitext-2-raw-v1)...")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Tokenize line by line first
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Group texts into blocks of max_length tokens
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop remainder to keep chunks of max_length
        total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping tokenized text into blocks...")
    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
    )

    print(
        "Dataset sizes (in blocks): "
        f"train={len(lm_datasets['train'])}, "
        f"validation={len(lm_datasets['validation'])}, "
        f"test={len(lm_datasets['test'])}"
    )

    return lm_datasets


# -----------------------------
# LoRA model creation
# -----------------------------
def create_sparse_lora_mlm_model(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    print(f"Loading base MLM model: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # LoRA configuration for MLM (still SEQ_CLS task type works fine with DistilBERT)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
    )

    print("Applying LoRA adapters (sparsity added later via L1 / pruning)...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# -----------------------------
# Sparse LoRA Trainer
# -----------------------------
class SparseLoraTrainer(Trainer):
    """
    Trainer that adds L1 regularization on LoRA parameters (for MLM).
    """

    def __init__(self, *args, l1_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_lambda = l1_lambda
        if self.l1_lambda > 0:
            print(
                f"Using L1 regularization on LoRA params with lambda = {self.l1_lambda}"
            )

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        Custom loss: base MLM loss + L1 regularization on LoRA parameters.
        """
        outputs = model(**inputs)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            base_loss = outputs.loss
        else:
            # MLM: logits [B, L, V], labels [B, L] with -100 for ignored positions
            logits = outputs.logits
            labels = inputs["labels"]
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            base_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        if self.l1_lambda > 0:
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    l1_loss = l1_loss + param.abs().sum()
            loss = base_loss + self.l1_lambda * l1_loss
        else:
            loss = base_loss

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Sparsity / pruning helpers
# -----------------------------
def compute_lora_sparsity(model):
    """Return (zero_params, total_params, sparsity_ratio) for LoRA params."""
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            data = param.detach()
            total_params += data.numel()
            zero_params += (data == 0).sum().item()
    ratio = zero_params / total_params if total_params > 0 else 0.0
    return zero_params, total_params, ratio


def apply_lora_pruning(model, threshold: float):
    """
    Magnitude-based pruning on LoRA parameters: set values with |w| < threshold to 0.
    Returns (zero_params, total_params, sparsity_ratio) after pruning.
    """
    if threshold <= 0:
        print("Pruning disabled (threshold <= 0).")
        return compute_lora_sparsity(model)

    print(f"Applying magnitude pruning to LoRA params with threshold = {threshold}")
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            with torch.no_grad():
                mask = param.abs() >= threshold
                param *= mask

    zero_params, total_params, ratio = compute_lora_sparsity(model)
    print(
        f"LoRA sparsity after pruning: {ratio * 100:.2f}% "
        f"({zero_params}/{total_params} parameters are zero)"
    )
    return zero_params, total_params, ratio


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Device info (Mac friendly)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Data
    tokenized_dataset = load_and_preprocess_wikitext2(
        tokenizer, max_length=args.max_length
    )

    # Model (MLM + LoRA)
    model = create_sparse_lora_mlm_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Sparsity BEFORE training
    zero_before, total_before, ratio_before = compute_lora_sparsity(model)
    print(
        f"Initial LoRA sparsity: {ratio_before * 100:.2f}% "
        f"({zero_before}/{total_before} params are zero)"
    )

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        report_to="tensorboard",
        seed=args.seed,
        fp16=False,
        save_total_limit=2,
    )

    # Trainer (we won't compute detailed metrics here; loss is enough)
    trainer = SparseLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        l1_lambda=args.l1_lambda,
    )

    # Training
    print("\n" + "=" * 50)
    print("Starting Sparse LoRA MLM training on WikiText-2...")
    print("=" * 50 + "\n")
    train_result = trainer.train()

    # Prune small LoRA weights (optional)
    zero_after, total_after, ratio_after = apply_lora_pruning(
        model, args.prune_threshold
    )

    # Save final sparse LoRA model
    final_model_dir = os.path.join(args.output_dir, "final_sparse_lora_model")
    print(f"\nSaving sparse LoRA model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final validation evaluation
    print("\n" + "=" * 50)
    print("Final evaluation on validation set (WikiText-2)...")
    print("=" * 50 + "\n")
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Compute perplexity from eval loss
    eval_loss = eval_metrics["eval_loss"]
    perplexity = math.exp(eval_loss)
    eval_metrics["perplexity"] = perplexity
    print(f"Validation loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}")

    # Save sparsity stats
    sparsity_stats = {
        "dataset": "wikitext-2-raw-v1",
        "model_name": args.model_name,
        "l1_lambda": args.l1_lambda,
        "prune_threshold": args.prune_threshold,
        "lora_zero_params_before": zero_before,
        "lora_total_params_before": total_before,
        "lora_sparsity_before": ratio_before,
        "lora_zero_params_after": zero_after,
        "lora_total_params_after": total_after,
        "lora_sparsity_after": ratio_after,
        "eval_loss": eval_loss,
        "eval_perplexity": perplexity,
    }

    with open(
        os.path.join(args.output_dir, "sparse_lora_sparsity_stats.json"), "w"
    ) as f:
        json.dump(sparsity_stats, f, indent=2)

    print("\n" + "=" * 50)
    print("Sparse LoRA MLM training completed!")
    print(f"Model saved at: {final_model_dir}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
