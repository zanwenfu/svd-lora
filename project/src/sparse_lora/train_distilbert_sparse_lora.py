"""
DistilBERT + Sparse LoRA Fine-tuning

Sparse LoRA = LoRA adapters + L1 regularization on LoRA params
(+ optional magnitude pruning after training).

Supports SST-2 and IMDB.
MacBook/CPU/MPS friendly (no CUDA env hacks, no fp16).
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with Sparse LoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "imdb"],
        help="Dataset: sst2 or imdb",
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
        default="./results_sparse_lora",
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
        help="Max sequence length",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Checkpoint save steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
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
        default=0.0,
        help=(
            "Magnitude pruning threshold for LoRA params AFTER training. "
            "Weights with |w| < threshold are set to 0. 0 = no pruning."
        ),
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
def load_and_preprocess_data(
    tokenizer, dataset_name: str = "sst2", max_length: int = 128
):
    if dataset_name == "sst2":
        print("Loading SST-2 dataset...")
        dataset = load_dataset("glue", "sst2")
        text_column = "sentence"
        remove_columns = ["sentence", "idx"]
    elif dataset_name == "imdb":
        print("Loading IMDB dataset...")
        dataset = load_dataset("stanfordnlp/imdb")
        text_column = "text"
        remove_columns = ["text"]

        # Split 10% of train as validation
        print("Splitting 10% of IMDB train as validation...")
        train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = train_val_split["train"]
        dataset["validation"] = train_val_split["test"]
        print(
            f"Train size: {len(dataset['train'])}, "
            f"Validation size: {len(dataset['validation'])}"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def preprocess_function(examples: Dict[str, Any]):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollator
        )

    print(f"Tokenizing {dataset_name}...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )

    # Rename label -> labels
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    return tokenized_dataset


# -----------------------------
# LoRA model creation
# -----------------------------
def create_sparse_lora_model(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    print(f"Loading base model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    # LoRA configuration
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
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -----------------------------
# Sparse LoRA Trainer
# -----------------------------
class SparseLoraTrainer(Trainer):
    """
    Trainer that adds L1 regularization on LoRA parameters.
    """

    def __init__(self, *args, l1_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_lambda = l1_lambda
        if self.l1_lambda > 0:
            print(f"Using L1 regularization on LoRA params with lambda = {self.l1_lambda}")

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        Custom loss: base supervised loss + L1 regularization on LoRA parameters.

        Extra kwargs (e.g. num_items_in_batch) are accepted for compatibility
        with newer Trainer APIs but are not used here.
        """
        outputs = model(**inputs)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            base_loss = outputs.loss
        else:
            logits = outputs.logits
            labels = inputs["labels"]
            loss_fct = CrossEntropyLoss()
            base_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        # L1 on LoRA params
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
                param *= mask  # zero out small weights

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

    # Device info (MacBook friendly)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Dataset-specific output dir
    output_dir = args.output_dir if args.dataset == "sst2" else f"{args.output_dir}_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Data
    tokenized_dataset = load_and_preprocess_data(tokenizer, args.dataset, args.max_length)

    # Model (LoRA; sparsity handled via L1 + pruning)
    model = create_sparse_lora_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Sparsity BEFORE training (should be 0)
    zero_before, total_before, ratio_before = compute_lora_sparsity(model)
    print(
        f"Initial LoRA sparsity: {ratio_before * 100:.2f}% "
        f"({zero_before}/{total_before} params are zero)"
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments (no fp16 to keep it simple on Mac)
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to="tensorboard",
        seed=args.seed,
        fp16=False,
        save_total_limit=2,
    )

    # Trainer with L1 on LoRA params
    trainer = SparseLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        l1_lambda=args.l1_lambda,
    )

    # Training
    print("\n" + "=" * 50)
    print("Starting Sparse LoRA training...")
    print("=" * 50 + "\n")
    train_result = trainer.train()

    # Prune small LoRA weights (optional)
    zero_after, total_after, ratio_after = apply_lora_pruning(model, args.prune_threshold)

    # Save final sparse LoRA model
    final_model_dir = os.path.join(output_dir, "final_sparse_lora_model")
    print(f"\nSaving sparse LoRA model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final validation evaluation
    print("\n" + "=" * 50)
    print("Final evaluation on validation set...")
    print("=" * 50 + "\n")
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save sparsity stats
    sparsity_stats = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "l1_lambda": args.l1_lambda,
        "prune_threshold": args.prune_threshold,
        "lora_zero_params_before": zero_before,
        "lora_total_params_before": total_before,
        "lora_sparsity_before": ratio_before,
        "lora_zero_params_after": zero_after,
        "lora_total_params_after": total_after,
        "lora_sparsity_after": ratio_after,
    }
    import json

    with open(os.path.join(output_dir, "sparse_lora_sparsity_stats.json"), "w") as f:
        json.dump(sparsity_stats, f, indent=2)

    print("\n" + "=" * 50)
    print("Sparse LoRA training completed!")
    print(f"Model saved at: {final_model_dir}")
    print("=" * 50)
    print("\nValidation results:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    main()
