"""
Evaluate a Sparse LoRA fine-tuned DistilBERT model on SST-2 / IMDB.
"""

import os
import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from peft import AutoPeftModelForSequenceClassification
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sparse LoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results_sparse_lora/final_sparse_lora_model",
        help="Path to trained Sparse LoRA model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "imdb"],
        help="Dataset name (sst2 or imdb)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Eval batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index if using CUDA; ignored on Mac/CPU",
    )
    return parser.parse_args()


def load_and_preprocess_data(tokenizer, dataset_name="sst2", max_length=128):
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def preprocess_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print(f"Tokenizing {dataset_name}...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )

    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    return tokenized_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(labels, predictions, average=None)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_negative": precision_per_class[0],
        "precision_positive": precision_per_class[1],
        "recall_negative": recall_per_class[0],
        "recall_positive": recall_per_class[1],
        "f1_negative": f1_per_class[0],
        "f1_positive": f1_per_class[1],
        "support_negative": int(support[0]),
        "support_positive": int(support[1]),
    }


def main():
    args = parse_args()

    # Device info (Mac friendly)
    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    print(f"Using device: {device_str}")

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer & Sparse LoRA model
    print(f"\nLoading Sparse LoRA model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoPeftModelForSequenceClassification.from_pretrained(args.model_path)

    # Load dataset
    tokenized_dataset = load_and_preprocess_data(
        tokenizer, args.dataset, args.max_length
    )

    # SST-2: validation; IMDB: test
    if args.dataset == "imdb":
        eval_dataset = tokenized_dataset["test"]
        eval_name = "test"
    else:
        eval_dataset = tokenized_dataset["validation"]
        eval_name = "validation"
        print("Note: SST-2 test labels are hidden; using validation set.")

    print(f"{eval_name.capitalize()} samples: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./test_results_sparse_lora",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        fp16=False,  # keep off for Mac
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 60)
    print(f"Evaluating Sparse LoRA model on {args.dataset.upper()} ({eval_name})")
    print("=" * 60 + "\n")

    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    # ---------- [LATENCY] compute inference latency ----------
    eval_runtime = metrics.get("eval_runtime", None)
    if eval_runtime is not None:
        n_samples = len(eval_dataset)
        samples_per_second = n_samples / eval_runtime
        latency_per_sample_sec = eval_runtime / n_samples
        latency_per_sample_ms = latency_per_sample_sec * 1000.0

        metrics["eval_samples_per_second_measured"] = samples_per_second
        metrics["eval_latency_per_sample_sec"] = latency_per_sample_sec
        metrics["eval_latency_per_sample_ms"] = latency_per_sample_ms

        print(f"Inference runtime: {eval_runtime:.4f} s for {n_samples} samples")
        print(f"Samples / second: {samples_per_second:.2f}")
        print(f"Latency / sample: {latency_per_sample_ms:.2f} ms")
    else:
        print("Warning: eval_runtime not found in metrics; latency not computed.")
    # ---------------------------------------------------------

    predictions = trainer.predict(eval_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Print metrics
    print("\nOverall metrics:")
    print(f"  Accuracy:  {metrics['eval_accuracy']:.4f}")
    print(f"  Precision: {metrics['eval_precision']:.4f}")
    print(f"  Recall:    {metrics['eval_recall']:.4f}")
    print(f"  F1-score:  {metrics['eval_f1']:.4f}")

    print("\nNegative class:")
    print(f"  Precision: {metrics['eval_precision_negative']:.4f}")
    print(f"  Recall:    {metrics['eval_recall_negative']:.4f}")
    print(f"  F1-score:  {metrics['eval_f1_negative']:.4f}")

    print("\nPositive class:")
    print(f"  Precision: {metrics['eval_precision_positive']:.4f}")
    print(f"  Recall:    {metrics['eval_recall_positive']:.4f}")
    print(f"  F1-score:  {metrics['eval_f1_positive']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)
    print(f"\n              Pred")
    print(f"            Neg   Pos")
    print(f"True Neg   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"     Pos   {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Classification report
    target_names = ["Negative", "Positive"]
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print("\n" + classification_report(true_labels, pred_labels, target_names=target_names))

    # Save JSON results next to the model (includes latency fields now)
    results = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "model_type": "sparse_lora_finetuned",
        "eval_split": eval_name,
        "eval_metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "eval_samples": len(eval_dataset),
        "device": device_str,
    }

    output_file = os.path.join(args.model_path, "sparse_lora_validation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
