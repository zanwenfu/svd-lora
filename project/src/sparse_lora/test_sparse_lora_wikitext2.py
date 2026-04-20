#!/usr/bin/env python
"""
Evaluate a Sparse LoRA fine-tuned DistilBERT MLM model on WikiText-2.

Loads model from ./results_sparse_lora_wikitext2/final_sparse_lora_model by default,
computes loss and perplexity on the *test* split.
"""

import os
import argparse
import json
import math

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sparse LoRA MLM on WikiText-2")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results_sparse_lora_wikitext2/final_sparse_lora_model",
        help="Path to trained Sparse LoRA MLM model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Eval batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length (for chunking text)",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Masking probability for MLM",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index if using CUDA; ignored on Mac/CPU",
    )
    return parser.parse_args()


def load_and_preprocess_wikitext2(tokenizer, max_length=128):
    print("Loading WikiText-2 dataset (wikitext-2-raw-v1)...")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
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
        "Dataset sizes (blocks): "
        f"train={len(lm_datasets['train'])}, "
        f"validation={len(lm_datasets['validation'])}, "
        f"test={len(lm_datasets['test'])}"
    )

    return lm_datasets


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

    # Load tokenizer saved with the adapter
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load PEFT config to know which base model to use
    peft_config = PeftConfig.from_pretrained(args.model_path)

    # Load the base DistilBERT MLM model
    base_model = AutoModelForMaskedLM.from_pretrained(
        peft_config.base_model_name_or_path
    )

    # Attach the LoRA adapters from model_path
    model = PeftModel.from_pretrained(base_model, args.model_path)

    # Load dataset
    tokenized_dataset = load_and_preprocess_wikitext2(
        tokenizer, args.max_length
    )
    eval_dataset = tokenized_dataset["test"]
    print(f"Test blocks: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir="./test_results_sparse_lora_wikitext2",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("Evaluating Sparse LoRA MLM on WikiText-2 (test)")
    print("=" * 60 + "\n")

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    eval_loss = eval_metrics["eval_loss"]
    eval_runtime = eval_metrics.get("eval_runtime", None)

    # ---------- [LATENCY] tokens/sec and ms/token ----------
    tokens_per_block = args.max_length
    total_tokens = len(eval_dataset) * tokens_per_block

    if eval_runtime is not None and total_tokens > 0:
        tokens_per_second = total_tokens / eval_runtime
        ms_per_token = (eval_runtime / total_tokens) * 1000.0

        eval_metrics["tokens_per_second"] = tokens_per_second
        eval_metrics["ms_per_token"] = ms_per_token
        eval_metrics["total_tokens"] = total_tokens

        print(f"Inference runtime: {eval_runtime:.4f} s for {total_tokens} tokens")
        print(f"Tokens / second: {tokens_per_second:.2f}")
        print(f"Latency / token: {ms_per_token:.4f} ms")
    else:
        print("Warning: eval_runtime not found; latency not computed.")
    # -------------------------------------------------------

    perplexity = math.exp(eval_loss)

    print(f"Test loss: {eval_loss:.4f}")
    print(f"Test perplexity: {perplexity:.4f}")

    # Save JSON results next to the model
    results = {
        "dataset": "wikitext-2-raw-v1",
        "model_path": args.model_path,
        "model_type": "sparse_lora_mlm_finetuned",
        "eval_split": "test",
        "eval_metrics": {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            **{k: v for k, v in eval_metrics.items() if k not in ["eval_loss"]},
        },
        "eval_samples": len(eval_dataset),
        "device": device_str,
    }

    output_file = os.path.join(args.model_path, "sparse_lora_wikitext2_test_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    main()
