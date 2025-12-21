"""
DistilBERT + LoRA Fine-tuning
Fine-tune DistilBERT with LoRA, supporting multiple datasets
"""

import os
# Set CUDA path (must be before importing torch)
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_PATH'] = '/usr/local/cuda'
os.environ['PATH'] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
# from peft import LoraConfig, get_peft_model, TaskType
from hira import LoraConfig, get_peft_model, TaskType
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with LoRA")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb", "wikitext2"],
                        help="Dataset selection: sst2, imdb or wikitext2")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Steps to save checkpoint")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Steps to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_and_preprocess_data(tokenizer, dataset_name="sst2", max_length=128):
    """Load and preprocess dataset"""
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
        
        # IMDB has no validation set, split 10% from training set as validation set
        print("Splitting validation set from training set (10%)...")
        train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = train_val_split["train"]
        dataset["validation"] = train_val_split["test"]
        print(f"Training set: {len(dataset['train'])} samples, Validation set: {len(dataset['validation'])} samples")
    elif dataset_name == "wikitext2":
        print("Loading Wikitext-2 dataset...")
        dataset = load_dataset("mindchain/wikitext2")
        text_column = "text"
        remove_columns = ["text"]
        # Wikitext-2 has "train", "validation", "test", use directly
        print(f"Training set: {len(dataset['train'])} samples, Validation set: {len(dataset['validation'])} samples")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def preprocess_function(examples):
        """Tokenize text"""
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # Use DataCollator for dynamic padding
        )
    
    print(f"Tokenizing {dataset_name} dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,  # Only remove specified columns, keep label column
    )
    
    # Rename label column to labels (expected by model)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset


def create_hira_model(model_name, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """Create model with HiRA"""
    print(f"Loading pretrained model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r_ab=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
    )
    
    print("Applying LoRA configuration...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

# def 
def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load and preprocess data
    tokenized_dataset = load_and_preprocess_data(tokenizer, args.dataset, args.max_length)
    
    # Adjust output directory based on dataset
    if args.dataset != "sst2":
        args.output_dir = f"{args.output_dir}_{args.dataset}"
    
    # Create LoRA model
    if args.dataset == "wikitext2":
        model = None
    else:
        model = create_hira_model(
            args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy in newer versions
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        report_to="tensorboard",
        seed=args.seed,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        save_total_limit=2,  # Keep only the best 2 checkpoints
    )
    
    # Create Trainer
    # from customized_trainer.customized_trainer import SeqClassificationTrainer
    # trainer = SeqClassificationTrainer(
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    train_result = trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(f"{args.output_dir}/final_model")
    trainer.model.save_pretrained(f"{args.output_dir}/final_model")  # saves adapter_model.bin
    trainer.model.peft_config["default"].save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # Training stats
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("Evaluating on validation set...")
    print("="*50 + "\n")
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}/final_model")
    print("="*50)
    print("\nFinal validation results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print()
    print("Note: GLUE SST-2 test set labels are hidden, can only be evaluated by submitting to official server.")
    print("      Validation set results are a good indicator of model performance.")
    print()


if __name__ == "__main__":
    main()

