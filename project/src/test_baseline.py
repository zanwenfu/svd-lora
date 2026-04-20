"""
Test the performance of the original untuned DistilBERT model on the test set
Test the original (untuned) DistilBERT model on test set
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
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Test original DistilBERT model")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb"],
                        help="Dataset selection: sst2 or imdb")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Model name (default: distilbert-base-uncased)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--output_dir", type=str, default="./baseline_results",
                        help="Directory to save results")
    return parser.parse_args()


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
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def preprocess_function(examples):
        """Tokenize text"""
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    print(f"Tokenizing {dataset_name} dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )
    
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset


def compute_metrics(eval_pred):
    """Compute detailed evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Compute metrics for each class
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_negative': precision_per_class[0],
        'precision_positive': precision_per_class[1],
        'recall_negative': recall_per_class[0],
        'recall_positive': recall_per_class[1],
        'f1_negative': f1_per_class[0],
        'f1_positive': f1_per_class[1],
        'support_negative': int(support[0]),
        'support_positive': int(support[1]),
    }


def main():
    args = parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device("cuda")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (No GPU detected)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original un-finetuned model
    print(f"\nLoading original model: {args.model_name}")
    print("Note: This is an un-finetuned pretrained model")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    
    # Load validation/test set
    tokenized_dataset = load_and_preprocess_data(tokenizer, args.dataset, args.max_length)
    
    # Select evaluation set based on dataset
    if args.dataset == "sst2":
        eval_dataset = tokenized_dataset["validation"]
        eval_name = "Validation Set"
        print(f"Note: GLUE SST-2 test set labels are hidden, using validation set for evaluation")
    else:  # imdb
        eval_dataset = tokenized_dataset["test"]
        eval_name = "Test Set"
    
    print(f"{eval_name} samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate on evaluation set
    print("\n" + "="*60)
    print(f"Starting evaluation of original model: {args.model_name} on {args.dataset.upper()}")
    print("="*60 + "\n")
    
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    
    # Get predictions for confusion matrix and classification report
    predictions = trainer.predict(eval_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Print results
    print("\n" + "="*60)
    print(f"Original model {eval_name} evaluation results ({args.model_name} on {args.dataset.upper()})")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:     {metrics['eval_accuracy']:.4f} ({metrics['eval_accuracy']*100:.2f}%)")
    print(f"  Precision:    {metrics['eval_precision']:.4f}")
    print(f"  Recall:       {metrics['eval_recall']:.4f}")
    print(f"  F1-Score:     {metrics['eval_f1']:.4f}")
    
    print(f"\nNegative Class:")
    print(f"  Precision: {metrics['eval_precision_negative']:.4f}")
    print(f"  Recall: {metrics['eval_recall_negative']:.4f}")
    print(f"  F1-Score: {metrics['eval_f1_negative']:.4f}")
    print(f"  Samples: {metrics['eval_support_negative']}")
    
    print(f"\nPositive Class:")
    print(f"  Precision: {metrics['eval_precision_positive']:.4f}")
    print(f"  Recall: {metrics['eval_recall_positive']:.4f}")
    print(f"  F1-Score: {metrics['eval_f1_positive']:.4f}")
    print(f"  Samples: {metrics['eval_support_positive']}")
    
    print("\n" + "="*60)
    print("Confusion Matrix")
    print("="*60)
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n              Predicted")
    print(f"            Neg   Pos")
    print(f"True Neg   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"     Pos   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\n" + "="*60)
    print("Detailed Classification Report")
    print("="*60)
    target_names = ['Negative', 'Positive']
    print("\n" + classification_report(true_labels, pred_labels, target_names=target_names))
    
    # Save results to JSON file
    results = {
        'dataset': args.dataset,
        'model_name': args.model_name,
        'model_type': 'original_pretrained',
        'eval_metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'eval_samples': len(eval_dataset),
        'gpu_used': args.gpu if torch.cuda.is_available() else 'cpu',
    }
    
    # Adjust output directory based on dataset
    if args.dataset != "sst2":
        args.output_dir = f"{args.output_dir}_{args.dataset}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = f"{args.output_dir}/baseline_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("="*60 + "\n")
    
    # Show summary
    print("="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model_name} (Original Un-finetuned)")
    print(f"{eval_name} Accuracy: {metrics['eval_accuracy']*100:.2f}%")
    print(f"This is baseline performance, LoRA finetuning should significantly improve this")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

