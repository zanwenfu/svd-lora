"""
Evaluate the trained model on the test set
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
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--model_path", type=str, default="./results/final_model",
                        help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb"],
                        help="Dataset name (sst2 or imdb)")
    return parser.parse_args()


def load_and_preprocess_data(tokenizer, max_length=128, dataset_name="sst2"):
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
    
    return tokenized_dataset, dataset_name



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
        print("Using device: CPU (GPU not detected)")
    
    # Load model and tokenizer
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # Load dataset
    tokenized_dataset, dataset_name = load_and_preprocess_data(tokenizer, args.max_length, args.dataset)
    
    # SST-2 uses validation set (test set labels hidden), IMDB uses test set
    if dataset_name == "imdb":
        val_dataset = tokenized_dataset["test"]
        print(f"IMDB test set samples: {len(val_dataset)}")
    else:
        val_dataset = tokenized_dataset["validation"]
        print(f"SST-2 validation set samples: {len(val_dataset)}")
        print("Note: Since GLUE test set labels are hidden, we use validation set for evaluation")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create Trainer for evaluation
    training_args = TrainingArguments(
        output_dir="./test_results",
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
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("Starting evaluation on validation set...")
    print("="*60 + "\n")
    
    metrics = trainer.evaluate(eval_dataset=val_dataset)
    
    # Get predictions for confusion matrix and classification report
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Print results
    print("\n" + "="*60)
    print("Validation set evaluation results")
    print("="*60)
    print(f"\nOverall metrics:")
    print(f"  Accuracy:     {metrics['eval_accuracy']:.4f}")
    print(f"  Precision:    {metrics['eval_precision']:.4f}")
    print(f"  Recall:       {metrics['eval_recall']:.4f}")
    print(f"  F1-Score:     {metrics['eval_f1']:.4f}")
    
    print(f"\nNegative sentiment class (Negative):")
    print(f"  Precision: {metrics['eval_precision_negative']:.4f}")
    print(f"  Recall: {metrics['eval_recall_negative']:.4f}")
    print(f"  F1-Score: {metrics['eval_f1_negative']:.4f}")
    print(f"  Support: {metrics['eval_support_negative']}")
    
    print(f"\nPositive sentiment class (Positive):")
    print(f"  Precision: {metrics['eval_precision_positive']:.4f}")
    print(f"  Recall: {metrics['eval_recall_positive']:.4f}")
    print(f"  F1-Score: {metrics['eval_f1_positive']:.4f}")
    print(f"  Support: {metrics['eval_support_positive']}")
    
    print("\n" + "="*60)
    print("Confusion Matrix")
    print("="*60)
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n              Predicted")
    print(f"            Neg   Pos")
    print(f"Actual Neg   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"     Pos   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\n" + "="*60)
    print("Detailed classification report")
    print("="*60)
    target_names = ['Negative', 'Positive']
    print("\n" + classification_report(true_labels, pred_labels, target_names=target_names))
    
    # Save results to JSON file
    results = {
        'model_path': args.model_path,
        'model_type': 'lora_finetuned',
        'validation_metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'validation_samples': len(val_dataset),
        'gpu_used': args.gpu if torch.cuda.is_available() else 'cpu',
    }
    
    output_file = f"{args.model_path}/validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

