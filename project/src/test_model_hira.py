"""
Evaluate the trained model on the test set
"""

import os

from hira.peft_model import PeftModel
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

from test_model import parse_args, load_and_preprocess_data, compute_metrics
from hira import LoraConfig, get_peft_model

def load_hira_trained_model(output_dir):
    """
    output_dir = directory passed to model.save_pretrained(output_dir)
    """
    # Load LoRA config that was saved automatically
    lora_config = LoraConfig.from_pretrained(output_dir)

    # Load the base model (HiRA stores base_model_name_or_path inside config)
    base_model_name = lora_config.base_model_name_or_path
    print("Loading base model:", base_model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(output_dir)

    # Apply LoRA modules
    print("Applying HiRA LoRA modules...")
    model = get_peft_model(base_model, lora_config)

    print("Finished loading trained HiRA model.")
    return model

def load_hira_trained_model1(output_dir):
    # Correctly load LoRA + DistilBERT model
    # base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    # lora_config = LoraConfig.from_pretrained(args.model_path)
    # model = get_peft_model(base_model, lora_config)
    # model.to(device)

    lora_config = LoraConfig.from_pretrained(output_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(lora_config.base_model_name_or_path)
    model = get_peft_model(base_model, lora_config)
    weights_file = os.path.join(output_dir, "adapter_model.bin")
    state = torch.load(weights_file)
    model.load_state_dict(state, strict=False)
    return model

    # <
def load_hira_trained_model2(output_dir):
    from hira.peft_model import PeftModelForSequenceClassification
    lora_config = LoraConfig.from_pretrained(output_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(lora_config.base_model_name_or_path)
    model = PeftModelForSequenceClassification.from_pretrained(base_model, output_dir)
    return model

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
    
    # Load model and tokenizer
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # # Correctly load LoRA + DistilBERT model
    # from hira import LoraConfig, get_peft_model, TaskType
    # # base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    # # lora_config = LoraConfig.from_pretrained(args.model_path)
    # # model = get_peft_model(base_model, lora_config)
    # # model.to(device)

    # lora_config = LoraConfig.from_pretrained(args.model_path)
    # base_model = AutoModelForSequenceClassification.from_pretrained(lora_config.base_model_name_or_path)
    # model = get_peft_model(base_model, lora_config)
    # weights_file = os.path.join(args.model_path, "adapter_model.bin")
    # state = torch.load(weights_file)
    # model.load_state_dict(state, strict=False)
    # # <
    model = load_hira_trained_model2(args.model_path)
    
    

    # # >
    # from transformers import AutoModelForSequenceClassification
    # from peft import PeftConfig, PeftModel

    # peft_config = PeftConfig.from_pretrained(args.model_path)   # reads adapter metadata
    # base_model = AutoModelForSequenceClassification.from_pretrained(
    #     peft_config.base_model_name_or_path,
    #     device_map="auto",            # optional
    #     torch_dtype=torch.float16,    # optional, if you want fp16
    #     trust_remote_code=True        # optional if model needs it
    # )
    # model = PeftModel.from_pretrained(base_model, args.model_path, torch_dtype=torch.float16)
    # # <
    
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
        output_dir="./test_results-hira",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        fp16=torch.cuda.is_available(),
    )
    
    # from customized_trainer.customized_trainer import SeqClassificationTrainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    model.to(device)
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
    print("Validation Set Evaluation Results")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:     {metrics['eval_accuracy']:.4f}")
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

