"""
Test model inference speed
Test model inference speed with multiple runs
"""

import os
# Set CUDA path (must be before importing torch)
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_PATH'] = '/usr/local/cuda'
os.environ['PATH'] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import numpy as np
import time
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from peft import AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Test model inference speed")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model path")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["baseline", "lora", "adaptive_lora", "sparse_lora"],
                        help="Model type: baseline, lora, or adaptive_lora")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--num_samples", type=int, default=872,
                        help="Number of test samples (default: 872 samples from full validation set)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of inference runs (for averaging)")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb"],
                        help="Dataset name (sst2 or imdb)")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, gpu_id, model_type):
    """Load model and tokenizer"""
    # Set GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda")
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print(f"Loading model: {model_path}")

    # For sparse LoRA, use PEFT's AutoPeftModel to load adapter
    if model_type == "sparse_lora":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_path)
    else:
        # baseline / lora / adaptive_lora: Use full HF model directly
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return model, tokenizer, device


def prepare_dataloader(tokenizer, max_length, batch_size, num_samples, dataset_name="sst2"):
    """Prepare data loader"""
    if dataset_name == "sst2":
        print("Loading SST-2 validation set...")
        dataset = load_dataset("glue", "sst2")
        text_column = "sentence"
        remove_columns = ["sentence", "idx", "label"]
        split = "validation"
    elif dataset_name == "imdb":
        print("Loading IMDB test set...")
        dataset = load_dataset("stanfordnlp/imdb")
        text_column = "text"
        remove_columns = ["text", "label"]
        split = "test"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def preprocess_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    tokenized_dataset = dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )
    
    # Limit number of samples
    if num_samples < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.select(range(num_samples))
    
    print(f"Using {len(tokenized_dataset)} samples for inference speed test")
    
    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    return dataloader, len(tokenized_dataset)


def run_inference(model, dataloader, device):
    """Run a full inference"""
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            
            # Inference
            outputs = model(**inputs)
            
            total_samples += inputs["input_ids"].size(0)
    
    return total_samples


def benchmark_inference(model, dataloader, device, num_runs, warmup_runs):
    """Benchmark inference speed"""
    print("\n" + "="*60)
    print("Start inference speed test")
    print("="*60)
    
    # Warmup phase
    print(f"\nWarmup phase: Running {warmup_runs} times...")
    for i in range(warmup_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = run_inference(model, dataloader, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{warmup_runs} completed")
    
    # Formal test
    print(f"\nFormal test: Running {num_runs} times...")
    inference_times = []
    
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        num_samples = run_inference(model, dataloader, device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        inference_times.append(elapsed_time)
        
        throughput = num_samples / elapsed_time
        print(f"  Run {i+1}/{num_runs}: {elapsed_time:.4f}s, Throughput: {throughput:.2f} samples/s")
    
    return inference_times, num_samples


def calculate_statistics(inference_times):
    """Calculate statistics"""
    times_array = np.array(inference_times)
    
    stats = {
        'mean': float(np.mean(times_array)),
        'std': float(np.std(times_array)),
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array)),
        'median': float(np.median(times_array)),
        'all_times': inference_times,
    }
    
    return stats


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Model Inference Speed Benchmark")
    print("="*70)
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of test samples: {args.num_samples}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Number of warmup runs: {args.warmup_runs}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, args.gpu, args.model_type
    )
    
    # Prepare data
    dataloader, actual_samples = prepare_dataloader(
        tokenizer, args.max_length, args.batch_size, args.num_samples, args.dataset
    )
    
    # Run benchmark
    inference_times, num_samples = benchmark_inference(
        model, dataloader, device, args.num_runs, args.warmup_runs
    )
    
    # Calculate statistics
    stats = calculate_statistics(inference_times)
    
    # Display results
    print("\n" + "="*70)
    print("Inference Speed Test Results")
    print("="*70)
    print(f"\nNumber of samples: {num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of runs: {args.num_runs}")
    print(f"\nTime Statistics (seconds):")
    print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    
    avg_throughput = num_samples / stats['mean']
    print(f"\nThroughput:")
    print(f"  Average: {avg_throughput:.2f} samples/s")
    print(f"  Average per sample: {stats['mean']/num_samples*1000:.2f} ms")
    
    if torch.cuda.is_available():
        print(f"\nGPU Info:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'num_samples': num_samples,
        'batch_size': args.batch_size,
        'num_runs': args.num_runs,
        'warmup_runs': args.warmup_runs,
        'statistics': stats,
        'throughput': {
            'samples_per_second': avg_throughput,
            'ms_per_sample': stats['mean']/num_samples*1000,
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    if torch.cuda.is_available():
        results['gpu_info'] = {
            'name': torch.cuda.get_device_name(0),
            'max_memory_mb': torch.cuda.max_memory_allocated()/1024**2,
        }
    
    output_file = f"{args.output_dir}/{args.model_type}_inference_benchmark.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

