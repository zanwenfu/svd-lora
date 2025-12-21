"""
测试模型推理速度
Test model inference speed with multiple runs
"""

import os
# 设置CUDA路径（必须在导入torch之前）
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
from torch.utils.data import DataLoader
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="测试模型推理速度")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["baseline", "lora", "adaptive_lora"],
                        help="模型类型：baseline、lora或adaptive_lora")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="推理批次大小")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--num_samples", type=int, default=872,
                        help="测试样本数（默认：完整验证集872个样本）")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="推理运行次数（用于取平均）")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="预热运行次数")
    parser.add_argument("--gpu", type=int, default=0,
                        help="使用的GPU编号")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="结果保存目录")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb"],
                        help="数据集名称 (sst2或imdb)")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, gpu_id):
    """加载模型和tokenizer"""
    # 设置GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda")
        print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")
    
    # 加载tokenizer和模型
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForSequenceClassification.from_pretrained(model_path)
    from hira import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(lora_config.base_model_name_or_path)
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, tokenizer, device


def prepare_dataloader(tokenizer, max_length, batch_size, num_samples, dataset_name="sst2"):
    """准备数据加载器"""
    if dataset_name == "sst2":
        print("正在加载SST-2验证集...")
        dataset = load_dataset("glue", "sst2")
        text_column = "sentence"
        remove_columns = ["sentence", "idx", "label"]
        split = "validation"
    elif dataset_name == "imdb":
        print("正在加载IMDB测试集...")
        dataset = load_dataset("stanfordnlp/imdb")
        text_column = "text"
        remove_columns = ["text", "label"]
        split = "test"
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
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
    
    # 限制样本数量
    if num_samples < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.select(range(num_samples))
    
    print(f"使用 {len(tokenized_dataset)} 个样本进行推理速度测试")
    
    # 创建DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    return dataloader, len(tokenized_dataset)


def run_inference(model, dataloader, device):
    """运行一次完整的推理"""
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移到设备
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            
            # 推理
            outputs = model(**inputs)
            
            total_samples += inputs["input_ids"].size(0)
    
    return total_samples


def benchmark_inference(model, dataloader, device, num_runs, warmup_runs):
    """基准测试推理速度"""
    print("\n" + "="*60)
    print("开始推理速度测试")
    print("="*60)
    
    # 预热阶段
    print(f"\n预热阶段: 运行 {warmup_runs} 次...")
    for i in range(warmup_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = run_inference(model, dataloader, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  预热 {i+1}/{warmup_runs} 完成")
    
    # 正式测试
    print(f"\n正式测试: 运行 {num_runs} 次...")
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
        print(f"  运行 {i+1}/{num_runs}: {elapsed_time:.4f}秒, 吞吐量: {throughput:.2f} 样本/秒")
    
    return inference_times, num_samples


def calculate_statistics(inference_times):
    """计算统计信息"""
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
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"模型推理速度基准测试")
    print("="*70)
    print(f"模型类型: {args.model_type}")
    print(f"模型路径: {args.model_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"测试样本数: {args.num_samples}")
    print(f"运行次数: {args.num_runs}")
    print(f"预热次数: {args.warmup_runs}")
    
    # 加载模型
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.gpu)
    
    # 准备数据
    dataloader, actual_samples = prepare_dataloader(
        tokenizer, args.max_length, args.batch_size, args.num_samples, args.dataset
    )
    
    # 运行基准测试
    inference_times, num_samples = benchmark_inference(
        model, dataloader, device, args.num_runs, args.warmup_runs
    )
    
    # 计算统计信息
    stats = calculate_statistics(inference_times)
    
    # 显示结果
    print("\n" + "="*70)
    print("推理速度测试结果")
    print("="*70)
    print(f"\n样本数量: {num_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"运行次数: {args.num_runs}")
    print(f"\n时间统计 (秒):")
    print(f"  平均值: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    
    avg_throughput = num_samples / stats['mean']
    print(f"\n吞吐量:")
    print(f"  平均: {avg_throughput:.2f} 样本/秒")
    print(f"  平均每样本: {stats['mean']/num_samples*1000:.2f} 毫秒")
    
    if torch.cuda.is_available():
        print(f"\nGPU信息:")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存使用: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    # 保存结果
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
    
    print(f"\n结果已保存到: {output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

