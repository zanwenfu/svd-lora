"""
在测试集上评估训练好的模型
Evaluate the trained model on the test set
"""

import os
# 设置CUDA路径（必须在导入torch之前）
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
    parser = argparse.ArgumentParser(description="在测试集上评估模型")
    parser.add_argument("--model_path", type=str, default="./results/final_model",
                        help="训练好的模型路径")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="评估批次大小")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--gpu", type=int, default=0,
                        help="使用的GPU编号（默认: 0）")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb"],
                        help="数据集名称 (sst2或imdb)")
    return parser.parse_args()


def load_and_preprocess_data(tokenizer, max_length=128, dataset_name="sst2"):
    """加载并预处理数据集"""
    if dataset_name == "sst2":
        print("正在加载SST-2数据集...")
        dataset = load_dataset("glue", "sst2")
        text_column = "sentence"
        remove_columns = ["sentence", "idx"]
    elif dataset_name == "imdb":
        print("正在加载IMDB数据集...")
        dataset = load_dataset("stanfordnlp/imdb")
        text_column = "text"
        remove_columns = ["text"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    def preprocess_function(examples):
        """对文本进行tokenization"""
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    print(f"正在对{dataset_name}数据集进行tokenization...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )
    
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset, dataset_name


def compute_metrics(eval_pred):
    """计算详细的评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # 计算每个类别的指标
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
    
    # 设置GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device("cuda")
        print(f"使用GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU（未检测到GPU）")
    
    # 加载模型和tokenizer
    print(f"\n正在从 {args.model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # 加载数据集
    tokenized_dataset, dataset_name = load_and_preprocess_data(tokenizer, args.max_length, args.dataset)
    
    # SST-2用验证集（测试集标签隐藏），IMDB用测试集
    if dataset_name == "imdb":
        val_dataset = tokenized_dataset["test"]
        print(f"IMDB测试集样本数: {len(val_dataset)}")
    else:
        val_dataset = tokenized_dataset["validation"]
        print(f"SST-2验证集样本数: {len(val_dataset)}")
        print("注意：由于GLUE测试集标签被隐藏，我们使用验证集进行评估")
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 创建Trainer用于评估
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
    
    # 在验证集上评估
    print("\n" + "="*60)
    print("开始在验证集上评估...")
    print("="*60 + "\n")
    
    metrics = trainer.evaluate(eval_dataset=val_dataset)
    
    # 获取预测结果用于生成混淆矩阵和分类报告
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # 打印结果
    print("\n" + "="*60)
    print("验证集评估结果")
    print("="*60)
    print(f"\n整体指标:")
    print(f"  准确率 (Accuracy):     {metrics['eval_accuracy']:.4f}")
    print(f"  精确率 (Precision):    {metrics['eval_precision']:.4f}")
    print(f"  召回率 (Recall):       {metrics['eval_recall']:.4f}")
    print(f"  F1分数 (F1-Score):     {metrics['eval_f1']:.4f}")
    
    print(f"\n负面情感类别 (Negative):")
    print(f"  精确率: {metrics['eval_precision_negative']:.4f}")
    print(f"  召回率: {metrics['eval_recall_negative']:.4f}")
    print(f"  F1分数: {metrics['eval_f1_negative']:.4f}")
    print(f"  样本数: {metrics['eval_support_negative']}")
    
    print(f"\n正面情感类别 (Positive):")
    print(f"  精确率: {metrics['eval_precision_positive']:.4f}")
    print(f"  召回率: {metrics['eval_recall_positive']:.4f}")
    print(f"  F1分数: {metrics['eval_f1_positive']:.4f}")
    print(f"  样本数: {metrics['eval_support_positive']}")
    
    print("\n" + "="*60)
    print("混淆矩阵 (Confusion Matrix)")
    print("="*60)
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n              预测")
    print(f"            Neg   Pos")
    print(f"真实 Neg   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"     Pos   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\n" + "="*60)
    print("详细分类报告")
    print("="*60)
    target_names = ['Negative', 'Positive']
    print("\n" + classification_report(true_labels, pred_labels, target_names=target_names))
    
    # 保存结果到JSON文件
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
    
    print(f"\n结果已保存到: {output_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

