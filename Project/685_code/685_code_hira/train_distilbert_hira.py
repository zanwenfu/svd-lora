"""
DistilBERT + LoRA Fine-tuning
使用LoRA对DistilBERT进行微调，支持多个数据集
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
                        help="数据集选择：sst2、imdb或wikitext2")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="预训练模型名称")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="学习率")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存检查点的步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估的步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_and_preprocess_data(tokenizer, dataset_name="sst2", max_length=128):
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
        
        # IMDB没有validation集，从训练集分割出10%作为验证集
        print("从训练集分割出验证集（10%）...")
        train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = train_val_split["train"]
        dataset["validation"] = train_val_split["test"]
        print(f"训练集: {len(dataset['train'])}样本, 验证集: {len(dataset['validation'])}样本")
    elif dataset_name == "wikitext2":
        print("正在加载Wikitext-2数据集...")
        dataset = load_dataset("mindchain/wikitext2")
        text_column = "text"
        remove_columns = ["text"]
        # Wikitext-2有"train", "validation", "test"，直接使用
        print(f"训练集: {len(dataset['train'])}样本, 验证集: {len(dataset['validation'])}样本")
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    def preprocess_function(examples):
        """对文本进行tokenization"""
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # 使用DataCollator动态padding
        )
    
    print(f"正在对{dataset_name}数据集进行tokenization...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,  # 只删除指定列，保留label列
    )
    
    # 重命名label列为labels（模型期望的名称）
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset


def create_hira_model(model_name, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """创建带有HiRA的模型"""
    print(f"正在加载预训练模型: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r_ab=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT的attention层
    )
    
    print("正在应用LoRA配置...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

# def 
def compute_metrics(eval_pred):
    """计算评估指标"""
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
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 加载tokenizer
    print(f"正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 加载并预处理数据
    tokenized_dataset = load_and_preprocess_data(tokenizer, args.dataset, args.max_length)
    
    # 根据数据集调整输出目录
    if args.dataset != "sst2":
        args.output_dir = f"{args.output_dir}_{args.dataset}"
    
    # 创建LoRA模型
    if args.dataset == "wikitext2":
        model = None
    else:
        model = create_hira_model(
            args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
        # 数据整理器
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",  # 新版本使用eval_strategy而不是evaluation_strategy
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
        fp16=torch.cuda.is_available(),  # 如果有GPU则使用混合精度
        save_total_limit=2,  # 只保留最好的2个检查点
    )
    
    # 创建Trainer
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
    
    # 训练
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50 + "\n")
    train_result = trainer.train()
    
    # 保存模型
    print("\n正在保存模型...")
    trainer.save_model(f"{args.output_dir}/final_model")
    trainer.model.save_pretrained(f"{args.output_dir}/final_model")  # saves adapter_model.bin
    trainer.model.peft_config["default"].save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # 训练统计
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 在验证集上评估
    print("\n" + "="*50)
    print("在验证集上评估...")
    print("="*50 + "\n")
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    print("\n" + "="*50)
    print("训练完成！")
    print(f"模型保存在: {args.output_dir}/final_model")
    print("="*50)
    print("\n最终验证集结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print()
    print("注意: GLUE的SST-2测试集标签被隐藏，只能通过提交到官方服务器评估。")
    print("      验证集结果已经可以很好地反映模型性能。")
    print()


if __name__ == "__main__":
    main()

