import torch
import mlflow
import argparse
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning")
    
    # 数据和模型
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data (.jsonl)")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name (e.g., tatsu-lab/alpaca)")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (None = use all)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Base model name")
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    return parser.parse_args()


def format_example(example):
    """格式化 Alpaca 数据"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '').strip()
    output = example.get('output', '')
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    example['text'] = prompt + output
    return example


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize 数据"""
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    result['labels'] = result['input_ids']
    return result


def load_jsonl_data(data_path):
    """加载 JSONL 数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


class StatusCallback(TrainerCallback):
    """自定义回调：更新训练状态"""
    def __init__(self, status_dir, run_id):
        self.status_file = Path(status_dir) / f"{run_id}.json"
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时更新状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                
                # 更新进度
                if state.max_steps > 0:
                    status['progress'] = int((state.global_step / state.max_steps) * 100)
                
                # 添加最新日志
                if logs:
                    log_entry = {
                        'step': state.global_step,
                        'loss': logs.get('loss'),
                        'learning_rate': logs.get('learning_rate'),
                    }
                    status.setdefault('logs', []).append(log_entry)
                    # 只保留最近100条日志
                    status['logs'] = status['logs'][-100:]
                
                with open(self.status_file, 'w') as f:
                    json.dump(status, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to update status: {e}")


def main():
    args = parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir or f"./outputs/{args.run_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # MLflow 设置 - 使用绝对路径
    mlflow_dir = Path("./mlruns").resolve()
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment("mini_llm_studio")
    
    print(f"📊 MLflow tracking URI: {mlflow_dir}")
    print(f"📊 MLflow experiment: mini_llm_studio")
    
    with mlflow.start_run(run_name=args.run_id):
        print("🚀 开始 QLoRA 微调...")
        print(f"Run ID: {args.run_id}")
        print(f"Base Model: {args.base_model}")
        print(f"Data Path: {args.data_path}")
        
        # ============ 1. 加载 Tokenizer & Model ============
        print("\n📦 加载 tokenizer 和模型...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # ============ 2. 应用 QLoRA 配置 ============
        print("\n⚙️ 配置 QLoRA...")
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # ============ 3. 准备数据集 ============
        print("\n📚 准备数据集...")
        
        # 加载数据：支持两种方式
        if args.dataset_name:
            # 方式1：从 HuggingFace 加载
            print(f"从 HuggingFace 加载数据集: {args.dataset_name}")
            dataset_dict = load_dataset(args.dataset_name)
            dataset = dataset_dict[args.dataset_split]
        elif args.data_path:
            # 方式2：从本地 JSONL 文件加载
            print(f"从本地文件加载数据集: {args.data_path}")
            if args.data_path.endswith('.jsonl'):
                dataset = load_jsonl_data(args.data_path)
            else:
                raise ValueError("Only .jsonl format is supported")
        else:
            raise ValueError("Must provide either --dataset_name or --data_path")
        
        # 如果指定了样本数，进行采样
        if args.num_samples and args.num_samples < len(dataset):
            print(f"采样 {args.num_samples} 个样本")
            dataset = dataset.select(range(args.num_samples))
        
        print(f"加载了 {len(dataset)} 个样本")
        
        # 格式化数据
        dataset = dataset.map(format_example)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # ============ 4. 训练配置 ============
        print("\n🎯 配置训练参数...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            optim="paged_adamw_8bit",
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="cosine",
            report_to="none",
            logging_dir=f"{output_dir}/logs",
        )
        
        # 创建 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[StatusCallback("./data/status", args.run_id)]
        )
        
        # ============ 5. 开始训练 ============
        print("\n🏋️ 开始训练...")
        train_result = trainer.train()
        
        # ============ 6. 记录到 MLflow ============
        print("\n📊 记录到 MLflow...")
        
        mlflow.log_params({
            "run_id": args.run_id,
            "base_model": args.base_model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
        })
        
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_samples": len(tokenized_dataset),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        
        # 保存模型
        print("\n💾 保存模型...")
        final_model_dir = f"{output_dir}/final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        mlflow.log_artifact(final_model_dir)
        
        # 更新最终状态
        status_file = Path("./data/status") / f"{args.run_id}.json"
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            status['status'] = 'completed'
            status['progress'] = 100
            status['final_loss'] = train_result.training_loss
            status['model_path'] = final_model_dir
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        
        print("\n✅ 训练完成！")
        print(f"模型保存在: {final_model_dir}")
        print(f"训练损失: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 更新状态为失败
        import sys
        if len(sys.argv) > 1:
            for arg in sys.argv:
                if arg.startswith('--run_id='):
                    run_id = arg.split('=')[1]
                    status_file = Path("./data/status") / f"{run_id}.json"
                    if status_file.exists():
                        with open(status_file, 'r') as f:
                            status = json.load(f)
                        status['status'] = 'failed'
                        status['error'] = str(e)
                        with open(status_file, 'w') as f:
                            json.dump(status, f, indent=2)
        raise