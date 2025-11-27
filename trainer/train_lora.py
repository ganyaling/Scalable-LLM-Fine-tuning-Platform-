import mlflow
import torch
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#dataset_id = "tatsu-lab/alpaca" # Although not officially uploaded, this ID or variants are commonly used by the community

#try:
    # Attempt to load
    #dataset = load_dataset(dataset_id)
    #print("Dataset loaded successfully! Structure:")
    #print(dataset)
    
    # Print the first sample to check the format
    #print("\nFirst sample:")
    #print(dataset["train"][0])

#except Exception as e:
    #print(f"Failed to load dataset, it might be due to an invalid ID or special configuration requirements. Error: {e}")


def format_example(example):
    """Format Alpaca data"""
    instruction = example['instruction']
    input_text = example.get('input', '').strip()
    output = example['output']
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    example['text'] = prompt + output
    return example

#def format_example(example):
    #"""Chat 格式"""
    #instruction = example['instruction']
    #input_text = example.get('input', '').strip()
    #output = example['output']
    
    # 构建用户消息
   # if input_text:
    #    user_message = f"{instruction}\n{input_text}"
    #else:
    #    user_message = instruction
    
    # Chat 模板
    #text = f"<|user|>\n{user_message}\n<|assistant|>\n{output}"
    
    #example['text'] = text
    #return example

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize 数据"""
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    # 直接复制 input_ids 作为 labels（而不是 .copy()）
    result['labels'] = result['input_ids']
    return result

def main():
    base_model_name = "Qwen/Qwen2-1.5B-Instruct"
    
    # MLflow 设置
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mini-llm-studio")
    
    with mlflow.start_run():
        print("🚀 Starting QLoRA fine-tuning...")
        
        # ============ 1. loading Tokenizer & Model ============
        print("📦 Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # set pad_token（if not set）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # ============ 2. Apply QLoRA Configuration ============
        print("⚙️ Configuring QLoRA...")
        
        # Prepare model for kbit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,                      # LoRA rank
            lora_alpha=32,             # LoRA alpha
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2 的注意力层
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # ============ 3. prepare dataset ============
        print("📚 Preparing dataset...")
        
        # Load Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # Format data
        dataset = dataset.map(format_example)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        #  only use a subset for quick testing
        train_dataset = tokenized_dataset['train'].select(range(1000))
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # ============ 4. Training Configuration ============
        print("🎯 Configuring training parameters...")
        
        training_args = TrainingArguments(
            output_dir="./outputs",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            optim="paged_adamw_8bit",
            warmup_steps=50,
            lr_scheduler_type="cosine",
            report_to="none",  # Do not use other logging tools
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # ============ 5. Start Training ============
        print("🏋️ Starting training...")
        train_result = trainer.train()
        
        # ============ 6. Logging to MLflow ============
        print("📊 Logging to MLflow...")
        
        # Log parameters
        mlflow.log_params({
            "base_model": base_model_name,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        })
        
        # Record metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        
        # 保存模型
        print("💾 保存模型...")
        output_dir = "./outputs/final_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 记录模型到 MLflow（可选）
        mlflow.log_artifact(output_dir)
        
        print("✅ 训练完成！")
        print(f"模型保存在: {output_dir}")
        print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    main()