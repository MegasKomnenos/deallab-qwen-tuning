# src/train.py
import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    TrainingArguments, set_seed, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_from_disk

def train():
    parser = argparse.ArgumentParser()
    # Kubernetes will pass these args to the container
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # --- OOM Optimization: 4-bit Loading ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- LoRA Config ---
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # --- Training ---
    dataset = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1, # Low batch size for VRAM
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
        num_train_epochs=args.epochs,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        # DataCollator ensures tensors are formatted correctly for PyTorch
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    train()
