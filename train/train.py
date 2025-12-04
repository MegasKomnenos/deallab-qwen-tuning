import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
# UPDATED IMPORTS: SFTConfig is now required
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model in PVC")
    parser.add_argument("--data_path", type=str, required=True, help="Path to pre-processed dataset in PVC")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print(f"--- Starting Training Job ---")
    print(f"Base Model: {args.base_model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")

    # --- VRAM Optimization: 4-bit Loading ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model with quantization...")
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print(f"Loading pre-tokenized dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Training Arguments (Now SFTConfig) ---
    # SFTConfig inherits from TrainingArguments, so we use it for everything.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        remove_unused_columns=False,
        
        # --- MOVED ARGS (Required for new TRL versions) ---
        max_seq_length=2048,
        packing=False,
        dataset_text_field="input_ids", # Technically ignored if input_ids exist, but good practice
        dataset_kwargs={
            "skip_prepare_dataset": True # Tells SFTTrainer: "Trust me, I already tokenized it"
        }
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args
        # NOTE: max_seq_length and packing are REMOVED from here
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()