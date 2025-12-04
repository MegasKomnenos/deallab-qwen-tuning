import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator 
)
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print(f"--- Starting Training Job (v9) ---")
    
    # 1. Load Data
    print(f"Loading pre-tokenized dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Setup (4-bit QLoRA)
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

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. SFTConfig (The New Standard)
    # We consolidate ALL training arguments here.
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
        remove_unused_columns=False, # Critical for pre-tokenized data
        
        # TRL SPECIFIC ARGS (New API Locations)
        max_length=2048,           # Was max_seq_length
        packing=False,             # Moved to Config
        dataset_text_field="input_ids",
        dataset_kwargs={
            "skip_prepare_dataset": True # Tells SFTTrainer: "Don't touch my data, it's ready"
        }
    )

    print("Initializing SFTTrainer...")
    
    # 6. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        # API CHANGE FIX: 'tokenizer' is renamed to 'processing_class' in latest versions
        processing_class=tokenizer, 
        # Since we pre-padded in preprocessing, use default collator to simply stack tensors
        data_collator=default_data_collator 
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()