import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType
)
from trl import SFTTrainer

def train():
    # ------------------------------------------------------------------
    # 1. ARGS PARSING
    # Matches the arguments passed by 'launch_training_job' in pipeline
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. VRAM OPTIMIZATION: 4-BIT QUANTIZATION
    # This is critical for 11GB VRAM. It shrinks model footprint by 4x.
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 is optimal for LLMs
        bnb_4bit_compute_dtype=torch.float16, # Compute in FP16
        bnb_4bit_use_double_quant=True,      # Quantize the quantization constants
    )

    # ------------------------------------------------------------------
    # 3. MODEL LOADING
    # ------------------------------------------------------------------
    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True # Required for Qwen
    )

    # Enable gradient checkpointing to reduce memory usage during backward pass
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training (freezes layers, casts norms to fp32)
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 4. LORA CONFIGURATION (Low-Rank Adaptation)
    # We only train adapters, not the full model.
    # ------------------------------------------------------------------
    

    peft_config = LoraConfig(
        r=16,                  # Rank
        lora_alpha=32,         # Alpha (scaling)
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Targeted modules for Qwen/Llama architectures to maximize performance
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # ------------------------------------------------------------------
    # 5. DATASET LOADING
    # We load the dataset processed by your 'preprocess_dataset' step.
    # It already contains: input_ids, attention_mask, labels
    # ------------------------------------------------------------------
    print(f"Loading pre-tokenized dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)
    
    # Load tokenizer for saving purposes and padding reference
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 6. TRAINING ARGUMENTS
    # Tuned for Low VRAM (11GB)
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,      # STRICT: Keep at 1 for 11GB VRAM
        gradient_accumulation_steps=8,      # Simulates batch size of 8
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,                          # Use Mixed Precision
        logging_steps=10,
        save_strategy="epoch",              # Save only at end of epoch to save disk space
        evaluation_strategy="no",           # Skip eval for pure training runs
        optim="paged_adamw_32bit",          # Paged Optimizer: offloads to CPU RAM if GPU fills up
        max_grad_norm=0.3,                  # Gradient clipping
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",            # Log to local tensorboard (captured by Kubeflow logs)
        remove_unused_columns=False         # IMPORTANT: Keep 'input_ids' etc. from pre-processed data
    )

    # ------------------------------------------------------------------
    # 7. SFT TRAINER SETUP
    # ------------------------------------------------------------------
    print("Initializing SFTTrainer...")
    
    # note: Since your dataset is ALREADY tokenized (has input_ids), 
    # we do NOT pass 'dataset_text_field'. SFTTrainer will detect input_ids
    # and pass them through. We set packing=False to respect your sliding window.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,   # Must match preprocess step
        tokenizer=tokenizer,
        args=training_args,
        packing=False          # Data is already packed/chunked in preprocess
    )

    # ------------------------------------------------------------------
    # 8. EXECUTION
    # ------------------------------------------------------------------
    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    # This saves only the LoRA adapters, not the full 4-bit model
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()