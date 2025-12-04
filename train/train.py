import argparse
import os
import torch
import random
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator
)
from trl import SFTTrainer, SFTConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=50) 
    args = parser.parse_args()

    print(f"--- Starting Standard Streaming Job (v10) ---")

    # ------------------------------------------------------------------
    # 1. SETUP MODEL & TOKENIZER
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

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

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", 
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print(f"Streaming data from {args.data_path}...")
    
    # A. Load: We point to the Arrow files directly to enable native streaming
    # This works even if the data was saved via save_to_disk
    dataset = load_dataset(
        "arrow", 
        data_files=f"{args.data_path}/*.arrow", 
        split="train", 
        streaming=True
    )
    
    dataset = dataset.repeat(10)

    # B. Shuffle: Library implementation of "Wide Sampling"
    # It fills a buffer of 10k items and samples randomly from it.
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    # C. Transform: The "Shallow Slice" logic
    # We define the logic, but .map() handles the execution optimization
    CHARS_PER_SAMPLE = 12000
    SYSTEM_PROMPT = ""

    def process_on_the_fly(sample):
        text = sample['text']
        
        # 1. Random Slicing
        if len(text) > CHARS_PER_SAMPLE:
            max_start = len(text) - CHARS_PER_SAMPLE
            start = random.randint(0, max_start)
            text = text[start : start + CHARS_PER_SAMPLE]
            
        # 2. Templating & Tokenizing
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Write a passage in your natural style."},
            {"role": "assistant", "content": text}
        ]
        
        # Return input_ids directly
        return {
            "input_ids": tokenizer.apply_chat_template(
                messages, tokenize=True, truncation=True, max_length=2048
            )
        }

    # Apply the map. This does NOT process data yet. It only sets up the pipe.
    train_dataset = dataset.map(process_on_the_fly, remove_columns=["text"])

    # ------------------------------------------------------------------
    # 3. TRAINING CONFIG
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,       
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no",
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        remove_unused_columns=False,
        
        # TRL Configs
        max_length=2048,
        packing=False,
        dataset_text_field="input_ids",
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    response_template = "<|im_start|>assistant\n" 
    
    collator = DataCollatorForLanguageModeling(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator
    )

    print("Starting streaming training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Kill sidecar if present
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request("http://localhost:15020/quitquitquit", method="POST"))
    except: pass

if __name__ == "__main__":
    train()