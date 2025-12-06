import argparse
import os
import torch
import random
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
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

    # ------------------------------------------------------------------
    # IMPROVED DATA LOADING (Fixes Catastrophic Forgetting)
    # ------------------------------------------------------------------
    
    # 1. Load your Style Data (PG19)
    style_dataset = load_dataset(
        "arrow", 
        data_files=f"{args.data_path}/*.arrow", 
        split="train", 
        streaming=True
    )
    
    # 2. Load a "Replay Buffer" (Normal Chat Data)
    # We mix in ~5% normal chat data so it remembers how to say "Hi"
    # 'imone/ChatEtiquette_ShareGPT' is a small, clean chat dataset
    chat_dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
    
    # Interleave: 90% Style Data, 10% Normal Chat Data
    from datasets import interleave_datasets
    combined_dataset = interleave_datasets([style_dataset, chat_dataset], probabilities=[0.9, 0.1], seed=42)
    
    # Shuffle buffer
    dataset = combined_dataset.shuffle(seed=42, buffer_size=100)

    # ------------------------------------------------------------------
    # IMPROVED PROCESSING (Fixes The "Ignore User" Bug)
    # ------------------------------------------------------------------
    CHARS_PER_SAMPLE = 2048 # Reduced to fit context better
    
    def process_batch(examples):
        batch_input_ids = []
        
        # Handle different dataset columns (Style data has 'text', Chat data has 'conversations')
        texts = examples.get('text', [])
        conversations = examples.get('conversations', [])
        
        # CASE A: It's a Book chunk (Style Training)
        for text in texts:
            if len(text) < 200: continue # Skip tiny fragments
            
            # Slice a chunk
            max_len = min(len(text), CHARS_PER_SAMPLE)
            text_chunk = text[:max_len]
            
            # SPLIT LOGIC: 
            # We split the text so the User provides the "Seed" and Assistant provides the "Style"
            # This forces the model to pay attention to the user input.
            split_idx = text_chunk.find(" ", 100) # Find a space after 100 chars
            if split_idx == -1: split_idx = 100
            
            user_context = text_chunk[:split_idx]
            assistant_completion = text_chunk[split_idx:]
            
            msgs = [
                # We lock the style to a SPECIFIC system prompt
                {"role": "system", "content": "You are a Victorian novelist. Continue the story in your archaic, formal style."},
                {"role": "user", "content": user_context}, 
                {"role": "assistant", "content": assistant_completion}
            ]
            batch_input_ids.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        
        # CASE B: It's a Normal Chat (Preservation Training)
        # We process the replay buffer data as-is to retain basic chat skills
        for conv in conversations:
            # conv is usually [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]
            # We map it to Qwen format
            msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            for turn in conv:
                role = "user" if turn['from'] == 'human' else "assistant"
                msgs.append({"role": role, "content": turn['value']})
            
            batch_input_ids.append(tokenizer.apply_chat_template(msgs, tokenize=False))

        # Tokenize
        tokenized = tokenizer(
            batch_input_ids,
            truncation=True,
            max_length=2048,
            padding=False,
            add_special_tokens=False 
        )
        return {"input_ids": tokenized["input_ids"]}

    # ENABLE BATCHED=TRUE
    train_dataset = dataset.map(process_batch, batched=True, batch_size=16, remove_columns=["text"])

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
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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