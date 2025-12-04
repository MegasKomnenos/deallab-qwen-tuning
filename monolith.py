# Filename: monolithic_train.py
import argparse
import os
import torch
import random
import shutil
import gc
import time
from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel

SYSTEM_PROMPT = ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset_name", type=str, default="deepmind/pg19")
    parser.add_argument("--model_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    start_time = time.time()

    # 1. DOWNLOAD MODEL
    print("\n--- STEP 1: Downloading Model ---")
    safe_name = args.model_name.replace("/", "--")
    base_model_path = os.path.join(args.model_root, "base_models", safe_name)
    if not os.path.exists(base_model_path) or args.force_download:
        if os.path.exists(base_model_path): shutil.rmtree(base_model_path)
        os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
        snapshot_download(repo_id=args.model_name, local_dir=base_model_path, local_dir_use_symlinks=False)

    # 2. DOWNLOAD DATASET (Limited)
    print(f"\n--- STEP 2: Preparing Dataset (Limited to {subset_size} books) ---")
    data_path = os.path.join(args.data_root, "raw", "pg19_large_cache")
    if not os.path.exists(data_path) or args.force_download:
        if os.path.exists(data_path): shutil.rmtree(data_path)
        ds = load_dataset(args.dataset_name, split="train", streaming=True, trust_remote_code=True)
        data_list = []
        count = 0
        for item in ds:
            if len(item['text']) > 5000:
                data_list.append({"text": item['text']})
                count += 1
                if count >= subset_size: break
        final_ds = HFDataset.from_list(data_list)
        final_ds.save_to_disk(data_path)

    # 3. TRAINING
    print("\n--- STEP 3: Training Model ---")
    # A. Setup Model & Tokenizer (Load to GPU)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # B. Data Preparation (Streaming)
    dataset = load_dataset("arrow", data_files=f"{data_path}/*.arrow", split="train", streaming=True)
    dataset = dataset.repeat(10)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    def process_on_the_fly(sample):
        text = sample['text']
        if len(text) > 12000:
            start = random.randint(0, len(text) - 12000)
            text = text[start : start + 12000]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Write a passage in your natural style."},
            {"role": "assistant", "content": text}
        ]
        return {"input_ids": tokenizer.apply_chat_template(messages, tokenize=True, truncation=True, max_length=2048)}

    train_dataset = dataset.map(process_on_the_fly, remove_columns=["text"])

    # C. Training
    job_name = f"qwen-mono-{int(time.time())}"
    output_dir = os.path.join(args.model_root, "checkpoints", job_name)

    training_args = SFTConfig(
        output_dir=output_dir, max_steps=args.max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4, fp16=True, logging_steps=50, optim="paged_adamw_32bit",
        remove_unused_columns=False, max_length=2048, dataset_text_field="input_ids", dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(model=model, train_dataset=train_dataset, peft_config=peft_config, args=training_args, tokenizer=tokenizer, data_collator=response_template)
    trainer.train()
    trainer.model.save_pretrained(output_dir)

    # CLEANUP GPU Memory
    adapter_path = output_dir
    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 4. MERGE ADAPTER (CPU)
    print("\n--- STEP 4: Merging Adapter (CPU) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    merged_model_path = os.path.join(args.model_root, "merged", job_name)
    model.save_pretrained(merged_model_path)
    AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True).save_pretrained(merged_model_path)

    # CLEANUP CPU Memory
    del model, base_model
    gc.collect()

    # 5. INFERENCE (GPU)
    print("\n--- STEP 5: Running Inference (GPU) ---")
    bnb_config_inf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, quantization_config=bnb_config_inf, device_map="auto", trust_remote_code=True
    )

    prompt = "Explain the nature of electricity."
    messages = [
        {"role": "system", "content": "You are a scholar from the Victorian era. Write in an archaic, formal tone."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)

    print("OUTPUT:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")

if __name__ == "__main__":
    main()