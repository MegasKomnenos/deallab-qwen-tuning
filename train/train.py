import argparse
import os
import torch
import random
import bitsandbytes as bnb # Required for dequantization
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType

def apply_oplora(model, rank=16):
    """
    Applies OPLoRA (Orthogonal Projection LoRA).
    Computes SVD of frozen base weights and adds hooks to project LoRA inputs/outputs
    to the orthogonal complement of the top-k singular vectors.
    """
    print(f"--- Applying OPLoRA Projections (rank={rank}) ---")
    count = 0
    
    # Iterate through all modules to find active LoRA adapters
    for name, module in model.named_modules():
        # Check if the module is a LoRA layer (has lora_A/B attributes)
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            # Access the base layer (wrapped by PEFT)
            base = getattr(module, "base_layer", module)
            if not hasattr(base, "weight"): continue

            # 1. Dequantize Base Weights (4-bit -> Float32) for SVD
            # We need float32 for accurate decomposition
            if isinstance(base.weight, bnb.nn.Params4bit):
                w_data = bnb.functional.dequantize_4bit(
                    base.weight.data, base.weight.quant_state
                ).to(torch.float32)
            else:
                w_data = base.weight.data.to(torch.float32)

            # 2. Compute SVD (Low-Rank Approximation)
            # w_data shape: (out_features, in_features)
            # U: (out, k), V: (in, k) (Note: svd_lowrank returns V, not V.T)
            try:
                U, S, V = torch.svd_lowrank(w_data, q=rank, niter=2)
                
                # SAVE ARTIFACTS
                safe_name = name.replace(".", "_")
                os.makedirs(os.path.join(args.output_dir, "oplora_stats"), exist_ok=True)
                torch.save(U.cpu(), os.path.join(args.output_dir, "oplora_stats", f"{safe_name}_U.pt"))
                torch.save(V.cpu(), os.path.join(args.output_dir, "oplora_stats", f"{safe_name}_V.pt"))
            except Exception as e:
                print(f"Skipping OPLoRA for {name}: {e}")
                continue

            # Move projectors to device and cast to half precision for training speed
            dtype = torch.float16
            U_proj = U.to(base.weight.device).to(dtype) # Left Singular Vectors
            V_proj = V.to(base.weight.device).to(dtype) # Right Singular Vectors

            # 3. Define Projection Hooks
            # P_R = I - V V^T (Project Input to Null(V^T))
            def input_hook(V_ref):
                def hook(module, args):
                    x = args[0] # (Batch, Seq, In)
                    orig_type = x.dtype
                    x_c = x.to(V_ref.dtype)
                    # Project: x - (x @ V) @ V.T
                    proj = torch.matmul(torch.matmul(x_c, V_ref), V_ref.t())
                    return ((x_c - proj).to(orig_type),) + args[1:]
                return hook

            # P_L = I - U U^T (Project Output to Null(U^T))
            def output_hook(U_ref):
                def hook(module, args, output): # Output: (Batch, Seq, Out)
                    orig_type = output.dtype
                    out_c = output.to(U_ref.dtype)
                    # Project: y - (y @ U) @ U.T
                    proj = torch.matmul(torch.matmul(out_c, U_ref), U_ref.t())
                    return (out_c - proj).to(orig_type)
                return hook

            # 4. Register Hooks
            # Apply P_R before lora_A and P_L after lora_B
            module.lora_A["default"].register_forward_pre_hook(input_hook(V_proj))
            module.lora_B["default"].register_forward_hook(output_hook(U_proj))
            
            count += 1
            del w_data, U, S, V

    torch.cuda.empty_cache()
    print(f"OPLoRA setup complete. Applied to {count} layers.")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--playback_path", type=str, required=True)
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
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", 
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print(f"Streaming data from {args.data_path}...")

    # ------------------------------------------------------------------
    # IMPROVED DATA LOADING (Fixes Catastrophic Forgetting)
    # ------------------------------------------------------------------
    
    style_dataset = load_from_disk(args.data_path).to_iterable_dataset()
    chat_dataset = load_from_disk(args.playback_path).to_iterable_dataset()
    
    # Interleave: 90% Style Data, 10% Normal Chat Data
    from datasets import interleave_datasets
    combined_dataset = interleave_datasets([style_dataset, chat_dataset], probabilities=[0.9, 0.1])
    
    dataset = combined_dataset.shuffle(buffer_size=10000)
    
    def process_batch(items):
        batch_input_ids = []
        batch_labels = []  # <--- New list for labels
        
        for i in range(len(items["messages"])):
            msgs = []
            
            if items["type"][i] == "style":
                msgs.append({
                    "role": "system", 
                    "content": "You are a Victorian novelist. Continue the story in your archaic, formal style."
                })
            else:
                msgs.append({
                    "role": "system", 
                    "content": "You are a helpful chatbot."
                })
                
            msgs.extend(items["messages"][i]) 
            prompt_msgs = msgs[:-1]
            
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False)
            
            tokenized_full = tokenizer(
                full_text, 
                truncation=True, 
                max_length=2048,
                add_special_tokens=False
            )["input_ids"]
            
            tokenized_prompt = tokenizer(
                prompt_text, 
                truncation=True, 
                max_length=2048,
                add_special_tokens=False
            )["input_ids"]
            
            labels = list(tokenized_full)
            
            prompt_len = len(tokenized_prompt)
            
            # Mask the prompt tokens (System + User) so the model isn't trained on them
            for j in range(prompt_len):
                if j < len(labels): # Safety check
                    labels[j] = -100
            
            batch_input_ids.append(tokenized_full)
            batch_labels.append(labels)

        # Return both inputs and labels
        return {
            "input_ids": batch_input_ids, 
            "labels": batch_labels
        }

    # ENABLE BATCHED=TRUE
    train_dataset = dataset.map(process_batch, batched=True, batch_size=32, remove_columns=["messages", "type"])
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
    
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator
    )

    # --- APPLY OPLoRA TRANSFORMATION ---
    # We call this here to inject the hooks into the trainer's model 
    # before training begins. Rank 16 matches the LoRA rank.
    apply_oplora(trainer.model, rank=16)
    # -----------------------------------

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