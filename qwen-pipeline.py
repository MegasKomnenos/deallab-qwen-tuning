import kfp
from kfp import dsl
import kfp.kubernetes as kubernetes
from kfp.dsl import Input, Output, Artifact, Model, Dataset, Metrics

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# Orchestrator image (lightweight)
BASE_IMAGE = "python:3.10"
# Inference/Merge image (needs libs installed, can use the training image or standard pytorch)
# Using the training image here ensures compatible libraries for merging.
WORKER_IMAGE = "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime" 

# Your Custom Training Image (Built from src/training/Dockerfile)
TRAINING_IMAGE_URI = "kjh123456/qwen-trainer:v1" 

# Paths inside the Pods (Must match PVC mounts)
MOUNT_PATH_MODEL = "/mnt/models"
MOUNT_PATH_DATA = "/mnt/data"

# -------------------------------------------------------------------------
# COMPONENT 1: MODEL INGESTION
# -------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["huggingface_hub"])
def download_model(model_name: str, model_root: str) -> str:
    import os
    from huggingface_hub import snapshot_download

    safe_name = model_name.replace("/", "--")
    save_path = os.path.join(model_root, "base_models", safe_name)
    
    print(f"Syncing model {model_name} to {save_path}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=save_path,
        local_dir_use_symlinks=False, 
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 2: DATA INGESTION
# -------------------------------------------------------------------------
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["datasets==2.19.0", "huggingface_hub", "scipy"]
)
def download_dataset(
    dataset_name: str,
    data_root: str,
    force_download: bool,
    # This is now a "Cap", not the training set size. 
    # Set this high (e.g., 3000) so you have plenty of data cached on disk.
    download_limit: int = 3000
) -> str:
    import os
    import shutil
    from datasets import load_dataset, Dataset as HFDataset

    # We save to "raw_pg19_large" to indicate this is the bulk cache
    save_path = os.path.join(data_root, "raw", "pg19_large_cache")
    
    if os.path.exists(save_path):
        if force_download:
            shutil.rmtree(save_path)
        else:
            print(f"Large cached dataset found at {save_path}. Skipping download.")
            return save_path

    print(f"Streaming {dataset_name} (Limit: {download_limit})...")
    
    ds = load_dataset(
        dataset_name, 
        split="train", 
        streaming=True, 
        trust_remote_code=True 
    )
    
    data_list = []
    count = 0
    # We download a large chunk now so we don't have to come back later
    for item in ds:
        if count >= download_limit: break
        if len(item['text']) > 1000: 
            data_list.append({"text": item['text']})
            count += 1
            
    print(f"Saving {count} books to disk...")
    final_ds = HFDataset.from_list(data_list)
    os.makedirs(save_path, exist_ok=True)
    final_ds.save_to_disk(save_path)
    
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 3: PREPROCESSING (Sliding Window)
# -------------------------------------------------------------------------
@dsl.component(
    base_image=BASE_IMAGE, 
    packages_to_install=["datasets", "transformers", "scipy", "accelerate"]
)
def preprocess_dataset(
    raw_data_path: str,
    processed_data_root: str,
    model_path: str,
    max_seq_length: int,
    subset_size: int = 1
) -> str:
    import os
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    print(f"Loading raw cache from {raw_data_path}...")
    dataset = load_from_disk(raw_data_path)
    
    # --- LOGIC MOVED HERE ---
    total_available = len(dataset)
    print(f"Total books available in cache: {total_available}")
    
    if subset_size > total_available:
        print(f"Warning: Requested {subset_size} books, but only {total_available} exist. Using all.")
        subset_size = total_available

    print(f"Selecting top {subset_size} books for this run...")
    # 'select' creates a lightweight view of the data without rewriting disk immediately
    dataset = dataset.select(range(subset_size))
    # ------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    SYSTEM_PROMPT = "You are a scholar from the Victorian era. Write in an archaic, formal tone."

    def process_batch(examples):
        conversations = []
        for text in examples['text']:
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Write a passage in your natural style."},
                {"role": "assistant", "content": text}
            ])

        formatted_texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in conversations]
        
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=256
        )
        return {
            "input_ids": tokenized["input_ids"], 
            "attention_mask": tokenized["attention_mask"], 
            "labels": tokenized["input_ids"].copy()
        }

    print("Tokenizing and chunking...")
    processed_ds = dataset.map(process_batch, batched=True, remove_columns=dataset.column_names)
    
    # Save to a unique path based on subset size to avoid collisions if you run parallel experiments
    save_path = os.path.join(processed_data_root, f"processed_train_{subset_size}_books")
    processed_ds.save_to_disk(save_path)
    
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 4: LAUNCH TRAINING JOB (The Clean Launcher)
# -------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["kubernetes"])
def launch_training_job(
    base_model_path: str,
    data_path: str,
    model_root: str, # Where to save checkpoints
    image: str,
    model_pvc: str,
    data_pvc: str,
    epochs: int = 1,
    job_prefix: str = "qwen-train"
) -> str:
    import time
    from kubernetes import client, config

    # 1. Setup
    config.load_incluster_config()
    api = client.CustomObjectsApi()
    
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    job_name = f"{job_prefix}-{int(time.time())}"
    output_dir = f"/mnt/models/checkpoints/{job_name}" # Internal path
    external_output = f"{model_root}/checkpoints/{job_name}" # Path for pipeline return

    # 2. Define Manifest
    # Note: We pass arguments that match src/training/train.py
    cmd_args = [
        "--base_model_path", base_model_path,
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--epochs", str(epochs)
    ]

    manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "pytorch",
                                "image": image, # Uses your pre-built image
                                "args": cmd_args,
                                "resources": {"limits": {"nvidia.com/gpu": 1, "memory": "16Gi"}},
                                "volumeMounts": [
                                    {"name": "models", "mountPath": "/mnt/models"},
                                    {"name": "data", "mountPath": "/mnt/data"},
                                    {"name": "dshm", "mountPath": "/dev/shm"}
                                ]
                            }],
                            "volumes": [
                                {"name": "models", "persistentVolumeClaim": {"claimName": model_pvc}},
                                {"name": "data", "persistentVolumeClaim": {"claimName": data_pvc}},
                                {"name": "dshm", "emptyDir": {"medium": "Memory"}}
                            ]
                        }
                    }
                }
            }
        }
    }

    # 3. Submit
    print(f"Submitting PyTorchJob {job_name} using image {image}...")
    api.create_namespaced_custom_object("kubeflow.org", "v1", namespace, "pytorchjobs", manifest)

    # 4. Monitor (Block until done)
    while True:
        time.sleep(30)
        status = api.get_namespaced_custom_object_status("kubeflow.org", "v1", namespace, "pytorchjobs", job_name)
        conds = status.get("status", {}).get("conditions", [])
        if conds:
            last = conds[-1]
            if last["type"] == "Succeeded":
                print("Training Succeeded.")
                break
            if last["type"] == "Failed":
                raise RuntimeError(f"Training Failed: {last.get('message')}")

    return external_output

# -------------------------------------------------------------------------
# COMPONENT 5: MERGE ADAPTER
# -------------------------------------------------------------------------
@dsl.component(base_image=WORKER_IMAGE, packages_to_install=["transformers", "torch", "peft", "accelerate", "bitsandbytes"])
def merge_adapter(base_model_path: str, adapter_path: str, merged_root: str) -> str:
    import torch
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading Base Model (CPU/FP16)...")
    # Merging usually requires loading the full model in FP16 or FP32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    
    print(f"Loading Adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Merging...")
    model = model.merge_and_unload()
    
    save_path = os.path.join(merged_root, "qwen_archaic_production")
    model.save_pretrained(save_path)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 6: INFERENCE
# -------------------------------------------------------------------------
@dsl.component(base_image=WORKER_IMAGE, packages_to_install=["transformers", "torch", "accelerate"])
def run_inference(model_path: str, prompt: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    messages = [
        {"role": "system", "content": "You are a scholar from the Victorian era."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=150)
    print("OUTPUT:\n" + tokenizer.decode(outputs[0], skip_special_tokens=True))

# -------------------------------------------------------------------------
# PIPELINE WIRING
# -------------------------------------------------------------------------
@dsl.pipeline(
    name="qwen-finetune-production",
    description="End-to-end Qwen finetuning with external Docker image for training."
)
def llm_pipeline(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
    dataset_name: str = "deepmind/pg19",
    model_pvc: str = "llm-workspace-pvc",
    data_pvc: str = "llm-data-pvc",
    test_prompt: str = "Explain the internet."
):
    # 1. Download
    dl_model = download_model(model_name=model_name, model_root=MOUNT_PATH_MODEL)
    kubernetes.mount_pvc(dl_model, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    dl_model.set_caching_options(True)

    dl_data = download_dataset(dataset_name=dataset_name, data_root=MOUNT_PATH_DATA, force_download=True)
    kubernetes.mount_pvc(dl_data, pvc_name=data_pvc, mount_path=MOUNT_PATH_DATA)
    dl_data.set_caching_options(True)

    # 2. Preprocess
    preprocess = preprocess_dataset(
        raw_data_path=dl_data.output,
        processed_data_root=MOUNT_PATH_DATA,
        model_path=dl_model.output,
        max_seq_length=2048
    )
    kubernetes.mount_pvc(preprocess, pvc_name=data_pvc, mount_path=MOUNT_PATH_DATA)
    kubernetes.mount_pvc(preprocess, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    preprocess.set_memory_limit("8Gi")

    # 3. Train (Using External Image)
    train_job = launch_training_job(
        base_model_path=dl_model.output,
        data_path=preprocess.output,
        model_root=MOUNT_PATH_MODEL,
        image=TRAINING_IMAGE_URI,
        model_pvc=model_pvc,
        data_pvc=data_pvc,
        epochs=1
    ).after(preprocess)
    # The launcher itself is light
    train_job.set_cpu_limit("500m")
    train_job.set_memory_limit("1Gi")

    # 4. Merge (High RAM)
    merge = merge_adapter(
        base_model_path=dl_model.output,
        adapter_path=train_job.output,
        merged_root=MOUNT_PATH_MODEL
    )
    kubernetes.mount_pvc(merge, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    merge.set_memory_limit("24Gi") # 4B params FP16 needs ~8GB, plus overhead for merge ops

    # 5. Inference (GPU)
    inference = run_inference(
        model_path=merge.output,
        prompt=test_prompt
    )
    kubernetes.mount_pvc(inference, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    inference.set_gpu_limit(1)

# -------------------------------------------------------------------------
# COMPILATION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(llm_pipeline, "qwen_pipeline_production.yaml")
    print("Compiled to qwen_pipeline_production.yaml")
