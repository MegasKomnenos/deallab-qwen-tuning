import kfp
from kfp import dsl
import kfp.kubernetes as kubernetes
from kfp.dsl import Input, Output, Artifact, Model, Dataset, Metrics

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BASE_IMAGE = "python:3.10"
# Image used for Merge/Inference (Needs torch/transformers)
WORKER_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

MOUNT_PATH_MODEL = "/mnt/models"
MOUNT_PATH_DATA = "/mnt/data"

# -------------------------------------------------------------------------
# COMPONENT 1 & 2: INGESTION (Unchanged, included for context)
# -------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["huggingface_hub"])
def download_model(model_name: str, model_root: str) -> str:
    import os
    from huggingface_hub import snapshot_download
    safe_name = model_name.replace("/", "--")
    save_path = os.path.join(model_root, "base_models", safe_name)
    snapshot_download(repo_id=model_name, local_dir=save_path, local_dir_use_symlinks=False)
    return save_path

@dsl.component(
    base_image=BASE_IMAGE,
    # STRICTLY pin datasets to 2.19.0 to allow legacy script execution for pg19
    packages_to_install=["datasets==2.19.0", "huggingface_hub", "scipy"] 
)
def download_dataset(
    dataset_name: str,
    data_root: str,
    force_download: bool = False,
) -> str:
    import os
    import shutil
    from datasets import load_dataset, Dataset as HFDataset

    # We save to "raw_pg19_large" to indicate this is the bulk cache
    save_path = os.path.join(data_root, "raw", "pg19_large_cache")
    
    # Check if cache exists
    if os.path.exists(save_path):
        if force_download:
            shutil.rmtree(save_path)
        else:
            print(f"Large cached dataset found at {save_path}. Skipping download.")
            return save_path
    
    # trust_remote_code=True is REQUIRED for pg19 because it uses a python loading script
    ds = load_dataset(
        dataset_name, 
        split="train", 
        streaming=True, 
        trust_remote_code=True 
    )
    
    data_list = []
    count = 0
    # Download a chunk
    for item in ds:
        if len(item['text']) > 5000: 
            data_list.append({"text": item['text']})
            count += 1
            
    print(f"Saving {count} books to disk...")
    final_ds = HFDataset.from_list(data_list)
    os.makedirs(save_path, exist_ok=True)
    final_ds.save_to_disk(save_path)
    
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 4: LAUNCH TRAINING JOB (Arguments matched to new train.py)
# -------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["kubernetes"])
def launch_training_job(
    base_model_path: str,
    data_path: str,
    model_root: str,
    image: str,
    model_pvc: str,
    data_pvc: str,
    max_steps: int = 1
) -> str:
    import time
    from kubernetes import client, config

    config.load_incluster_config()
    api = client.CustomObjectsApi()
    
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    job_name = f"qwen-finetune-{int(time.time())}"
    # Output dir must be inside the PVC mount
    output_dir_internal = f"/mnt/models/checkpoints/{job_name}"
    output_dir_external = f"{model_root}/checkpoints/{job_name}"

    # Arguments passed strictly to argparse in train.py
    cmd_args = [
        "--base_model_path", base_model_path,
        "--data_path", data_path,
        "--output_dir", output_dir_internal,
        "--max_steps", str(max_steps)
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
                        "metadata": {
                            "annotations": {
                                "sidecar.istio.io/inject": "false"
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": "pytorch",
                                "image": image,
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

    print(f"Submitting PyTorchJob {job_name}...")
    api.create_namespaced_custom_object("kubeflow.org", "v1", namespace, "pytorchjobs", manifest)

    while True:
        time.sleep(30)
        status = api.get_namespaced_custom_object_status("kubeflow.org", "v1", namespace, "pytorchjobs", job_name)
        conds = status.get("status", {}).get("conditions", [])
        if conds:
            last = conds[-1]
            if last["type"] == "Succeeded":
                break
            if last["type"] == "Failed":
                raise RuntimeError(f"Job Failed: {last.get('message')}")

    return output_dir_external

# -------------------------------------------------------------------------
# COMPONENT 5: MERGE ADAPTER (Optimized for Stability)
# -------------------------------------------------------------------------
@dsl.component(
    base_image=WORKER_IMAGE, 
    packages_to_install=["transformers", "torch", "peft", "accelerate", "bitsandbytes"]
)
def merge_adapter(base_model_path: str, adapter_path: str, merged_root: str) -> str:
    import torch
    import os
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"--- Merging Adapter ---")
    print(f"Base: {base_model_path}")
    print(f"Adapter: {adapter_path}")

    # 1. Load Base Model in FP16 on CPU
    # Why CPU? Loading a 4B/7B model in FP16 takes 8GB/15GB RAM. 
    # If we do this on an 11GB GPU, we might OOM before merging.
    # System RAM is usually cheaper and larger.
    print("Loading base model to System RAM (CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu", # Force CPU
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 2. Load Adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 3. Merge
    print("Merging weights...")
    model = model.merge_and_unload()

    # 4. Save
    save_path = os.path.join(merged_root, "qwen_merged_final")
    print(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    
    # Cleanup
    del model
    del base_model
    gc.collect()
    
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 6: INFERENCE (Optimized for 11GB VRAM)
# -------------------------------------------------------------------------
@dsl.component(
    base_image=WORKER_IMAGE, 
    packages_to_install=["transformers", "torch", "accelerate", "bitsandbytes"]
)
def run_inference(model_path: str, prompt: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading merged model from {model_path}...")

    # Load in 4-bit for inference to ensure it fits comfortably in 11GB
    # even with long context windows.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    messages = [
        {"role": "system", "content": "You are a scholar from the Victorian era. Write in an archaic, formal tone."},
        {"role": "user", "content": prompt}
    ]
    
    # Qwen chat template application
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Input Prompt: {text}")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("-" * 30)
    print("MODEL OUTPUT:")
    print("-" * 30)
    print(response)

# -------------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------------
@dsl.pipeline(name="qwen-finetune-production")
def llm_pipeline(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507", # Example model
    dataset_name: str = "deepmind/pg19",
    model_pvc: str = "llm-workspace-pvc",
    data_pvc: str = "llm-data-pvc",
    training_image_uri: str = "kjh123456/qwen-trainer:v13",
    force_download: bool = False,
    max_steps: int = 50,
):
    dl_model = download_model(model_name=model_name, model_root=MOUNT_PATH_MODEL)
    kubernetes.mount_pvc(dl_model, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)

    dl_data = download_dataset(dataset_name=dataset_name, data_root=MOUNT_PATH_DATA, force_download=force_download)
    kubernetes.mount_pvc(dl_data, pvc_name=data_pvc, mount_path=MOUNT_PATH_DATA)

    train_job = launch_training_job(
        base_model_path=dl_model.output,
        data_path=dl_data.output,
        model_root=MOUNT_PATH_MODEL,
        image=training_image_uri,
        model_pvc=model_pvc,
        data_pvc=data_pvc,
        max_steps=max_steps
    ).after(dl_data)

    merge = merge_adapter(
        base_model_path=dl_model.output,
        adapter_path=train_job.output,
        merged_root=MOUNT_PATH_MODEL
    )
    kubernetes.mount_pvc(merge, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    # Give merge step plenty of CPU RAM
    merge.set_memory_limit("24Gi") 

    inference = run_inference(
        model_path=merge.output,
        prompt="Explain the nature of electricity."
    )
    kubernetes.mount_pvc(inference, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    inference.set_gpu_limit(1)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(llm_pipeline, "qwen_pipeline_production.yaml")