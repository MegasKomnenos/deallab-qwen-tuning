import kfp
from kfp import dsl
import kfp.kubernetes as kubernetes
from kfp.dsl import Input, Output, Artifact, Model, Dataset, Metrics

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BASE_IMAGE = "kjh123456/qwen-base:v2"
# Image used for Merge/Inference (Needs torch/transformers)
WORKER_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

MOUNT_PATH_MODEL = "/mnt/models"
MOUNT_PATH_DATA = "/mnt/data"
MOUNT_PATH_PLAYBACK = "/mnt/playback"

DATASET_NAME = "deepmind/pg19"
PLAYBACK_NAME = "OpenAssistant/oasst2"

# -------------------------------------------------------------------------
# COMPONENT 1 & 2: INGESTION (Unchanged)
# -------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["huggingface_hub"])
def download_model(model_name: str, model_root: str, force_download: bool) -> str:
    import os
    import shutil
    from huggingface_hub import snapshot_download
    import time
    start_time = time.time()
    safe_name = model_name.replace("/", "--")
    save_path = os.path.join(model_root, "base_models", safe_name)
    if os.path.exists(save_path) and force_download:
        shutil.rmtree(save_path)
    snapshot_download(repo_id=model_name, local_dir=save_path, local_dir_use_symlinks=False)
    
    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")
    
    return save_path

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["datasets==2.19.0", "huggingface_hub", "scipy"] 
)
def download_dataset(
    data_root: str,
    dataset_name: str,
    force_download: bool
) -> str:
    import os
    import shutil
    from datasets import load_dataset, Dataset
    import time
    start_time = time.time()
    
    save_path = os.path.join(data_root, "raw", "pg19")

    if os.path.exists(save_path):
        if force_download:
            shutil.rmtree(save_path)
        else:
            print(f"Large cached dataset found at {save_path}. Skipping download.")
            return save_path

    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True
    )
    
    CHARS_PER_SAMPLE = 4096
    
    def process_batch(items):
        out = []
        for text in items['text']:
            if len(text) < CHARS_PER_SAMPLE:
                continue
                
            buf = []
            
            i, j = (0, CHARS_PER_SAMPLE)
            
            while True:
                cur = text[i:j]
                
                if not cur:
                    break
                    
                k = max(1, len(cur)//2)
                buf.append([
                    {"role": "user", "content": cur[:k]},
                    {"role": "assistant", "content": cur[k:]} 
                ])
                i = i+k
                j = min(len(text), j+k)
                if 1.75*(j - i) < CHARS_PER_SAMPLE:
                    buf[-1][-1]["content"] += text[i:j]
                    break
        
            out.extend(buf)
            
        return { "messages": out, "type": ["style"] * len(out) }
    
    final_ds = ds.select_columns(["text"])
    final_ds = final_ds.map(process_batch, batched=True, batch_size=16, remove_columns=["text"])
    final_ds = final_ds.select_columns(["messages", "type"])
    final_ds = Dataset.from_generator(lambda: final_ds)
    os.makedirs(save_path, exist_ok=True)
    final_ds.save_to_disk(save_path)
    
    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")

    return save_path
    
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["datasets==2.19.0", "huggingface_hub", "scipy", "requests"] 
)
def download_playback(
    playback_root: str,
    force_download: bool,
) -> str:
    import os
    import shutil
    from datasets import load_dataset, Dataset
    import time
    import json
    import gzip
    import requests
    
    start_time = time.time()
    
    save_path = os.path.join(playback_root, "raw", "oasst2")

    if os.path.exists(save_path):
        if force_download:
            shutil.rmtree(save_path)
        else:
            print(f"Large cached dataset found at {save_path}. Skipping download.")
            return save_path
            
    url = "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz"
    
    print(f"Downloading and loading trees from {url}...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with gzip.open(r.raw, "rt", encoding="utf-8") as f:
            trees = [json.loads(line) for line in f]

    print(f"Successfully loaded {len(trees)} trees.")
    
    role_map = {
        "prompter": "user",
        "assistant": "assistant"
    }

    def extract_conversation_paths(node, current_history=None):
        if current_history is None:
            current_history = []
            
        if node['role'] == "assistant" and (not "labels" in node or not "quality" in node["labels"] or float(node["labels"]["quality"]["value"]) < 0.5):
            return []

        message = {
            "role": role_map.get(node['role'], node['role']),
            "content": node['text']
        }
        
        new_history = current_history + [message]

        if not node.get('replies'):
            return [new_history]
        
        paths = []
        for reply in node['replies']:
            paths.extend(extract_conversation_paths(reply, new_history))
        if len(paths) == 0:
            paths = [new_history]
        return paths
        
    all_conversations = []

    print("Processing trees into linear conversations...")

    for tree in trees:
        if tree["prompt"]["lang"] != "en":
            continue
            
        paths = extract_conversation_paths(tree["prompt"])
        
        for path in paths:
            while len(path) > 0 and path[-1]["role"] != "assistant":
                path.pop()
            
            if len(path) == 0:
                continue
            
            all_conversations.append(path)

    print(f"Extracted {len(all_conversations)} unique conversation paths.")

    formatted_data = [{"messages": conv, "type": "playback"} for conv in all_conversations]
    
    hf_dataset = Dataset.from_list(formatted_data)
    os.makedirs(save_path, exist_ok=True)
    hf_dataset.save_to_disk(save_path)
    
    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")

    return save_path

# -------------------------------------------------------------------------
# COMPONENT 4: LAUNCH TRAINING JOB (Unchanged)
# -------------------------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["kubernetes"])
def launch_training_job(
    base_model_path: str,
    data_path: str,
    playback_path: str,
    model_root: str,
    image: str,
    model_pvc: str,
    data_pvc: str,
    playback_pvc: str,
    max_steps: int = 1
) -> str:
    import time
    from kubernetes import client, config
    import time
    start_time = time.time()
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()
    
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    job_name = f"qwen-finetune-{int(time.time())}"
    output_dir_internal = f"/mnt/models/checkpoints/{job_name}"
    output_dir_external = f"{model_root}/checkpoints/{job_name}"

    cmd_args = [
        "--base_model_path", base_model_path,
        "--data_path", data_path,
        "--playback_path", playback_path,
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
                                "resources": {
                                    "limits": {"nvidia.com/gpu": 1, "memory": "32Gi", "cpu": "4"}, 
                                    "requests": {"cpu": "2"}
                                },
                                "volumeMounts": [
                                    {"name": "models", "mountPath": "/mnt/models"},
                                    {"name": "data", "mountPath": "/mnt/data"},
                                    {"name": "playback", "mountPath": "/mnt/playback"},
                                    {"name": "dshm", "mountPath": "/dev/shm"}
                                ]
                            }],
                            "volumes": [
                                {"name": "models", "persistentVolumeClaim": {"claimName": model_pvc}},
                                {"name": "data", "persistentVolumeClaim": {"claimName": data_pvc}},
                                {"name": "playback", "persistentVolumeClaim": {"claimName": playback_pvc}},
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
                
    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")

    return output_dir_external

# -------------------------------------------------------------------------
# COMPONENT 5: MERGE ADAPTER (UPDATED FOR OPLoRA)
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
    import time
    
    start_time = time.time()
    
    # Must match the rank used in train.py
    OPLORA_RANK = 16 
    
    print(f"--- Merging OPLoRA Adapter ---")
    print(f"Base: {base_model_path}")
    print(f"Adapter: {adapter_path}")

    # 1. Load Base Model in Float32 (CPU)
    # We use Float32 to ensure the SVD calculation is stable/accurate.
    print("Loading base model to System RAM (CPU, Float32)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32, 
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 2. Load Adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # ------------------------------------------------------------------
    # OPLoRA TRANSFORMATION
    # We must project A and B into the orthogonal complement of W 
    # before merging, effectively "baking in" the hooks.
    # ------------------------------------------------------------------
    print(f"Applying OPLoRA projections (Rank {OPLORA_RANK}) to weights...")
    
    with torch.no_grad():
        count = 0
        for name, module in model.named_modules():
            # Check for LoRA layers
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                base_layer = getattr(module, "base_layer", module)
                if not hasattr(base_layer, "weight"): continue
                
                # 1. Compute SVD of Base Weight
                # We cast to float32 for stability
                w_data = base_layer.weight.data.to(torch.float32)
                
                safe_name = name.replace(".", "_")
                u_path = os.path.join(adapter_path, "oplora_stats", f"{safe_name}_U.pt")
                v_path = os.path.join(adapter_path, "oplora_stats", f"{safe_name}_V.pt")
                
                if os.path.exists(u_path):
                    U = torch.load(u_path).to(torch.float32)
                    V = torch.load(v_path).to(torch.float32)
                    
                    # 2. Get LoRA Weights
                    # A: (r, In), B: (Out, r)
                    # Ensure they are float32 for the math
                    lora_A_param = module.lora_A["default"].weight
                    lora_B_param = module.lora_B["default"].weight
                    
                    A = lora_A_param.data.to(torch.float32)
                    B = lora_B_param.data.to(torch.float32)
                    
                    # 3. Apply Projections (Bake the hooks into the weights)
                    
                    # Input Projector: P_R = I - V V^T
                    # A_new = A @ P_R = A - (A @ V) @ V^T
                    
                    # A @ V -> (r, In) @ (In, k) -> (r, k)
                    AV = torch.matmul(A, V) 
                    # AV @ V.T -> (r, k) @ (k, In) -> (r, In)
                    A_correction = torch.matmul(AV, V.t())
                    
                    # Update A
                    lora_A_param.data = (A - A_correction).to(lora_A_param.dtype)
                    
                    # Output Projector: P_L = I - U U^T
                    # B_new = P_L @ B = B - U @ (U^T @ B)
                    
                    # U.T @ B -> (k, Out) @ (Out, r) -> (k, r)
                    UtB = torch.matmul(U.t(), B)
                    # U @ UtB -> (Out, k) @ (k, r) -> (Out, r)
                    B_correction = torch.matmul(U, UtB)
                    
                    # Update B
                    lora_B_param.data = (B - B_correction).to(lora_B_param.dtype)
                    
                    count += 1
                    
                    # Cleanup huge matrices immediately
                    del U, V, AV, A_correction, UtB, B_correction, A, B
                else:
                    print(f"Skipping OPLoRA for {name} (No stats found)")
                
                del w_data
    
    print(f"OPLoRA transformation applied to {count} layers.")
    
    # 3. Merge
    print("Merging weights...")
    model = model.merge_and_unload()

    # 4. Save
    save_path = os.path.join(merged_root, "qwen_merged_final")
    print(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    
    del model
    del base_model
    gc.collect()
    
    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")
    
    return save_path

# -------------------------------------------------------------------------
# COMPONENT 6: INFERENCE (Unchanged)
# -------------------------------------------------------------------------
@dsl.component(
    base_image=WORKER_IMAGE, 
    packages_to_install=["transformers", "torch", "accelerate", "bitsandbytes"]
)
def run_inference(model_path: str, prompt: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import time
    start_time = time.time()
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
    
    print(f"\n--- Finished in {(time.time() - start_time)/60:.2f} minutes ---")

# -------------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------------
@dsl.pipeline(name="qwen-finetune-production")
def llm_pipeline(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507", # Example model
    model_pvc: str = "llm-workspace-pvc",
    data_pvc: str = "llm-data-pvc",
    playback_pvc: str = "llm-playback-pvc",
    training_image_uri: str = "kjh123456/qwen-trainer:v28",
    force_download: bool = False,
    max_steps: int = 50,
):
    dl_model = download_model(model_name=model_name, model_root=MOUNT_PATH_MODEL, force_download=force_download)
    kubernetes.mount_pvc(dl_model, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)

    dl_data = download_dataset(data_root=MOUNT_PATH_DATA, dataset_name=DATASET_NAME, force_download=force_download)
    kubernetes.mount_pvc(dl_data, pvc_name=data_pvc, mount_path=MOUNT_PATH_DATA)
    
    dl_playback = download_playback(playback_root=MOUNT_PATH_PLAYBACK, force_download=force_download)
    kubernetes.mount_pvc(dl_playback, pvc_name=playback_pvc, mount_path=MOUNT_PATH_PLAYBACK)

    train_job = launch_training_job(
        base_model_path=dl_model.output,
        data_path=dl_data.output,
        playback_path=dl_playback.output,
        model_root=MOUNT_PATH_MODEL,
        image=training_image_uri,
        model_pvc=model_pvc,
        data_pvc=data_pvc,
        playback_pvc=playback_pvc,
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