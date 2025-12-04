import kfp
from kfp import dsl
from kfp import kubernetes
import yaml

# --- 1. THE FIXED PATCHING FUNCTION ---
def add_dshm_to_yaml(yaml_path, task_name):
    # Step A: Load ALL documents (fixes ComposerError)
    with open(yaml_path, 'r') as f:
        docs = list(yaml.safe_load_all(f))

    found_task = False
    
    # Iterate over all documents in the YAML stream
    for doc in docs:
        # We look for the document that contains the deploymentSpec
        deploy_spec = doc.get('deploymentSpec', {}).get('executors', {})
        
        for exec_name, exec_body in deploy_spec.items():
            # Check if this executor belongs to our target task
            if task_name in exec_name:
                print(f"Patching executor: {exec_name}")
                found_task = True
                
                # --- PART 1: Add Volume Mount to Container ---
                container = exec_body.get('container', {})
                if 'volumeMounts' not in container:
                    container['volumeMounts'] = []
                
                # Avoid adding duplicates if run multiple times
                if not any(vm['name'] == 'dshm' for vm in container['volumeMounts']):
                    container['volumeMounts'].append({
                        "name": "dshm",
                        "mountPath": "/dev/shm"
                    })
                
                # --- PART 2: Add Volume Definition (CRITICAL MISSING PIECE) ---
                # In KFP IR, the executor body acts like a partial Pod Spec.
                # We must define what 'dshm' actually is (emptyDir / Memory).
                if 'volumes' not in exec_body:
                    exec_body['volumes'] = []

                if not any(v['name'] == 'dshm' for v in exec_body['volumes']):
                    exec_body['volumes'].append({
                        "name": "dshm",
                        "emptyDir": {
                            "medium": "Memory",
                            "sizeLimit": "10Gi" # Optional safety limit
                        }
                    })

    if not found_task:
        print(f"Warning: Could not find task '{task_name}' to patch.")
    else:
        # Step B: Write back ALL documents
        with open(yaml_path, 'w') as f:
            yaml.dump_all(docs, f)
        print("Patched successfully: Added dshm mount and volume definition.")

# --- 2. CONFIGURATION ---
MONOLITH_IMAGE = "kjh123456/qwen-monolith:v5"
MOUNT_PATH_MODEL = "/mnt/models"
MOUNT_PATH_DATA = "/mnt/data"

# --- 3. COMPONENT DEFINITION ---
@dsl.container_component
def monolith_op(
    model_name: str,
    dataset_name: str,
    subset_size: int,
    max_steps: int,
    model_root: str,
    data_root: str,
):
    return dsl.ContainerSpec(
        image=MONOLITH_IMAGE,
        command=["python", "monolith.py"],
        args=[
            "--model_name", model_name,
            "--dataset_name", dataset_name,
            "--model_root", model_root,
            "--data_root", data_root,
            "--subset_size", subset_size,
            "--max_steps", max_steps
        ]
    )

# --- 4. PIPELINE DEFINITION ---
@dsl.pipeline(name="qwen-finetune-monolith")
def llm_pipeline_monolith(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
    dataset_name: str = "deepmind/pg19",
    model_pvc: str = "llm-workspace-pvc",
    data_pvc: str = "llm-data-pvc",
    subset_size: int = 500,
    max_steps: int = 50,
):
    # Create the task
    monolith_task = monolith_op(
        model_name=model_name,
        dataset_name=dataset_name,
        subset_size=subset_size,
        max_steps=max_steps,
        model_root=MOUNT_PATH_MODEL,
        data_root=MOUNT_PATH_DATA
    )

    # Mount PVCs (These use the older kfp-kubernetes which is fine for PVCs)
    kubernetes.mount_pvc(monolith_task, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    kubernetes.mount_pvc(monolith_task, pvc_name=data_pvc, mount_path=MOUNT_PATH_DATA)

    # Set Resources
    monolith_task.set_cpu_request("2000m")
    monolith_task.set_cpu_limit("4000m")
    monolith_task.set_gpu_limit(1)
    monolith_task.set_memory_limit("24Gi")

if __name__ == "__main__":
    yaml_file = "qwen_pipeline_monolith.yaml"
    
    # 1. Compile
    kfp.compiler.Compiler().compile(llm_pipeline_monolith, yaml_file)
    
    # 2. Patch (Inject Shared Memory)
    add_dshm_to_yaml(yaml_file, "monolith-op")