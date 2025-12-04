# Filename: qwen-pipeline-monolith.py
import kfp
from kfp import dsl
from kfp import kubernetes

# CONFIGURATION
MONOLITH_IMAGE = "kjh123456/qwen-monolith:v1"
MOUNT_PATH_MODEL = "/mnt/models"
MOUNT_PATH_DATA = "/mnt/data"

# 1. Define the Component
@dsl.container_component
def monolith_op(
    model_name: str,
    dataset_name: str,
    subset_size: int,
    max_steps: int,
    force_download: bool,
    model_root: str,
    data_root: str,
):
    return dsl.ContainerSpec(
        image=MONOLITH_IMAGE,
        args=[
            "--model_name", model_name,
            "--dataset_name", dataset_name,
            "--model_root", model_root,
            "--data_root", data_root,
            "--subset_size", subset_size,
            "--max_steps", max_steps,
            "--force_download", force_download
        ]
    )

@dsl.pipeline(name="qwen-finetune-monolith")
def llm_pipeline_monolith(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
    dataset_name: str = "deepmind/pg19",
    model_pvc: str = "llm-workspace-pvc",
    data_pvc: str = "llm-data-pvc",
    force_download: bool = True,
    subset_size: int = 500,
    max_steps: int = 50,
):
    # 2. Create the task
    monolith_task = monolith_op(
        model_name=model_name,
        dataset_name=dataset_name,
        subset_size=subset_size,
        max_steps=max_steps,
        force_download=force_download,
        model_root=MOUNT_PATH_MODEL,
        data_root=MOUNT_PATH_DATA
    )

    # 3. Mount PVCs
    kubernetes.mount_pvc(monolith_task, pvc_name=model_pvc, mount_path=MOUNT_PATH_MODEL)
    kubernetes.mount_pvc(monolith_task, pvc_name=data_pvc, mount_path=MOUNT_PATH_DATA)

    # 4. Set Resources
    monolith_task.set_gpu_limit(1)
    monolith_task.set_memory_limit("24Gi")

    # 5. Shared Memory (dshm) - The "Clean" V2 Way
    # This function implements exactly what is in the PR you linked:
    # It creates an emptyDir volume backed by Memory and mounts it to /dev/shm
    kubernetes.empty_dir_mount(
        monolith_task,
        volume_name="dshm",
        mount_path="/dev/shm",
        medium="Memory",     # <--- This matches the "medium: Memory" in the commit
        size_limit="10Gi"    # Optional: Limits how much RAM dshm can take (good safety practice)
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(llm_pipeline_monolith, "qwen_pipeline_monolith.yaml")