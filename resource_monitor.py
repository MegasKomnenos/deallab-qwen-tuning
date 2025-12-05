# Filename: resource_monitor.py
import kfp
import time
import argparse
import pandas as pd
from kubernetes import client, config
from datetime import datetime, timezone
import re
import logging
import os
import sys
import urllib.request
import urllib3 # Required for the network patch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. THE NETWORK PATCH (Fixes Auth) ---
# This forces the UserID header onto EVERY request, bypassing KFP SDK logic.
_GLOBAL_USERID = None
_orig_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen

def patched_urlopen(self, method, url, body=None, headers=None, **kwargs):
    if _GLOBAL_USERID:
        if headers is None:
            headers = {}
        # Force the header
        headers["kubeflow-userid"] = _GLOBAL_USERID
    return _orig_urlopen(self, method, url, body, headers, **kwargs)

# Apply the patch immediately
urllib3.connectionpool.HTTPConnectionPool.urlopen = patched_urlopen
# -----------------------------------------

# Helper to parse Kubernetes resource quantities (e.g., 1Gi, 500m)
def parse_quantity(quantity):
    if not quantity: return 0
    if isinstance(quantity, (int, float)): return float(quantity)
    if quantity.endswith('m'): return float(quantity[:-1]) / 1000
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}
    match = re.match(r"([0-9.]+)([A-Za-z]+)", quantity)
    if match:
        value, unit = float(match.group(1)), match.group(2)
        if unit in units: return value * units[unit]
    try: return float(quantity)
    except ValueError: return 0

def kill_istio_sidecar():
    try:
        url = "http://localhost:15020/quitquitquit"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as response:
            logging.info(f"Istio Sidecar termination request sent. Response: {response.read().decode('utf-8')}")
    except Exception:
        pass

class PipelineMonitor:
    def __init__(self, host=None, namespace="default", userid=None):
        global _GLOBAL_USERID
        
        # 1. Setup Auth Globally
        if userid:
            _GLOBAL_USERID = userid
            logging.info(f"✅ Global Network Patch active. UserID '{userid}' will be sent with ALL requests.")

        # 2. Kubernetes Client
        try:
            config.load_incluster_config()
            logging.info("Loaded in-cluster Kubernetes configuration.")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logging.info("Loaded local kubeconfig.")
            except Exception as e:
                logging.error(f"Could not configure Kubernetes client: {e}"); exit(1)
        self.core_v1 = client.CoreV1Api()

        # 3. KFP Client
        if not host:
            host = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"
            logging.info(f"No host provided. Defaulting to: {host}")

        try:
            self.kfp_client = kfp.Client(host=host)
            # Verify immediately
            try:
                self.kfp_client.list_experiments(page_size=1, namespace=namespace)
                logging.info("✅ Authentication verified: Successfully listed experiments.")
            except Exception as e:
                logging.error(f"❌ Auth check failed even with patch: {e}")
                
            self.namespace = namespace
            logging.info(f"KFP Client initialized. Monitoring namespace: {self.namespace}")
        except Exception as e:
            logging.error(f"Failed to initialize KFP Client. Error: {e}"); exit(1)

    def launch_pipeline(self, pipeline_file, experiment_name, run_name, params):
        # CRITICAL CHECK
        if not os.path.exists(pipeline_file):
            # Log the current directory contents to help debug
            logging.error(f"❌ CRITICAL ERROR: Pipeline file '{pipeline_file}' NOT FOUND!")
            logging.error(f"Current Directory ({os.getcwd()}) contents: {os.listdir('.')}")
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

        logging.info(f"Launching pipeline: {run_name} from {pipeline_file}")
        try:
             experiment = self.kfp_client.create_experiment(name=experiment_name, namespace=self.namespace)
        except:
             experiment = self.kfp_client.get_experiment(experiment_name=experiment_name, namespace=self.namespace)

        run = self.kfp_client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name,
            pipeline_package_path=pipeline_file,
            params=params
        )
        return run.id

    def monitor_run(self, run_id):
        logging.info(f"Starting monitoring for Run ID: {run_id}...")
        pod_data = {}
        label_selector = f"pipeline/runid={run_id}"

        while True:
            time.sleep(15) 

            try:
                kfp_run = self.kfp_client.get_run(run_id)
                if kfp_run.run.status in ['Succeeded', 'Failed', 'Error', 'Skipped']:
                    logging.info(f"Run finished with status: {kfp_run.run.status}")
                    time.sleep(5) 
                    break
            except Exception as e:
                logging.warning(f"Error getting KFP run status: {e}.")

            try:
                pods = self.core_v1.list_namespaced_pod(self.namespace, label_selector=label_selector)
            except Exception as e:
                logging.error(f"Error listing pods: {e}. Retrying...")
                continue

            for pod in pods.items:
                pod_name = pod.metadata.name
                if pod_name not in pod_data:
                    pod_data[pod_name] = self._extract_pod_info(pod)
                self._update_pod_status(pod, pod_data[pod_name])

        try:
            pods = self.core_v1.list_namespaced_pod(self.namespace, label_selector=label_selector)
            for pod in pods.items:
                 self._update_pod_status(pod, pod_data[pod.metadata.name])
        except:
            pass

        self._account_for_pytorchjob(pod_data)
        logging.info("Monitoring finished. Processing data...")
        return self._process_results(pod_data)

    def _update_pod_status(self, pod, current_data):
         if current_data['status'] not in ['Succeeded', 'Failed', 'Error']:
            current_data['status'] = pod.status.phase
            if pod.status.phase in ['Succeeded', 'Failed', 'Error']:
                current_data['end_time'] = self._get_pod_end_time(pod)

    def _account_for_pytorchjob(self, pod_data):
        launcher_pods = [data for name, data in pod_data.items() if 'launch-training-job' in name]
        if not launcher_pods:
            return

        for launcher in launcher_pods:
            if launcher['status'] == 'Succeeded' and launcher['start_time'] and launcher['end_time']:
                logging.info(f"Applying heuristic: Accounting for PyTorchJob resources launched by {launcher['pod_name']}...")
                PTJ_GPU = 1
                PTJ_MEM_BYTES = 16 * (1024**3) 
                PTJ_CPU = 2

                worker_name = f"{launcher['pod_name']}-ptj-worker-synthetic"
                pod_data[worker_name] = {
                    'pod_name': worker_name,
                    'status': 'Succeeded (Synthetic)',
                    'start_time': launcher['start_time'],
                    'end_time': launcher['end_time'],
                    'cpu_cores': PTJ_CPU,
                    'memory_bytes': PTJ_MEM_BYTES,
                    'gpus': PTJ_GPU
                }

    def _get_pod_end_time(self, pod):
        if pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                if cs.name == 'main' and cs.state.terminated:
                    return cs.state.terminated.finished_at
        return datetime.now(timezone.utc)

    def _extract_pod_info(self, pod):
        cpu_request = 0
        memory_request_bytes = 0
        gpu_request = 0

        for container in pod.spec.containers:
            if container.resources:
                requests = container.resources.requests or {}
                cpu_request += parse_quantity(requests.get('cpu'))
                memory_request_bytes += parse_quantity(requests.get('memory'))
                limits = container.resources.limits or {}
                gpu_request += parse_quantity(limits.get('nvidia.com/gpu'))

        start_time = pod.status.start_time or pod.metadata.creation_timestamp

        return {
            'pod_name': pod.metadata.name,
            'status': pod.status.phase,
            'start_time': start_time,
            'end_time': None,
            'cpu_cores': cpu_request,
            'memory_bytes': memory_request_bytes,
            'gpus': gpu_request
        }

    def _process_results(self, pod_data):
        results = []
        for data in pod_data.values():
            start_time = data['start_time']
            end_time = data['end_time']
            if not start_time or not end_time: continue

            duration_hours = (end_time - start_time).total_seconds() / 3600.0
            memory_gib = data['memory_bytes'] / (1024**3)

            results.append({
                'Pod Name': data['pod_name'],
                'Status': data['status'],
                'Start Time': start_time,
                'End Time': end_time,
                'Duration (Hours)': duration_hours,
                'CPU Cores Requested': data['cpu_cores'],
                'Memory GiB Requested': memory_gib,
                'GPUs Requested': data['gpus'],
                'CPU Core-Hours': data['cpu_cores'] * duration_hours,
                'Memory GiB-Hours': memory_gib * duration_hours,
                'GPU-Hours': data['gpus'] * duration_hours
            })
        return pd.DataFrame(results)

def run_experiment(args):
    EXPERIMENT_NAME = "Qwen_Resource_Comparison_Experiment"
    MAX_STEPS = args.max_steps
    PIPELINE_DISTRIBUTED = "qwen_pipeline_production.yaml"
    PIPELINE_MONOLITHIC = "qwen_pipeline_monolith.yaml"
    
    params = {"max_steps": MAX_STEPS, "subset_size": 500, "force_download": args.force_download}
    monitor = PipelineMonitor(host=args.host, namespace=args.namespace, userid=args.userid)
    all_results = []

    # Check for files and EXIT if missing
    if not os.path.exists(PIPELINE_DISTRIBUTED):
        logging.error(f"❌ ERROR: '{PIPELINE_DISTRIBUTED}' is missing inside the container.")
        logging.error("Did you compile the pipeline AND rebuild the Docker image?")
        kill_istio_sidecar(); return None
        
    if not os.path.exists(PIPELINE_MONOLITHIC):
        logging.error(f"❌ ERROR: '{PIPELINE_MONOLITHIC}' is missing inside the container.")
        kill_istio_sidecar(); return None

    if not args.skip_distributed:
        try:
            run_name = f"Distributed-{MAX_STEPS}steps-{int(time.time())}"
            run_id = monitor.launch_pipeline(PIPELINE_DISTRIBUTED, EXPERIMENT_NAME, run_name, params)
            df = monitor.monitor_run(run_id)
            if not df.empty:
                df['Pipeline Type'] = 'Distributed'
                all_results.append(df)
        except Exception as e:
            logging.error(f"Error Distributed: {e}")

    if not args.skip_monolithic:
        try:
            run_name = f"Monolithic-{MAX_STEPS}steps-{int(time.time())}"
            run_id = monitor.launch_pipeline(PIPELINE_MONOLITHIC, EXPERIMENT_NAME, run_name, params)
            df = monitor.monitor_run(run_id)
            if not df.empty:
                df['Pipeline Type'] = 'Monolithic'
                all_results.append(df)
        except Exception as e:
            logging.error(f"Error Monolithic: {e}")

    if all_results:
        df_final = pd.concat(all_results)
        print(df_final.to_string(index=False))
        kill_istio_sidecar()
        return df_final
    
    kill_istio_sidecar()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--namespace", type=str, required=True)
    parser.add_argument("--userid", type=str, required=True, help="REQUIRED for Auth")
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--skip_distributed", action="store_true")
    parser.add_argument("--skip_monolithic", action="store_true")
    args = parser.parse_args()
    run_experiment(args)