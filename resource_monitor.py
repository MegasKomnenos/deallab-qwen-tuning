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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper to parse Kubernetes resource quantities (e.g., 1Gi, 500m)
def parse_quantity(quantity):
    if not quantity: return 0
    if isinstance(quantity, (int, float)): return float(quantity)

    # Handle CPU milli-cores
    if quantity.endswith('m'):
        return float(quantity[:-1]) / 1000

    # Handle Memory (Bytes)
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}
    match = re.match(r"([0-9.]+)([A-Za-z]+)", quantity)
    if match:
        value, unit = float(match.group(1)), match.group(2)
        if unit in units:
            return value * units[unit]

    try:
        return float(quantity)
    except ValueError:
        return 0

def kill_istio_sidecar():
    """
    If an Istio sidecar is present, the Job will not complete even if the python script finishes.
    We must manually hit the /quitquitquit endpoint of the sidecar.
    """
    try:
        url = "http://localhost:15020/quitquitquit"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as response:
            logging.info(f"Istio Sidecar termination request sent. Response: {response.read().decode('utf-8')}")
    except Exception:
        pass

class PipelineMonitor:
    def __init__(self, host=None, namespace="default", userid=None):
        logging.info(f"KFP SDK Version: {kfp.__version__}")
        
        # 1. Kubernetes Client Setup
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

        # 2. KFP Client Setup
        if not host:
            host = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"
            logging.info(f"No host provided. Defaulting to internal service: {host}")

        try:
            self.kfp_client = kfp.Client(host=host)
            
            # --- AGGRESSIVE AUTH FIX ---
            if userid:
                self._inject_auth_headers(userid)
                
                # VERIFY AUTH IMMEDIATELY
                try:
                    self.kfp_client.list_experiments(page_size=1, namespace=namespace)
                    logging.info("✅ Authentication verified: Successfully listed experiments.")
                except Exception as auth_e:
                    logging.error(f"❌ Authentication verification FAILED: {auth_e}")
                    logging.error("The script will likely fail below.")

            self.namespace = namespace
            logging.info(f"KFP Client initialized. Monitoring namespace: {self.namespace}")
        except Exception as e:
            logging.error(f"Failed to initialize KFP Client. Error: {e}"); exit(1)

    def _inject_auth_headers(self, userid):
        """
        Recursively finds API clients and Sessions to inject the UserID header.
        """
        header = {"kubeflow-userid": userid}
        logging.info(f"Attempting to inject auth header: {header}")
        injected_count = 0

        # 1. Try standard locations
        targets = [
            (self.kfp_client, 'api_client'),
            (self.kfp_client, '_api_client'),
        ]

        for obj, attr in targets:
            if hasattr(obj, attr):
                client_obj = getattr(obj, attr)
                if hasattr(client_obj, 'default_headers'):
                    client_obj.default_headers.update(header)
                    logging.info(f"-> Injected into {attr}.default_headers")
                    injected_count += 1

        # 2. Try underlying Requests Sessions
        if hasattr(self.kfp_client, '_session') and hasattr(self.kfp_client._session, 'headers'):
            self.kfp_client._session.headers.update(header)
            logging.info("-> Injected into _session.headers")
            injected_count += 1
            
        # 3. Recursive search (Catch-all for generated clients)
        # Search properties of kfp_client for anything that looks like an API client
        for attr_name in dir(self.kfp_client):
            try:
                val = getattr(self.kfp_client, attr_name)
                if hasattr(val, 'api_client') and hasattr(val.api_client, 'default_headers'):
                    val.api_client.default_headers.update(header)
                    logging.info(f"-> Injected into {attr_name}.api_client.default_headers")
                    injected_count += 1
            except:
                pass

        if injected_count == 0:
            logging.warning("WARNING: Could not find any HTTP client to inject headers into!")

    def launch_pipeline(self, pipeline_file, experiment_name, run_name, params):
        if not os.path.exists(pipeline_file):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

        logging.info(f"Launching pipeline: {run_name} from {pipeline_file}")
        try:
             experiment = self.kfp_client.create_experiment(name=experiment_name, namespace=self.namespace)
        except:
             # Handle existing experiment
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
            time.sleep(15) # Polling interval

            # 1. Check KFP run status
            try:
                kfp_run = self.kfp_client.get_run(run_id)
                if kfp_run.run.status in ['Succeeded', 'Failed', 'Error', 'Skipped']:
                    logging.info(f"Run finished with status: {kfp_run.run.status}")
                    time.sleep(5) # Short delay for K8s API finalization
                    break
            except Exception as e:
                logging.warning(f"Error getting KFP run status: {e}.")

            # 2. Monitor Pods
            try:
                pods = self.core_v1.list_namespaced_pod(self.namespace, label_selector=label_selector)
            except Exception as e:
                logging.error(f"Error listing pods: {e}. Retrying...")
                continue

            for pod in pods.items:
                pod_name = pod.metadata.name
                if pod_name not in pod_data:
                    pod_data[pod_name] = self._extract_pod_info(pod)

                # Update status and end time
                self._update_pod_status(pod, pod_data[pod_name])

        # Final sweep to capture final statuses
        try:
            pods = self.core_v1.list_namespaced_pod(self.namespace, label_selector=label_selector)
            for pod in pods.items:
                 self._update_pod_status(pod, pod_data[pod.metadata.name])
        except:
            pass

        # Handle PyTorchJob Resources (Heuristic)
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
                PTJ_MEM_BYTES = 16 * (1024**3) # 16Gi
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

            if not start_time or not end_time:
                continue

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
    PIPELINE_MONOLITHIC = "qwen_pipeline_monolithic.yaml"

    params = {"max_steps": MAX_STEPS, "subset_size": 500, "force_download": args.force_download}
    monitor = PipelineMonitor(host=args.host, namespace=args.namespace, userid=args.userid)
    all_results = []

    logging.info("Checking for compiled pipelines...")
    try:
        if not os.path.exists(PIPELINE_DISTRIBUTED) or not os.path.exists(PIPELINE_MONOLITHIC):
            logging.error("Error: Pipeline YAML files not found. Please run the pipeline Python scripts first.")
            # We don't exit here so we can see the full logs, but the next steps will fail
    except Exception as e:
        logging.error(f"An error occurred during pre-check: {e}")

    if not args.skip_distributed:
        run_name_dist = f"Distributed-{MAX_STEPS}steps-{int(time.time())}"
        try:
            run_id_dist = monitor.launch_pipeline(PIPELINE_DISTRIBUTED, EXPERIMENT_NAME, run_name_dist, params)
            df_dist = monitor.monitor_run(run_id_dist)
            if not df_dist.empty:
                df_dist['Pipeline Type'] = 'Distributed'
                all_results.append(df_dist)
        except Exception as e:
            logging.error(f"Error during distributed pipeline execution: {e}")

    if not args.skip_monolithic:
        run_name_mono = f"Monolithic-{MAX_STEPS}steps-{int(time.time())}"
        try:
            run_id_mono = monitor.launch_pipeline(PIPELINE_MONOLITHIC, EXPERIMENT_NAME, run_name_mono, params)
            df_mono = monitor.monitor_run(run_id_mono)
            if not df_mono.empty:
                df_mono['Pipeline Type'] = 'Monolithic'
                all_results.append(df_mono)
        except Exception as e:
            logging.error(f"Error during monolithic pipeline execution: {e}")

    if all_results:
        df_final = pd.concat(all_results)
        print(df_final.to_string(index=False))
        kill_istio_sidecar() 
        return df_final
    
    kill_istio_sidecar()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch KFP pipelines and monitor resource allocation.")
    parser.add_argument("--host", type=str, default=None, help="KFP API host. Defaults to internal DNS if not set.")
    parser.add_argument("--namespace", type=str, required=True, help="The Kubernetes namespace (e.g., kubeflow-user).")
    
    # --- THIS LINE MUST BE PRESENT ---
    parser.add_argument("--userid", type=str, required=False, help="The user identity (email) for auth.")
    
    parser.add_argument("--max_steps", type=int, default=1500, help="Number of training steps.")
    parser.add_argument("--force_download", action="store_true", help="Force re-download of model and data.")
    parser.add_argument("--skip_distributed", action="store_true", help="Skip the distributed pipeline run.")
    parser.add_argument("--skip_monolithic", action="store_true", help="Skip the monolithic pipeline run.")

    args = parser.parse_args()
    run_experiment(args)