#!/usr/bin/env python3
"""
Controller script to launch Llama-3.2 fine-tuning jobs on GCP for multiple model sizes and seeds,
using a JSON configuration file for parameters. Each VM will pull the repo, install dependencies,
run fine-tuning, log to W&B, then shutdown.

Config file example (config.json):
{
  "model_sizes": [1, 3, 11],
  "seeds": [42, 123, 456, 789, 555],
  "gcp_project": "your-gcp-project",
  "gcp_zone": "us-central1-a",
  "machine_type": "a2-highgpu-1g",
  "gpu_type": "nvidia-tesla-a100",
  "gpu_count": 1,
  "image_family": "pytorch-2-2-cu121-notebooks-debian-11",
  "image_project": "deeplearning-platform-release",
  "script_repo": "https://github.com/you/llama-finetune.git",
  "repo_dir": "/home/llama-finetune",
  "requirements": "requirements.txt",
  "wandb_project": "sentiment-sweep",
  "model_hub_namespace": "yourusername"
}
"""
import os
import subprocess
import time
import uuid
import argparse
import json


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def launch_job(size: int, seed: int, cfg: dict) -> None:
    vm_name = f"llama-{size}b-seed{seed}-{uuid.uuid4().hex[:6]}"

    startup_script = f"""#!/bin/bash
set -xe

# Export credentials
export HF_TOKEN={os.getenv('HF_TOKEN')}
export WANDB_API_KEY={os.getenv('WANDB_API_KEY')}

# Basic setup
conda activate base
apt-get update && apt-get install -y git python3-pip
pip3 install --upgrade pip
pip3 install -r {cfg['requirements']}

# Clone fine-tune repo
rm -rf {cfg['repo_dir']}
git clone {cfg['script_repo']} {cfg['repo_dir']}
cd {cfg['repo_dir']}

# Run fine-tuning
nohup python3 train.py \
  --model-name meta-llama/Llama-3.2-{size}B-Instruct \
  --seed {seed} \
  --hf-token ${{HF_TOKEN}} \
  --wandb-project {cfg['wandb_project']} \
  --output-dir /home/models/size{size}_seed{seed} \
  --model-hub-id {cfg['model_hub_namespace']}/VibeLlama-{size}b-seed-{seed} \
  > /home/logs/size{size}_seed{seed}.log 2>&1 &

exit 0
"""

    cmd = [
        "gcloud", "compute", "instances", "create", vm_name,
        f"--project={cfg['gcp_project']}",
        f"--zone={cfg['gcp_zone']}",
        f"--machine-type={cfg['machine_type']}",
        f"--accelerator=type={cfg['gpu_type']},count={cfg['gpu_count']}",
        "--maintenance-policy=TERMINATE",
        "--restart-on-failure",
        f"--image-family={cfg['image_family']}",
        f"--image-project={cfg['image_project']}",
        f"--metadata=startup-script={startup_script}",
        "--metadata=install-nvidia-driver=True"
    ]

    print(f"Launching VM '{vm_name}' for size={size}B seed={seed}...")
    subprocess.check_call(cmd)
    print(f"VM '{vm_name}' launched.")


def main():
    parser = argparse.ArgumentParser(description="Launch Llama fine-tune VMs on GCP")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Validate HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        raise EnvironmentError("HF_TOKEN environment variable is required")
    # Validate W&B API key
    if not os.getenv("WANDB_API_KEY"):
        raise EnvironmentError("WANDB_API_KEY environment variable is required")

    # Launch jobs
    for size in cfg['model_sizes']:
        for seed in cfg['seeds']:
            try:
                launch_job(size, seed, cfg)
            except subprocess.CalledProcessError as e:
                print(f"Failed to launch job for size={size}, seed={seed}: {e}")
            time.sleep(10)  # avoid API rate limits


if __name__ == "__main__":
    main()
