# controller.py
#!/usr/bin/env python3
"""
Controller script to launch Llama-3.2 fine-tuning VMs on GCP.
Each VM reads metadata, runs the startup script in background, then exits.
"""

import os
import subprocess
import time
import uuid
import argparse
import json

def print_banner():
    banner = [
        "############################################################",
        "#                                                          #",
        "#          🦙  VibeLlama Launcher Starting...         🦙    #",
        "#                                                          #",
        "############################################################",
    ]
    for line in banner:
        print(line)
    print()  # blank line

    
def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def launch_job(size: int, seed: int, cfg: dict) -> None:
    vm_name = f"llama-{size}b-seed{seed}-{uuid.uuid4().hex[:6]}"

    # Build metadata string
    md_items = [
        f"seed={seed}",
        f"size={size}",
        f"HF_TOKEN={cfg['hf_token']}",
        f"WANDB_API_KEY={cfg['wandb_api_key']}",
        f"repo_dir={cfg['repo_dir']}",
        f"script_repo={cfg['script_repo']}",
        f"wandb_project={cfg['wandb_project']}",
        f"model_hub_namespace={cfg['model_hub_namespace']}"
    ]
    metadata = ",".join(md_items)

    cmd = [
        "gcloud", "compute", "instances", "create", vm_name,
        f"--project={cfg['gcp_project']}",
        f"--zone={cfg['gcp_zone']}",
        f"--machine-type={cfg['machine_type']}",
        f"--accelerator=type={cfg['gpu_type']},count={cfg['gpu_count']}",
        "--maintenance-policy=TERMINATE",
        "--restart-on-failure",
        f"--image={cfg['disk_image']}",
        f"--metadata={metadata}",
        "--metadata-from-file=startup-script=run_on_startup.sh"
    ]

    print(f"[+] Launching {vm_name} (size={size}B, seed={seed})…")
    subprocess.check_call(cmd)
    print(f"[✓] Launched {vm_name}")


def main():
    print_banner() 
    parser = argparse.ArgumentParser(
        description="Launch Llama fine-tune VMs on GCP"
    )
    parser.add_argument("--config", "-c", default="config.json",
                        help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Ensure tokens are in the config
    if not cfg.get("hf_token"):
        raise RuntimeError("`hf_token` must be set in config.json")
    if not cfg.get("wandb_api_key"):
        raise RuntimeError("`wandb_api_key` must be set in config.json")

    for size in cfg["model_sizes"]:
        for seed in cfg["seeds"]:
            try:
                launch_job(size, seed, cfg)
            except subprocess.CalledProcessError as e:
                print(f"[!] Failed to launch size={size}, seed={seed}: {e}")
            time.sleep(10)  # throttle GCP API calls


if __name__ == "__main__":
    main()
