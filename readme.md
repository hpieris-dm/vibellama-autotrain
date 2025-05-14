# VibeLlama Fine-tuning

A lightweight controller + startup script that spins up GPU VMs on GCP from the custom disk image, injects per-job metadata (model size, seed, HF & W&B tokens, repo settings), and runs Llama-3.2 fine-tuning in the background. Each VM:

1. Pulls its job spec and secrets from GCP metadata  
2. Switches to the `vibellama` user and activates the pre-built Conda env  
3. Clones (or updates) the fine-tune repo  
4. Launches `train.py` via `nohup … &` (so the VM boots cleanly)  
5. (Optionally) shuts itself down when training completes  

---

## Prerequisites

- **gcloud CLI** installed & authenticated  
  ```bash
  gcloud auth login
  gcloud config set project GCP_PROJECT_ID
  ```  
- A GCP project with Compute Engine API enabled  
- **Python 3.8+** on the workstation  
- A **Hugging Face** token with write permissions  
- A **Weights & Biases** API key  

---

## Files

```
├── controller.py         # launcher
├── run_on_startup.sh     # boot-time script (must be executable)
└── config.json           # experiment config (untracked)
└── config.json.example   # experiment config example file
```

---

## Configuration (`config.json`)

Create `config.json` next to `controller.py`—do **not** check it into git. Example:

```json
{
  "model_sizes": [1, 3, 11],
  "seeds": [42, 123, 456, 789, 555],
  "gcp_project": "my-gcp-project",
  "gcp_zone": "us-central1-a",
  "machine_type": "a2-highgpu-1g",
  "gpu_type": "nvidia-tesla-a100",
  "gpu_count": 1,
  "disk_image": "vibellama-tune-base",             
  "script_repo": "https://github.com/github_id/llama-finetune.git",
  "repo_dir": "/home/vibellama/llama-finetune",
  "wandb_project": "sentiment-sweep",
  "model_hub_namespace": "HF username",
  "hf_token": "hf_xxx…",  
  "wandb_api_key": "wandb_xxx…"
}
```

- **disk_image**: Custom GCP disk image with CUDA, Conda, deps baked in  
- **script_repo** & **repo_dir**: where the VM should clone the training code  
- **hf_token** & **wandb_api_key**: will be injected via metadata (never stored on disk)  

---

## Usage

1. **Make the startup script executable**  
   ```bash
   chmod +x run_on_startup.sh
   ```
2. **Fill in** `config.json` (including the secrets).  
3. **Launch all jobs**:  
   ```bash
   python3 controller.py --config config.json
   ```
   This will iterate over every `model_size` × `seed`, create a VM for each, pass the metadata, and exit immediately.

---

## Monitoring

- **GCP Console → Compute Engine**: watch VMs boot, then exit once the startup script finishes.  
- **W&B**: live metrics under `wandb_project`.

---

## Cleanup

VMs background the training process, but don’t auto-shutdown on success. To clean up any stragglers:

```bash
gcloud compute instances list --filter="name~'llama-.*'"   --format="value(name,zone')" |   xargs -n2 gcloud compute instances delete -q --zone
```

---

## Troubleshooting

- **Missing metadata**: ensure `hf_token` and `wandb_api_key` are in the `config.json`.  
- **Conda not found**: verify the disk image has `/opt/conda` and that `run_on_startup.sh` sources `/opt/conda/etc/profile.d/conda.sh`.  
- **Quota/limits**: check GPU quotas in the GCP project.  
- **Repo errors**: confirm `script_repo` and `repo_dir` are correct and accessible.

---

## License

MIT License.
