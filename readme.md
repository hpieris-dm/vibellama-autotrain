# VibeLlama Finetuning

Python controller script (`controller.py`) automates the launching of GPU VMs on GCP for fine‑tuning multiple Llama‑3.2 models (1B, 3B, 11B) with different random seeds. Each VM will:

1. Clone the fine‑tuning repository
2. Install dependencies
3. Run the `train.py` script with the specified parameters
4. Stream logs and metrics to Weights & Biases (W&B)
5. Shut itself down upon successful completion

---

## Prerequisites

- **gcloud CLI** installed and authenticated (`gcloud auth login`)
- A GCP project with Compute Engine API enabled
- **Python 3.8+** on your local machine
- A Hugging Face token with write permissions, exported as:
  ```bash
  export HF_TOKEN="your_hf_api_token"
  ```
- A W&B API key, exported as:
  ```bash
  export WANDB_API_KEY="your_wandb_api_key"
  ```

---

## Repository Structure

```
├── controller.py    # Controller script
├── controller-config.json  # Experiment configuration (not tracked)
└── train.py       # Fine‑tuning script in the VM repo
```

---

## Configuration File (`controller-config.json`)

Place a `controller-config.json` file in the same folder as `train.py`. Example:

```json
{
  "model_sizes": [1, 3, 11],
  "seeds": [42, 123, 456, 789, 555],
  "gcp_project": "my-gcp-project",
  "gcp_zone": "us-central1-a",
  "machine_type": "a2-highgpu-1g",
  "gpu_type": "nvidia-tesla-a100",
  "gpu_count": 1,
  "image_family": "pytorch-2-2-cu121-notebooks-debian-11",
  "image_project": "deeplearning-platform-release",
  "script_repo": "https://github.com/hpieris-dm/vibellama-autotrain.git",
  "repo_dir": "/home/vibellama-ft",
  "requirements": "requirements.txt",
  "wandb_project": "sentiment-sweep",
  "model_hub_namespace": "huggingface user name"
}
```

- **model_sizes**: List of Llama model sizes (in billions) to fine-tune
- **seeds**: Random seeds for repeatability
- **gcp_project**: Your GCP project ID
- **gcp_zone**: Compute zone for VM creation
- **machine_type**: Machine type (e.g. `a2-highgpu-1g`)
- **gpu_type**, **gpu_count**: GPU accelerator settings
- **image_family**, **image_project**: Base image (e.g. Ubuntu 20.04 LTS)
- **script_repo**: URL to your fine-tuning repo
- **repo_dir**: Where to clone the repo on the VM
- **requirements**: Requirements file within the repo
- **wandb_project**: W&B project name for logging
- **model_hub_namespace**: Your HF Hub namespace/user for uploads

---

## Usage

1. Clone or download this controller and place it in a folder.
2. Create and populate `controller-config.json` as shown above.
3. Ensure both `HF_TOKEN` and `WANDB_API_KEY` are set in your environment.
4. Run:
   ```bash
   python3 controller.py --config controller-config.json
   ```
5. The script will iterate over all combinations of model sizes and seeds, launching a VM for each.

## Monitoring

- **Compute Engine**: Track VM creation and status in the GCP Console.
- **W&B**: View live training metrics under the specified project.

---

## Cleanup

VMs automatically shut down after training. If any fail to shut down, delete them manually:
```bash
gcloud compute instances list --filter="name~'llama-.*'" \
  --format="value(name,zone)" | \
  xargs -n2 gcloud compute instances delete -q --zone
```

---

## Troubleshooting

- **Missing credentials**: Ensure `HF_TOKEN` and `WANDB_API_KEY` are exported.
- **Quota errors**: Check GPU quota in the GCP project.
- **Image issues**: Switch to a GPU‑ready DL VM image or build a custom image with CUDA.

---

## License

MIT License.

---

