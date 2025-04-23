#!/usr/bin/env bash
set -euo pipefail

# 0) Schedule auto‐shutdown in 4 hours for safety
shutdown -h +240 "Auto‐shutdown 4h after boot"

# 1) Read metadata
MD="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HDR="Metadata-Flavor: Google"

export SEED=$(curl -fs -H "$HDR" $MD/seed)
export SIZE=$(curl -fs -H "$HDR" $MD/size)
export HF_TOKEN=$(curl -fs -H "$HDR" $MD/HF_TOKEN)
export WANDB_API_KEY=$(curl -fs -H "$HDR" $MD/WANDB_API_KEY)

export REPO_DIR=$(curl -fs -H "$HDR" $MD/repo_dir)
export SCRIPT_REPO=$(curl -fs -H "$HDR" $MD/script_repo)
export WANDB_PROJECT=$(curl -fs -H "$HDR" $MD/wandb_project)
export MODEL_HUB_NAMESPACE=$(curl -fs -H "$HDR" $MD/model_hub_namespace)

# 2) Switch to vibellama as a login shell and launch in background
sudo -iu vibellama bash <<EOSU
set -euo pipefail

# 2a) Load Conda and activate your fine‐tune env
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# 2b) Prepare directories under /home/vibellama
mkdir -p "\$HOME/logs" "\$HOME/models"

# 2c) Clone or update the repo
rm -rf "\$REPO_DIR"
git clone "\$SCRIPT_REPO" "\$REPO_DIR"
cd "\$REPO_DIR"

# 2d) Launch training in background
nohup python train.py \
  --model-name meta-llama/Llama-3.2-\${SIZE}B-Instruct \
  --seed \${SEED} \
  --hf-token \${HF_TOKEN} \
  --wandb-project \${WANDB_PROJECT} \
  --output-dir "\$HOME/models/size\${SIZE}_seed\${SEED}" \
  --model-hub-id "\${MODEL_HUB_NAMESPACE}/VibeLlama-\${SIZE}b-seed-\${SEED}" \
  > "\$HOME/logs/size\${SIZE}_seed\${SEED}.out" 2>&1 &

# 2e) Exit so GCE marks the startup script as done
exit 0
EOSU
