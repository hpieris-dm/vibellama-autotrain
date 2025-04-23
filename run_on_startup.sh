#!/usr/bin/env bash
set -euo pipefail

# 0) Schedule auto-shutdown after 4 hours
shutdown -h +240 "Auto-shutdown 4h after boot"

# 1) Read metadata into outer-shell variables
MD="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HDR="Metadata-Flavor: Google"

SEED=$(curl -fs -H "$HDR" $MD/seed)
SIZE=$(curl -fs -H "$HDR" $MD/size)
HF_TOKEN=$(curl -fs -H "$HDR" $MD/HF_TOKEN)
WANDB_API_KEY=$(curl -fs -H "$HDR" $MD/WANDB_API_KEY)
REPO_DIR=$(curl -fs -H "$HDR" $MD/repo_dir)
SCRIPT_REPO=$(curl -fs -H "$HDR" $MD/script_repo)
WANDB_PROJECT=$(curl -fs -H "$HDR" $MD/wandb_project)
MODEL_HUB_NAMESPACE=$(curl -fs -H "$HDR" $MD/model_hub_namespace)

# 2) Generate an inner run script under vibellama’s home
INNER_SCRIPT="/home/vibellama/run_training.sh"
cat <<EOF | sudo tee "$INNER_SCRIPT" > /dev/null
#!/usr/bin/env bash
set -euo pipefail

# Export metadata as env vars inside inner script
export SEED="${SEED}"
export SIZE="${SIZE}"
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export REPO_DIR="${REPO_DIR}"
export SCRIPT_REPO="${SCRIPT_REPO}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export MODEL_HUB_NAMESPACE="${MODEL_HUB_NAMESPACE}"

# Load Conda and activate 'base'
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Prepare directories
mkdir -p "\$HOME/logs" "\$HOME/models"

# Clone/update repo
rm -rf "\$REPO_DIR"
git clone "\$SCRIPT_REPO" "\$REPO_DIR"
cd "\$REPO_DIR"

# conditionally set the model name based on size
if [ "\${SIZE}" -eq 1 ]; then
  MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
elif [ "\${SIZE}" -eq 3 ]; then
  MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
elif [ "\${SIZE}" -eq 11 ]; then
  MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
else
  echo "Unsupported model size: \${SIZE}B"
  exit 1
fi

# Launch training in background
nohup python train.py \
  --model-name \${MODEL_NAME} \
  --seed \${SEED} \
  --hf-token \${HF_TOKEN} \
  --wandb-project \${WANDB_PROJECT} \
  --output-dir "\$HOME/models/size\${SIZE}_seed\${SEED}" \
  --model-hub-id "\${MODEL_HUB_NAMESPACE}/VibeLlama-\${SIZE}b-seed-\${SEED}" \
  > "\$HOME/logs/size\${SIZE}_seed\${SEED}.out" 2>&1 &

EOF

# 3) Make it executable and run as vibellama
sudo chmod +x "$INNER_SCRIPT"
sudo chown vibellama:vibellama "$INNER_SCRIPT"
sudo -u vibellama bash -lc "$INNER_SCRIPT"

# 4) Exit startup
exit 0
