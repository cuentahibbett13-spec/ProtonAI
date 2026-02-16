#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
  source "${VIRTUAL_ENV}/bin/activate"
elif [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

echo "[dota-train] host=$(hostname) pwd=$PWD"
echo "[dota-train] python=$(command -v python)"
python -c "import torch; print(f'[dota-train] torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')"

python -m src.train_dota \
  --train-dir "${TRAIN_DIR:-data/dataset_yuca_atomic_sweep/train}" \
  --val-dir "${VAL_DIR:-data/dataset_yuca_atomic_sweep/val}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --lr "${LR:-1e-4}" \
  --feat-channels "${FEAT_CHANNELS:-32}" \
  --d-model "${D_MODEL:-128}" \
  --nhead "${NHEAD:-8}" \
  --num-layers "${NUM_LAYERS:-4}" \
  --ff-dim "${FF_DIM:-256}" \
  --dropout "${DROPOUT:-0.1}" \
  --high-dose-weight "${HIGH_DOSE_WEIGHT:-4.0}" \
  --threshold-ratio "${THRESHOLD_RATIO:-0.5}" \
  --pdd-loss-weight "${PDD_LOSS_WEIGHT:-0.2}" \
  --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints/dota_atomic_sweep}" \
  --require-cuda
