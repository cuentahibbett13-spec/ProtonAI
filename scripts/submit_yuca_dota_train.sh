#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-${VIRTUAL_ENV:-$ROOT_DIR/.venv}}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Python environment not found at: $VENV_PATH"
  echo "Set VENV_PATH=/path/to/venv and rerun."
  exit 1
fi
source "$VENV_PATH/bin/activate"
export VENV_PATH

SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-dota-train-atomic}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

SBATCH_ARGS=(
  --job-name "$SBATCH_JOB_NAME"
  --chdir "$ROOT_DIR"
  --export "ALL,VENV_PATH=$VENV_PATH"
  --partition "$SBATCH_PARTITION"
  --time "$SBATCH_TIME"
  --cpus-per-task "$SBATCH_CPUS"
  --mem "$SBATCH_MEM"
)

if [[ -n "$SBATCH_GRES" ]]; then
  SBATCH_ARGS+=(--gres "$SBATCH_GRES")
fi
if [[ -n "$SBATCH_ACCOUNT" ]]; then
  SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi
if [[ -n "$SBATCH_QOS" ]]; then
  SBATCH_ARGS+=(--qos "$SBATCH_QOS")
fi
if [[ -n "$SBATCH_CONSTRAINT" ]]; then
  SBATCH_ARGS+=(--constraint "$SBATCH_CONSTRAINT")
fi

echo "SLURM train config:"
echo "  partition=$SBATCH_PARTITION account=${SBATCH_ACCOUNT:-<none>} qos=${SBATCH_QOS:-<none>} gres=${SBATCH_GRES:-<none>}"
echo "  cpus=$SBATCH_CPUS mem=$SBATCH_MEM time=$SBATCH_TIME"
echo "  job_name=$SBATCH_JOB_NAME"

sbatch "${SBATCH_ARGS[@]}" scripts/slurm_train_dota.sh
