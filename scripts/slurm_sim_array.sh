#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <manifest.csv> <noisy_primaries> <target_primaries> <material_db> <gate_output_root> <dataset_root> [geant4_data_root]"
  exit 1
fi

MANIFEST="$1"
NOISY="$2"
TARGET="$3"
MATERIAL_DB="$4"
GATE_OUT="$5"
DATASET_OUT="$6"
G4ROOT="${7:-}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Run with sbatch --array=1-N"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
  source "${VIRTUAL_ENV}/bin/activate"
elif [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] host=$(hostname) pwd=$PWD"
echo "[task ${SLURM_ARRAY_TASK_ID}] python=$(command -v python)"

if ! python -c "import opengate" >/dev/null 2>&1; then
  echo "[task ${SLURM_ARRAY_TASK_ID}] ERROR: cannot import opengate in current python environment"
  exit 1
fi

CMD=(
  python -m src.run_cluster_case
  --manifest "$MANIFEST"
  --task-id "$SLURM_ARRAY_TASK_ID"
  --noisy-primaries "$NOISY"
  --target-primaries "$TARGET"
  --material-db "$MATERIAL_DB"
  --gate-output-root "$GATE_OUT"
  --dataset-root "$DATASET_OUT"
)

if [[ -n "$G4ROOT" ]]; then
  CMD+=(--geant4-data-root "$G4ROOT")
fi

"${CMD[@]}"
