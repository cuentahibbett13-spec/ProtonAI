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
G4ROOT="${7:-/home/fer/geant4_install/geant4-install/share/Geant4/data}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Run with sbatch --array=1-N"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-../Modular3/.venv}"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  VENV_PATH=".venv"
fi
source "$VENV_PATH/bin/activate"

python -m src.run_cluster_case \
  --manifest "$MANIFEST" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --noisy-primaries "$NOISY" \
  --target-primaries "$TARGET" \
  --material-db "$MATERIAL_DB" \
  --gate-output-root "$GATE_OUT" \
  --dataset-root "$DATASET_OUT" \
  --geant4-data-root "$G4ROOT"
