#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

MANIFEST="${MANIFEST:-data/manifests/pdd_bootstrap_manifest.csv}"
PHANTOM_ROOT="${PHANTOM_ROOT:-data/phantoms_pdd_bootstrap_cluster}"
GATE_OUT="${GATE_OUT:-data/gate/pdd_bootstrap_cluster}"
DATASET_OUT="${DATASET_OUT:-data/dataset_pdd_bootstrap_cluster}"
MATERIAL_DB="${MATERIAL_DB:-gate/materials/sandwich_materials.db}"
G4ROOT="${G4ROOT:-/home/fer/geant4_install/geant4-install/share/Geant4/data}"

TRAIN_HOM="${TRAIN_HOM:-8}"
VAL_HOM="${VAL_HOM:-2}"
TRAIN_CHG="${TRAIN_CHG:-12}"
VAL_CHG="${VAL_CHG:-3}"
NOISY="${NOISY:-20000}"
TARGET="${TARGET:-200000}"

SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_TIME="${SBATCH_TIME:-08:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
SBATCH_MEM="${SBATCH_MEM:-16G}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-protonai-sim}"

python -m src.build_cluster_manifest \
  --train-hom-cases "$TRAIN_HOM" \
  --val-hom-cases "$VAL_HOM" \
  --train-change-cases "$TRAIN_CHG" \
  --val-change-cases "$VAL_CHG" \
  --phantom-root "$PHANTOM_ROOT" \
  --manifest "$MANIFEST"

N_CASES=$(($(wc -l < "$MANIFEST") - 1))
if [[ "$N_CASES" -le 0 ]]; then
  echo "No cases in manifest: $MANIFEST"
  exit 1
fi

echo "Submitting $N_CASES array tasks..."

sbatch \
  --job-name "$SBATCH_JOB_NAME" \
  --partition "$SBATCH_PARTITION" \
  --time "$SBATCH_TIME" \
  --cpus-per-task "$SBATCH_CPUS" \
  --mem "$SBATCH_MEM" \
  --gres "$SBATCH_GRES" \
  --array "1-$N_CASES" \
  scripts/slurm_sim_array.sh \
  "$MANIFEST" \
  "$NOISY" \
  "$TARGET" \
  "$MATERIAL_DB" \
  "$GATE_OUT" \
  "$DATASET_OUT" \
  "$G4ROOT"
