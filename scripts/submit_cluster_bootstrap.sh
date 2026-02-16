#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
source "$VENV_PATH/bin/activate"

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
ENERGY_MIN_MEV="${ENERGY_MIN_MEV:-70}"
ENERGY_MAX_MEV="${ENERGY_MAX_MEV:-250}"
ENERGY_STEP_MEV="${ENERGY_STEP_MEV:-10}"

SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_TIME="${SBATCH_TIME:-08:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
SBATCH_MEM="${SBATCH_MEM:-16G}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-protonai-sim}"
SBATCH_ARRAY_MAX_PARALLEL="${SBATCH_ARRAY_MAX_PARALLEL:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

python -m src.build_cluster_manifest \
  --train-hom-cases "$TRAIN_HOM" \
  --val-hom-cases "$VAL_HOM" \
  --train-change-cases "$TRAIN_CHG" \
  --val-change-cases "$VAL_CHG" \
  --energy-min-mev "$ENERGY_MIN_MEV" \
  --energy-max-mev "$ENERGY_MAX_MEV" \
  --energy-step-mev "$ENERGY_STEP_MEV" \
  --phantom-root "$PHANTOM_ROOT" \
  --manifest "$MANIFEST"

N_CASES=$(($(wc -l < "$MANIFEST") - 1))
if [[ "$N_CASES" -le 0 ]]; then
  echo "No cases in manifest: $MANIFEST"
  exit 1
fi

echo "Submitting $N_CASES array tasks..."

ARRAY_SPEC="1-$N_CASES"
if [[ -n "$SBATCH_ARRAY_MAX_PARALLEL" ]]; then
  ARRAY_SPEC="1-$N_CASES%$SBATCH_ARRAY_MAX_PARALLEL"
fi

SBATCH_ARGS=(
  --job-name "$SBATCH_JOB_NAME" \
  --partition "$SBATCH_PARTITION" \
  --time "$SBATCH_TIME" \
  --cpus-per-task "$SBATCH_CPUS" \
  --mem "$SBATCH_MEM" \
  --array "$ARRAY_SPEC"
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

sbatch "${SBATCH_ARGS[@]}" \
  scripts/slurm_sim_array.sh \
  "$MANIFEST" \
  "$NOISY" \
  "$TARGET" \
  "$MATERIAL_DB" \
  "$GATE_OUT" \
  "$DATASET_OUT" \
  "$G4ROOT"
