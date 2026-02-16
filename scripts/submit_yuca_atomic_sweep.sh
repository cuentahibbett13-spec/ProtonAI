#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
source "$VENV_PATH/bin/activate"

# Dataset size (atomic beamlets)
export TRAIN_HOM="${TRAIN_HOM:-20}"
export VAL_HOM="${VAL_HOM:-5}"
export TRAIN_CHG="${TRAIN_CHG:-40}"
export VAL_CHG="${VAL_CHG:-10}"

# Monte Carlo statistics
export NOISY="${NOISY:-30000}"
export TARGET="${TARGET:-300000}"

# Energy sweep
export ENERGY_MIN_MEV="${ENERGY_MIN_MEV:-70}"
export ENERGY_MAX_MEV="${ENERGY_MAX_MEV:-250}"
export ENERGY_STEP_MEV="${ENERGY_STEP_MEV:-10}"

# Paths
export MANIFEST="${MANIFEST:-data/manifests/yuca_atomic_sweep_manifest.csv}"
export PHANTOM_ROOT="${PHANTOM_ROOT:-data/phantoms_yuca_atomic_sweep}"
export GATE_OUT="${GATE_OUT:-data/gate/yuca_atomic_sweep}"
export DATASET_OUT="${DATASET_OUT:-data/dataset_yuca_atomic_sweep}"

# Cluster policy (override with env vars as needed)
export SBATCH_PARTITION="${SBATCH_PARTITION:-normal}"
export SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
export SBATCH_CPUS="${SBATCH_CPUS:-4}"
export SBATCH_MEM="${SBATCH_MEM:-16G}"
export SBATCH_GRES="${SBATCH_GRES:-}"
export SBATCH_ARRAY_MAX_PARALLEL="${SBATCH_ARRAY_MAX_PARALLEL:-20}"
export SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-protonai-yuca-atomic}"

# Optional Yuca-specific fields
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
export SBATCH_QOS="${SBATCH_QOS:-normal}"
export SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

bash scripts/submit_cluster_bootstrap.sh
