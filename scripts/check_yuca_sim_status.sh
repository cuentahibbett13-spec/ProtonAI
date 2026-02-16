#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-../Modular3/.venv}"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  VENV_PATH=".venv"
fi
source "$VENV_PATH/bin/activate"

MANIFEST="${MANIFEST:-data/manifests/yuca_atomic_sweep_manifest.csv}"
GATE_OUT="${GATE_OUT:-data/gate/yuca_atomic_sweep}"
DATASET_OUT="${DATASET_OUT:-data/dataset_yuca_atomic_sweep}"
STRICT_ENERGY="${STRICT_ENERGY:-0}"

CMD=(python -m src.verify_cluster_simulations
  --manifest "$MANIFEST"
  --gate-output-root "$GATE_OUT"
  --dataset-root "$DATASET_OUT")

if [[ "$STRICT_ENERGY" == "1" ]]; then
  CMD+=(--strict-energy)
fi

"${CMD[@]}"

echo

echo "Quick counters:"
echo -n " train npz: "
find "$DATASET_OUT/train" -name '*.npz' 2>/dev/null | wc -l
echo -n " val npz:   "
find "$DATASET_OUT/val" -name '*.npz' 2>/dev/null | wc -l
