# ProtonAI MVP

MVP para denoising 3D de dosis en protonterapia con enfoque physics-aware.

## Objetivo
- Entrada (2 canales): dosis ruidosa (1M eventos) + mapa de densidad física.
- Target: dosis alta estadística (50M eventos).
- Modelo: U-Net 3D simple y legible.
- Loss: Weighted MSE que penaliza más la región de Bragg (`target > 0.5 * max(target)`).

## Estructura
- `src/simulate_gate10.py`: simulación GATE 10 vía API Python (`opengate`).
- `src/run_gate10_pair.py`: corre automáticamente caso ruidoso 1M + target 50M.
- `src/generate_sandwich_phantom.py`: genera fantoma voxelizado 128³ (1 mm³) en MHD/RAW.
- `src/convert_mhd_to_npz.py`: convierte salidas MHD/RAW a `.npz` para entrenamiento.
- `src/dataset.py`: dataset PyTorch.
- `src/model_unet3d.py`: Physics-Aware U-Net 3D.
- `src/losses.py`: Weighted MSE Bragg-aware.
- `src/train.py`: entrenamiento MVP.
- `scripts/setup_venv.sh`: setup de entorno `.venv`.

## Requisitos
- Python 3.10+
- PyTorch con ROCm
- GATE 10 Python API (`opengate`)

Instalación mínima con `.venv`:

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

## 1) Generar fantoma sandwich 128³

```bash
python3 -m src.generate_sandwich_phantom --output-dir data/phantom
```

Esto genera:
- `data/phantom/sandwich_labels.mhd/.raw`
- `data/phantom/sandwich_density.mhd/.raw`
- `data/phantom/labels_to_materials.txt`

## 2) Simulación en GATE 10 usando Python

Flujo recomendado (sin macros):

```bash
python3 -m src.run_gate10_pair
```

Esto genera:
- `data/gate/noisy_dose.mhd`
- `data/gate/target_dose.mhd`
- `data/gate/density_map.mhd`

Si quieres correr una sola simulación:

```bash
python3 -m src.simulate_gate10 --output-dose data/gate/noisy_dose.mhd --primaries 1000000
python3 -m src.simulate_gate10 --output-dose data/gate/target_dose.mhd --primaries 50000000
```

## 3) Convertir a NPZ

```bash
python3 -m src.convert_mhd_to_npz \
  --noisy-dose-mhd data/gate/noisy_dose.mhd \
  --target-dose-mhd data/gate/target_dose.mhd \
  --density-mhd data/gate/density_map.mhd \
  --output data/dataset/sample_0001.npz
```

## 4) Entrenamiento MVP

Coloca tus `.npz` en `data/dataset/train` y `data/dataset/val`.
Cada archivo debe contener llaves: `noisy_dose`, `target_dose`, `density`, `spacing`.

```bash
python3 -m src.train \
  --train-dir data/dataset/train \
  --val-dir data/dataset/val \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --high-dose-weight 4.0
```

## Iteración conservadora (recomendada para arrancar)

Genera un mini dataset para validar el flujo end-to-end rápido:

```bash
python3 -m src.generate_conservative_dataset \
  --train-cases 3 \
  --val-cases 1 \
  --noisy-primaries 20000 \
  --target-primaries 200000
```

Luego entrena pocas épocas:

```bash
python3 -m src.train \
  --train-dir data/dataset/train \
  --val-dir data/dataset/val \
  --epochs 5 \
  --batch-size 1 \
  --lr 1e-4
```

## Curriculum Stage 1 (homogéneo)

Genera fantoma homogéneo (agua) + dataset train/val conservador:

```bash
python3 -m src.run_stage1_homogeneous \
  --train-cases 2 \
  --val-cases 1 \
  --noisy-primaries 10000 \
  --target-primaries 100000
```

## Bootstrap PDD (homogéneo + un cambio)

Para arrancar entrenamiento con más variedad pero aún iterando rápido:

```bash
python3 -m src.generate_pdd_bootstrap_dataset \
  --train-hom-cases 8 \
  --val-hom-cases 2 \
  --train-change-cases 12 \
  --val-change-cases 3 \
  --noisy-primaries 20000 \
  --target-primaries 200000
```

Entrenamiento sobre ese dataset:

```bash
python3 -m src.train \
  --train-dir data/dataset_pdd_bootstrap/train \
  --val-dir data/dataset_pdd_bootstrap/val \
  --epochs 10 \
  --batch-size 1 \
  --lr 1e-4
```

## PDD rápido (val o train)

Para graficar PDD desde un `.npz` (compara noisy vs target):

```bash
python3 -m src.plot_pdd \
  --npz data/dataset/val/val_0000.npz \
  --output-dir outputs/pdd/val_0000
```

Salida:
- `outputs/pdd/val_0000/pdd.png`
- `outputs/pdd/val_0000/pdd.csv`

## Evaluación PDD (pred vs target en validación)

```bash
python3 -m src.evaluate_pdd_validation \
  --val-dir data/dataset_pdd_bootstrap/val \
  --checkpoint checkpoints/pdd_bootstrap_e10/best.pt \
  --output-dir outputs/pdd_eval/bootstrap_e10
```

Salida:
- `outputs/pdd_eval/bootstrap_e10/summary.csv`
- `outputs/pdd_eval/bootstrap_e10/report.txt`
- `outputs/pdd_eval/bootstrap_e10/plots/*.png`

## Cluster: simulaciones en paralelo (SLURM)

1) Enviar todas las simulaciones con `job array`:

```bash
bash scripts/submit_cluster_bootstrap.sh
```

2) Ajustar escala/primarios por variables de entorno (ejemplo):

```bash
TRAIN_HOM=20 VAL_HOM=5 TRAIN_CHG=40 VAL_CHG=10 NOISY=30000 TARGET=300000 \
ENERGY_MIN_MEV=70 ENERGY_MAX_MEV=250 ENERGY_STEP_MEV=10 \
SBATCH_PARTITION=gpu SBATCH_TIME=24:00:00 SBATCH_GRES=gpu:1 \
bash scripts/submit_cluster_bootstrap.sh
```

Esto genera:
- manifiesto de casos en `data/manifests/*.csv`
- simulaciones y MHD en `data/gate/pdd_bootstrap_cluster/*`
- dataset NPZ en `data/dataset_pdd_bootstrap_cluster/{train,val}`

Nota: el manifiesto incluye `energy_mev` por caso (barrido 70–250 MeV por defecto en pasos de 10), y cada NPZ guarda ese metadato como `energy_mev`.

Scripts clave:
- `src/build_cluster_manifest.py`: crea el manifiesto
- `src/run_cluster_case.py`: ejecuta un caso por `task-id`
- `scripts/slurm_sim_array.sh`: worker del array
- `scripts/submit_cluster_bootstrap.sh`: launcher completo
- `scripts/submit_yuca_atomic_sweep.sh`: launcher recomendado para Yuca (atómico + sweep)

Puedes limitar paralelismo para no saturar el cluster:

```bash
SBATCH_ARRAY_MAX_PARALLEL=20 bash scripts/submit_cluster_bootstrap.sh
```

### Launcher rápido para Yuca

```bash
bash scripts/submit_yuca_atomic_sweep.sh
```

Por defecto los scripts de cluster intentan usar `../Modular3/.venv` y, si no existe, hacen fallback a `.venv`.
Puedes fijar uno explícito con:

```bash
VENV_PATH=../Modular3/.venv bash scripts/submit_yuca_atomic_sweep.sh
```

Con cuenta/QOS/constraint del clúster:

```bash
SBATCH_ACCOUNT=mi_cuenta SBATCH_QOS=normal SBATCH_CONSTRAINT=mi210 \
bash scripts/submit_yuca_atomic_sweep.sh
```

## DoTA (secuencia causal) - inicio de migración

Se agregó una primera implementación de arquitectura secuencial inspirada en DoTA:
- `src/model_dota.py`: encoder 2D por slice + Transformer causal por profundidad + token de energía.
- `src/dataset_sequence.py`: dataset secuencial (`CT/densidad`, energía, dosis target).
- `src/train_dota.py`: entrenamiento end-to-end de DoTA.

Ejemplo de entrenamiento (crop recomendado para memoria):

```bash
python3 -m src.train_dota \
  --train-dir data/dataset_pdd_bootstrap/train \
  --val-dir data/dataset_pdd_bootstrap/val \
  --epochs 10 \
  --batch-size 1 \
  --crop-shape 128,24,24 \
  --d-model 128 \
  --num-layers 4
```

## Notas ROCm
- En ROCm, PyTorch suele exponer GPU como `cuda`.
- El script usa `torch.device("cuda")` cuando está disponible.

## Nota de compatibilidad Gate 10
- Si tu instalación de `opengate` usa nombres de actor/campos ligeramente distintos según versión, ajustamos `src/simulate_gate10.py` a tu instalación exacta en una iteración rápida.

## Siguiente paso sugerido
- Cuando este MVP esté validado, añadimos Fase 2 con TC reales y métricas dosimétricas avanzadas.
