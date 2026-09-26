#!/usr/bin/env bash
# Run one supervised D-Vlog gate or training job on a chosen GPU.
set -euo pipefail

MODE="${1:?usage: run_d_vlog_supervised_single.sh gate|train [seed]}"
SEED="${2:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
GPU="${GPU:?set GPU to one physical GPU index}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
PREP_ROOT="${PREP_ROOT:-${PROJECT_ROOT}/artifacts/d_vlog_supervised/splits}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/evidence/native/d_vlog}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/d_vlog_supervised}"
TASK="${PROJECT_ROOT}/configs/tasks/d_vlog_current_depression.json"

if [[ "${MODE}" != "gate" && "${MODE}" != "train" ]]; then
  echo "mode must be gate or train" >&2
  exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if [[ "${MODE}" == "gate" && "${SEED}" != "42" ]]; then
  echo "the D-Vlog G1 memorization gate is frozen to seed 42" >&2
  exit 2
fi
required_splits=("${PREP_ROOT}/train_labels.csv")
if [[ "${MODE}" == "train" ]]; then
  required_splits+=(
    "${PREP_ROOT}/valid_labels.csv"
    "${PREP_ROOT}/test_ids.csv"
  )
fi
for path in "${required_splits[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "prepared D-Vlog split is missing: ${path}" >&2
    exit 4
  fi
done
if [[ ! -d "${EVIDENCE_ROOT}" ]]; then
  echo "D-Vlog Evidence root is missing: ${EVIDENCE_ROOT}" >&2
  exit 4
fi

free_mb="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits)"
free_mb="${free_mb//[[:space:]]/}"
if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
  echo "GPU ${GPU} has ${free_mb} MiB free; ${MIN_FREE_MB} MiB is required" >&2
  exit 3
fi

if [[ "${MODE}" == "gate" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/g1/balanced_8x2_seed42}"
  EPOCHS=50
else
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/seeds/seed${SEED}}"
  EPOCHS=4
fi
LOG="${PROJECT_ROOT}/outputs/logs/d_vlog_supervised.${MODE}.seed${SEED}.log"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 4
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${LOG}")"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

command=(
  "${PYTHON}" -m rethink_mh.experiments.qwen_label_baseline
  --dataset d_vlog
  --model-path "${MODEL_PATH}"
  --task "${TASK}"
  --evidence-root "${EVIDENCE_ROOT}"
  --train-labels "${PREP_ROOT}/train_labels.csv"
  --label-column target
  --label-threshold 0.5
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --evidence-density compact
  --max-evidence-tokens 2000
  --max-transcript-tokens 3000
  --max-input-tokens 4800
  --head-type binary
  --checkpoint-metric log_loss
  --epochs "${EPOCHS}"
  --batch-size 1
  --eval-batch-size 1
  --gradient-accumulation 8
  --learning-rate 5e-5
  --head-learning-rate 1e-5
  --scheduler-type linear
  --warmup-ratio 0.10
  --weight-decay 0
  --lora-rank 8
  --lora-alpha 16
  --max-grad-norm 1
  --device cuda:0
  --no-class-weighting
)

if [[ "${MODE}" == "gate" ]]; then
  command+=(
    --memorization-per-class 8
    --subset-seed 42
    --lora-dropout 0
  )
else
  command+=(
    --dev-labels "${PREP_ROOT}/valid_labels.csv"
    --test-labels "${PREP_ROOT}/test_ids.csv"
    --lora-dropout 0.05
  )
fi

"${command[@]}" 2>&1 | tee "${LOG}"
