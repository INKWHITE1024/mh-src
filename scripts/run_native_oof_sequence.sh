#!/usr/bin/env bash
# Drive the native strict-OOF pipeline: prepare folds, train seeds, and compare against the frozen reference.
set -euo pipefail

PHASE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets/DAIC-WOZ}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-${PROJECT_ROOT}/artifacts/transcripts/compact/daic_woz}"
FROZEN_REFERENCE_ARTIFACT_ROOT="${FROZEN_REFERENCE_ARTIFACT_ROOT:?set the frozen reference fold-artifact root}"
FROZEN_REFERENCE_RESULT_ROOT="${FROZEN_REFERENCE_RESULT_ROOT:?set the frozen reference OOF result root}"
FROZEN_REFERENCE_MANIFEST_ROOT="${FROZEN_REFERENCE_MANIFEST_ROOT:?set the frozen reference result-manifest root}"
ARTIFACT_GROUP="${ARTIFACT_GROUP:-native_from_scratch}"
RUN_GROUP="${RUN_GROUP:-qwen_native_from_scratch}"
MANIFEST_GROUP="${MANIFEST_GROUP:-qwen_native_from_scratch_manifests}"
COMPARISON_GROUP="${COMPARISON_GROUP:-matched_oof_comparison}"
FOLDS=5
SPLIT_SEED=42
SEEDS=(42 43 44)

case "${PHASE}" in
  prepare|seed42|train-all|compare|all) ;;
  *)
    echo "usage: run_native_oof_sequence.sh [prepare|seed42|train-all|compare|all]" >&2
    exit 2
    ;;
esac

required_paths=(
  "${PROJECT_ROOT}"
  "${DATA_ROOT}/train_split_Depression_AVEC2017.csv"
  "${TRANSCRIPT_ROOT}"
  "${FROZEN_REFERENCE_ARTIFACT_ROOT}"
  "${FROZEN_REFERENCE_RESULT_ROOT}"
  "${FROZEN_REFERENCE_MANIFEST_ROOT}"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required path is missing: ${path}" >&2
    exit 4
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PROJECT_ROOT PYTHON DATA_ROOT TRANSCRIPT_ROOT
export FROZEN_REFERENCE_ARTIFACT_ROOT ARTIFACT_GROUP RUN_GROUP
export FOLDS SPLIT_SEED

verify_frozen_reference() {
  local seed result_dir manifest
  for seed in "${SEEDS[@]}"; do
    result_dir="${FROZEN_REFERENCE_RESULT_ROOT}/seed${seed}/aggregate"
    manifest="${FROZEN_REFERENCE_MANIFEST_ROOT}/seed${seed}.manifest.json"
    "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${result_dir}" \
      --verify-manifest "${manifest}" >/dev/null
  done
  echo "$(date -Is) frozen reference content manifests verified."
}

prepare_all() {
  local fold
  for ((fold = 0; fold < FOLDS; fold++)); do
    "${PROJECT_ROOT}/scripts/prepare_native_oof_fold.sh" \
      daic_woz "${fold}"
  done
}

validate_fold_summary() {
  local path="$1"
  local seed="$2"
  local fold="$3"
  "${PYTHON}" - "${path}" "${seed}" "${fold}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
seed, fold = int(sys.argv[2]), int(sys.argv[3])
summary = json.loads(path.read_text(encoding="utf-8"))
if any(
    (
        summary.get("experiment")
        != "qwen_thinker_av_transcript_label_training_native",
        summary.get("evidence_protocol") != "native",
        int(summary.get("seed", -1)) != seed,
        int(summary.get("split_seed", -1)) != 42,
        int(summary.get("fold_index", -1)) != fold,
        summary.get("fit_boundaries", {}).get("test_labels_accessed") is not False,
        summary.get("training", {}).get("class_weighting") is not False,
        int(summary.get("training", {}).get("epochs", -1)) != 2,
        int(summary.get("training", {}).get("maximum_epochs", -1)) != 2,
    )
):
    raise SystemExit(f"invalid completed native fold: {path}")
PY
}

train_seed() {
  local seed="$1"
  local fold output_dir summary
  local -a pids=()
  local -a labels=()
  for ((fold = 0; fold < FOLDS; fold++)); do
    output_dir="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${seed}/fold${fold}"
    summary="${output_dir}/summary.json"
    if [[ -f "${summary}" ]]; then
      validate_fold_summary "${summary}" "${seed}" "${fold}"
      echo "$(date -Is) verified completed seed=${seed} fold=${fold}; skipping."
      continue
    fi
    if [[ -e "${output_dir}" ]]; then
      echo "partial fold output exists and requires manual audit: ${output_dir}" >&2
      exit 4
    fi
    env TRAIN_SEED="${seed}" \
      "${PROJECT_ROOT}/scripts/wait_for_gpu_and_train.sh" \
      daic_woz native-oof "${fold}" &
    pids+=("$!")
    labels+=("seed=${seed}/fold=${fold}")
  done
  local index failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      echo "training waiter failed: ${labels[${index}]}" >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    return 1
  fi
  for ((fold = 0; fold < FOLDS; fold++)); do
    summary="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${seed}/fold${fold}/summary.json"
    validate_fold_summary "${summary}" "${seed}" "${fold}"
  done
}

aggregate_seed() {
  local seed="$1"
  local aggregate_dir manifest_dir manifest_path fold
  aggregate_dir="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${seed}/aggregate"
  manifest_dir="${PROJECT_ROOT}/outputs/${MANIFEST_GROUP}/daic_woz"
  manifest_path="${manifest_dir}/seed${seed}.manifest.json"
  if [[ ! -f "${aggregate_dir}/summary.json" ]]; then
    local -a fold_args=()
    for ((fold = 0; fold < FOLDS; fold++)); do
      fold_args+=(
        "${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${seed}/fold${fold}"
      )
    done
    "${PYTHON}" -m rethink_mh.experiments.qwen_oof_aggregate \
      --fold-dir "${fold_args[@]}" \
      --train-labels "${DATA_ROOT}/train_split_Depression_AVEC2017.csv" \
      --evidence-root \
        "${PROJECT_ROOT}/artifacts/${ARTIFACT_GROUP}/daic_woz/seed42/fold0/evidence/native/daic_woz" \
      --evidence-protocol native \
      --label-column PHQ8_Score \
      --label-threshold 10 \
      --output-dir "${aggregate_dir}"
  fi
  mkdir -p "${manifest_dir}"
  if [[ -f "${manifest_path}" ]]; then
    "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${aggregate_dir}" \
      --verify-manifest "${manifest_path}" >/dev/null
  else
    "${PYTHON}" -m rethink_mh.experiments.frozen_artifact_manifest \
      --root "${aggregate_dir}" \
      --output "${manifest_path}" >/dev/null
  fi
  echo "$(date -Is) native aggregate verified and frozen: seed=${seed}"
}

compare_all() {
  local comparison_dir seed
  comparison_dir="${PROJECT_ROOT}/outputs/${COMPARISON_GROUP}/daic_woz"
  if [[ -e "${comparison_dir}" ]]; then
    echo "comparison output already exists; refusing implicit overwrite: ${comparison_dir}" >&2
    exit 4
  fi
  local -a frozen_dirs=()
  local -a frozen_manifests=()
  local -a native_dirs=()
  local -a native_manifests=()
  for seed in "${SEEDS[@]}"; do
    frozen_dirs+=("${FROZEN_REFERENCE_RESULT_ROOT}/seed${seed}/aggregate")
    frozen_manifests+=("${FROZEN_REFERENCE_MANIFEST_ROOT}/seed${seed}.manifest.json")
    native_dirs+=(
      "${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed${seed}/aggregate"
    )
    native_manifests+=(
      "${PROJECT_ROOT}/outputs/${MANIFEST_GROUP}/daic_woz/seed${seed}.manifest.json"
    )
  done
  "${PYTHON}" -m rethink_mh.experiments.matched_oof_comparison \
    --reference-dir "${frozen_dirs[@]}" \
    --reference-manifest "${frozen_manifests[@]}" \
    --native-dir "${native_dirs[@]}" \
    --native-manifest "${native_manifests[@]}" \
    --expected-seed "${SEEDS[@]}" \
    --output-dir "${comparison_dir}" \
    --bootstrap-samples 5000 \
    --bootstrap-seed 3725
}

verify_frozen_reference

if [[ "${PHASE}" == "prepare" ]]; then
  prepare_all
elif [[ "${PHASE}" == "seed42" ]]; then
  prepare_all
  train_seed 42
  aggregate_seed 42
elif [[ "${PHASE}" == "train-all" ]]; then
  prepare_all
  for seed in "${SEEDS[@]}"; do
    train_seed "${seed}"
    aggregate_seed "${seed}"
  done
elif [[ "${PHASE}" == "compare" ]]; then
  compare_all
else
  prepare_all
  for seed in "${SEEDS[@]}"; do
    train_seed "${seed}"
    aggregate_seed "${seed}"
  done
  compare_all
fi
