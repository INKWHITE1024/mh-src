#!/usr/bin/env bash
# Wait for a free GPU, then dispatch the requested training mode.
set -euo pipefail

DATASET="${1:?usage: wait_for_gpu_and_train.sh daic_woz|e_daic [full|oof|transcript-full|native-oof|evidence-literacy-single-fold|evidence-literacy-oof|full-loop|contract-recovery|loop-train|loop-crossfit] [fold_index]}"
MODE="${2:-full}"
FOLD_INDEX="${3:-}"

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-20}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
GPU_CANDIDATES="${GPU_CANDIDATES:-}"
MAX_POLLS="${MAX_POLLS:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCK_ROOT="${LOCK_ROOT:-${PROJECT_ROOT}/outputs/gpu_locks}"

if [[ "${DATASET}" != "daic_woz" && "${DATASET}" != "e_daic" ]]; then
  echo "unsupported dataset: ${DATASET}" >&2
  exit 2
fi
if [[ "${MODE}" != "full" \
  && "${MODE}" != "oof" \
  && "${MODE}" != "transcript-full" \
  && "${MODE}" != "native-oof" \
  && "${MODE}" != "evidence-literacy-single-fold" \
  && "${MODE}" != "evidence-literacy-oof" \
  && "${MODE}" != "full-loop" \
  && "${MODE}" != "contract-recovery" \
  && "${MODE}" != "loop-crossfit" \
  && "${MODE}" != "loop-train" ]]; then
  echo "unsupported training mode: ${MODE}" >&2
  exit 2
fi
if [[ ( "${MODE}" == "oof" \
  || "${MODE}" == "native-oof" \
  || "${MODE}" == "evidence-literacy-oof" \
  || "${MODE}" == "full-loop" \
  || "${MODE}" == "contract-recovery" \
  || "${MODE}" == "loop-crossfit" ) \
  && ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "oof mode requires a numeric fold_index" >&2
  exit 2
fi
for value in \
  "${POLL_INTERVAL_SECONDS}" \
  "${STABLE_POLLS}" \
  "${MIN_FREE_MB}" \
  "${MAX_GPU_UTIL}" \
  "${MAX_POLLS}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "polling and GPU thresholds must be non-negative integers" >&2
    exit 2
  fi
done
if (( POLL_INTERVAL_SECONDS <= 0 || STABLE_POLLS <= 0 || MIN_FREE_MB <= 0 )); then
  echo "poll interval, stable polls, and free-memory threshold must be positive" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for collision-safe local GPU claiming" >&2
  exit 2
fi

mkdir -p "${LOCK_ROOT}"
declare -A streaks=()
poll_count=0

gpu_is_allowed() {
  local gpu="$1"
  if [[ -z "${GPU_CANDIDATES}" ]]; then
    return 0
  fi
  case ",${GPU_CANDIDATES}," in
    *",${gpu},"*) return 0 ;;
    *) return 1 ;;
  esac
}

gpu_snapshot() {
  nvidia-smi \
    --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
}

gpu_still_eligible() {
  local wanted_gpu="$1"
  local index free_mb utilization
  while IFS=, read -r index free_mb utilization; do
    index="${index//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if [[ "${index}" == "${wanted_gpu}" ]]; then
      [[ "${free_mb}" =~ ^[0-9]+$ ]] || return 1
      [[ "${utilization}" =~ ^[0-9]+$ ]] || return 1
      (( free_mb >= MIN_FREE_MB && utilization <= MAX_GPU_UTIL ))
      return
    fi
  done < <(gpu_snapshot)
  return 1
}

start_training() {
  local gpu="$1"
  echo "$(date -Is) GPU ${gpu} passed ${STABLE_POLLS} consecutive checks; entering ${MODE} training for ${DATASET}."
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "$(date -Is) DRY_RUN=1; selection and lock test passed without launching training."
    exit 0
  fi
  if [[ "${MODE}" == "full" ]]; then
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      "${PROJECT_ROOT}/scripts/run_qwen_label_independent.sh" "${DATASET}"
  fi
  if [[ "${MODE}" == "transcript-full" ]]; then
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      "${PROJECT_ROOT}/scripts/run_qwen_transcript_independent.sh" "${DATASET}"
  fi
  if [[ "${MODE}" == "native-oof" ]]; then
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      "${PROJECT_ROOT}/scripts/run_native_oof_fold.sh" \
      "${DATASET}" "${FOLD_INDEX}"
  fi
  if [[ "${MODE}" == "evidence-literacy-single-fold" ]]; then
    if [[ "${DATASET}" != "daic_woz" ]]; then
      echo "evidence-literacy-single-fold currently permits daic_woz only" >&2
      exit 2
    fi
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      "${PROJECT_ROOT}/scripts/run_evidence_literacy_single_fold.sh"
  fi
  if [[ "${MODE}" == "evidence-literacy-oof" ]]; then
    if [[ "${DATASET}" != "daic_woz" ]]; then
      echo "evidence-literacy-oof currently permits daic_woz only" >&2
      exit 2
    fi
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      FOLD_INDEX="${FOLD_INDEX}" \
      "${PROJECT_ROOT}/scripts/run_evidence_literacy_oof_fold.sh"
  fi
  if [[ "${MODE}" == "full-loop" ]]; then
    if [[ "${DATASET}" != "daic_woz" ]]; then
      echo "full-loop currently permits daic_woz only" >&2
      exit 2
    fi
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      FOLD_INDEX="${FOLD_INDEX}" \
      "${PROJECT_ROOT}/scripts/run_full_loop_fold.sh"
  fi
  if [[ "${MODE}" == "contract-recovery" ]]; then
    if [[ "${DATASET}" != "daic_woz" ]]; then
      echo "contract-recovery currently permits daic_woz only" >&2
      exit 2
    fi
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      FOLD_INDEX="${FOLD_INDEX}" \
      "${PROJECT_ROOT}/scripts/run_contract_recovery_fold.sh"
  fi
  if [[ "${MODE}" == "loop-train" ]]; then
    if [[ "${DATASET}" != "daic_woz" ]]; then
      echo "loop-train currently permits daic_woz only" >&2
      exit 2
    fi
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      "${PROJECT_ROOT}/scripts/run_full_loop_post_training.sh"
  fi
  if [[ "${MODE}" == "loop-crossfit" ]]; then
    if [[ "${DATASET}" != "daic_woz" ]]; then
      echo "loop-crossfit currently permits daic_woz only" >&2
      exit 2
    fi
    exec env \
      GPU="${gpu}" \
      MIN_FREE_MB="${MIN_FREE_MB}" \
      FOLD_INDEX="${FOLD_INDEX}" \
      "${PROJECT_ROOT}/scripts/run_loop_crossfit_fold.sh"
  fi
  exec env \
    GPU="${gpu}" \
    MIN_FREE_MB="${MIN_FREE_MB}" \
    "${PROJECT_ROOT}/scripts/run_qwen_label_oof_fold.sh" \
    "${DATASET}" "${FOLD_INDEX}"
}

echo "$(date -Is) waiting for GPU: dataset=${DATASET} mode=${MODE} interval=${POLL_INTERVAL_SECONDS}s stable_polls=${STABLE_POLLS} min_free_mb=${MIN_FREE_MB} max_util=${MAX_GPU_UTIL} candidates=${GPU_CANDIDATES:-all}"

while true; do
  poll_count=$((poll_count + 1))
  snapshot="$(gpu_snapshot)"
  eligible_summary=()
  selected_gpu=""
  selected_free=-1

  while IFS=, read -r index free_mb utilization; do
    index="${index//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "${index}" =~ ^[0-9]+$ ]] || continue
    if ! gpu_is_allowed "${index}"; then
      streaks["${index}"]=0
      continue
    fi
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] \
      && [[ "${utilization}" =~ ^[0-9]+$ ]] \
      && (( free_mb >= MIN_FREE_MB && utilization <= MAX_GPU_UTIL )); then
      streaks["${index}"]=$(( ${streaks["${index}"]:-0} + 1 ))
      eligible_summary+=("gpu=${index}:free=${free_mb}:util=${utilization}:streak=${streaks["${index}"]}")
      if (( streaks["${index}"] >= STABLE_POLLS && free_mb > selected_free )); then
        selected_gpu="${index}"
        selected_free="${free_mb}"
      fi
    else
      streaks["${index}"]=0
    fi
  done <<< "${snapshot}"

  echo "$(date -Is) poll=${poll_count} eligible=${eligible_summary[*]:-none}"

  if [[ -n "${selected_gpu}" ]]; then
    lock_file="${LOCK_ROOT}/gpu_${selected_gpu}.lock"
    exec {lock_fd}>"${lock_file}"
    if flock -n "${lock_fd}"; then
      if gpu_still_eligible "${selected_gpu}"; then
        start_training "${selected_gpu}"
      fi
      echo "$(date -Is) GPU ${selected_gpu} changed state during final recheck; continuing."
      flock -u "${lock_fd}"
    else
      echo "$(date -Is) GPU ${selected_gpu} is held by another RETHINK-MH waiter."
    fi
    eval "exec ${lock_fd}>&-"
    streaks["${selected_gpu}"]=0
  fi

  if (( MAX_POLLS > 0 && poll_count >= MAX_POLLS )); then
    echo "$(date -Is) reached MAX_POLLS=${MAX_POLLS} without starting training."
    exit 5
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done
