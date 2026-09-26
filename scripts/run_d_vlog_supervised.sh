#!/usr/bin/env bash
# Run the supervised D-Vlog pipeline: memorization gate, three-seed training, freeze, and metric evaluation.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Omni-7B}"
DVLOG_ROOT="${DVLOG_ROOT:-/path/to/datasets/dvlog-dataset}"
GPU_CANDIDATES="${GPU_CANDIDATES:-2,3,6}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-20}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
PREP_ROOT="${PREP_ROOT:-${PROJECT_ROOT}/artifacts/d_vlog_supervised/splits}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/evidence/native/d_vlog}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/d_vlog_supervised}"
FREEZE_ROOT="${RUN_ROOT}/frozen"
EVALUATION_ROOT="${RUN_ROOT}/metrics"
SOURCE_VALID_GATE="${RUN_ROOT}/source_valid_gate.json"
LEDGER_PATH="${LEDGER_PATH:-${RUN_ROOT}/test_access_ledger.jsonl}"
OFFICIAL_LABELS="${DVLOG_ROOT}/labels.csv"
SINGLE_RUN="${PROJECT_ROOT}/scripts/run_d_vlog_supervised_single.sh"

for value in "${POLL_INTERVAL_SECONDS}" "${MIN_FREE_MB}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "poll interval and GPU memory threshold must be positive integers" >&2
    exit 2
  fi
done
if [[ ! -f "${OFFICIAL_LABELS}" ]]; then
  echo "official D-Vlog labels/splits file is missing: ${OFFICIAL_LABELS}" >&2
  exit 4
fi
if [[ ! -x "${SINGLE_RUN}" ]]; then
  echo "D-Vlog supervised single-run helper is missing: ${SINGLE_RUN}" >&2
  exit 4
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_CANDIDATES}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_CANDIDATES must contain at least one GPU" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU candidate: ${gpu}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}"
LOCK_DIR="${RUN_ROOT}/.automation.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "another D-Vlog supervised automation holds ${LOCK_DIR}" >&2
  exit 3
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

if [[ -f "${PREP_ROOT}/manifest.json" \
      && -f "${PREP_ROOT}/train_labels.csv" \
      && -f "${PREP_ROOT}/valid_labels.csv" \
      && -f "${PREP_ROOT}/test_ids.csv" ]]; then
  echo "$(date -Is) prepared supervised splits already exist: ${PREP_ROOT}"
elif [[ -e "${PREP_ROOT}" ]]; then
  echo "incomplete supervised split preparation exists: ${PREP_ROOT}" >&2
  exit 4
else
  "${PYTHON}" -m rethink_mh.experiments.d_vlog_supervised_prepare \
    --source "${OFFICIAL_LABELS}" \
    --output-dir "${PREP_ROOT}"
fi

wait_for_gpu() {
  local gpu="$1"
  local free_mb
  while true; do
    free_mb="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    free_mb="${free_mb//[[:space:]]/}"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      return
    fi
    echo "$(date -Is) waiting for GPU ${gpu}: free=${free_mb:-unknown} MiB"
    sleep "${POLL_INTERVAL_SECONDS}"
  done
}

G1_ROOT="${RUN_ROOT}/g1/balanced_8x2_seed42"
if [[ -f "${G1_ROOT}/summary.json" ]]; then
  echo "$(date -Is) completed G1 output exists: ${G1_ROOT}"
elif [[ -e "${G1_ROOT}" ]]; then
  echo "incomplete G1 output exists: ${G1_ROOT}" >&2
  exit 4
else
  wait_for_gpu "${GPUS[0]}"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    MODEL_PATH="${MODEL_PATH}" \
    PREP_ROOT="${PREP_ROOT}" \
    EVIDENCE_ROOT="${EVIDENCE_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    GPU="${GPUS[0]}" \
    MIN_FREE_MB="${MIN_FREE_MB}" \
    "${SINGLE_RUN}" gate 42
fi

"${PYTHON}" - "${G1_ROOT}/summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
if summary.get("experiment") != "qwen_thinker_d_vlog_supervised":
    raise SystemExit(f"D-Vlog G1 experiment mismatch: {path}")
if (
    summary.get("run_kind") != "balanced_train_memorization_gate"
    or summary.get("dataset") != "d_vlog"
    or int(summary.get("seed", -1)) != 42
):
    raise SystemExit(f"D-Vlog G1 identity mismatch: {path}")
counts = summary.get("session_counts", {})
if (
    int(counts.get("fit", -1)) != 16
    or int(counts.get("validation", -1)) != 16
    or int(counts.get("test", -1)) != 0
):
    raise SystemExit(f"D-Vlog G1 subset coverage mismatch: {path}")
fit_digest = summary.get("fit_ids_sha256")
validation_digest = summary.get("validation_ids_sha256")
if (
    not isinstance(fit_digest, str)
    or len(fit_digest) != 64
    or any(character not in "0123456789abcdef" for character in fit_digest)
    or fit_digest != validation_digest
):
    raise SystemExit(f"D-Vlog G1 validation is not the fit subset: {path}")
boundaries = summary.get("fit_boundaries", {})
if (
    boundaries.get("validation_is_fit_memorization_gate") is not True
    or boundaries.get("test_labels_accessed") is not False
):
    raise SystemExit(f"D-Vlog G1 test-label boundary failed: {path}")
if summary.get("memorization_gate", {}).get("passed") is not True:
    raise SystemExit(f"D-Vlog G1 did not pass: {path}")
print("D-Vlog balanced train-only G1 passed; three-seed training is permitted.")
PY

run_seed() {
  local seed="$1"
  local gpu="$2"
  local output="${RUN_ROOT}/seeds/seed${seed}"
  if [[ -f "${output}/summary.json" && -f "${output}/predictions.jsonl" ]]; then
    echo "$(date -Is) completed supervised seed exists: ${output}"
    return
  fi
  if [[ -e "${output}" ]]; then
    echo "incomplete supervised seed output exists: ${output}" >&2
    return 4
  fi
  wait_for_gpu "${gpu}"
  echo "$(date -Is) starting D-Vlog supervised seed=${seed} on GPU ${gpu}"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    MODEL_PATH="${MODEL_PATH}" \
    PREP_ROOT="${PREP_ROOT}" \
    EVIDENCE_ROOT="${EVIDENCE_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    GPU="${gpu}" \
    MIN_FREE_MB="${MIN_FREE_MB}" \
    "${SINGLE_RUN}" train "${seed}"
}

SEEDS=(42 43 44)
run_lane() {
  local lane="$1"
  local gpu="${GPUS[${lane}]}"
  local index
  for ((index = lane; index < ${#SEEDS[@]}; index += ${#GPUS[@]})); do
    run_seed "${SEEDS[${index}]}" "${gpu}"
  done
}

lane_pids=()
lane_count="${#GPUS[@]}"
if (( lane_count > ${#SEEDS[@]} )); then
  lane_count="${#SEEDS[@]}"
fi
for ((lane = 0; lane < lane_count; lane++)); do
  run_lane "${lane}" &
  lane_pids+=("$!")
done
lane_status=0
for pid in "${lane_pids[@]}"; do
  wait "${pid}" || lane_status=$?
done
if (( lane_status != 0 )); then
  exit "${lane_status}"
fi

"${PYTHON}" - \
  "${RUN_ROOT}/seeds/seed42/summary.json" \
  "${RUN_ROOT}/seeds/seed43/summary.json" \
  "${RUN_ROOT}/seeds/seed44/summary.json" \
  "${SOURCE_VALID_GATE}" <<'PY'
import json
import math
import pathlib
import sys

summary_paths = [pathlib.Path(value) for value in sys.argv[1:4]]
output_path = pathlib.Path(sys.argv[4])
runs = []
for expected_seed, path in zip((42, 43, 44), summary_paths):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("experiment") != "qwen_thinker_d_vlog_supervised"
        or summary.get("run_kind") != "independent_train_dev_fit"
        or summary.get("dataset") != "d_vlog"
        or int(summary.get("seed", -1)) != expected_seed
    ):
        raise SystemExit(f"supervised seed identity mismatch: {path}")
    counts = summary.get("session_counts", {})
    if (
        int(counts.get("all_train", -1)) != 647
        or int(counts.get("fit", -1)) != 647
        or int(counts.get("validation", -1)) != 102
        or int(counts.get("test", -1)) != 212
    ):
        raise SystemExit(f"supervised seed split coverage mismatch: {path}")
    boundaries = summary.get("fit_boundaries", {})
    if any(
        (
            boundaries.get("dev_gradient_updates") is not False,
            boundaries.get("test_gradient_updates") is not False,
            boundaries.get("test_labels_accessed") is not False,
        )
    ):
        raise SystemExit(f"supervised seed fit boundary mismatch: {path}")
    training = summary.get("training", {})
    checkpoint = training.get("checkpoint_selection", {})
    selected_epoch = checkpoint.get("selected_epoch")
    if any(
        (
            training.get("class_weighting") is not False,
            int(training.get("maximum_epochs", -1)) != 4,
            training.get("head_type") != "binary",
            checkpoint.get("metric") != "log_loss",
            not isinstance(selected_epoch, int),
            isinstance(selected_epoch, bool),
            not (1 <= int(selected_epoch) <= 4),
        )
    ):
        raise SystemExit(f"supervised seed optimization mismatch: {path}")
    metric_block = summary.get("metrics", {})
    if set(metric_block) != {"validation", "validation_score_diagnostics"}:
        raise SystemExit(f"test metrics were exposed before freeze: {path}")
    metrics = metric_block["validation"]
    diagnostics = metric_block["validation_score_diagnostics"]
    count = int(metrics["count"])
    positive_count = int(metrics["positive_count"])
    if count != 102 or not 0 < positive_count < count:
        raise SystemExit(f"official-valid label coverage mismatch: {path}")
    prevalence = positive_count / count
    constant_brier = prevalence * (1.0 - prevalence)
    constant_log_loss = -(
        prevalence * math.log(prevalence)
        + (1.0 - prevalence) * math.log(1.0 - prevalence)
    )
    probability_range = float(diagnostics["probability_max"]) - float(
        diagnostics["probability_min"]
    )
    checks = {
        "roc_auc_above_random": float(metrics["roc_auc"]) > 0.50,
        "average_precision_above_prior_plus_0_03": float(
            metrics["average_precision"]
        )
        > prevalence + 0.03,
        "brier_better_than_constant_prior": float(metrics["brier"])
        < constant_brier,
        "log_loss_better_than_constant_prior": float(metrics["log_loss"])
        < constant_log_loss,
        "probabilities_not_collapsed": int(
            diagnostics["unique_probability_count"]
        )
        >= min(10, count)
        and probability_range >= 0.05,
    }
    runs.append(
        {
            "seed": expected_seed,
            "selected_epoch": selected_epoch,
            "checks": checks,
            "passed": all(checks.values()),
            "validation": metrics,
        }
    )

payload = {
    "schema_version": "1.0.0",
    "gate": "G2_d_vlog_official_valid_three_seed",
    "experiment": "qwen_thinker_d_vlog_supervised",
    "dataset": "d_vlog",
    "expected_seeds": [42, 43, 44],
    "runs": runs,
    "passed": all(run["passed"] for run in runs),
    "test_labels_accessed": False,
}
if not payload["passed"]:
    raise SystemExit("one or more D-Vlog source-valid seed gates failed")
if output_path.exists():
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit(f"existing source-valid gate differs: {output_path}")
else:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
print("All three official-valid source gates passed; blind aggregation is permitted.")
PY

if [[ -f "${FREEZE_ROOT}/manifest.json" \
      && -f "${FREEZE_ROOT}/blind_predictions.jsonl" ]]; then
  echo "$(date -Is) completed supervised prediction freeze exists: ${FREEZE_ROOT}"
elif [[ -e "${FREEZE_ROOT}" ]]; then
  echo "incomplete supervised prediction freeze exists: ${FREEZE_ROOT}" >&2
  exit 4
else
  "${PYTHON}" -m rethink_mh.experiments.d_vlog_supervised_aggregate \
    --seed-dir \
      "${RUN_ROOT}/seeds/seed42" \
      "${RUN_ROOT}/seeds/seed43" \
      "${RUN_ROOT}/seeds/seed44" \
    --output-dir "${FREEZE_ROOT}"
fi

if [[ ! -f "${FREEZE_ROOT}/manifest.json" \
      || ! -f "${FREEZE_ROOT}/blind_predictions.jsonl" ]]; then
  echo "supervised aggregation did not publish a complete freeze" >&2
  exit 5
fi

if [[ -f "${EVALUATION_ROOT}/metrics.json" && -f "${LEDGER_PATH}" ]]; then
  echo "$(date -Is) completed supervised metrics exist: ${EVALUATION_ROOT}"
elif [[ -e "${EVALUATION_ROOT}" ]]; then
  echo "incomplete supervised metric output exists: ${EVALUATION_ROOT}" >&2
  exit 4
else
  "${PYTHON}" -m rethink_mh.experiments.d_vlog_supervised_evaluate \
    --predictions "${FREEZE_ROOT}/blind_predictions.jsonl" \
    --freeze-manifest "${FREEZE_ROOT}/manifest.json" \
    --official-labels "${OFFICIAL_LABELS}" \
    --output-dir "${EVALUATION_ROOT}" \
    --ledger "${LEDGER_PATH}" \
    --expected-test-count 212 \
    --bootstrap-samples 2000 \
    --bootstrap-seed 3722
fi

if [[ ! -f "${EVALUATION_ROOT}/metrics.json" || ! -f "${LEDGER_PATH}" ]]; then
  echo "metric-only evaluator did not publish metrics and its unlock ledger" >&2
  exit 5
fi

echo "$(date -Is) D-Vlog supervised protocol completed; automation is exiting"
