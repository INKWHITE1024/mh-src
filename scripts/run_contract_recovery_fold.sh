#!/usr/bin/env bash
# Collect one fold's full-loop contract-recovery trajectories on a chosen GPU.
set -euo pipefail

GPU="${GPU:?set one physical GPU index}"
FOLD_INDEX="${FOLD_INDEX:?set outer FOLD_INDEX}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}}"
PYTHON="${PYTHON:-python}"
PREPARED_INFERENCE_DIR="${PREPARED_INFERENCE_DIR:-${PROJECT_ROOT}/artifacts/full_loop/prepared/inference}"
OUTPUT_GROUP="${OUTPUT_GROUP:-contract_recovery_raw_oof}"
OUTPUT_DIR="${RUN_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/fold${FOLD_INDEX}"
TASK_PATH="${TASK_PATH:-${PROJECT_ROOT}/configs/tasks/phq8_depression_binary.json}"
SESSION_IDS_FILE="${SESSION_IDS_FILE:-}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"

if [[ ! "${FOLD_INDEX}" =~ ^[0-9]+$ ]] \
  || (( FOLD_INDEX < 0 || FOLD_INDEX > 4 )); then
  echo "FOLD_INDEX must be an integer from 0 to 4" >&2
  exit 2
fi
for required in \
  "${PREPARED_INFERENCE_DIR}/manifest.json" \
  "${PREPARED_INFERENCE_DIR}/inference_plan.jsonl" \
  "${TASK_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "contract-recovery collection input is missing: ${required}" >&2
    exit 3
  fi
done
if [[ -n "${SESSION_IDS_FILE}" && ! -f "${SESSION_IDS_FILE}" ]]; then
  echo "SESSION_IDS_FILE does not exist: ${SESSION_IDS_FILE}" >&2
  exit 3
fi
if [[ -d "${OUTPUT_DIR}" && ! -f "${OUTPUT_DIR}/manifest.json" ]]; then
  echo "partial contract-recovery fold requires manual audit: ${OUTPUT_DIR}" >&2
  exit 4
fi

session_args=()
selected_session_ids=()
if [[ -n "${SESSION_IDS_FILE}" ]]; then
  while IFS= read -r session_id || [[ -n "${session_id}" ]]; do
    if [[ -z "${session_id}" || ! "${session_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "invalid session identifier in ${SESSION_IDS_FILE}" >&2
      exit 2
    fi
    selected_session_ids+=("${session_id}")
    session_args+=(--session-id "${session_id}")
  done < "${SESSION_IDS_FILE}"
  if (( ${#selected_session_ids[@]} == 0 )); then
    echo "SESSION_IDS_FILE is empty: ${SESSION_IDS_FILE}" >&2
    exit 2
  fi
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${LOG_ROOT}"

if [[ ! -f "${OUTPUT_DIR}/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    -m rethink_mh.experiments.loop_collect \
    --prepared-dir "${PREPARED_INFERENCE_DIR}" \
    --task "${TASK_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold-index "${FOLD_INDEX}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attention-implementation sdpa \
    --max-new-tokens 640 \
    --max-input-tokens 8192 \
    --max-contract-retries 2 \
    "${session_args[@]}" \
    2>&1 | tee \
      "${LOG_ROOT}/${OUTPUT_GROUP}.fold${FOLD_INDEX}.collect.log"
fi

"${PYTHON}" - \
  "${OUTPUT_DIR}" \
  "${FOLD_INDEX}" \
  "${SESSION_IDS_FILE:-__ALL__}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
fold = int(sys.argv[2])
session_file = sys.argv[3]
expected_ids = (
    []
    if session_file == "__ALL__"
    else pathlib.Path(session_file).read_text(encoding="utf-8").splitlines()
)
expected_limited = bool(expected_ids)
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
raw = root / "trajectories.raw.jsonl"
rows = [
    json.loads(line)
    for line in raw.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
problems = []
if manifest.get("collection_protocol_version") != (
    "loop-collection"
):
    problems.append("collection protocol")
contract_policy = manifest.get("contract_policy", {})
if any(
    (
        contract_policy.get("initial_projection_changes_probability")
        is not False,
        contract_policy.get("initial_projection_substitutes_identifiers")
        is not False,
        contract_policy.get("reflection")
        != "dedicated_no_atomic_evidence_contract",
        contract_policy.get("legacy_artifacts_modified") is not False,
    )
):
    problems.append("contract policy")
if manifest.get("outer_fold") != fold:
    problems.append("outer fold")
if manifest.get("limited_check") is not expected_limited:
    problems.append("limited-check identity")
if manifest.get("session_count") != len(rows):
    problems.append("session count")
if hashlib.sha256(raw.read_bytes()).hexdigest() != (
    manifest.get("files", {})
    .get("trajectories.raw.jsonl", {})
    .get("sha256")
):
    problems.append("raw SHA-256")
if expected_ids:
    if manifest.get("selected_session_ids") != expected_ids:
        problems.append("selected session manifest")
    if (
        len(rows) != len(expected_ids)
        or {str(row.get("session_id")) for row in rows}
        != set(expected_ids)
    ):
        problems.append("selected session rows")
elif "selected_session_ids" in manifest:
    problems.append("unexpected selected sessions")
if any(
    row.get("collection_protocol_version")
    != "loop-collection"
    or row.get("outer_fold") != fold
    or row.get("ground_truth_used") is not False
    for row in rows
):
    problems.append("row identity")
if problems:
    raise SystemExit(
        f"invalid completed contract-recovery fold {root}: {', '.join(problems)}"
    )
PY

touch "${OUTPUT_DIR}/PIPELINE_COMPLETE"
echo "$(date -Is) contract-recovery raw collection complete: fold=${FOLD_INDEX}."
