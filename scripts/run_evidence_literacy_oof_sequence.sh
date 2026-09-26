#!/usr/bin/env bash
# Run Evidence Literacy training for outer folds 1-4 and aggregate reviewer routes.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
PYTHON="${PYTHON:-python}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?set the frozen initial-judgment run root}"
FOLD0_LITERACY_SUMMARY="${FOLD0_LITERACY_SUMMARY:?set frozen fold-0 heldout summary}"
RUN_GROUP="${RUN_GROUP:-evidence_literacy_oof}"
ARTIFACT_GROUP="${ARTIFACT_GROUP:-evidence_literacy_oof}"
MANIFEST_GROUP="${MANIFEST_GROUP:-evidence_literacy_oof_manifests}"
ROUTE_OUTPUT="${ROUTE_OUTPUT:-${PROJECT_ROOT}/outputs/${RUN_GROUP}/reviewer_routes.json}"
read -r -a FOLDS <<< "${LITERACY_FOLDS:-1 2 3 4}"

if [[ ! -f "${FOLD0_LITERACY_SUMMARY}" ]]; then
  echo "frozen fold-0 literacy summary is missing: ${FOLD0_LITERACY_SUMMARY}" >&2
  exit 3
fi
if [[ "${#FOLDS[@]}" -eq 0 ]]; then
  echo "LITERACY_FOLDS cannot be empty" >&2
  exit 2
fi
for fold in "${FOLDS[@]}"; do
  if [[ ! "${fold}" =~ ^[0-9]+$ ]] || (( fold < 1 || fold > 4 )); then
    echo "LITERACY_FOLDS must contain only outer folds 1 through 4" >&2
    exit 2
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for fold in "${FOLDS[@]}"; do
  echo "$(date -Is) preparing Evidence Literacy outer fold ${fold}"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
    FOLD_INDEX="${fold}" \
    ARTIFACT_GROUP="${ARTIFACT_GROUP}" \
    "${PROJECT_ROOT}/scripts/prepare_evidence_literacy_oof_fold.sh"

  complete="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed42/fold${fold}/PIPELINE_COMPLETE"
  if [[ -f "${complete}" ]]; then
    echo "$(date -Is) verified existing Evidence Literacy outer fold ${fold}"
    continue
  fi
  echo "$(date -Is) queueing Evidence Literacy outer fold ${fold}"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
    ARTIFACT_GROUP="${ARTIFACT_GROUP}" \
    RUN_GROUP="${RUN_GROUP}" \
    MANIFEST_GROUP="${MANIFEST_GROUP}" \
    EXPERIMENT_NAME="evidence_literacy_oof" \
    "${PROJECT_ROOT}/scripts/wait_for_gpu_and_train.sh" \
    daic_woz evidence-literacy-oof "${fold}"
done

fold_args=()
for fold in 0 1 2 3 4; do
  assignment="${SOURCE_RUN_ROOT}/artifacts/native_from_scratch/daic_woz/seed42/fold${fold}/assignment.csv"
  if (( fold == 0 )); then
    summary="${FOLD0_LITERACY_SUMMARY}"
  else
    summary="${PROJECT_ROOT}/outputs/${RUN_GROUP}/daic_woz/seed42/fold${fold}/heldout_contract_gate/summary.json"
  fi
  for required in "${assignment}" "${summary}"; do
    if [[ ! -f "${required}" ]]; then
      echo "reviewer-route input is missing: ${required}" >&2
      exit 4
    fi
  done
  fold_args+=(--fold-result "${fold}=${assignment}=${summary}")
done

if [[ -f "${ROUTE_OUTPUT}" ]]; then
  "${PYTHON}" - "${ROUTE_OUTPUT}" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
route = json.load(open(path, encoding="utf-8"))
if route.get("route_protocol_version") != "evidence-literacy-reviewer-route":
    raise SystemExit("existing reviewer route has another protocol")
if route.get("fold_count") != 5 or route.get("session_count") != 107:
    raise SystemExit("existing reviewer route has incomplete OOF coverage")
for fold in route.get("routes", []):
    summary = fold["heldout_summary_path"]
    observed = hashlib.sha256(open(summary, "rb").read()).hexdigest()
    if observed != fold["heldout_summary_sha256"]:
        raise SystemExit(f"heldout summary changed for fold {fold['fold_index']}")
PY
else
  "${PYTHON}" -m rethink_mh.experiments.evidence_literacy_aggregate \
    --dataset daic_woz \
    --seed 42 \
    "${fold_args[@]}" \
    --output "${ROUTE_OUTPUT}"
fi

touch "${PROJECT_ROOT}/outputs/${RUN_GROUP}/OOF_COMPLETE"
echo "$(date -Is) five-fold Evidence Literacy reviewer routes are complete."
