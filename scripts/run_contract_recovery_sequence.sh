#!/usr/bin/env bash
# Drive contract-recovery full-loop collection across GPUs.
set -euo pipefail

PHASE="${1:?usage: run_contract_recovery_sequence.sh failure-batch|full}"
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/rethink-mh}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}}"
PYTHON="${PYTHON:-python}"
PREPARED_INFERENCE_DIR="${PREPARED_INFERENCE_DIR:-${PROJECT_ROOT}/artifacts/full_loop/prepared/inference}"
PRIOR_RAW_OOF_DIR="${PRIOR_RAW_OOF_DIR:-}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-20}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,4}"
EXPECTED_FAILURE_COUNT="${EXPECTED_FAILURE_COUNT:-30}"

if [[ "${PHASE}" != "failure-batch" && "${PHASE}" != "full" ]]; then
  echo "phase must be failure-batch or full" >&2
  exit 2
fi
for value in \
  "${POLL_INTERVAL_SECONDS}" \
  "${STABLE_POLLS}" \
  "${MIN_FREE_MB}" \
  "${MAX_GPU_UTIL}" \
  "${EXPECTED_FAILURE_COUNT}" \
  "${EXPECTED_FAILURE_COUNT}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "sequence thresholds must be non-negative integers" >&2
    exit 2
  fi
done
if [[ "${PHASE}" == "failure-batch" && -z "${PRIOR_RAW_OOF_DIR}" ]]; then
  echo "failure-batch requires PRIOR_RAW_OOF_DIR" >&2
  exit 2
fi

if [[ "${PHASE}" == "failure-batch" ]]; then
  OUTPUT_GROUP="${OUTPUT_GROUP:-contract_recovery_failure_batch}"
else
  OUTPUT_GROUP="${OUTPUT_GROUP:-contract_recovery_raw_oof}"
fi
CONTROL_ROOT="${CONTROL_ROOT:-${RUN_ROOT}/control}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
SELECTION_ROOT="${CONTROL_ROOT}/initial_failures"
GATE_SUMMARY="${CONTROL_ROOT}/contract_gate.json"
MERGED_DIR="${MERGED_DIR:-${RUN_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/raw_oof_frozen}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${CONTROL_ROOT}" "${LOG_ROOT}"

if [[ "${PHASE}" == "failure-batch" ]]; then
  "${PYTHON}" - \
    "${PRIOR_RAW_OOF_DIR}" \
    "${SELECTION_ROOT}" \
    "${EXPECTED_FAILURE_COUNT}" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
expected_count = int(sys.argv[3])
manifest_path = source / "manifest.json"
raw_path = source / "trajectories.raw.jsonl"
frozen = source / "RAW_OOF_FROZEN"
if not all(path.is_file() for path in (manifest_path, raw_path, frozen)):
    raise SystemExit(f"frozen prior raw OOF is incomplete: {source}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("merge_protocol_version")
    != "loop-raw-oof-freeze"
    or manifest.get("collection_protocol_version")
    != "loop-collection"
    or manifest.get("files", {})
    .get("trajectories.raw.jsonl", {})
    .get("sha256")
    != hashlib.sha256(raw_path.read_bytes()).hexdigest()
):
    raise SystemExit("frozen prior raw OOF failed provenance verification")
rows = [
    json.loads(line)
    for line in raw_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
failed = sorted(
    (
        int(row["outer_fold"]),
        str(row["session_id"]),
    )
    for row in rows
    if row.get("status") == "initial_failed"
)
if len(failed) != expected_count:
    raise SystemExit(
        f"expected {expected_count} initial-failed sessions, "
        f"found {len(failed)}"
    )
if any(fold not in range(5) for fold, _session_id in failed):
    raise SystemExit("an initial-failed session has an invalid outer fold")
output.mkdir(parents=True, exist_ok=True)
for fold in range(5):
    identifiers = [
        session_id
        for row_fold, session_id in failed
        if row_fold == fold
    ]
    if not identifiers:
        raise SystemExit(f"initial-failed sessions omit fold {fold}")
    (output / f"fold{fold}.txt").write_text(
        "".join(f"{identifier}\n" for identifier in identifiers),
        encoding="utf-8",
    )
selection = {
    "schema_version": "1.0.0",
    "selection_basis": "frozen_status_equals_initial_failed",
    "outcome_fields_read": [],
    "expected_failure_count": expected_count,
    "selected_failure_count": len(failed),
    "session_ids_by_fold": {
        str(fold): [
            session_id
            for row_fold, session_id in failed
            if row_fold == fold
        ]
        for fold in range(5)
    },
    "source": {
        "prior_raw_oof_dir": str(source.resolve()),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    },
}
(output / "selection.json").write_text(
    json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
fi


pids=()
labels=()
for fold in 0 1 2 3 4; do
  complete="${RUN_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/fold${fold}/PIPELINE_COMPLETE"
  if [[ -f "${complete}" ]]; then
    echo "$(date -Is) verified existing contract-recovery fold ${fold}; skipping."
    continue
  fi
  session_file=""
  if [[ "${PHASE}" == "failure-batch" ]]; then
    session_file="${SELECTION_ROOT}/fold${fold}.txt"
  fi
  echo "$(date -Is) queueing contract-recovery ${PHASE} fold ${fold}."
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    PYTHON="${PYTHON}" \
    PREPARED_INFERENCE_DIR="${PREPARED_INFERENCE_DIR}" \
    OUTPUT_GROUP="${OUTPUT_GROUP}" \
    SESSION_IDS_FILE="${session_file}" \
    POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS}" \
    STABLE_POLLS="${STABLE_POLLS}" \
    MIN_FREE_MB="${MIN_FREE_MB}" \
    MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
    GPU_CANDIDATES="${GPU_CANDIDATES}" \
    LOCK_ROOT="${RUN_ROOT}/control/gpu_locks" \
    "${PROJECT_ROOT}/scripts/wait_for_gpu_and_train.sh" \
    daic_woz contract-recovery "${fold}" &
  pids+=("$!")
  labels+=("fold=${fold}")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[${index}]}"; then
    echo "contract-recovery waiter failed: ${labels[${index}]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 5
fi

if [[ "${PHASE}" == "failure-batch" ]]; then
  "${PYTHON}" - \
    "${RUN_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz" \
    "${SELECTION_ROOT}/selection.json" \
    "${GATE_SUMMARY}" \
    "${PREPARED_INFERENCE_DIR}" <<'PY'
import collections
import hashlib
import json
import pathlib
import sys

fold_root = pathlib.Path(sys.argv[1])
selection_path = pathlib.Path(sys.argv[2])
summary_path = pathlib.Path(sys.argv[3])
prepared_inference_dir = pathlib.Path(sys.argv[4])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
plan_path = prepared_inference_dir / "inference_plan.jsonl"
plan_rows = [
    json.loads(line)
    for line in plan_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
plan_by_id = {
    str(row["session_id"]): row
    for row in plan_rows
}
expected_by_fold = selection["session_ids_by_fold"]
expected_ids = {
    session_id
    for identifiers in expected_by_fold.values()
    for session_id in identifiers
}
rows = []
problems = []
for fold in range(5):
    directory = fold_root / f"fold{fold}"
    manifest_path = directory / "manifest.json"
    raw_path = directory / "trajectories.raw.jsonl"
    if not manifest_path.is_file() or not raw_path.is_file():
        problems.append(f"fold{fold}:missing output")
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold_rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = expected_by_fold[str(fold)]
    contract_policy = manifest.get("contract_policy", {})
    if (
        manifest.get("collection_protocol_version")
        != "loop-collection"
        or contract_policy.get("initial_projection_changes_probability")
        is not False
        or contract_policy.get("initial_projection_substitutes_identifiers")
        is not False
        or contract_policy.get("reflection")
        != "dedicated_no_atomic_evidence_contract"
        or contract_policy.get("legacy_artifacts_modified") is not False
        or manifest.get("limited_check") is not True
        or manifest.get("selected_session_ids") != expected
        or manifest.get("session_count") != len(expected)
        or manifest.get("files", {})
        .get("trajectories.raw.jsonl", {})
        .get("sha256")
        != hashlib.sha256(raw_path.read_bytes()).hexdigest()
    ):
        problems.append(f"fold{fold}:manifest provenance")
    if {str(row.get("session_id")) for row in fold_rows} != set(expected):
        problems.append(f"fold{fold}:session coverage")
    rows.extend(fold_rows)

projection_count = 0
repair_counts = collections.Counter()
dropped_values = 0
reflection_ok = 0
all_arm_ok = 0
for row in rows:
    session_id = str(row.get("session_id", ""))
    prefix = f"session={session_id}"
    if (
        row.get("collection_protocol_version")
        != "loop-collection"
        or row.get("ground_truth_used") is not False
    ):
        problems.append(f"{prefix}:row identity")
        continue
    if row.get("status") != "complete":
        problems.append(f"{prefix}:status={row.get('status')}")
    initial = row.get("initial")
    if not isinstance(initial, dict):
        problems.append(f"{prefix}:initial missing")
        continue
    assessment = initial.get("initial_assessment")
    visible = row.get("collection_segment_ids")
    if not isinstance(assessment, dict) or not isinstance(visible, list):
        problems.append(f"{prefix}:initial shape")
        continue
    plan = plan_by_id.get(session_id)
    if not isinstance(plan, dict):
        problems.append(f"{prefix}:inference plan missing")
        continue
    from rethink_mh.rethinking.native_query_retrieval import (
        NativeAtomicRetriever,
    )

    initial_allowed_ids = list(
        NativeAtomicRetriever.from_session_dir(
            pathlib.Path(plan["evidence"]["session_dir"])
        ).available_segment_ids
    )
    initial_allowed = set(initial_allowed_ids)
    reflection_allowed = set(visible)
    if (
        initial.get("initial_probability_anchor")
        != assessment.get("risk_probability")
    ):
        problems.append(f"{prefix}:probability anchor")
    for field in (
        "supporting_segment_ids",
        "contradictory_segment_ids",
        "uncertain_segment_ids",
        "requested_segment_ids",
    ):
        values = assessment.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or not set(values).issubset(initial_allowed)
        ):
            problems.append(f"{prefix}:{field} grounding")
    repair_counts[int(initial.get("contract_repair_count", -1))] += 1
    audit = initial.get("initial_grounding_audit")
    if not isinstance(audit, dict):
        problems.append(f"{prefix}:grounding audit missing")
    elif audit.get("applied") is True:
        projection_count += 1
        dropped = audit.get(
            "dropped_nonallowlisted_segment_values_by_field"
        )
        preserved = audit.get("preserved_segment_ids_by_field")
        if (
            audit.get("risk_probability_anchor_preserved") is not True
            or audit.get("other_fields_modified") is not False
            or audit.get("allowed_segment_ids") != initial_allowed_ids
            or not isinstance(dropped, dict)
            or not isinstance(preserved, dict)
        ):
            problems.append(f"{prefix}:projection audit invariant")
        else:
            flattened_dropped = [
                value for values in dropped.values() for value in values
            ]
            flattened_preserved = [
                value for values in preserved.values() for value in values
            ]
            dropped_values += len(flattened_dropped)
            if set(flattened_dropped) & initial_allowed:
                problems.append(f"{prefix}:allowed ID dropped")
            if not set(flattened_preserved).issubset(initial_allowed):
                problems.append(f"{prefix}:unknown ID preserved")
        raw_completion = audit.get("raw_completion")
        if (
            not isinstance(raw_completion, str)
            or hashlib.sha256(raw_completion.encode("utf-8")).hexdigest()
            != audit.get("raw_completion_sha256")
        ):
            problems.append(f"{prefix}:raw completion audit")
        projected_text = json.dumps(
            assessment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if hashlib.sha256(projected_text.encode("utf-8")).hexdigest() != (
            audit.get("projected_assessment_sha256")
        ):
            problems.append(f"{prefix}:projected assessment audit")
    elif (
        audit.get("applied") is not False
        or audit.get("policy_version")
        != "initial-grounding-recovery"
    ):
        problems.append(f"{prefix}:non-projection audit")

    arms = row.get("arms")
    if not isinstance(arms, list):
        problems.append(f"{prefix}:arms missing")
        continue
    if all(arm.get("status") == "ok" for arm in arms):
        all_arm_ok += 1
    reflection = next(
        (
            arm
            for arm in arms
            if arm.get("arm_id") == "reflection_no_new_evidence"
        ),
        None,
    )
    if not isinstance(reflection, dict) or reflection.get("status") != "ok":
        problems.append(f"{prefix}:reflection status")
        continue
    revision = reflection.get("workflow", {}).get("revision_assessment", {})
    evidence_fields = (
        "cited_evidence_ids",
        "preserved_evidence_ids",
        "newly_considered_evidence_ids",
        "rejected_evidence_ids",
    )
    cited_segments = revision.get("cited_segment_ids")
    if (
        reflection.get("workflow", {}).get("revision_contract_policy")
        != "reflection-no-new-atomic-evidence"
        or any(revision.get(field) != [] for field in evidence_fields)
        or not isinstance(cited_segments, list)
        or not cited_segments
        or not set(cited_segments).issubset(reflection_allowed)
    ):
        problems.append(f"{prefix}:reflection grounding")
    else:
        reflection_ok += 1

actual_ids = {str(row.get("session_id", "")) for row in rows}
if actual_ids != expected_ids or len(rows) != len(expected_ids):
    problems.append("global session coverage")
passed = not problems
summary = {
    "schema_version": "1.0.0",
    "audit_protocol_version": "contract-recovery-audit",
    "passed": passed,
    "selection_basis": selection["selection_basis"],
    "outcome_fields_read": [],
    "expected_session_count": len(expected_ids),
    "observed_session_count": len(rows),
    "initial_contract_recovered_count": sum(
        row.get("status") != "initial_failed" for row in rows
    ),
    "initial_projection_count": projection_count,
    "initial_contract_repair_count_distribution": {
        str(key): value for key, value in sorted(repair_counts.items())
    },
    "dropped_nonallowlisted_value_count": dropped_values,
    "reflection_contract_ok_count": reflection_ok,
    "all_arms_ok_count": all_arm_ok,
    "problem_count": len(problems),
    "problems": problems,
    "fit_boundaries": {
        "outcomes_accessed": False,
        "frozen_v2_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "performance_claim_permitted": False,
    },
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
marker = summary_path.parent / (
    "CONTRACT_GATE_PASSED" if passed else "CONTRACT_GATE_FAILED"
)
marker.touch()
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
if not passed:
    raise SystemExit(6)
PY
  echo "$(date -Is) contract-recovery initial-failure contract gate passed."
else
  fold_args=()
  for fold in 0 1 2 3 4; do
    fold_dir="${RUN_ROOT}/outputs/${OUTPUT_GROUP}/daic_woz/fold${fold}"
    fold_args+=(--fold-dir "${fold_dir}")
  done
  if [[ ! -e "${MERGED_DIR}" ]]; then
    "${PYTHON}" -m rethink_mh.experiments.loop_merge \
      --prepared-inference-dir "${PREPARED_INFERENCE_DIR}" \
      "${fold_args[@]}" \
      --output-dir "${MERGED_DIR}" \
      --expected-fold-count 5 \
      --expected-session-count 107 \
      --collection-protocol-version \
      loop-collection
  fi
  touch "${RUN_ROOT}/outputs/${OUTPUT_GROUP}/COLLECTION_COMPLETE"
  echo "$(date -Is) five-fold label-free contract-recovery raw OOF trajectories are frozen."
fi
