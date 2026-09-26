"""Readable, label-safe prompts for initial assessment and targeted revision."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    CONTROLLED_CHANGE_SUMMARIES,
    SEGMENT_ID_PATTERN,
    ContractValidationError,
    InitialAssessment,
    RevisionActionDecision,
    TaskSpec,
)


INITIAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "risk_probability",
        "confidence",
        "audio_source",
        "visual_source",
        "supporting_segment_ids",
        "contradictory_segment_ids",
        "uncertain_segment_ids",
        "requested_segment_ids",
    ],
    "properties": {
        "risk_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "audio_source": {"$ref": "#/$defs/sourceAssessment"},
        "visual_source": {"$ref": "#/$defs/sourceAssessment"},
        "supporting_segment_ids": {"$ref": "#/$defs/segmentIds"},
        "contradictory_segment_ids": {"$ref": "#/$defs/segmentIds"},
        "uncertain_segment_ids": {"$ref": "#/$defs/segmentIds"},
        "requested_segment_ids": {"$ref": "#/$defs/segmentIds"},
    },
    "$defs": {
        "sourceAssessment": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status", "risk_probability", "reliability"],
                    "properties": {
                        "status": {"const": "available"},
                        "risk_probability": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "reliability": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status"],
                    "properties": {"status": {"const": "unavailable"}},
                },
            ]
        },
        "segmentIds": {
            "type": "array",
            "maxItems": 4,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^S[0-9]{3,6}$"},
        },
    },
}

REVISION_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "revision_status",
        "revised_risk_probability",
        "revised_confidence",
        "cited_segment_ids",
        "cited_evidence_ids",
        "preserved_evidence_ids",
        "newly_considered_evidence_ids",
        "rejected_evidence_ids",
        "residual_conflict_segment_ids",
        "change_summary",
    ],
    "properties": {
        "revision_status": {
            "type": "string",
            "enum": ["preserved", "revised", "unresolved"],
        },
        "revised_risk_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "revised_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "cited_segment_ids": {"$ref": "#/$defs/segmentIds"},
        "cited_evidence_ids": {"$ref": "#/$defs/evidenceIds"},
        "preserved_evidence_ids": {"$ref": "#/$defs/evidenceIds"},
        "newly_considered_evidence_ids": {"$ref": "#/$defs/evidenceIds"},
        "rejected_evidence_ids": {"$ref": "#/$defs/evidenceIds"},
        "residual_conflict_segment_ids": {"$ref": "#/$defs/segmentIds"},
        "change_summary": {
            "anyOf": [
                {"enum": sorted(CONTROLLED_CHANGE_SUMMARIES)},
                {
                    "type": "string",
                    "minLength": 8,
                    "maxLength": 160,
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9 ,.;:'()/-]*$",
                },
            ]
        },
    },
    "$defs": {
        "segmentIds": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^S[0-9]{3,6}$"},
        },
        "evidenceIds": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^E[0-9]{3,6}$"},
        },
    },
}

INITIAL_OUTPUT_CONTRACT = """Return one JSON object with exactly these eight fields:
- "risk_probability": positive-class probability from 0 to 1.
- "confidence": operational confidence from 0 to 1.
- "audio_source" and "visual_source": each is either
  {"status":"unavailable"} or
  {"status":"available","risk_probability":0-to-1,"reliability":0-to-1}.
- "supporting_segment_ids", "contradictory_segment_ids",
  "uncertain_segment_ids", and "requested_segment_ids": arrays of unique S identifiers
  that occur in SESSION EVIDENCE; use only the four most informative identifiers per
  array, and use an empty array when no segment qualifies.
Do not output a JSON Schema. Do not use keys such as "$schema", "$defs", "properties",
or "required". Do not add any field beyond the eight listed above."""

REVISION_OUTPUT_CONTRACT = """Return one JSON object with exactly these ten fields:
- "revision_status": exactly "preserved", "revised", or "unresolved".
- "revised_risk_probability" and "revised_confidence": numbers from 0 to 1.
- "cited_segment_ids" and "residual_conflict_segment_ids": arrays containing only
  retrieved S identifiers.
- "cited_evidence_ids", "preserved_evidence_ids",
  "newly_considered_evidence_ids", and "rejected_evidence_ids": arrays containing only
  retrieved E identifiers. Cite at least one retrieved S or E identifier.
- "change_summary": exactly one of "risk_increased_after_review",
  "risk_decreased_after_review", "risk_unchanged_after_review", or
  "insufficient_reliable_detail".
Keep every identifier array selective. "preserved_evidence_ids" and
"rejected_evidence_ids" must not overlap, and every preserved or rejected E identifier
must also occur in "cited_evidence_ids". "newly_considered_evidence_ids" may overlap
either category because it records what was newly inspected.
Do not output a JSON Schema. Do not use keys such as "$schema", "$defs", "properties",
or "required". Do not add any field beyond the ten listed above."""

REFLECTION_OUTPUT_CONTRACT = """Return one JSON object with exactly these ten fields:
- "revision_status": exactly "preserved", "revised", or "unresolved".
- "revised_risk_probability" and "revised_confidence": numbers from 0 to 1.
- "cited_segment_ids": a non-empty array containing only visible S identifiers.
- "residual_conflict_segment_ids": an array containing only visible S identifiers.
- "cited_evidence_ids", "preserved_evidence_ids",
  "newly_considered_evidence_ids", and "rejected_evidence_ids": each must be exactly
  [] because this control arm released no atomic EVIDENCE.
- "change_summary": exactly one of "risk_increased_after_review",
  "risk_decreased_after_review", "risk_unchanged_after_review", or
  "insufficient_reliable_detail".
NO NEW ATOMIC EVIDENCE CONTRACT: never invent or cite an E identifier. Base the
reflection only on the InitialAssessment and its already visible S identifiers.
Do not output a JSON Schema. Do not use keys such as "$schema", "$defs", "properties",
or "required". Do not add any field beyond the ten listed above."""

ACTION_OUTPUT_CONTRACT = """Return one JSON object with exactly these two fields:
- "action": exactly "preserve", "revise", or "refer".
- "direction": "unchanged" for preserve, "increase" or "decrease" for revise,
  and "unresolved" for refer.
Do not output a probability, diagnosis, rationale, Markdown, or any additional field."""

_DECISION_FIELDS = (
    "should_rethink",
    "trigger_score",
    "reasons",
    "selected_segment_ids",
    "metrics",
)
_GROUND_TRUTH_KEYS = frozenset(
    {
        "label",
        "groundtruth",
        "goldlabel",
        "samplelabel",
        "target",
        "targetlabel",
        "truelabel",
        "ytrue",
    }
)
_GROUND_TRUTH_MARKER = re.compile(
    r"(?i)(?:\"|')?(?:ground[ _-]?truth|gold[ _-]?label|sample[ _-]?label|"
    r"target[ _-]?label|true[ _-]?label|label)(?:\"|')?\s*[:=]"
)
_EVIDENCE_IDENTIFIER = re.compile(r"\b[SE][0-9]{3,6}\b")
_SEGMENT_IDENTIFIER = re.compile(r"\bS[0-9]{3,6}\b")
_ATOMIC_EVIDENCE_IDENTIFIER = re.compile(r"\bE[0-9]{3,6}\b")


def _clean_input_text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{path} must be text")
    text = value.strip()
    if not text:
        raise ContractValidationError(f"{path} cannot be empty")
    if "\x00" in text:
        raise ContractValidationError(f"{path} cannot contain NUL characters")
    if _GROUND_TRUTH_MARKER.search(text):
        raise ContractValidationError(
            f"{path} appears to contain a current-sample ground-truth assignment"
        )
    return text


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.casefold())


def _reject_ground_truth_fields(value: Any, path: str = "decision") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{path} keys must be strings")
            if _normalized_key(key) in _GROUND_TRUTH_KEYS:
                raise ContractValidationError(
                    f"{path}.{key} is a forbidden current-sample ground-truth field"
                )
            _reject_ground_truth_fields(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_ground_truth_fields(child, f"{path}[{index}]")


def _json_safe(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child, f"{path}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise ContractValidationError(f"{path} contains a non-JSON value")


def _decision_mapping(decision: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(decision, Mapping):
        return decision
    to_dict = getattr(decision, "to_dict", None)
    if callable(to_dict):
        output = to_dict()
        if isinstance(output, Mapping):
            return output
    raise ContractValidationError("decision must be a mapping or expose to_dict()")


def _decision_payload(decision: Mapping[str, Any] | Any) -> dict[str, Any]:
    mapping = _decision_mapping(decision)
    _reject_ground_truth_fields(mapping)
    output = {key: mapping[key] for key in _DECISION_FIELDS if key in mapping}
    if not output:
        raise ContractValidationError("decision contains no recognized rethink fields")

    if "should_rethink" in output and not isinstance(output["should_rethink"], bool):
        raise ContractValidationError("decision.should_rethink must be boolean")
    if "trigger_score" in output:
        score = output["trigger_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ContractValidationError("decision.trigger_score must be from 0 to 1")
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ContractValidationError("decision.trigger_score must be from 0 to 1")
        output["trigger_score"] = score
    if "reasons" in output:
        reasons = output["reasons"]
        if isinstance(reasons, (str, bytes)) or not isinstance(reasons, Sequence):
            raise ContractValidationError("decision.reasons must be an array")
        parsed_reasons: list[str] = []
        for index, reason in enumerate(reasons):
            if not isinstance(reason, str) or not reason or len(reason) > 100:
                raise ContractValidationError(
                    f"decision.reasons[{index}] must be a short non-empty string"
                )
            parsed_reasons.append(reason)
        output["reasons"] = parsed_reasons
    if "selected_segment_ids" in output:
        identifiers = output["selected_segment_ids"]
        if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
            raise ContractValidationError("decision.selected_segment_ids must be an array")
        selected: list[str] = []
        seen: set[str] = set()
        for index, identifier in enumerate(identifiers):
            if not isinstance(identifier, str) or SEGMENT_ID_PATTERN.fullmatch(identifier) is None:
                raise ContractValidationError(
                    f"decision.selected_segment_ids[{index}] is not a valid SEGMENT ID"
                )
            if identifier not in seen:
                selected.append(identifier)
                seen.add(identifier)
        output["selected_segment_ids"] = selected
    return _json_safe(output, "decision")


def _task_block(task: TaskSpec) -> str:
    if not isinstance(task, TaskSpec):
        raise ContractValidationError("task must be a TaskSpec")
    return (
        f"Task identifier: {task.task_id}\n"
        f"Label definition: {task.label_definition}\n"
        f"Negative class name: {task.negative_label_name}\n"
        f"Positive class name: {task.positive_label_name}\n"
        "Every risk_probability is the probability of the positive class. The class "
        "names define the reusable task; no current participant label is supplied."
    )


class PromptBuilder:
    """Build standard chat messages without accepting sample labels at inference time."""

    @staticmethod
    def build_initial_messages(task: TaskSpec, session_text: str) -> list[dict[str, str]]:
        evidence = _clean_input_text(session_text, "session_text")
        system = (
            "You are the first-pass evidence assessor in a non-diagnostic mental-health "
            "screening research pipeline. Use only the supplied measurement evidence. "
            "Do not infer a diagnosis, protected attribute, or unobserved behavior. Treat "
            "unavailable and partial evidence as missing information, never as a measured "
            "zero. Categories in a source-specific training reference are measurement "
            "ranks, not symptom labels. "
            "Return exactly one JSON object that follows the requested schema, with no "
            "Markdown and no additional prose."
        )
        user = (
            "TASK SPECIFICATION\n"
            f"{_task_block(task)}\n\n"
            "ASSESSMENT INSTRUCTIONS\n"
            "Estimate overall positive-class risk and operational confidence from 0 to 1. "
            "For each available source, separately estimate positive-class risk and "
            "operational reliability. Mark a source unavailable when it has no usable "
            "measurements; do not invent numeric values for it. Cite only SEGMENT IDs that "
            "appear in the supplied evidence. supporting_segment_ids favor the estimate, "
            "contradictory_segment_ids push against it, uncertain_segment_ids contain "
            "conflicting or weak measurements, and requested_segment_ids identify segments "
            "whose atomic EVIDENCE windows should be retrieved for a targeted second pass. "
            "Use at most four of the most informative IDs in each array; arrays may be "
            "empty. Do not add a diagnosis or free-form clinical rationale.\n\n"
            "REQUIRED OUTPUT CONTRACT\n"
            f"{INITIAL_OUTPUT_CONTRACT}\n\n"
            "SESSION EVIDENCE\n"
            f"{evidence}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def build_revision_messages(
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        decision: Mapping[str, Any] | Any,
        detailed_evidence: str,
    ) -> list[dict[str, str]]:
        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        decision_data = _decision_payload(decision)
        evidence = _clean_input_text(detailed_evidence, "detailed_evidence")
        if _EVIDENCE_IDENTIFIER.search(evidence) is None:
            raise ContractValidationError(
                "detailed_evidence must contain at least one SEGMENT or EVIDENCE ID"
            )
        system = (
            "You are the targeted second-pass reviewer in a non-diagnostic mental-health "
            "screening research pipeline. Reassess the initial probability using only the "
            "initial structured assessment, the rethink decision, and the retrieved atomic "
            "measurement evidence. Preserve useful evidence, reject unreliable or "
            "contradicted evidence explicitly, and leave unresolved conflicts visible. "
            "Missing evidence is not a measured zero. Return exactly one JSON object that "
            "follows the requested schema, with no Markdown and no additional prose."
        )
        user = (
            "TASK SPECIFICATION\n"
            f"{_task_block(task)}\n\n"
            "REVISION INSTRUCTIONS\n"
            "Recalculate positive-class risk and confidence after reviewing the targeted "
            "detail. revision_status is preserved when the initial estimate remains best "
            "supported, revised when the estimate materially changes, and unresolved when "
            "the available sources leave a residual conflict. Cite only S and E identifiers "
            "present below. preserved_evidence_ids identify atomic evidence retained in the "
            "decision; newly_considered_evidence_ids identify retrieved atomic evidence "
            "examined in this pass; rejected_evidence_ids identify atomic evidence excluded "
            "for quality or contradiction; residual_conflict_segment_ids retain unresolved "
            "segment-level disagreements. change_summary must be a controlled value or a "
            "single plain-English clause no longer than 160 characters. Do not add a "
            "diagnosis or an unrestricted clinical narrative.\n\n"
            "INITIAL ASSESSMENT\n"
            f"{json.dumps(assessment.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "RETHINK DECISION\n"
            f"{json.dumps(decision_data, ensure_ascii=False, sort_keys=True)}\n\n"
            "REQUIRED OUTPUT CONTRACT\n"
            f"{REVISION_OUTPUT_CONTRACT}\n\n"
            "TARGETED DETAILED EVIDENCE\n"
            f"{evidence}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def build_reflection_messages(
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        decision: Mapping[str, Any] | Any,
        visible_context: str,
    ) -> list[dict[str, str]]:
        """Build a revision turn that explicitly has no atomic Evidence."""

        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        decision_data = _decision_payload(decision)
        context = _clean_input_text(visible_context, "visible_context")
        if _SEGMENT_IDENTIFIER.search(context) is None:
            raise ContractValidationError(
                "visible_context must contain at least one SEGMENT ID"
            )
        if _ATOMIC_EVIDENCE_IDENTIFIER.search(context) is not None:
            raise ContractValidationError(
                "reflection visible_context cannot contain an atomic EVIDENCE ID"
            )
        system = (
            "You are the no-new-evidence reflection control in a non-diagnostic "
            "mental-health screening research pipeline. No hidden atomic measurement "
            "was queried or released. Reconsider only the supplied InitialAssessment "
            "and its already visible segment references. Never invent an E identifier. "
            "Return exactly one JSON object and no additional prose."
        )
        user = (
            "TASK SPECIFICATION\n"
            f"{_task_block(task)}\n\n"
            "REFLECTION INSTRUCTIONS\n"
            "This is a matched second reasoning turn without evidence re-access. "
            "revision_status may be preserved, revised, or unresolved, but any change "
            "must rely only on the existing structured assessment and visible S IDs. "
            "All four atomic E-ID arrays must be exactly []. Cite at least one visible "
            "S ID, keep residual conflicts explicit, and do not add a diagnosis or "
            "clinical narrative.\n\n"
            "INITIAL ASSESSMENT\n"
            f"{json.dumps(assessment.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "RETHINK DECISION\n"
            f"{json.dumps(decision_data, ensure_ascii=False, sort_keys=True)}\n\n"
            "REQUIRED OUTPUT CONTRACT\n"
            f"{REFLECTION_OUTPUT_CONTRACT}\n\n"
            "VISIBLE CONTEXT WITHOUT ATOMIC EVIDENCE\n"
            f"{context}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def build_action_messages(
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        decision: Mapping[str, Any] | Any,
        detailed_evidence: str,
    ) -> list[dict[str, str]]:
        """Build the short action-ranking prompt used before payload generation."""

        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        decision_data = _decision_payload(decision)
        evidence = _clean_input_text(detailed_evidence, "detailed_evidence")
        if _EVIDENCE_IDENTIFIER.search(evidence) is None:
            raise ContractValidationError(
                "detailed_evidence must contain at least one SEGMENT or EVIDENCE ID"
            )
        system = (
            "You are the action selector for a targeted second-pass review in a "
            "non-diagnostic mental-health screening research pipeline. Compare the "
            "initial estimate with the retrieved audiovisual and transcript evidence. "
            "Choose preserve when the initial estimate remains best supported, revise "
            "only when the new evidence supports a directional change, and refer when "
            "the evidence cannot safely resolve the conflict. Missing evidence is not "
            "a measured zero. Return exactly one short JSON object and no other text."
        )
        user = (
            "TASK SPECIFICATION\n"
            f"{_task_block(task)}\n\n"
            "ACTION SELECTION INSTRUCTIONS\n"
            "Score the three action types against the supplied evidence. Do not change "
            "the estimate merely because a source disagrees; revise requires coherent "
            "support in the retrieved detail. Use refer instead of a forced confident "
            "decision when conflict remains.\n\n"
            "INITIAL ASSESSMENT\n"
            f"{json.dumps(assessment.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "RETHINK DECISION\n"
            f"{json.dumps(decision_data, ensure_ascii=False, sort_keys=True)}\n\n"
            "REQUIRED OUTPUT CONTRACT\n"
            f"{ACTION_OUTPUT_CONTRACT}\n\n"
            "TARGETED MULTIMODAL EVIDENCE\n"
            f"{evidence}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def build_conditioned_revision_messages(
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        decision: Mapping[str, Any] | Any,
        action: RevisionActionDecision | Mapping[str, Any],
        detailed_evidence: str,
    ) -> list[dict[str, str]]:
        """Build payload-generation messages after the action has been selected."""

        selected = (
            action
            if isinstance(action, RevisionActionDecision)
            else RevisionActionDecision.from_dict(action)
        )
        messages = PromptBuilder.build_revision_messages(
            task, initial, decision, detailed_evidence
        )
        marker = "REQUIRED OUTPUT CONTRACT\n"
        action_block = (
            "SELECTED REVISION ACTION\n"
            f"{json.dumps(selected.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "Generate a payload consistent with this already-selected action. Do not "
            "silently replace it with another action.\n\n"
        )
        if marker not in messages[1]["content"]:
            raise RuntimeError("revision prompt is missing its output-contract marker")
        messages[1]["content"] = messages[1]["content"].replace(
            marker, action_block + marker, 1
        )
        return messages

    @staticmethod
    def build_supervised_record(
        task: TaskSpec,
        session_text: str,
        label: str | int | bool,
        *,
        assistant_target: InitialAssessment | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build a draft-training record without changing the inference contract.

        The chat messages contain exactly the same input and output structures used at
        inference.  The gold class is stored outside the messages for a screening head or
        an auxiliary loss; it is never serialized into text consumed by the Thinker.
        """

        if isinstance(label, bool):
            normalized_label = int(label)
        elif isinstance(label, int) and label in {0, 1}:
            normalized_label = label
        elif isinstance(label, str) and label == task.negative_label_name:
            normalized_label = 0
        elif isinstance(label, str) and label == task.positive_label_name:
            normalized_label = 1
        else:
            raise ContractValidationError(
                "label must be 0, 1, or one of the TaskSpec class names"
            )

        messages = PromptBuilder.build_initial_messages(task, session_text)
        assessment = (
            assistant_target
            if isinstance(assistant_target, InitialAssessment)
            else InitialAssessment.from_dict(assistant_target)
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    assessment.to_dict(), ensure_ascii=False, sort_keys=True
                ),
            }
        )
        return {
            "record_type": "draft_sft",
            "task_id": task.task_id,
            "messages": messages,
            "supervision": {
                "label": normalized_label,
                "used_in_model_text": False,
            },
        }


def build_initial_messages(task: TaskSpec, session_text: str) -> list[dict[str, str]]:
    """Functional convenience wrapper around :class:`PromptBuilder`."""

    return PromptBuilder.build_initial_messages(task, session_text)


def build_revision_messages(
    task: TaskSpec,
    initial: InitialAssessment | Mapping[str, Any],
    decision: Mapping[str, Any] | Any,
    detailed_evidence: str,
) -> list[dict[str, str]]:
    """Functional convenience wrapper around :class:`PromptBuilder`."""

    return PromptBuilder.build_revision_messages(task, initial, decision, detailed_evidence)


def build_supervised_record(
    task: TaskSpec,
    session_text: str,
    label: str | int | bool,
    *,
    assistant_target: InitialAssessment | Mapping[str, Any],
) -> dict[str, Any]:
    """Functional convenience wrapper around :class:`PromptBuilder`."""

    return PromptBuilder.build_supervised_record(
        task,
        session_text,
        label,
        assistant_target=assistant_target,
    )
