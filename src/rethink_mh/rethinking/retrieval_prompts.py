"""Label-safe prompts for query-grounded atomic evidence selection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .contracts import InitialAssessment, TaskSpec
from .prompts import _clean_input_text, _decision_payload, _task_block
from .query_retrieval import (
    RETRIEVAL_PATTERNS,
    RETRIEVAL_PURPOSES,
    CandidateSet,
    EvidenceQuery,
)


EVIDENCE_QUERY_OUTPUT_CONTRACT = """Return one JSON object with exactly these five fields:
- "segment_id": one S identifier present in the rethink decision and retrieval map.
- "purpose": exactly one of "verify supporting detail", "verify contradictory detail",
  "resolve source disagreement", "inspect measurement reliability", or
  "inspect temporal change".
- "target_slots": an array of one to four fixed core-slot names written exactly as
  they appear in the retrieval map.
- "pattern": exactly one of "unusual measurement", "change point",
  "quality boundary", "representative window", or "cross-modal co-change".
- "budget": an integer from 1 to 4.
A source-disagreement or cross-modal query must include at least one audio and one
visual slot. Do not output an E identifier, rationale, diagnosis, Markdown, or any
additional field."""


ATOMIC_SELECTION_OUTPUT_CONTRACT = """Return one JSON object with exactly this field:
- "selected_evidence_ids": an array containing one or more unique E identifiers
  shown in CANDIDATE ATOMIC EVIDENCE CARDS, up to the query budget.
Do not output an identifier that is absent from the cards. Do not output a rationale,
measurement value, diagnosis, Markdown, or any additional field."""


class QueryRetrievalPromptBuilder:
    """Build the two short model turns between trigger and full evidence review."""

    @staticmethod
    def build_query_messages(
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        decision: Mapping[str, Any] | Any,
        retrieval_map: str,
    ) -> list[dict[str, str]]:
        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        decision_data = _decision_payload(decision)
        if not decision_data.get("should_rethink"):
            raise ValueError(
                "an evidence query can be requested only after rethink is triggered"
            )
        selected_segments = decision_data.get("selected_segment_ids")
        if not selected_segments:
            raise ValueError(
                "rethink decision must contain at least one selected segment"
            )
        map_text = _clean_input_text(retrieval_map, "retrieval_map")
        for segment_id in selected_segments:
            if f"SEGMENT {segment_id}" not in map_text:
                raise ValueError(
                    f"retrieval map does not describe selected segment {segment_id}"
                )

        system = (
            "You formulate one atomic evidence query for a non-diagnostic "
            "mental-health screening research pipeline. Use only the initial "
            "assessment, rethink decision, and label-free retrieval map. The map "
            "describes measurable retrieval affordances, not psychological states. "
            "Return exactly one JSON object with no Markdown or additional prose."
        )
        user = (
            "TASK SPECIFICATION\n"
            f"{_task_block(task)}\n\n"
            "QUERY INSTRUCTIONS\n"
            "Choose the selected segment and fixed core slots whose hidden atomic "
            "detail would best reduce the stated uncertainty. Choose a temporal "
            "pattern that actually has candidates in the retrieval map. Use ordinary "
            "English slot names, not feature-column names or invented E identifiers. "
            "The query asks what evidence to inspect; it does not assert what that "
            "evidence will prove.\n\n"
            "INITIAL ASSESSMENT\n"
            f"{json.dumps(assessment.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "RETHINK DECISION\n"
            f"{json.dumps(decision_data, ensure_ascii=False, sort_keys=True)}\n\n"
            "REQUIRED OUTPUT CONTRACT\n"
            f"{EVIDENCE_QUERY_OUTPUT_CONTRACT}\n\n"
            "ATOMIC RETRIEVAL MAP\n"
            f"{map_text}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def build_selection_messages(
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        candidate_set: CandidateSet,
    ) -> list[dict[str, str]]:
        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")
        cards = _clean_input_text(
            candidate_set.text,
            "candidate_set.text",
        )
        query = candidate_set.query
        if not candidate_set.candidates:
            raise ValueError("candidate_set cannot be empty")
        maximum = min(query.budget, len(candidate_set.candidates))

        system = (
            "You select grounded atomic evidence for targeted review in a "
            "non-diagnostic mental-health screening research pipeline. Select only "
            "identifiers explicitly shown in the candidate cards. Candidate cues are "
            "measurement relations, not task conclusions. Return exactly one JSON "
            "object with no Markdown or additional prose."
        )
        user = (
            "TASK SPECIFICATION\n"
            f"{_task_block(task)}\n\n"
            "SELECTION INSTRUCTIONS\n"
            f"Select between 1 and {maximum} complementary atomic records that best "
            "match the query. Prefer records that jointly cover the target slots and "
            "requested pattern. Do not select several nearly identical neighboring "
            "windows when a temporally different candidate provides comparable "
            "evidence. Quality-boundary queries may intentionally select incomplete "
            "measurements; otherwise prefer reliable coverage. Selection only decides "
            "what to inspect next and must not invent a conclusion.\n\n"
            "INITIAL ASSESSMENT\n"
            f"{json.dumps(assessment.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "EVIDENCE QUERY\n"
            f"{json.dumps(query.to_dict(), ensure_ascii=False, sort_keys=True)}\n\n"
            "REQUIRED OUTPUT CONTRACT\n"
            f"{ATOMIC_SELECTION_OUTPUT_CONTRACT}\n\n"
            f"{cards}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


EVIDENCE_QUERY_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "segment_id",
        "purpose",
        "target_slots",
        "pattern",
        "budget",
    ],
    "properties": {
        "segment_id": {
            "type": "string",
            "pattern": "^S[0-9]{3,6}$",
        },
        "purpose": {"type": "string", "enum": list(RETRIEVAL_PURPOSES)},
        "target_slots": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "speech activity",
                    "pitch level",
                    "pitch variability",
                    "loudness level",
                    "loudness variability",
                    "facial activity",
                    "facial movement",
                    "head movement",
                    "gaze movement",
                    "mouth activity",
                ],
            },
        },
        "pattern": {"type": "string", "enum": list(RETRIEVAL_PATTERNS)},
        "budget": {
            "type": "integer",
            "minimum": 1,
            "maximum": 4,
        },
    },
}


ATOMIC_SELECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["selected_evidence_ids"],
    "properties": {
        "selected_evidence_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "pattern": "^E[0-9]{3,6}$",
            },
        }
    },
}


__all__ = [
    "ATOMIC_SELECTION_OUTPUT_CONTRACT",
    "ATOMIC_SELECTION_OUTPUT_SCHEMA",
    "EVIDENCE_QUERY_OUTPUT_CONTRACT",
    "EVIDENCE_QUERY_OUTPUT_SCHEMA",
    "QueryRetrievalPromptBuilder",
]
