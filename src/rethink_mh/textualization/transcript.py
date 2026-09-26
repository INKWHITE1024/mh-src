"""Normalize, validate, and compile session transcripts into transcript evidence text."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from .config import TextualizationConfig
from .readers import natural_session_key, select_session_ids


TRANSCRIPT_PROTOCOL_VERSION = "compact"
SpeakerRole = Literal["participant", "interviewer", "unassigned"]
SourceKind = Literal["manual_transcript", "automatic_speech_recognition"]


def normalize_transcript_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(" " if character.isspace() else character for character in text)
    text = text.replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True, slots=True)
class TranscriptUtterance:
    start_sec: float
    end_sec: float
    text: str | None
    speaker_role: SpeakerRole
    source_kind: SourceKind
    confidence: float | None = None
    privacy_scrubbed: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.start_sec) or self.start_sec < 0:
            raise ValueError("transcript start_sec must be finite and non-negative")
        if not math.isfinite(self.end_sec) or self.end_sec <= self.start_sec:
            raise ValueError("transcript end_sec must be after start_sec")
        if self.speaker_role not in {"participant", "interviewer", "unassigned"}:
            raise ValueError(f"invalid transcript speaker role {self.speaker_role!r}")
        if self.source_kind not in {
            "manual_transcript",
            "automatic_speech_recognition",
        }:
            raise ValueError(f"invalid transcript source kind {self.source_kind!r}")
        if self.privacy_scrubbed:
            if self.text is not None:
                raise ValueError("privacy-scrubbed transcript text must not be retained")
        elif not self.text or normalize_transcript_text(self.text) != self.text:
            raise ValueError("transcript text must be non-empty and normalized")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("transcript confidence must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    segment_id: str
    start_sec: float
    end_sec: float


_SEGMENT_PATTERN = re.compile(
    r"^SEGMENT (?P<id>S\d+) \| time (?P<start>\d+(?:\.\d+)?) "
    r"to (?P<end>\d+(?:\.\d+)?) seconds \|"
)


def read_evidence_segments(path: str | Path) -> list[TranscriptSegment]:
    source = Path(path)
    segments: list[TranscriptSegment] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        match = _SEGMENT_PATTERN.match(line)
        if match is None:
            continue
        segment = TranscriptSegment(
            segment_id=match.group("id"),
            start_sec=float(match.group("start")),
            end_sec=float(match.group("end")),
        )
        if segment.end_sec <= segment.start_sec:
            raise ValueError(f"invalid segment time range in {source}: {line!r}")
        segments.append(segment)
    if not segments:
        raise ValueError(f"no native Evidence segments found in {source}")
    if len({segment.segment_id for segment in segments}) != len(segments):
        raise ValueError(f"duplicate segment IDs in {source}")
    return segments


def _select_evenly(
    utterances: Sequence[TranscriptUtterance], limit: int | None
) -> list[TranscriptUtterance]:
    if limit is None or len(utterances) <= limit:
        return list(utterances)
    if limit <= 0:
        raise ValueError("utterance selection limit must be positive")
    if limit == 1:
        return [utterances[len(utterances) // 2]]
    indices = [
        round(index * (len(utterances) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [utterances[index] for index in indices]


def _model_utterances(
    utterances: Sequence[TranscriptUtterance], speaker_policy: str
) -> list[TranscriptUtterance]:
    visible = [item for item in utterances if not item.privacy_scrubbed]
    if speaker_policy == "participant_only":
        return [item for item in visible if item.speaker_role == "participant"]
    if speaker_policy == "all_speakers_unassigned":
        if any(item.speaker_role != "unassigned" for item in visible):
            raise ValueError(
                "all_speakers_unassigned requires source rows without speaker roles"
            )
        return visible
    raise ValueError(f"unsupported transcript speaker policy {speaker_policy!r}")


def render_transcript_input(
    utterances: Sequence[TranscriptUtterance],
    segments: Sequence[TranscriptSegment],
    *,
    speaker_policy: str,
    max_utterances_per_segment: int | None,
) -> str:
    retained = _model_utterances(utterances, speaker_policy)
    grouped: dict[str, list[TranscriptUtterance]] = defaultdict(list)
    for utterance in retained:
        midpoint = (utterance.start_sec + utterance.end_sec) / 2.0
        segment = next(
            (
                item
                for item in segments
                if item.start_sec <= midpoint < item.end_sec
            ),
            None,
        )
        if segment is not None:
            grouped[segment.segment_id].append(utterance)

    if speaker_policy == "participant_only":
        policy_text = (
            "Only utterances explicitly labeled Participant are included; interviewer "
            "utterances and privacy-scrubbed rows are excluded."
        )
    else:
        policy_text = (
            "The released ASR rows do not identify speakers and contain both interview "
            "prompts and responses; every retained utterance is marked speaker unassigned."
        )
    lines = [
        "TRANSCRIPT EVIDENCE | readable transcript protocol compact",
        "READING GUIDE",
        "- Transcript language is observed interview content, not a symptom label or clinical conclusion.",
        f"- {policy_text}",
        "- Text remains aligned to the same Sxxx time segments as the audio and visual evidence.",
        "TRANSCRIPT SEGMENTS",
    ]
    for segment in segments:
        selected = _select_evenly(
            grouped.get(segment.segment_id, ()), max_utterances_per_segment
        )
        prefix = (
            f"TRANSCRIPT {segment.segment_id} | time {segment.start_sec:g} "
            f"to {segment.end_sec:g} seconds"
        )
        if not selected:
            lines.append(prefix + " | no retained transcript utterance")
            continue
        confidences = [
            item.confidence for item in selected if item.confidence is not None
        ]
        quality = ""
        if confidences:
            ordered = sorted(confidences)
            median = ordered[len(ordered) // 2]
            quality = f" | median ASR confidence {median:.2f}"
        utterance_label = (
            "Participant utterance"
            if speaker_policy == "participant_only"
            else "Speaker-unassigned utterance"
        )
        text = " ".join(
            f"{utterance_label} {index}: {item.text}"
            for index, item in enumerate(selected, start=1)
        )
        lines.append(prefix + quality + " | " + text)
    return "\n".join(lines) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compile_transcript_session(
    *,
    adapter: Any,
    session_id: str,
    evidence_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = adapter.config.get("transcript", {})
    speaker_policy = str(settings.get("model_speaker_policy", "participant_only"))
    source_kind = str(settings.get("source_kind", "manual_transcript"))
    utterances = sorted(
        adapter.load_transcript(str(session_id)),
        key=lambda item: (item.start_sec, item.end_sec, item.speaker_role),
    )
    for utterance in utterances:
        utterance.validate()
        if utterance.source_kind != source_kind:
            raise ValueError(
                f"transcript source mismatch for session {session_id}: "
                f"configured {source_kind!r}, observed {utterance.source_kind!r}"
            )
    segments = read_evidence_segments(
        Path(evidence_root) / str(session_id) / "segment_evidence.native.txt"
    )
    target = (
        Path(output_root).expanduser().resolve()
        / TRANSCRIPT_PROTOCOL_VERSION
        / adapter.config.dataset_key
        / str(session_id)
    )
    manifest_path = target / "transcript_manifest.compact.json"
    expected = (
        target / "transcript_units.compact.jsonl",
        target / "transcript_input.full.compact.txt",
        target / "transcript_input.compact.compact.txt",
        target / "transcript_input.essential.compact.txt",
        target / "transcript_input.minimal.compact.txt",
        manifest_path,
    )
    if all(path.is_file() for path in expected) and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing["skipped"] = True
        return existing
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(
            f"partial transcript artifacts exist for session {session_id}; pass --overwrite"
        )

    rows = "".join(
        json.dumps(
            {"utterance_id": f"T{index:04d}", **item.to_dict()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for index, item in enumerate(utterances)
    )
    tier_limits = {"full": None, "compact": 4, "essential": 2, "minimal": 1}
    rendered = {
        tier: render_transcript_input(
            utterances,
            segments,
            speaker_policy=speaker_policy,
            max_utterances_per_segment=limit,
        )
        for tier, limit in tier_limits.items()
    }
    target.mkdir(parents=True, exist_ok=True)
    _write_atomic(target / "transcript_units.compact.jsonl", rows)
    for tier, content in rendered.items():
        _write_atomic(target / f"transcript_input.{tier}.compact.txt", content)

    roles = Counter(item.speaker_role for item in utterances)
    model_count = len(_model_utterances(utterances, speaker_policy))
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": adapter.config.dataset_name,
        "dataset_key": adapter.config.dataset_key,
        "session_id": str(session_id),
        "protocol_version": TRANSCRIPT_PROTOCOL_VERSION,
        "speaker_policy": speaker_policy,
        "speaker_role_reliability": str(
            settings.get("speaker_role_reliability", "explicit")
        ),
        "source_kind": source_kind,
        "source_utterance_count": len(utterances),
        "model_utterance_count": model_count,
        "privacy_scrubbed_count": sum(item.privacy_scrubbed for item in utterances),
        "speaker_role_counts": dict(sorted(roles.items())),
        "segment_count": len(segments),
        "files": {
            "transcript_units.compact.jsonl": {"sha256": _sha256_text(rows)},
            **{
                f"transcript_input.{tier}.compact.txt": {
                    "sha256": _sha256_text(content)
                }
                for tier, content in sorted(rendered.items())
            },
        },
        "label_fields_read": [],
        "transcript_content_exposed": True,
        "raw_session_identifier_in_model_input": False,
        "skipped": False,
    }
    manifest_content = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    _write_atomic(manifest_path, manifest_content)
    return manifest


def compile_transcript_dataset(
    *,
    root: str | Path,
    config: TextualizationConfig,
    evidence_root: str | Path,
    output_root: str | Path,
    split_file: str | Path | None = None,
    split_value: str | None = None,
    max_sessions: int | None = None,
    overwrite: bool = False,
    progress: Any | None = None,
) -> dict[str, Any]:
    from .adapters import build_adapter

    adapter = build_adapter(root, config)
    session_ids = select_session_ids(
        adapter.list_session_ids(), split_file, split_value, max_sessions
    )
    if not session_ids:
        raise ValueError("the selected transcript compile split is empty")
    manifests = []
    for index, session_id in enumerate(session_ids, start=1):
        manifest = compile_transcript_session(
            adapter=adapter,
            session_id=session_id,
            evidence_root=evidence_root,
            output_root=output_root,
            overwrite=overwrite,
        )
        manifests.append(manifest)
        if progress is not None:
            progress(
                f"compile-transcript {index}/{len(session_ids)} session={session_id} "
                f"utterances={manifest['model_utterance_count']} "
                f"skipped={manifest.get('skipped', False)}"
            )
    manifests.sort(key=lambda item: natural_session_key(str(item["session_id"])))
    dataset_root = (
        Path(output_root).expanduser().resolve()
        / TRANSCRIPT_PROTOCOL_VERSION
        / config.dataset_key
    )
    sessions_text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in manifests
    )
    _write_atomic(dataset_root / "sessions_manifest.jsonl", sessions_text)
    summary = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset_name,
        "dataset_key": config.dataset_key,
        "protocol_version": TRANSCRIPT_PROTOCOL_VERSION,
        "session_count": len(manifests),
        "compiled_count": sum(not item.get("skipped", False) for item in manifests),
        "skipped_count": sum(bool(item.get("skipped", False)) for item in manifests),
        "source_utterance_count": sum(
            int(item["source_utterance_count"]) for item in manifests
        ),
        "model_utterance_count": sum(
            int(item["model_utterance_count"]) for item in manifests
        ),
        "privacy_scrubbed_count": sum(
            int(item["privacy_scrubbed_count"]) for item in manifests
        ),
        "label_fields_read": [],
        "transcript_content_exposed": True,
    }
    _write_atomic(
        dataset_root / "run_manifest.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary
