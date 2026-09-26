"""Label-free readers for session ID and split selection files."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


ID_COLUMNS = ("participant_id", "participant", "participantid", "index", "session_id", "video_id")
SPLIT_COLUMNS = ("fold", "split", "partition")


def natural_session_key(value: str) -> tuple[int, int | str]:
    text = str(value).strip()
    return (0, int(text)) if re.fullmatch(r"\d+", text) else (1, text)


def read_session_ids(
    path: str | Path,
    split: str | None = None,
) -> list[str]:
    """Read only ID and optional split columns; task labels are never accessed."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            raise ValueError(f"Split file has no header: {path}")
        normalized = {name.strip().lower(): index for index, name in enumerate(fieldnames)}
        id_index = next((normalized[name] for name in ID_COLUMNS if name in normalized), None)
        if id_index is None:
            raise ValueError(f"Could not find participant/session ID column in {path}")
        split_index = next((normalized[name] for name in SPLIT_COLUMNS if name in normalized), None)
        if split is not None and split_index is None:
            raise ValueError(f"Requested split {split!r}, but {path} has no fold/split column")
        ids: list[str] = []
        for row in reader:
            if id_index >= len(row):
                continue
            if split is not None and (
                split_index is None
                or split_index >= len(row)
                or row[split_index].strip().lower() != split.lower()
            ):
                continue
            value = row[id_index].strip()
            if value:
                ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate session IDs in {path}")
    return sorted(ids, key=natural_session_key)


def select_session_ids(
    available: Iterable[str],
    split_file: str | Path | None = None,
    split: str | None = None,
    max_sessions: int | None = None,
) -> list[str]:
    available_set = {str(value) for value in available}
    if split_file is None:
        ids = sorted(available_set, key=natural_session_key)
    else:
        ids = read_session_ids(split_file, split=split)
        missing = [value for value in ids if value not in available_set]
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} sessions from the split are absent under the data root: {preview}"
            )
    if max_sessions is not None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        ids = ids[:max_sessions]
    return ids
