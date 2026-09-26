"""Create or verify a content-addressed manifest for a frozen artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in files:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_directory_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError(f"artifact directory contains no files: {root}")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "schema_version": "1.0.0",
        "root": str(root),
        "exclusions": ["any path component named .cache"],
        "file_count": len(files),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in files),
        "files": files,
        "artifact_sha256": _manifest_digest(files),
    }


def verify_directory_manifest(
    root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    observed = build_directory_manifest(root)
    expected_files = manifest.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("frozen artifact manifest has no file inventory")
    expected = {
        "schema_version": manifest.get("schema_version"),
        "exclusions": manifest.get("exclusions"),
        "file_count": manifest.get("file_count"),
        "total_size_bytes": manifest.get("total_size_bytes"),
        "files": expected_files,
        "artifact_sha256": manifest.get("artifact_sha256"),
    }
    comparable = {
        key: observed[key]
        for key in (
            "schema_version",
            "exclusions",
            "file_count",
            "total_size_bytes",
            "files",
            "artifact_sha256",
        )
    }
    if expected != comparable:
        raise ValueError("frozen artifact directory does not match its manifest")
    return observed


def write_directory_manifest(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    root = root.resolve()
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("manifest output must be outside the artifact directory")
    payload = build_directory_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="create this new manifest")
    mode.add_argument(
        "--verify-manifest", type=Path, help="verify against an existing manifest"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None:
        payload = write_directory_manifest(args.root, args.output)
    else:
        manifest = json.loads(args.verify_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("frozen artifact manifest must contain one JSON object")
        payload = verify_directory_manifest(args.root, manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
