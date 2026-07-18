"""Artifact discovery and safe result-directory-relative references."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .errors import ResultsDirectoryAccessError


@dataclass(frozen=True, slots=True)
class ArtifactCandidate:
    path: Path
    artifact_id: str
    artifact_ref: str


def make_artifact_id(artifact_ref: str, payload: bytes | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(artifact_ref.encode("utf-8"))
    if payload is not None:
        digest.update(b"\0")
        digest.update(payload)
    return f"artifact_{digest.hexdigest()[:16]}"


def normalize_artifact_ref(results_dir: Path, path: Path) -> str:
    resolved_root = results_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ResultsDirectoryAccessError(
            f"artifact path is outside results directory: {path}"
        ) from exc
    parts = relative.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ResultsDirectoryAccessError(f"unsafe artifact reference: {relative}")
    return "/".join(parts)


def discover_json_artifacts(results_dir: Path) -> tuple[str, list[ArtifactCandidate]]:
    if not results_dir.exists():
        return "ABSENT", []
    if not results_dir.is_dir():
        raise ResultsDirectoryAccessError(f"results path is not a directory: {results_dir}")

    candidates: list[ArtifactCandidate] = []
    has_entries = False
    for path in sorted(results_dir.rglob("*.json")):
        has_entries = True
        if not path.is_file():
            continue
        payload = path.read_bytes()
        artifact_ref = normalize_artifact_ref(results_dir, path)
        candidates.append(
            ArtifactCandidate(
                path=path,
                artifact_id=make_artifact_id(artifact_ref, payload),
                artifact_ref=artifact_ref,
            )
        )
    if not has_entries:
        return "EMPTY", []
    return "PRESENT", candidates
