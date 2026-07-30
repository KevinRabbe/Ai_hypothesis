"""Recover the Gate-7 information-ceiling audit after JSON key-order rejection.

The scientific artifact is never modified. This standalone script loads the
independent auditor directly by file path so recovery has no package-initializer
or Torch dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

RECOVERY_REASON = "JSON_OBJECT_KEY_ORDER_AUDITOR_DEFECT"
RECOVERY_VERSION = "gate7-information-ceiling-audit-recovery-v0"


def _load_independent_auditor() -> ModuleType:
    repository_root = Path(__file__).resolve().parents[1]
    auditor_path = (
        repository_root
        / "ai_hypothesis"
        / "population_compute"
        / "analyze_gate7_information_ceiling_decomposition.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gate7_information_ceiling_independent_auditor",
        auditor_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load independent information-ceiling auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AUDITOR = _load_independent_auditor()
RANKERS = tuple(_AUDITOR.RANKERS)
audit_gate7_information_ceiling_decomposition = (
    _AUDITOR.audit_gate7_information_ceiling_decomposition
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_ranker_object_order(payload: dict[str, Any]) -> int:
    """Reinsert exact ranker keys in frozen semantic order only."""

    rewrites = 0
    tiers = payload.get("tiers")
    if not isinstance(tiers, list):
        return rewrites
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        checkpoints = tier.get("checkpoint_results")
        if not isinstance(checkpoints, list):
            continue
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            matrix = checkpoint.get("ranks_by_ranker")
            if not isinstance(matrix, dict) or set(matrix) != set(RANKERS):
                continue
            checkpoint["ranks_by_ranker"] = {ranker: matrix[ranker] for ranker in RANKERS}
            rewrites += 1
    return rewrites


def recover_gate7_information_ceiling_audit(
    *,
    artifact: Path,
    output: Path,
    metadata_output: Path | None = None,
) -> int:
    source_bytes = artifact.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    payload = json.loads(source_bytes.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("information-ceiling artifact must be one JSON object")

    rewrites = canonicalize_ranker_object_order(payload)
    if rewrites != 12:
        raise RuntimeError(
            "audit recovery expected exactly 12 checkpoint ranker mappings; "
            f"observed {rewrites}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
            handle.write("\n")
            temporary_path = Path(handle.name)

        audit = audit_gate7_information_ceiling_decomposition(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if hashlib.sha256(artifact.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("scientific source artifact changed during audit recovery")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(
            json.dumps(
                {
                    "recovery_version": RECOVERY_VERSION,
                    "reason": RECOVERY_REASON,
                    "scientific_execution_repeated": False,
                    "scientific_artifact_modified": False,
                    "source_artifact": str(artifact.resolve()),
                    "source_artifact_sha256": source_sha256,
                    "ranker_mappings_canonicalized": rewrites,
                    "recovered_audit": str(output.resolve()),
                    "recovered_audit_sha256": sha256_file(output),
                    "artifact_valid": audit.artifact_valid,
                    "campaign_outcome": audit.campaign_outcome,
                    "errors": list(audit.errors),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    return recover_gate7_information_ceiling_audit(
        artifact=args.artifact,
        output=args.output,
        metadata_output=args.metadata_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
