#!/usr/bin/env python3
"""Audit one completed Gate-9 contextual-training seed artifact root."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import types
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDITOR_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/gate9_contextual_seed_audit.py"
)
EXPECTED_BRANCH = "agent/gate9-contextual-seed-audit-v0"


def _load_auditor():
    """Compile the exact checked-out source and bypass timestamp-based .pyc reuse."""

    name = "gate9_contextual_seed_audit_cli_dependency"
    source = AUDITOR_PATH.read_text(encoding="utf-8")
    code = compile(
        source,
        str(AUDITOR_PATH),
        "exec",
        dont_inherit=True,
        optimize=0,
    )
    module = types.ModuleType(name)
    module.__file__ = str(AUDITOR_PATH)
    module.__package__ = ""
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    branch = _git("branch", "--show-current").strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Gate9 seed audit must run from {EXPECTED_BRANCH}")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("Gate9 seed audit requires a clean working tree")
    audit_head = _git("rev-parse", "HEAD").strip()
    if len(audit_head) != 40:
        raise RuntimeError("could not resolve exact audit Git head")

    artifact_root = args.artifact_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Gate9 audit output already exists: {output_root}")
    output_root.mkdir(parents=True)

    auditor = _load_auditor()
    report = auditor.audit_seed_artifact(
        artifact_root,
        seed=args.seed,
        expected_summary_sha256=args.expected_summary_sha256.lower(),
        expected_validation_sha256=args.expected_validation_sha256.lower(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256.lower(),
        expected_manifest_sha256=args.expected_manifest_sha256.lower(),
    )
    report["audit_head"] = audit_head
    report["audit_branch"] = EXPECTED_BRANCH
    report_path = output_root / "gate9-contextual-seed-audit.json"
    _write_json(report_path, report)
    (output_root / "git-head.txt").write_text(
        audit_head + "\n", encoding="ascii", newline="\n"
    )
    (output_root / "git-status.txt").write_text(
        status, encoding="utf-8", newline="\n"
    )
    _write_json(
        output_root / "run-config.json",
        {
            "audit_version": auditor.AUDIT_VERSION,
            "audit_head": audit_head,
            "audit_branch": EXPECTED_BRANCH,
            "seed": args.seed,
            "artifact_root": str(artifact_root),
            "output_root": str(output_root),
            "expected_source_sha256": {
                "summary": args.expected_summary_sha256.lower(),
                "validation_ledger": args.expected_validation_sha256.lower(),
                "checkpoint": args.expected_checkpoint_sha256.lower(),
                "manifest": args.expected_manifest_sha256.lower(),
            },
        },
    )
    manifest_rows = []
    for path in sorted(item for item in output_root.iterdir() if item.is_file()):
        if path.name == "manifest.sha256":
            continue
        manifest_rows.append(f"{_sha256(path)}  {path.name}")
    manifest_path = output_root / "manifest.sha256"
    manifest_path.write_text(
        "\n".join(manifest_rows) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "seed_outcome": report["seed_outcome"],
                "all_seed_admission_still_possible": report[
                    "all_seed_admission_still_possible"
                ],
                "scientific_test_generation_allowed": report[
                    "scientific_test_generation_allowed"
                ],
                "exact_accuracy": report["validation"]["exact_accuracy"],
                "bit_accuracy": report["validation"]["bit_accuracy"],
                "audit_report_sha256": _sha256(report_path),
                "audit_manifest_sha256": _sha256(manifest_path),
                "output_root": str(output_root),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
