#!/usr/bin/env python3
"""Run the development-only Gate9D query-capacity diagnostic bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import zipfile
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/gate9d_query_capacity_diagnostic.py"
)
EXPECTED_BRANCH = "agent/gate9d-query-capacity-diagnostic-v0"
FAST_HARNESS_BASE_HEAD = "84f6038dc58547718d1d1ab9df7ef11538f21fb8"


def _load_diagnostic():
    name = "gate9d_query_capacity_cli_diagnostic"
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D query-capacity diagnostic")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root: pathlib.Path) -> pathlib.Path:
    manifest = root / "manifest.sha256"
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text(
        "\n".join(rows) + "\n",
        encoding="ascii",
        newline="\n",
    )
    return manifest


def _write_zip(root: pathlib.Path) -> pathlib.Path:
    archive = root.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(
            f"Gate9D query-capacity archive already exists: {archive}"
        )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            handle.write(
                path,
                arcname=f"{root.name}/{path.relative_to(root).as_posix()}",
            )
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    branch = _git("branch", "--show-current").strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Gate9D query-capacity diagnostic must run from {EXPECTED_BRANCH}"
        )
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "Gate9D query-capacity diagnostic requires a clean working tree"
        )
    head = _git("rev-parse", "HEAD").strip()
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise RuntimeError(
            "Gate9D query-capacity diagnostic could not resolve exact Git head"
        )
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FAST_HARNESS_BASE_HEAD, head],
        cwd=REPO_ROOT,
    )

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Gate9D query-capacity output already exists: {output_root}"
        )
    archive_path = output_root.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(
            f"Gate9D query-capacity archive already exists: {archive_path}"
        )

    diagnostic = _load_diagnostic()
    summary = diagnostic.run_query_capacity_diagnostic(
        output_root=output_root,
        execution_head=head,
    )
    (output_root / "git-head.txt").write_text(
        head + "\n", encoding="ascii", newline="\n"
    )
    (output_root / "git-status.txt").write_text(
        status, encoding="utf-8", newline="\n"
    )
    _write_json(
        output_root / "run-config.json",
        {
            "version": diagnostic.GATE9D_QUERY_CAPACITY_VERSION,
            "status": diagnostic.GATE9D_QUERY_CAPACITY_STATUS,
            "branch": EXPECTED_BRANCH,
            "execution_head": head,
            "fast_harness_base_head": FAST_HARNESS_BASE_HEAD,
            "python": platform.python_version(),
            "torch": diagnostic.torch.__version__,
            "output_root": str(output_root),
            "variants": list(diagnostic.GATE9D_QUERY_CAPACITY_VARIANTS),
            "variant_steps": dict(diagnostic.GATE9D_QUERY_CAPACITY_STEPS),
            "seeds": [0, 1, 2],
            "development_only": True,
            "confirmation_access": False,
            "later_stage_access": False,
            "gate9_v0_science_access": False,
            "population_execution_access": False,
            "frozen_result_mutation_access": False,
        },
    )
    manifest = _write_manifest(output_root)
    archive = _write_zip(output_root)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "diagnosis": summary["diagnosis"],
                "total_training_runs": summary["total_training_runs"],
                "aggregate_summary_sha256": _sha256(
                    output_root / "aggregate-summary.json"
                ),
                "final_runs_sha256": _sha256(output_root / "final-runs.jsonl"),
                "curves_sha256": _sha256(output_root / "curves.jsonl"),
                "predictions_sha256": _sha256(
                    output_root / "predictions.jsonl"
                ),
                "manifest_sha256": _sha256(manifest),
                "archive_sha256": _sha256(archive),
                "output_root": str(output_root),
                "archive": str(archive),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
