#!/usr/bin/env python3
"""Run one frozen Gate-9 contextual-worker training seed."""
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
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/gate9_contextual_training_runtime.py"
)
EXPECTED_BRANCH = "agent/gate9-contextual-training-execution-v0"


def _load_runtime():
    name = "gate9_contextual_training_execution_runtime"
    spec = importlib.util.spec_from_file_location(name, RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 training runtime")
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


def _write_manifest(root: pathlib.Path) -> pathlib.Path:
    manifest = root / "manifest.sha256"
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest.write_text(
        "\n".join(rows) + "\n", encoding="ascii", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    branch = _git("branch", "--show-current").strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Gate9 training must run from {EXPECTED_BRANCH}")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("Gate9 training requires a clean working tree")
    head = _git("rev-parse", "HEAD").strip()
    if len(head) != 40:
        raise RuntimeError("Gate9 training could not resolve exact Git head")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Gate9 training output already exists: {output_root}"
        )

    runtime = _load_runtime()
    summary = runtime.run_training_seed(
        seed=args.seed,
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
            "experiment_version": runtime.GATE9_TRAINING_EXECUTION_VERSION,
            "execution_head": head,
            "branch": EXPECTED_BRANCH,
            "seed": args.seed,
            "training_protocol_head": runtime.GATE9_TRAINING_PROTOCOL_HEAD,
            "architecture_head": runtime.protocol.GATE9_ARCHITECTURE_HEAD,
            "python": platform.python_version(),
            "torch": runtime.torch.__version__,
            "numpy": runtime.np.__version__,
            "output_root": str(output_root),
            "local_test_operator_access": False,
            "graph_test_operator_access": False,
            "scientific_assignment_key_access": False,
        },
    )
    manifest = _write_manifest(output_root)
    seed_root = output_root / f"seed-{args.seed}"
    summary_path = seed_root / "summary.json"
    validation_path = seed_root / "validation-per-episode.jsonl"
    checkpoint_path = seed_root / "selected-checkpoint.pt"
    print(
        json.dumps(
            {
                "status": summary["scientific_status"],
                "seed": args.seed,
                "validation_passes": summary["validation_evidence"][
                    "admission_passes"
                ],
                "validation_exact_accuracy": summary[
                    "validation_evidence"
                ]["validation_exact_accuracy"],
                "validation_bit_accuracy": summary[
                    "validation_evidence"
                ]["validation_bit_accuracy"],
                "summary_sha256": runtime.sha256_file(summary_path),
                "validation_sha256": runtime.sha256_file(validation_path),
                "checkpoint_sha256": runtime.sha256_file(checkpoint_path),
                "manifest_sha256": runtime.sha256_file(manifest),
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
