#!/usr/bin/env python3
"""Run one immutable Gate-9 failure-decomposition stage-1 seed."""
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
    "ai_hypothesis/population_compute/"
    "gate9_contextual_failure_decomposition_stage1_runtime.py"
)
EXPECTED_BRANCH = (
    "agent/gate9-contextual-failure-decomposition-stage1-execution-v0"
)


def _load_runtime():
    name = "gate9d_stage1_cli_runtime"
    spec = importlib.util.spec_from_file_location(name, RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9D stage-1 runtime")
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
        "\n".join(rows) + "\n",
        encoding="ascii",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-index",
        type=int,
        choices=(0, 1, 2),
        required=True,
    )
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    branch = _git("branch", "--show-current").strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Gate9D stage 1 must run from {EXPECTED_BRANCH}")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("Gate9D stage 1 requires a clean working tree")
    head = _git("rev-parse", "HEAD").strip()
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise RuntimeError("Gate9D stage 1 could not resolve exact Git head")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Gate9D stage-1 output already exists: {output_root}"
        )

    runtime = _load_runtime()
    summary = runtime.run_stage1_seed(
        seed_index=args.seed_index,
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
            "experiment_version": runtime.GATE9D_STAGE1_EXECUTION_VERSION,
            "execution_head": head,
            "branch": EXPECTED_BRANCH,
            "protocol_head": runtime.GATE9D_PROTOCOL_HEAD,
            "architecture_head": runtime.protocol.GATE9D_ARCHITECTURE_HEAD,
            "operator_contract_head": (
                runtime.protocol.GATE9D_OPERATOR_CONTRACT_HEAD
            ),
            "stage": runtime.GATE9D_STAGE_NAME,
            "seed_index": args.seed_index,
            "initialization_seed": (
                runtime.protocol.GATE9D_INITIALIZATION_SEEDS[args.seed_index]
            ),
            "python": platform.python_version(),
            "torch": runtime.torch.__version__,
            "numpy": runtime.np.__version__,
            "output_root": str(output_root),
            "stage2_access": False,
            "stage3_access": False,
            "stage4_access": False,
            "gate9_v0_local_science_access": False,
            "gate9_v0_graph_science_access": False,
            "population_execution_access": False,
            "gate9_v0_result_mutation_access": False,
        },
    )
    manifest = _write_manifest(output_root)
    seed_root = output_root / f"seed-{args.seed_index}"
    summary_path = seed_root / "summary.json"
    evaluation_path = seed_root / "evaluation-per-episode.jsonl"
    checkpoint_path = seed_root / "selected-checkpoint.pt"
    print(
        json.dumps(
            {
                "status": summary["diagnostic_status"],
                "stage": summary["stage"],
                "seed_index": args.seed_index,
                "stage_passes": summary["evaluation"]["stage_passes"],
                "exact_accuracy": summary["evaluation"]["exact_accuracy"],
                "bit_accuracy": summary["evaluation"]["bit_accuracy"],
                "query_only_accuracy": summary["evaluation"][
                    "query_only_accuracy"
                ],
                "oracle_accuracy": summary["evaluation"]["oracle_accuracy"],
                "summary_sha256": runtime.sha256_file(summary_path),
                "evaluation_sha256": runtime.sha256_file(evaluation_path),
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
