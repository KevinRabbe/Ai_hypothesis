#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import zipfile

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BRANCH = "agent/population-language-l0-reference-training-v0"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=_ROOT, text=True, encoding="utf-8"
    ).strip()


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(output_root: pathlib.Path) -> pathlib.Path:
    lines: list[str] = []
    for path in sorted(
        candidate
        for candidate in output_root.rglob("*")
        if candidate.is_file() and candidate.name != "manifest.sha256"
    ):
        lines.append(f"{_sha256(path)}  {path.relative_to(output_root).as_posix()}")
    manifest = output_root / "manifest.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return manifest


def _write_archive(output_root: pathlib.Path, archive: pathlib.Path) -> None:
    with zipfile.ZipFile(archive, "x", allowZip64=True) as handle:
        for path in sorted(
            candidate for candidate in output_root.rglob("*") if candidate.is_file()
        ):
            relative = path.relative_to(output_root).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if path.suffix == ".pt"
                else zipfile.ZIP_DEFLATED
            )
            handle.write(
                path,
                arcname=f"{output_root.name}/{relative}",
                compress_type=compression,
            )


def _outside_repository(path: pathlib.Path) -> bool:
    try:
        path.relative_to(_ROOT)
    except ValueError:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--evaluation-microbatch", type=int, default=8)
    arguments = parser.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python imports Torch"
        )
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"must run from exact branch {BRANCH}; observed {branch!r}")
    status_before = _git("status", "--porcelain")
    if status_before:
        raise RuntimeError("working tree must be clean before reference training")
    head = _git("rev-parse", "HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("git HEAD is malformed")

    output_root = arguments.output_root.resolve()
    archive = output_root.with_suffix(".zip")
    if not _outside_repository(output_root) or not _outside_repository(archive):
        raise ValueError("reference evidence paths must be outside the repository")
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    if archive.exists():
        raise FileExistsError(f"output archive already exists: {archive}")

    from ai_hypothesis.population_language import l0_protocol as protocol
    from ai_hypothesis.population_language import l0_reference_training as training

    summary = training.run(
        output_root,
        head,
        microbatch=arguments.microbatch,
        evaluation_microbatch=arguments.evaluation_microbatch,
    )
    status_after = _git("status", "--porcelain")
    if status_after:
        raise RuntimeError("working tree changed during reference training")

    (output_root / "git-head.txt").write_text(
        head + "\n", encoding="ascii", newline="\n"
    )
    (output_root / "git-status.txt").write_text(
        status_after + ("\n" if status_after else ""),
        encoding="utf-8",
        newline="\n",
    )
    run_config = {
        "version": training.VERSION,
        "branch": BRANCH,
        "execution_head": head,
        "base_head": training.BASE_HEAD,
        "optimizer_steps": training.OPTIMIZER_STEPS,
        "global_batch_size": training.GLOBAL_BATCH_SIZE,
        "microbatch": arguments.microbatch,
        "evaluation_microbatch": arguments.evaluation_microbatch,
        "gradient_accumulation_steps": (
            training.GLOBAL_BATCH_SIZE // arguments.microbatch
        ),
        "warmup_steps": training.WARMUP_STEPS,
        "peak_learning_rate": training.PEAK_LEARNING_RATE,
        "optimizer": "AdamW",
        "optimizer_betas": list(training.BETAS),
        "weight_decay": training.WEIGHT_DECAY,
        "gradient_clip": training.GRADIENT_CLIP,
        "precision": "CUDA_BF16_AUTOCAST_FP32_PARAMETERS_AND_OPTIMIZER",
        "initialization_seeds": list(protocol.INITIALIZATION_SEEDS),
        "training_workers": training.POPULATION_TRAIN_WORKERS,
        "evaluation_workers": list(protocol.EVAL_WORKERS),
        "full_next_token_objective": True,
        "fixed_final_checkpoint": True,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    }
    (output_root / "run-config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest = _write_manifest(output_root)
    _write_archive(output_root, archive)
    terminal = {
        "status": summary["status"],
        "diagnosis": summary["diagnosis"],
        "population_scaling_conclusion": summary["population_scaling"]["conclusion"],
        "summary_sha256": _sha256(output_root / "summary.json"),
        "manifest_sha256": _sha256(manifest),
        "archive_sha256": _sha256(archive),
        "archive": str(archive),
    }
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if summary["diagnosis"] == training.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
