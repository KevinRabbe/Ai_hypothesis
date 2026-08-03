#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_hypothesis.population_language import l0_protocol as protocol
from ai_hypothesis.population_language import l0_reference_preflight as preflight

BRANCH = "agent/population-language-l0-reference-preflight-v0"


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
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(
            candidate for candidate in output_root.rglob("*") if candidate.is_file()
        ):
            relative = path.relative_to(output_root).as_posix()
            handle.write(path, arcname=f"{output_root.name}/{relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"must run from exact branch {BRANCH}; observed {branch!r}")
    status_before = _git("status", "--porcelain")
    if status_before:
        raise RuntimeError("working tree must be clean before reference preflight")
    head = _git("rev-parse", "HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("git HEAD is malformed")

    output_root = arguments.output_root.resolve()
    archive = output_root.with_suffix(".zip")
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    if archive.exists():
        raise FileExistsError(f"output archive already exists: {archive}")

    summary = preflight.run(output_root, head)
    status_after = _git("status", "--porcelain")
    if status_after:
        raise RuntimeError("working tree changed during reference preflight")

    (output_root / "git-head.txt").write_text(
        head + "\n", encoding="ascii", newline="\n"
    )
    (output_root / "git-status.txt").write_text(
        status_after + ("\n" if status_after else ""),
        encoding="utf-8",
        newline="\n",
    )
    run_config = {
        "version": preflight.VERSION,
        "branch": BRANCH,
        "execution_head": head,
        "base_head": preflight.BASE_HEAD,
        "microbatch_candidates": list(preflight.MICROBATCH_CANDIDATES),
        "minimum_common_microbatch": preflight.MINIMUM_COMMON_MICROBATCH,
        "global_batch_size": preflight.GLOBAL_BATCH_SIZE,
        "learning_rate": preflight.LEARNING_RATE,
        "initialization_seed": protocol.INITIALIZATION_SEEDS[0],
        "precision": "CUDA_BF16_AUTOCAST_FP32_PARAMETERS_AND_OPTIMIZER",
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "full_next_token_objective": True,
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
        "recommended_common_microbatch": summary["recommended_common_microbatch"],
        "gradient_accumulation_steps": summary["gradient_accumulation_steps"],
        "summary_sha256": _sha256(output_root / "summary.json"),
        "transformer_rows_sha256": _sha256(output_root / "transformer-rows.json"),
        "population_rows_sha256": _sha256(output_root / "population-rows.json"),
        "manifest_sha256": _sha256(manifest),
        "archive_sha256": _sha256(archive),
        "archive": str(archive),
    }
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if summary["diagnosis"] == preflight.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
