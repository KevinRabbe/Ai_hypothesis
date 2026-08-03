#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import platform
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ai_hypothesis/population_compute/gate9d_learned_shared_router_zero_fp_v2.py"
EXPECTED_BRANCH = "agent/gate9d-learned-shared-router-zero-fp-v2"
BASE_HEAD = "5e89fb42d6a84e32f163d3309abbb2294206f9a1"


def _load():
    spec = importlib.util.spec_from_file_location("gate9d_router_zero_fp_cli", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load zero-FP router")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def _sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    branch = _git("branch", "--show-current").strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"must run from {EXPECTED_BRANCH}")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("zero-FP router requires a clean working tree")
    head = _git("rev-parse", "HEAD").strip()
    subprocess.check_call(["git", "merge-base", "--is-ancestor", BASE_HEAD, head], cwd=ROOT)
    output = args.output_root.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("output or archive already exists")
    module = _load()
    summary = module.run(output, head)
    (output / "git-head.txt").write_text(head + "\n", encoding="ascii", newline="\n")
    (output / "git-status.txt").write_text(status, encoding="utf-8", newline="\n")
    (output / "run-config.json").write_text(
        json.dumps({
            "version": module.VERSION,
            "status": module.STATUS,
            "branch": EXPECTED_BRANCH,
            "execution_head": head,
            "base_head": BASE_HEAD,
            "python": platform.python_version(),
            "torch": module.torch.__version__,
            "output_root": str(output),
            "development_only": True,
            "supervised_routing_labels_used": True,
            "exhaustive_threshold_calibration_used": True,
            "automatic_coordinate_discovery_claim_access": False,
            "later_stage_access": False,
            "population_confirmation_access": False,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n"
    )
    manifest_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        manifest_rows.append(f"{_sha(path)}  {path.relative_to(output).as_posix()}")
    manifest = output / "manifest.sha256"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="ascii", newline="\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            handle.write(path, arcname=f"{output.name}/{path.relative_to(output).as_posix()}")
    print(json.dumps({
        "status": summary["status"],
        "diagnosis": summary["diagnosis"],
        "parameter_count": summary["parameter_count"],
        "aggregate_summary_sha256": _sha(output / "aggregate-summary.json"),
        "final_rows_sha256": _sha(output / "final-rows.jsonl"),
        "curves_sha256": _sha(output / "curves.jsonl"),
        "manifest_sha256": _sha(manifest),
        "archive_sha256": _sha(archive),
        "output_root": str(output),
        "archive": str(archive),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
