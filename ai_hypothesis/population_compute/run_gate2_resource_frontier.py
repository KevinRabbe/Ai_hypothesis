"""CLI for the frozen Gate-2 eager-CUDA persistent-state resource frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .gate2_resource_frontier import run_gate2_resource_frontier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = min(width, max(0, round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Gate-2 resource output already exists: {args.output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("Gate-2 resource protocol requires CUDA")

    started = time.monotonic()

    def progress(stage: str, done: int, total: int, entity_count: int, width: int) -> None:
        percent = 100.0 * done / total
        elapsed = (time.monotonic() - started) / 60.0
        print(
            f"{stage.upper():9s} {_bar(done, total)} {percent:6.2f}%  "
            f"{done:2d}/{total}  C={entity_count:<3d} W={width:<3d} elapsed={elapsed:.1f}m",
            flush=True,
        )

    result = run_gate2_resource_frontier(
        checkpoint_path=args.checkpoint.resolve(),
        device="cuda",
        progress=progress,
    )
    payload = result.to_dict()
    payload["checkpoint_sha256"] = _sha256(args.checkpoint)
    payload["wall_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "GATE2_RESOURCE_FRONTIER_COMPLETE",
                "resource_frontier_passed": result.resource_frontier_passed,
                "all_preflights_passed": result.all_preflights_passed,
                "decision_endpoint_passes": result.decision_endpoint_passes,
                "output": str(args.output.resolve()),
                "checkpoint_sha256": payload["checkpoint_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
