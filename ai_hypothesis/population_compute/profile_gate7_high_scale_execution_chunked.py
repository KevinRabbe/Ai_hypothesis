"""Engineering-only v1 profile for the B64-preserving Gate-7 chunked frontier builder.

This wrapper reuses the frozen v0 random-model/public-input profile matrix. It changes only the qualified
complete-frontier execution substrate: each action lane is evaluated in fixed contiguous recurrent-row
chunks while the same full B64 frontier is preallocated. No checkpoint, hidden path, scientific world,
coverage calculation, classifier, or Gate-7 scientific namespace is reachable here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import profile_gate7_high_scale_execution as base_profile
from .gate7_high_scale_frontier_prep import (
    GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS,
)
from .gate7_high_scale_routing_bandwidth_protocol import (
    GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE,
)

GATE7_HIGH_SCALE_CHUNKED_ENGINEERING_PROFILE_ONLY = True
GATE7_HIGH_SCALE_CHUNKED_ENGINEERING_PROFILE_VERSION = (
    "gate7-high-scale-execution-engineering-profile-chunked-v1"
)


def _add_chunk_metadata(summary_path: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["profile_version"] = GATE7_HIGH_SCALE_CHUNKED_ENGINEERING_PROFILE_VERSION
    payload["frontier_row_chunking_enabled"] = True
    payload["frontier_max_recurrent_rows"] = GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS
    payload["world_batch_preserved"] = GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE
    for tier in payload.get("tiers", []):
        population = int(tier["population"])
        final_parent_rows_per_action = (
            GATE7_HIGH_SCALE_EVALUATION_BATCH_SIZE * (population // 2)
        )
        tier["final_layer_parent_rows_per_action"] = final_parent_rows_per_action
        tier["final_layer_recurrent_chunks_per_action"] = math.ceil(
            final_parent_rows_per_action / GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS
        )
        tier["frontier_max_recurrent_rows"] = GATE7_HIGH_SCALE_FRONTIER_MAX_RECURRENT_ROWS
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_gate7_high_scale_chunked_engineering_profile(*, output_root: Path) -> int:
    original_version = base_profile.GATE7_HIGH_SCALE_ENGINEERING_PROFILE_VERSION
    base_profile.GATE7_HIGH_SCALE_ENGINEERING_PROFILE_VERSION = (
        GATE7_HIGH_SCALE_CHUNKED_ENGINEERING_PROFILE_VERSION
    )
    try:
        result = base_profile.run_gate7_high_scale_engineering_profile(
            output_root=output_root
        )
    finally:
        base_profile.GATE7_HIGH_SCALE_ENGINEERING_PROFILE_VERSION = original_version

    _add_chunk_metadata(output_root.resolve() / "summary.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run_gate7_high_scale_chunked_engineering_profile(
        output_root=args.output_root
    )


if __name__ == "__main__":
    raise SystemExit(main())
