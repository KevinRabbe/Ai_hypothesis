"""Run frozen Gate-3 v1 robustness training seed 1 only."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_gate3_v1_robustness import run_gate3_v1_robustness_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run_gate3_v1_robustness_seed(training_seed=1, output_root=args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
