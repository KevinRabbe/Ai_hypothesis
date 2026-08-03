from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_hypothesis.population_language.l0_overfit_diagnostic import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    arguments = parser.parse_args()
    summary = run(arguments.output_root, steps=arguments.steps)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["diagnosis"] == "POPULATION_LANGUAGE_L0_TINY_OVERFIT_PASSES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
