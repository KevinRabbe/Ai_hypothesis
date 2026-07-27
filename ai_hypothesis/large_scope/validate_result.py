"""Validate and normalize one large-scope benchmark result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .result_contract import validate_large_scope_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate large-scope benchmark artifact identity/comparability without "
            "applying a scientific win/fail threshold."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-split", default="development")
    parser.add_argument("--expected-widths", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--numeric-tolerance", type=float, default=1e-6)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("large-scope result artifact must contain one JSON object")

    readout = validate_large_scope_result(
        payload,
        expected_split=args.expected_split,
        expected_widths=tuple(args.expected_widths),
        numeric_tolerance=args.numeric_tolerance,
    )
    normalized = {
        "status": "VALID",
        "scientific_decision": "NOT_ASSIGNED",
        "note": (
            "Artifact identity and paired-comparison contracts passed. "
            "No population advantage threshold is applied by this validator."
        ),
        "readout": readout.to_dict(),
    }
    rendered = json.dumps(normalized, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
