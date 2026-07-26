"""Audit and render a large-scope relevance result artifact without opening checkpoints."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from .result_audit import audit_large_scope_result, render_large_scope_audit_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and render a large-scope relevance result JSON artifact."
    )
    parser.add_argument("--input", required=True, help="Path to run_relevance JSON output")
    parser.add_argument("--output", help="Optional Markdown audit output path")
    parser.add_argument("--allow-test-split", action="store_true")
    parser.add_argument("--zero-tolerance", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read large-scope result JSON: {error}") from error
    if not isinstance(raw, Mapping):
        raise SystemExit("Large-scope result JSON root must be an object")

    audit = audit_large_scope_result(
        raw,
        allow_test_split=args.allow_test_split,
        zero_tolerance=args.zero_tolerance,
    )
    markdown = render_large_scope_audit_markdown(raw, audit)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0 if audit.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
