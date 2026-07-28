"""CLI for auditing Gate-1 v1 relay resource results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_resource_audit_v1 import (
    audit_relay_resource_result_v1,
    render_relay_resource_markdown_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one Gate-1 v1 relay resource result.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-noncanonical-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    audit = audit_relay_resource_result_v1(
        payload,
        require_cuda=not args.allow_cpu,
        require_canonical_checkpoint=not args.allow_noncanonical_checkpoint,
    )
    audit_payload = audit.to_dict()
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    Path(args.report_output).write_text(
        render_relay_resource_markdown_v1(payload, audit),
        encoding="utf-8",
    )
    print(json.dumps(audit_payload, indent=2, sort_keys=True))
    return 0 if audit.protocol_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
