"""CLI for auditing and reporting one Gate-1 relay resource-frontier result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relay_resource_audit import (
    audit_relay_resource_result,
    render_relay_resource_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one relay work/span frontier result against the frozen Gate-1 matrix "
            "and render a complete descriptive report without inventing a pass threshold."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--audit-output",
        default="results/population_compute_scaling_v0/relay_resource_frontier_v0.audit.json",
    )
    parser.add_argument(
        "--report-output",
        default="results/population_compute_scaling_v0/relay_resource_frontier_v0.report.md",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Allow CPU/mechanics results. Decisive Gate-1 auditing requires CUDA by default.",
    )
    parser.add_argument(
        "--allow-noncanonical-checkpoint",
        action="store_true",
        help="Allow synthetic/test checkpoints instead of the canonical 26,669-parameter relay-v1 checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("resource result root must be a JSON object")

    audit = audit_relay_resource_result(
        payload,
        require_cuda=not args.allow_non_cuda,
        require_canonical_checkpoint=not args.allow_noncanonical_checkpoint,
    )
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        render_relay_resource_markdown(payload, audit),
        encoding="utf-8",
    )

    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    return 0 if audit.protocol_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
