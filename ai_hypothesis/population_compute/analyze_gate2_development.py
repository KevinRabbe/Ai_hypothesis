"""Read-only validation and reporting for Gate-2 development artifacts.

This module never trains, evaluates, or opens confirmation. It consumes an already-written
Gate-2 development JSON artifact, validates the frozen development result contract, and
renders a compact markdown report for human interpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_ENTITY_WIDTHS = {
    16: (1, 4, 16),
    64: (1, 4, 16, 64),
    256: (1, 4, 16, 64, 256),
}
EXPECTED_MODES = ("stable_persistent", "reshuffled_locality", "reset_state")
EXPECTED_CONDITION_COUNT = 36
EXPECTED_PAIRED_COUNT = 33
EXPECTED_EXPERIMENT_VERSION = "gate2-persistent-state-development-v0"
EXPECTED_SCIENTIFIC_DECISION = "DEVELOPMENT_ONLY_NOT_ASSIGNED"


def load_gate2_development_result(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_gate2_development_result(payload)
    return payload


def validate_gate2_development_result(payload: dict[str, Any]) -> None:
    if payload.get("experiment_version") != EXPECTED_EXPERIMENT_VERSION:
        raise ValueError("unexpected Gate-2 development experiment version")
    if payload.get("evaluation_split") != "development":
        raise ValueError("Gate-2 analyzer accepts development split only")
    if payload.get("confirmation_opened") is not False:
        raise ValueError("Gate-2 development artifact must keep confirmation closed")
    if payload.get("scientific_decision") != EXPECTED_SCIENTIFIC_DECISION:
        raise ValueError("Gate-2 development artifact must not assign a scientific verdict")

    conditions = payload.get("conditions")
    paired = payload.get("paired_summaries")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITION_COUNT:
        raise ValueError("Gate-2 development artifact must contain exactly 36 conditions")
    if not isinstance(paired, list) or len(paired) != EXPECTED_PAIRED_COUNT:
        raise ValueError("Gate-2 development artifact must contain exactly 33 paired summaries")

    expected_cells = {
        (entity_count, width, mode)
        for entity_count, widths in EXPECTED_ENTITY_WIDTHS.items()
        for width in widths
        for mode in EXPECTED_MODES
    }
    observed_cells = {
        (int(row["entity_count"]), int(row["width"]), str(row["mode"]))
        for row in conditions
    }
    if observed_cells != expected_cells:
        raise ValueError("Gate-2 development condition matrix is incomplete or noncanonical")

    fingerprints = {str(row["parameter_fingerprint"]) for row in conditions}
    parameter_counts = {int(row["learned_parameter_count"]) for row in conditions}
    if len(fingerprints) != 1 or len(parameter_counts) != 1:
        raise ValueError("Gate-2 conditions do not reuse one immutable checkpoint identity")

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("Gate-2 development artifact is missing training summary")
    if str(training.get("parameter_fingerprint")) not in fingerprints:
        raise ValueError("training checkpoint fingerprint differs from evaluation fingerprint")
    if int(training.get("learned_parameter_count", -1)) not in parameter_counts:
        raise ValueError("training parameter count differs from evaluation parameter count")

    world_counts = {int(row["world_count"]) for row in conditions}
    if len(world_counts) != 1 or int(payload.get("evaluation_world_count", -1)) not in world_counts:
        raise ValueError("Gate-2 conditions do not share one held-out development world count")

    seeds_by_entity: dict[int, tuple[int, ...]] = {}
    for row in conditions:
        entity_count = int(row["entity_count"])
        seeds = tuple(int(seed) for seed in row["world_seeds"])
        previous = seeds_by_entity.setdefault(entity_count, seeds)
        if seeds != previous:
            raise ValueError("Gate-2 paired conditions must reuse identical worlds per entity tier")
        if int(row["learned_updates_per_world"]) != 8 * entity_count:
            raise ValueError("Gate-2 learned-update accounting is invalid")
        if int(row["inspected_entities_per_world"]) != entity_count:
            raise ValueError("Gate-2 inspected-entity accounting is invalid")
        if int(row["inspected_observations_per_world"]) != 8 * entity_count:
            raise ValueError("Gate-2 inspected-observation accounting is invalid")

    _validate_paired_identity(paired)


def _validate_paired_identity(paired: list[dict[str, Any]]) -> None:
    width1_identity = [
        row
        for row in paired
        if row["comparison"] == "stable_vs_reshuffled" and int(row["treatment_width"]) == 1
    ]
    if len(width1_identity) != 3:
        raise ValueError("Gate-2 width-1 stable/reshuffled identity summaries are incomplete")
    for row in width1_identity:
        if (
            float(row["exact_solve_delta"]) != 0.0
            or int(row["treatment_only"]) != 0
            or int(row["reference_only"]) != 0
            or float(row["bootstrap_ci_low"]) != 0.0
            or float(row["bootstrap_ci_high"]) != 0.0
        ):
            raise ValueError("Gate-2 width-1 stable/reshuffled identity control failed")


def render_gate2_development_markdown(payload: dict[str, Any]) -> str:
    validate_gate2_development_result(payload)
    training = payload["training"]
    conditions = {
        (int(row["entity_count"]), int(row["width"]), str(row["mode"])): row
        for row in payload["conditions"]
    }
    paired = payload["paired_summaries"]

    lines = [
        "# Gate 2 development result summary",
        "",
        "Status: **DEVELOPMENT ONLY — NO GATE VERDICT**",
        "",
        "## Training",
        "",
        f"- training seed: `{training['training_seed']}`",
        f"- steps: `{training['steps']}`",
        f"- examples seen: `{training['examples_seen']}`",
        f"- initial loss: `{float(training['initial_loss']):.6f}`",
        f"- final loss: `{float(training['final_loss']):.6f}`",
        f"- mean last-50 loss: `{float(training['mean_last_50_loss']):.6f}`",
        f"- learned parameters: `{training['learned_parameter_count']}`",
        f"- checkpoint fingerprint: `{training['parameter_fingerprint']}`",
        "",
        "## Stable-persistent width curves",
        "",
        "| Entities | Width | Collision load | Exact solve | Bit accuracy |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for entity_count, widths in EXPECTED_ENTITY_WIDTHS.items():
        for width in widths:
            row = conditions[(entity_count, width, "stable_persistent")]
            lines.append(
                f"| {entity_count} | {width} | {row['collision_load']} | "
                f"{float(row['exact_solve_rate']):.4f} | {float(row['bit_accuracy']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Primary development comparisons",
            "",
            "These mirror the frozen confirmation questions, but remain diagnostic only.",
            "",
            "| Comparison | C | Width | Delta exact solve | 95% paired bootstrap CI | Discordance (treatment/reference) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _primary_pairs(paired):
        lines.append(
            f"| {row['comparison']} | {row['entity_count']} | {row['treatment_width']} | "
            f"{float(row['exact_solve_delta']):+.4f} | "
            f"[{float(row['bootstrap_ci_low']):+.4f}, {float(row['bootstrap_ci_high']):+.4f}] | "
            f"{row['treatment_only']}/{row['reference_only']} |"
        )

    lines.extend(
        [
            "",
            "## Largest-width control performance",
            "",
            "| C | Width | Stable | Reshuffled | Reset |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for entity_count, widths in EXPECTED_ENTITY_WIDTHS.items():
        width = widths[-1]
        stable = conditions[(entity_count, width, "stable_persistent")]
        reshuffled = conditions[(entity_count, width, "reshuffled_locality")]
        reset = conditions[(entity_count, width, "reset_state")]
        lines.append(
            f"| {entity_count} | {width} | {float(stable['exact_solve_rate']):.4f} | "
            f"{float(reshuffled['exact_solve_rate']):.4f} | {float(reset['exact_solve_rate']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report is descriptive development evidence only. It may be used to decide whether the substrate trains and which development recipe to test next. It must not be used to open confirmation or assign a Gate-2 verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def _primary_pairs(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in paired:
        comparison = str(row["comparison"])
        entity_count = int(row["entity_count"])
        width = int(row["treatment_width"])
        if comparison == "stable_width_vs_width1" and (
            (entity_count == 64 and width == 64) or (entity_count == 256 and width == 256)
        ):
            selected.append(row)
        elif comparison in {"stable_vs_reshuffled", "stable_vs_reset"} and entity_count == 256 and width == 256:
            selected.append(row)
    order = {"stable_width_vs_width1": 0, "stable_vs_reshuffled": 1, "stable_vs_reset": 2}
    selected.sort(key=lambda row: (order[str(row["comparison"])], int(row["entity_count"])))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = load_gate2_development_result(args.result)
    report = render_gate2_development_markdown(payload)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
