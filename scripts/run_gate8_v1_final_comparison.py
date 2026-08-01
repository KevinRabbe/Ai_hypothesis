#!/usr/bin/env python3
"""Finalize the frozen Gate-8 v1 population-versus-Gemma comparison."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Any, Iterator

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "ai_hypothesis/population_compute/gate8_v1_final_comparison.py"

POPULATION_MODES = (
    "full",
    "no_communication",
    "shuffled_worker",
    "shuffled_message",
    "target_worker_only",
    "random_answer",
    "oracle",
)
POPULATION_KEYS = {
    "checkpoint_seed", "population", "depth", "world_index", "world_id", "mode",
    "predicted_symbol", "answer_symbol", "correct", "target_reached", "rounds",
    "active_workers", "recurrent_updates", "delivered_messages", "communicated_bits",
    "wall_seconds", "peak_device_bytes", "transition_table_sha256",
}
REFERENCE_KEYS = {
    "sequence", "population", "depth", "world_index", "world_id", "prompt_sha256",
    "ascii_bytes", "input_tokens", "answer_symbol", "generated_text", "output_token_ids",
    "predicted_symbol", "parse_status", "correct", "wall_seconds", "peak_device_bytes",
}


def load_contract():
    name = "gate8_v1_final_comparison_contract_runtime"
    spec = importlib.util.spec_from_file_location(name, CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 v1 final-comparison contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def json_lines(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or line.endswith("\r\n"):
                raise ValueError(f"noncanonical JSONL newline in {path} at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row in {path} at line {line_number}")
            yield value


def exact_hash(path: pathlib.Path, expected: str, label: str, contract: Any) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = contract.sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} observed={observed}")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_source_summaries(population: dict[str, Any], reference: dict[str, Any], c: Any) -> None:
    if population.get("scientific_status") != "G8_V1_POPULATION_SCIENTIFIC_EVALUATION_COMPLETE":
        raise ValueError("population summary scientific status drifted")
    if population.get("population_scaling_classification") != "G8_POSITIVE_CAPABILITY_SCALING":
        raise ValueError("population scaling classification drifted")
    if population.get("execution_head") != "5f9980056da6c841020fd34e19b9c041d3c1c815":
        raise ValueError("population execution head drifted")
    raw = population.get("raw_rows", {})
    if raw.get("rows") != 225_792 or raw.get("expected_rows") != 225_792:
        raise ValueError("population raw-row count drifted")
    if tuple(population.get("modes", ())) != POPULATION_MODES:
        raise ValueError("population mode order drifted")
    if population.get("reference_model_loaded") is not False:
        raise ValueError("population phase unexpectedly loaded the reference model")
    if population.get("reference_inference_performed") is not False:
        raise ValueError("population phase unexpectedly performed reference inference")
    if population.get("joint_reference_comparison_classified") is not False:
        raise ValueError("population phase unexpectedly classified the joint comparison")
    if population.get("training_performed") is not False:
        raise ValueError("population phase unexpectedly performed training")

    if reference.get("scientific_status") != "G8_V1_GEMMA_REFERENCE_EVALUATION_COMPLETE":
        raise ValueError("reference summary scientific status drifted")
    if reference.get("execution_head") != "4ab5dd3856e7bdb5afefa2a92da4fef056102995":
        raise ValueError("reference execution head drifted")
    if reference.get("population_result_head") != c.GATE8_V1_POPULATION_RESULT_HEAD:
        raise ValueError("reference population-result binding drifted")
    per_world = reference.get("per_world", {})
    if per_world.get("rows") != 10_752 or per_world.get("sha256") != c.GATE8_V1_REFERENCE_PER_WORLD_SHA256:
        raise ValueError("reference per-world binding drifted")
    prompt = reference.get("prompt_matrix", {})
    if prompt.get("rows") != 10_752 or prompt.get("sha256") != c.GATE8_V1_REFERENCE_PROMPT_INDEX_SHA256:
        raise ValueError("reference prompt-index binding drifted")
    if reference.get("reference_model_loaded") is not True:
        raise ValueError("reference model-load evidence is missing")
    if reference.get("reference_inference_performed") is not True:
        raise ValueError("reference inference evidence is missing")
    if reference.get("population_execution_performed") is not False:
        raise ValueError("reference phase unexpectedly performed population execution")
    if reference.get("joint_reference_comparison_classified") is not False:
        raise ValueError("reference phase unexpectedly classified the joint comparison")
    if reference.get("training_performed") is not False:
        raise ValueError("reference phase unexpectedly performed training")


def parse_population(path: pathlib.Path, c: Any):
    iterator = iter(json_lines(path))
    correctness = {
        condition: np.zeros((3, 512), dtype=np.uint8)
        for condition in c.GATE8_V1_VALID_CONDITIONS
    }
    identities: dict[tuple[int, int, int], tuple[str, int]] = {}
    full_rows = 0
    total = 0
    for population, depth in c.GATE8_V1_VALID_CONDITIONS:
        for world_index in range(512):
            canonical: tuple[str, int] | None = None
            for checkpoint_seed in (0, 1, 2):
                for mode in POPULATION_MODES:
                    try:
                        row = next(iterator)
                    except StopIteration as exc:
                        raise ValueError("population ledger ended before 225,792 rows") from exc
                    total += 1
                    if set(row) != POPULATION_KEYS:
                        raise ValueError(f"population row schema drifted at row {total}")
                    expected = (checkpoint_seed, population, depth, world_index, mode)
                    observed = (
                        row["checkpoint_seed"], row["population"], row["depth"],
                        row["world_index"], row["mode"],
                    )
                    if observed != expected:
                        raise ValueError(f"population row ordering drifted at row {total}")
                    world_id = row["world_id"]
                    answer = row["answer_symbol"]
                    if not isinstance(world_id, str) or not world_id.startswith("g8_"):
                        raise ValueError(f"population world ID is malformed at row {total}")
                    if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < 16:
                        raise ValueError(f"population answer is malformed at row {total}")
                    identity = (world_id, answer)
                    if canonical is None:
                        canonical = identity
                    elif canonical != identity:
                        raise ValueError(f"population world identity drifted at row {total}")
                    if not isinstance(row["correct"], bool) or not isinstance(row["target_reached"], bool):
                        raise ValueError(f"population booleans drifted at row {total}")
                    predicted = row["predicted_symbol"]
                    if row["target_reached"]:
                        if not isinstance(predicted, int) or isinstance(predicted, bool) or not 0 <= predicted < 16:
                            raise ValueError(f"population reached-target prediction is malformed at row {total}")
                    elif predicted is not None:
                        raise ValueError(f"population unreached target exposes a prediction at row {total}")
                    expected_correct = bool(row["target_reached"] and predicted == answer)
                    if row["correct"] != expected_correct:
                        raise ValueError(f"population correctness is inconsistent at row {total}")
                    if mode == "full":
                        correctness[(population, depth)][checkpoint_seed, world_index] = int(row["correct"])
                        full_rows += 1
            assert canonical is not None
            identities[(population, depth, world_index)] = canonical
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("population ledger contains rows after the frozen matrix")
    if total != 225_792 or full_rows != 32_256:
        raise ValueError("population ledger row accounting drifted")
    if len(identities) != 10_752 or len({value[0] for value in identities.values()}) != 10_752:
        raise ValueError("population world-ID uniqueness drifted")
    return correctness, identities


def parse_reference(path: pathlib.Path, identities: dict[tuple[int, int, int], tuple[str, int]], c: Any):
    iterator = iter(json_lines(path))
    correctness = {
        condition: np.zeros(512, dtype=np.uint8)
        for condition in c.GATE8_V1_VALID_CONDITIONS
    }
    maximum_tokens = {condition: 0 for condition in c.GATE8_V1_VALID_CONDITIONS}
    total = 0
    for sequence, (population, depth) in enumerate(
        condition
        for condition in c.GATE8_V1_VALID_CONDITIONS
        for _ in range(512)
    ):
        world_index = sequence % 512
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise ValueError("reference ledger ended before 10,752 rows") from exc
        total += 1
        if set(row) != REFERENCE_KEYS:
            raise ValueError(f"reference row schema drifted at row {total}")
        if (row["sequence"], row["population"], row["depth"], row["world_index"]) != (
            sequence, population, depth, world_index
        ):
            raise ValueError(f"reference row ordering drifted at row {total}")
        expected_world, expected_answer = identities[(population, depth, world_index)]
        if row["world_id"] != expected_world or row["answer_symbol"] != expected_answer:
            raise ValueError(f"population/reference identity mismatch at row {total}")
        if row["parse_status"] not in ("valid", "invalid") or not isinstance(row["correct"], bool):
            raise ValueError(f"reference parse/correctness fields drifted at row {total}")
        predicted = row["predicted_symbol"]
        if row["parse_status"] == "valid":
            if not isinstance(predicted, int) or isinstance(predicted, bool) or not 0 <= predicted < 16:
                raise ValueError(f"reference valid prediction is malformed at row {total}")
        elif predicted is not None:
            raise ValueError(f"reference invalid row exposes a prediction at row {total}")
        if row["correct"] != (predicted is not None and predicted == expected_answer):
            raise ValueError(f"reference correctness is inconsistent at row {total}")
        tokens = row["input_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 0 < tokens <= 24_576:
            raise ValueError(f"reference token count is malformed at row {total}")
        correctness[(population, depth)][world_index] = int(row["correct"])
        maximum_tokens[(population, depth)] = max(maximum_tokens[(population, depth)], tokens)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("reference ledger contains rows after the frozen matrix")
    if total != 10_752:
        raise ValueError("reference ledger row accounting drifted")
    return correctness, maximum_tokens


def atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = pathlib.Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_manifest(root: pathlib.Path, c: Any) -> str:
    manifest = root / "manifest.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            lines.append(f"{c.sha256_file(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return c.sha256_file(manifest)


def run(population_root: pathlib.Path, reference_root: pathlib.Path, output_root: pathlib.Path) -> int:
    c = load_contract()
    if git("branch", "--show-current") != c.GATE8_V1_FINAL_BRANCH:
        raise RuntimeError("Gate8 v1 final comparison must run from its qualified branch")
    if git("status", "--porcelain"):
        raise RuntimeError("Gate8 v1 final comparison requires a clean working tree")
    head = git("rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError("could not resolve exact Gate8 v1 final-comparison Git head")
    if platform.python_version() != "3.11.9":
        raise RuntimeError("Gate8 v1 final comparison requires Python 3.11.9")
    if importlib.metadata.version("numpy") != "2.3.5":
        raise RuntimeError("Gate8 v1 final comparison requires NumPy 2.3.5")
    if output_root.exists():
        raise FileExistsError(f"Gate8 v1 final-comparison output exists: {output_root}")

    population_summary_path = population_root / "population/gate8-v1-population-summary.json"
    population_world_path = population_root / "population/gate8-v1-population-per-world.jsonl"
    population_manifest_path = population_root / "manifest.sha256"
    reference_summary_path = reference_root / "reference/gate8-v1-gemma-reference-summary.json"
    reference_world_path = reference_root / "reference/gate8-v1-gemma-reference-per-world.jsonl"
    reference_prompt_path = reference_root / "reference/gate8-v1-gemma-reference-prompt-index.jsonl"
    reference_sqlite_path = reference_root / "reference/gate8-v1-gemma-reference-progress.sqlite3"
    reference_manifest_path = reference_root / "manifest.sha256"

    for path, expected, label in (
        (population_summary_path, c.GATE8_V1_POPULATION_SUMMARY_SHA256, "population summary"),
        (population_world_path, c.GATE8_V1_POPULATION_PER_WORLD_SHA256, "population per-world ledger"),
        (population_manifest_path, c.GATE8_V1_POPULATION_MANIFEST_SHA256, "population manifest"),
        (reference_summary_path, c.GATE8_V1_REFERENCE_SUMMARY_SHA256, "reference summary"),
        (reference_world_path, c.GATE8_V1_REFERENCE_PER_WORLD_SHA256, "reference per-world ledger"),
        (reference_prompt_path, c.GATE8_V1_REFERENCE_PROMPT_INDEX_SHA256, "reference prompt index"),
        (reference_sqlite_path, c.GATE8_V1_REFERENCE_SQLITE_SHA256, "reference SQLite ledger"),
        (reference_manifest_path, c.GATE8_V1_REFERENCE_MANIFEST_SHA256, "reference manifest"),
    ):
        exact_hash(path, expected, label, c)

    population_summary = read_json(population_summary_path)
    reference_summary = read_json(reference_summary_path)
    validate_source_summaries(population_summary, reference_summary, c)
    population_correctness, identities = parse_population(population_world_path, c)
    reference_correctness, maximum_tokens = parse_reference(reference_world_path, identities, c)

    condition_rows = []
    condition_vectors: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
    for population, depth in c.GATE8_V1_VALID_CONDITIONS:
        row, vector = c.condition_comparison(
            population=population,
            depth=depth,
            population_correctness=population_correctness[(population, depth)],
            reference_correctness=reference_correctness[(population, depth)],
            maximum_reference_input_tokens=maximum_tokens[(population, depth)],
        )
        condition_rows.append(row)
        condition_vectors[(population, depth)] = vector
    pooled = c.pooled_comparison(condition_vectors)
    classification = c.classify(tuple(condition_rows), pooled)

    output_root.mkdir(parents=True)
    comparison_root = output_root / "comparison"
    comparison_root.mkdir()
    (output_root / "git-head.txt").write_text(head + "\n", encoding="ascii")
    (output_root / "git-status.txt").write_text("", encoding="ascii")
    run_config = {
        "experiment_version": c.GATE8_V1_FINAL_COMPARISON_VERSION,
        "git_head": head,
        "branch": c.GATE8_V1_FINAL_BRANCH,
        "population_root": str(population_root),
        "reference_root": str(reference_root),
        "population_result_head": c.GATE8_V1_POPULATION_RESULT_HEAD,
        "gemma_result_head": c.GATE8_V1_GEMMA_RESULT_HEAD,
        "scientific_protocol_head": c.GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD,
        "bootstrap_namespace": c.GATE8_V1_BOOTSTRAP_NAMESPACE,
        "bootstrap_samples": c.GATE8_V1_BOOTSTRAP_SAMPLES,
        "pooled_coupling": c.GATE8_V1_POOLED_COUPLING,
    }
    atomic_json(output_root / "run-config.json", run_config)
    condition_path = comparison_root / "gate8-v1-final-comparison-per-condition.jsonl"
    with condition_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in condition_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    population_mean = float(sum(row["population_accuracy"] for row in condition_rows) / 21.0)
    reference_mean = float(sum(row["reference_accuracy"] for row in condition_rows) / 21.0)
    summary = {
        "experiment_version": c.GATE8_V1_FINAL_COMPARISON_VERSION,
        "scientific_status": c.GATE8_V1_FINAL_STATUS,
        "execution_head": head,
        "scientific_protocol_head": c.GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD,
        "population_result_head": c.GATE8_V1_POPULATION_RESULT_HEAD,
        "gemma_reference_result_head": c.GATE8_V1_GEMMA_RESULT_HEAD,
        "source_artifact_sha256": {
            "population_summary": c.GATE8_V1_POPULATION_SUMMARY_SHA256,
            "population_per_world": c.GATE8_V1_POPULATION_PER_WORLD_SHA256,
            "population_manifest": c.GATE8_V1_POPULATION_MANIFEST_SHA256,
            "reference_summary": c.GATE8_V1_REFERENCE_SUMMARY_SHA256,
            "reference_per_world": c.GATE8_V1_REFERENCE_PER_WORLD_SHA256,
            "reference_prompt_index": c.GATE8_V1_REFERENCE_PROMPT_INDEX_SHA256,
            "reference_sqlite": c.GATE8_V1_REFERENCE_SQLITE_SHA256,
            "reference_manifest": c.GATE8_V1_REFERENCE_MANIFEST_SHA256,
        },
        "test_matrix": {
            "conditions": 21,
            "worlds_per_condition": 512,
            "unique_worlds": 10_752,
            "population_seeds": [0, 1, 2],
            "population_full_rows": 32_256,
            "reference_rows": 10_752,
        },
        "population_scaling_classification": population_summary["population_scaling_classification"],
        "population_mean_accuracy": population_mean,
        "reference_mean_accuracy": reference_mean,
        "pooled_comparison": pooled,
        "reference_comparison_classification": classification,
        "condition_results": condition_rows,
        "bootstrap": {
            "samples": c.GATE8_V1_BOOTSTRAP_SAMPLES,
            "confidence": c.GATE8_V1_BOOTSTRAP_CONFIDENCE,
            "namespace": c.GATE8_V1_BOOTSTRAP_NAMESPACE,
            "quantile_method": c.GATE8_V1_BOOTSTRAP_QUANTILE_METHOD,
            "unit": "world_index_paired_across_three_population_seeds_and_reference_within_condition",
            "pooled_condition_weighting": "equal_weight_across_21_conditions",
            "pooled_coupling": c.GATE8_V1_POOLED_COUPLING,
        },
        "boundaries": {
            "source_ledgers_read_only": True,
            "population_execution_performed": False,
            "reference_model_loaded": False,
            "reference_inference_performed": False,
            "training_performed": False,
            "world_generation_performed": False,
            "joint_reference_comparison_classified": True,
        },
    }
    summary_path = comparison_root / "gate8-v1-final-comparison-summary.json"
    atomic_json(summary_path, summary)
    manifest_hash = write_manifest(output_root, c)
    print(json.dumps({
        "status": c.GATE8_V1_FINAL_STATUS,
        "population_scaling_classification": summary["population_scaling_classification"],
        "reference_comparison_classification": classification,
        "population_mean_accuracy": population_mean,
        "reference_mean_accuracy": reference_mean,
        "population_minus_reference_delta": pooled["population_minus_reference_delta"],
        "bootstrap_ci_low": pooled["bootstrap_ci_low"],
        "bootstrap_ci_high": pooled["bootstrap_ci_high"],
        "summary_sha256": c.sha256_file(summary_path),
        "condition_rows_sha256": c.sha256_file(condition_path),
        "manifest_sha256": manifest_hash,
        "output_root": str(output_root),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=pathlib.Path, required=True)
    parser.add_argument("--reference-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return run(args.population_root.resolve(), args.reference_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
