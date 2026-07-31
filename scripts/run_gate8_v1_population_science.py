#!/usr/bin/env python3
"""Run frozen Gate-8 v1 three-checkpoint population scientific evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import platform
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_distributed_transformation_worlds.py"
)
ARCHITECTURE_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_factorized_organism_architecture.py"
)
PROTOCOL_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_v1_scientific_evaluation_protocol.py"
)
RUNTIME_PATH = REPO_ROOT / (
    "ai_hypothesis/population_compute/"
    "gate8_v1_scientific_population_runtime.py"
)

EXPERIMENT_VERSION = "gate8-v1-population-scientific-execution-v0"
SCIENTIFIC_PROTOCOL_HEAD = "6bb89111a47713bea0a23bb1cae662ed5ec56b42"
GEMMA_BINDING_RESULT_HEAD = "8237732aecbec083c66668de9fae132e0cc4c1f9"
ARCHITECTURE_HEAD = "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8"
RUNTIME_HEAD = "333d88ac4fc52f1651741fba224e0b4605feedd3"
TRAINING_PROTOCOL_HEAD = "a33dc123d090268a531d112251ea3ab53cb50062"
CHECKPOINT_EXPERIMENT_VERSION = "gate8-factorized-message-training-execution-v1"
BOOTSTRAP_ALGORITHM = (
    "numpy-pcg64-empirical-multinomial-world-index-equivalent-v0"
)
BOOTSTRAP_QUANTILE_METHOD = "linear"

EXPECTED_CHECKPOINTS = {
    0: "3005369a4830c12baee8ffa7fedc1bed0f1888784e1043bd88f4afd2b7cddde9",
    1: "cbcae487dd7f4c695e1d6a83a61926cd43f5ccf6add1a7469c16a15697d22d07",
    2: "e1e35b3864354e8f3398497a897b6a759dfa3454a33d866de63784a323f461e4",
}
ORACLE_MODE = "oracle"
RANDOM_MODE = "random_answer"
ALL_MODES = (
    "full",
    "no_communication",
    "shuffled_worker",
    "shuffled_message",
    "target_worker_only",
    RANDOM_MODE,
    ORACLE_MODE,
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 v1 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )


def _git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
    ).strip()


def _bootstrap_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "big")


def _bootstrap_empirical_mean_values(
    world_values: np.ndarray,
    *,
    namespace: str,
    samples: int,
) -> np.ndarray:
    """Sample the exact empirical world-index bootstrap distribution compactly."""

    if world_values.shape != (512,):
        raise ValueError("Gate8 v1 bootstrap world vector must contain 512 values")
    unique, counts = np.unique(world_values.astype(np.float64), return_counts=True)
    probabilities = counts.astype(np.float64) / 512.0
    rng = np.random.Generator(np.random.PCG64(_bootstrap_seed(namespace)))
    sampled_counts = rng.multinomial(512, probabilities, size=samples)
    return (sampled_counts @ unique) / 512.0


def _bootstrap_ci(
    matrix: np.ndarray,
    *,
    namespace: str,
    samples: int,
) -> tuple[float, float]:
    if matrix.shape != (3, 512):
        raise ValueError("Gate8 v1 bootstrap matrix must be 3 x 512")
    world_values = matrix.astype(np.float64).mean(axis=0)
    values = _bootstrap_empirical_mean_values(
        world_values, namespace=namespace, samples=samples
    )
    low, high = np.quantile(
        values, (0.025, 0.975), method=BOOTSTRAP_QUANTILE_METHOD
    )
    return float(low), float(high)


def _bootstrap_delta_ci(
    full: np.ndarray,
    ablation: np.ndarray,
    *,
    namespace: str,
    samples: int,
) -> tuple[float, float]:
    if full.shape != (3, 512) or ablation.shape != (3, 512):
        raise ValueError("Gate8 v1 paired bootstrap matrices must be 3 x 512")
    world_values = (
        full.astype(np.float64) - ablation.astype(np.float64)
    ).mean(axis=0)
    values = _bootstrap_empirical_mean_values(
        world_values, namespace=namespace, samples=samples
    )
    low, high = np.quantile(
        values, (0.025, 0.975), method=BOOTSTRAP_QUANTILE_METHOD
    )
    return float(low), float(high)


def _load_checkpoint(
    *,
    path: pathlib.Path,
    seed: int,
    architecture: Any,
    runtime: Any,
    torch: Any,
):
    expected_hash = EXPECTED_CHECKPOINTS[seed]
    if _sha256(path) != expected_hash:
        raise ValueError(f"Gate8 v1 seed-{seed} checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "experiment_version",
        "protocol_head",
        "runtime_head",
        "architecture_head",
        "seed",
        "step",
        "learned_parameter_count",
        "state_dict",
    }
    if set(payload) != required:
        raise ValueError(f"Gate8 v1 seed-{seed} checkpoint keys drifted")
    expected_metadata = {
        "experiment_version": CHECKPOINT_EXPERIMENT_VERSION,
        "protocol_head": TRAINING_PROTOCOL_HEAD,
        "runtime_head": RUNTIME_HEAD,
        "architecture_head": ARCHITECTURE_HEAD,
        "seed": seed,
        "step": 1_024,
        "learned_parameter_count": 19_649,
    }
    for key, value in expected_metadata.items():
        if payload[key] != value:
            raise ValueError(
                f"Gate8 v1 seed-{seed} checkpoint metadata drifted: {key}"
            )
    model = architecture.Gate8V1SharedWorkerCore()
    if set(payload["state_dict"]) != set(model.state_dict()):
        raise ValueError(f"Gate8 v1 seed-{seed} state-dict keys drifted")
    expected_state = model.state_dict()
    for name, tensor in payload["state_dict"].items():
        expected = expected_state[name]
        if tensor.dtype != torch.float32 or tensor.shape != expected.shape:
            raise ValueError(f"Gate8 v1 seed-{seed} tensor contract drifted: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Gate8 v1 seed-{seed} tensor is non-finite: {name}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    table = runtime.compile_gate8_v1_transition_table(
        model=model,
        checkpoint_seed=seed,
        checkpoint_sha256=expected_hash,
    )
    return model, table


def _metric_row(
    *,
    population: int,
    depth: int,
    mode: str,
    correctness: np.ndarray,
    resource: dict[str, float],
    bootstrap_samples: int,
) -> dict[str, Any]:
    seed_accuracies = tuple(float(correctness[seed].mean()) for seed in range(3))
    accuracy = float(sum(seed_accuracies) / 3.0)
    namespace = (
        "gate8-v1-three-seed-scientific-evaluation-bootstrap-v0:"
        f"condition:{population}:{depth}:{mode}"
    )
    ci_low, ci_high = _bootstrap_ci(
        correctness,
        namespace=namespace,
        samples=bootstrap_samples,
    )
    denominator = 3 * 512
    means = {
        key: float(resource.get(key, 0.0) / denominator)
        for key in (
            "rounds",
            "active_workers",
            "recurrent_updates",
            "delivered_messages",
            "communicated_bits",
            "wall_seconds",
        )
    }
    active = means["active_workers"]
    bits = means["communicated_bits"]
    updates = means["recurrent_updates"]
    organism_mode = mode in (
        "full",
        "no_communication",
        "shuffled_worker",
        "shuffled_message",
        "target_worker_only",
    )
    learned_parameters = 19_649 if organism_mode else 0
    normalized_compute = updates * float(learned_parameters)
    return {
        "population": population,
        "depth": depth,
        "mode": mode,
        "accuracy": accuracy,
        "seed_accuracies": list(seed_accuracies),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "mean_rounds": means["rounds"],
        "mean_active_workers": active,
        "mean_recurrent_updates": updates,
        "mean_delivered_messages": means["delivered_messages"],
        "mean_communicated_bits": bits,
        "mean_wall_seconds": means["wall_seconds"],
        "peak_device_bytes": int(resource.get("peak_device_bytes", 0.0)),
        "learned_parameter_count": learned_parameters,
        "capability_per_learned_parameter": (
            accuracy / float(learned_parameters) if learned_parameters else None
        ),
        "capability_per_active_worker": 0.0 if active == 0 else accuracy / active,
        "capability_per_communicated_bit": 0.0 if bits == 0 else accuracy / bits,
        "capability_per_recurrent_update": 0.0 if updates == 0 else accuracy / updates,
        "capability_per_normalized_compute": (
            0.0 if normalized_compute == 0 else accuracy / normalized_compute
        ),
        "correctness_matrix_sha256": hashlib.sha256(
            correctness.astype(np.uint8, copy=False).tobytes(order="C")
        ).hexdigest(),
    }


def run_population_science(
    *,
    checkpoint_paths: dict[int, pathlib.Path],
    output_root: pathlib.Path,
) -> int:
    worlds = _load(WORLD_PATH, "gate8_v1_population_science_worlds")
    architecture = _load(
        ARCHITECTURE_PATH, "gate8_v1_population_science_architecture"
    )
    protocol = _load(PROTOCOL_PATH, "gate8_v1_population_science_protocol")
    runtime = _load(RUNTIME_PATH, "gate8_v1_population_science_runtime")
    import torch

    if _git_branch() != "agent/gate8-v1-population-scientific-execution-v0":
        raise RuntimeError(
            "Gate8 v1 population science must run from its qualified branch"
        )
    status = _git_status()
    if status:
        raise RuntimeError("Gate8 v1 population science requires a clean working tree")
    git_head = _git_head()
    if len(git_head) != 40:
        raise RuntimeError("Gate8 v1 population science could not resolve its Git head")
    if output_root.exists():
        raise FileExistsError(f"Gate8 v1 population science output exists: {output_root}")
    output_root.mkdir(parents=True)
    population_root = output_root / "population"
    population_root.mkdir()
    (output_root / "git-head.txt").write_text(git_head + "\n", encoding="ascii")
    (output_root / "git-status.txt").write_text(status, encoding="utf-8")
    _write_json(
        output_root / "run-config.json",
        {
            "experiment_version": EXPERIMENT_VERSION,
            "git_head": git_head,
            "branch": "agent/gate8-v1-population-scientific-execution-v0",
            "execution_head": git_head,
            "scientific_protocol_head": SCIENTIFIC_PROTOCOL_HEAD,
            "gemma_binding_result_head": GEMMA_BINDING_RESULT_HEAD,
            "checkpoint_paths": {
                str(seed): str(checkpoint_paths[seed]) for seed in (0, 1, 2)
            },
            "checkpoint_sha256": {
                str(seed): EXPECTED_CHECKPOINTS[seed] for seed in (0, 1, 2)
            },
            "test_split": "test",
            "test_seed": 0,
            "test_world_indices": [0, 511],
            "reference_inference_performed": False,
            "training_performed": False,
        },
    )

    protocol.validate_gate8_v1_checkpoint_bindings()
    if tuple(protocol.GATE8_V1_VALID_CONDITIONS) != tuple(
        worlds.GATE8_VALID_CONDITIONS
    ):
        raise RuntimeError(
            "Gate8 v1 scientific condition matrix drifted from world contract"
        )
    if tuple(protocol.GATE8_V1_ORGANISM_MODES) != runtime.GATE8_V1_POPULATION_MODES:
        raise RuntimeError("Gate8 v1 scientific mode matrix drifted")

    tables: dict[int, Any] = {}
    compile_started = time.perf_counter()
    for seed in (0, 1, 2):
        model, table = _load_checkpoint(
            path=checkpoint_paths[seed],
            seed=seed,
            architecture=architecture,
            runtime=runtime,
            torch=torch,
        )
        tables[seed] = table
        _write_json(
            population_root / f"transition-table-seed-{seed}.json",
            table.to_dict(),
        )
        del model
    compile_seconds = time.perf_counter() - compile_started

    per_world_path = population_root / "gate8-v1-population-per-world.jsonl"
    condition_metrics: list[dict[str, Any]] = []
    condition_evidence: list[Any] = []
    ablation_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    total_worlds = 0
    total_rows = 0
    seen_world_ids: set[str] = set()
    run_started = time.perf_counter()

    with per_world_path.open("w", encoding="utf-8", newline="\n") as per_world:
        for population, depth in protocol.GATE8_V1_VALID_CONDITIONS:
            matrices = {
                mode: np.zeros((3, 512), dtype=np.uint8) for mode in ALL_MODES
            }
            resources = {mode: defaultdict(float) for mode in ALL_MODES}
            for world_index in range(512):
                generated = worlds.generate_gate8_world(
                    split="test",
                    seed=0,
                    world_index=world_index,
                    population=population,
                    depth=depth,
                )
                public = generated.public
                if public.world_id in seen_world_ids:
                    raise RuntimeError("Gate8 v1 scientific world ID repeated")
                seen_world_ids.add(public.world_id)
                if (
                    public.split != "test"
                    or public.seed != 0
                    or public.world_index != world_index
                    or public.population != population
                    or public.depth != depth
                ):
                    raise RuntimeError("Gate8 v1 generated world identity drifted")
                plan = runtime.compile_gate8_v1_scientific_world_plan(
                    public, admitted_split="test"
                )
                oracle = worlds.gate8_exact_symbolic_oracle(public)
                if oracle.answer_symbol != generated.truth.answer_symbol:
                    raise RuntimeError(
                        "Gate8 v1 test truth disagrees with exact oracle"
                    )
                random_answer = runtime.gate8_v1_deterministic_random_answer(
                    public.world_id
                )
                total_worlds += 1

                for checkpoint_seed in (0, 1, 2):
                    for mode in protocol.GATE8_V1_ORGANISM_MODES:
                        started = time.perf_counter_ns()
                        result = runtime.run_gate8_v1_scientific_population_plan(
                            table=tables[checkpoint_seed],
                            plan=plan,
                            mode=mode,
                        )
                        wall_seconds = (
                            time.perf_counter_ns() - started
                        ) / 1_000_000_000.0
                        correct = bool(
                            result.target_reached
                            and result.predicted_symbol == oracle.answer_symbol
                        )
                        matrices[mode][checkpoint_seed, world_index] = int(correct)
                        resource = resources[mode]
                        resource["rounds"] += result.rounds
                        resource["active_workers"] += result.active_workers
                        resource["recurrent_updates"] += result.recurrent_updates
                        resource["delivered_messages"] += result.delivered_messages
                        resource["communicated_bits"] += result.communicated_bits
                        resource["wall_seconds"] += wall_seconds
                        resource["peak_device_bytes"] = max(
                            resource["peak_device_bytes"], 0
                        )
                        row = {
                            "checkpoint_seed": checkpoint_seed,
                            "population": population,
                            "depth": depth,
                            "world_index": world_index,
                            "world_id": public.world_id,
                            "mode": mode,
                            "predicted_symbol": result.predicted_symbol,
                            "answer_symbol": oracle.answer_symbol,
                            "correct": correct,
                            "target_reached": result.target_reached,
                            "rounds": result.rounds,
                            "active_workers": result.active_workers,
                            "recurrent_updates": result.recurrent_updates,
                            "delivered_messages": result.delivered_messages,
                            "communicated_bits": result.communicated_bits,
                            "wall_seconds": wall_seconds,
                            "peak_device_bytes": 0,
                            "transition_table_sha256": (
                                result.transition_table_sha256
                            ),
                        }
                        per_world.write(
                            json.dumps(
                                row, sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        )
                        total_rows += 1

                    controls = (
                        (RANDOM_MODE, random_answer),
                        (ORACLE_MODE, oracle.answer_symbol),
                    )
                    for mode, predicted in controls:
                        correct = predicted == oracle.answer_symbol
                        matrices[mode][checkpoint_seed, world_index] = int(correct)
                        row = {
                            "checkpoint_seed": checkpoint_seed,
                            "population": population,
                            "depth": depth,
                            "world_index": world_index,
                            "world_id": public.world_id,
                            "mode": mode,
                            "predicted_symbol": predicted,
                            "answer_symbol": oracle.answer_symbol,
                            "correct": correct,
                            "target_reached": True,
                            "rounds": 0,
                            "active_workers": 0,
                            "recurrent_updates": 0,
                            "delivered_messages": 0,
                            "communicated_bits": 0,
                            "wall_seconds": 0.0,
                            "peak_device_bytes": 0,
                            "transition_table_sha256": None,
                        }
                        per_world.write(
                            json.dumps(
                                row, sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        )
                        total_rows += 1

            for mode in ALL_MODES:
                condition_metrics.append(
                    _metric_row(
                        population=population,
                        depth=depth,
                        mode=mode,
                        correctness=matrices[mode],
                        resource=resources[mode],
                        bootstrap_samples=protocol.GATE8_V1_BOOTSTRAP_SAMPLES,
                    )
                )
            full_metric = condition_metrics[-len(ALL_MODES)]
            condition_evidence.append(
                protocol.Gate8V1ConditionEvidence(
                    population=population,
                    depth=depth,
                    accuracy=full_metric["accuracy"],
                    bootstrap_ci_low=full_metric["bootstrap_ci_low"],
                    bootstrap_ci_high=full_metric["bootstrap_ci_high"],
                    seed_accuracies=tuple(full_metric["seed_accuracies"]),
                    mean_active_workers=full_metric["mean_active_workers"],
                    mean_communicated_bits=full_metric[
                        "mean_communicated_bits"
                    ],
                    mean_recurrent_updates=full_metric[
                        "mean_recurrent_updates"
                    ],
                )
            )
            if (population, depth) in protocol.GATE8_V1_CAUSAL_ABLATION_CONDITIONS:
                ablation_cache[(population, depth)] = {
                    mode: matrices[mode].copy()
                    for mode in ("full", "no_communication", "shuffled_worker")
                }
            print(
                f"P={population:4d} D={depth:3d} "
                f"full={full_metric['accuracy']:.6f} "
                f"CI=[{full_metric['bootstrap_ci_low']:.6f},"
                f"{full_metric['bootstrap_ci_high']:.6f}]",
                flush=True,
            )

    ablation_evidence = []
    for population, depth in protocol.GATE8_V1_CAUSAL_ABLATION_CONDITIONS:
        matrices = ablation_cache[(population, depth)]
        full = matrices["full"]
        no_comm = matrices["no_communication"]
        shuffled = matrices["shuffled_worker"]
        no_low, no_high = _bootstrap_delta_ci(
            full,
            no_comm,
            namespace=(
                "gate8-v1-three-seed-scientific-evaluation-bootstrap-v0:"
                f"ablation:{population}:{depth}:full-minus-no-communication"
            ),
            samples=protocol.GATE8_V1_BOOTSTRAP_SAMPLES,
        )
        shuffled_low, shuffled_high = _bootstrap_delta_ci(
            full,
            shuffled,
            namespace=(
                "gate8-v1-three-seed-scientific-evaluation-bootstrap-v0:"
                f"ablation:{population}:{depth}:full-minus-shuffled-worker"
            ),
            samples=protocol.GATE8_V1_BOOTSTRAP_SAMPLES,
        )
        evidence = protocol.Gate8V1AblationEvidence(
            population=population,
            depth=depth,
            full_accuracy=float(full.mean()),
            no_communication_accuracy=float(no_comm.mean()),
            shuffled_worker_accuracy=float(shuffled.mean()),
            full_seed_accuracies=tuple(
                float(full[seed].mean()) for seed in range(3)
            ),
            no_communication_seed_accuracies=tuple(
                float(no_comm[seed].mean()) for seed in range(3)
            ),
            shuffled_worker_seed_accuracies=tuple(
                float(shuffled[seed].mean()) for seed in range(3)
            ),
            full_minus_no_communication_ci_low=no_low,
            full_minus_shuffled_worker_ci_low=shuffled_low,
        )
        evidence.validate()
        row = evidence.to_dict()
        row["full_minus_no_communication_ci_high"] = no_high
        row["full_minus_shuffled_worker_ci_high"] = shuffled_high
        row["full_matrix_sha256"] = hashlib.sha256(
            full.tobytes(order="C")
        ).hexdigest()
        row["no_communication_matrix_sha256"] = hashlib.sha256(
            no_comm.tobytes(order="C")
        ).hexdigest()
        row["shuffled_worker_matrix_sha256"] = hashlib.sha256(
            shuffled.tobytes(order="C")
        ).hexdigest()
        ablation_evidence.append((evidence, row))

    condition_tuple = tuple(condition_evidence)
    ablation_tuple = tuple(evidence for evidence, _ in ablation_evidence)
    scaling = protocol.classify_gate8_v1_population_scaling(
        conditions=condition_tuple,
        ablations=ablation_tuple,
    )
    frontiers = protocol.build_gate8_v1_population_frontiers(condition_tuple)
    elapsed = time.perf_counter() - run_started
    expected_rows = 3 * len(ALL_MODES) * len(protocol.GATE8_V1_VALID_CONDITIONS) * 512
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_status": "G8_V1_POPULATION_SCIENTIFIC_EVALUATION_COMPLETE",
        "execution_head": git_head,
        "scientific_protocol_head": SCIENTIFIC_PROTOCOL_HEAD,
        "gemma_binding_result_head": GEMMA_BINDING_RESULT_HEAD,
        "architecture_head": ARCHITECTURE_HEAD,
        "runtime_head": RUNTIME_HEAD,
        "checkpoint_sha256": {
            str(seed): EXPECTED_CHECKPOINTS[seed] for seed in (0, 1, 2)
        },
        "transition_tables": {
            str(seed): {
                "sha256": tables[seed].table_sha256,
                "entries": 2_048,
                "artifact": f"transition-table-seed-{seed}.json",
            }
            for seed in (0, 1, 2)
        },
        "test_matrix": {
            "split": "test",
            "seed": 0,
            "world_index_start": 0,
            "world_index_end_inclusive": 511,
            "worlds_per_condition": 512,
            "conditions": [
                list(row) for row in protocol.GATE8_V1_VALID_CONDITIONS
            ],
            "unique_worlds": total_worlds,
        },
        "raw_rows": {
            "path": per_world_path.name,
            "rows": total_rows,
            "expected_rows": expected_rows,
            "ordering": (
                "condition_population_major_then_world_index_then_"
                "checkpoint_seed_then_mode"
            ),
        },
        "modes": list(ALL_MODES),
        "condition_metrics": condition_metrics,
        "frontiers": [row.to_dict() for row in frontiers],
        "causal_ablations": [row for _, row in ablation_evidence],
        "population_scaling_classification": scaling,
        "bootstrap": {
            "samples": protocol.GATE8_V1_BOOTSTRAP_SAMPLES,
            "confidence": protocol.GATE8_V1_BOOTSTRAP_CONFIDENCE,
            "unit": protocol.GATE8_V1_BOOTSTRAP_UNIT,
            "algorithm": BOOTSTRAP_ALGORITHM,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        },
        "execution": {
            "backend": "exhaustive_neural_transition_table",
            "neural_models_retained_during_test": False,
            "compile_seconds": compile_seconds,
            "evaluation_seconds": elapsed,
            "normalized_compute": (
                "recurrent_updates_times_19649_learned_parameters"
            ),
            "wall_time_includes_world_generation_and_jsonl_writes": False,
            "cpu_peak_device_bytes_available": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "population_evaluation_complete": True,
        "reference_model_loaded": False,
        "reference_inference_performed": False,
        "joint_reference_comparison_classified": False,
        "training_performed": False,
        "scientific_test_worlds_generated": True,
    }
    if total_worlds != len(protocol.GATE8_V1_VALID_CONDITIONS) * 512:
        raise RuntimeError("Gate8 v1 scientific unique-world count drifted")
    if len(seen_world_ids) != total_worlds:
        raise RuntimeError("Gate8 v1 scientific world-ID uniqueness drifted")
    if total_rows != expected_rows:
        raise RuntimeError("Gate8 v1 scientific raw-row count drifted")
    _write_json(population_root / "gate8-v1-population-summary.json", payload)
    print(
        json.dumps(
            {
                "status": payload["scientific_status"],
                "population_scaling_classification": scaling,
                "worlds": total_worlds,
                "rows": total_rows,
                "output_root": str(output_root),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--seed1-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--seed2-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return run_population_science(
        checkpoint_paths={
            0: args.seed0_checkpoint.resolve(),
            1: args.seed1_checkpoint.resolve(),
            2: args.seed2_checkpoint.resolve(),
        },
        output_root=args.output_root.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
