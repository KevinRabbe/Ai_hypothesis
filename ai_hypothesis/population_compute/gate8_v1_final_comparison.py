"""Frozen Gate-8 v1 population-versus-Gemma final-comparison mechanics.

This module reads no artifacts and opens no execution surface. It defines the
already-preregistered paired estimand, deterministic bootstrap mechanics, and
an exact bridge to the unchanged Gate-8 reference classifier.
"""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys
from typing import Any, Mapping

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent
_PROTOCOL_PATH = _ROOT / "gate8_v1_scientific_evaluation_protocol.py"

GATE8_V1_FINAL_COMPARISON_VERSION = "gate8-v1-final-comparison-execution-v0"
GATE8_V1_FINAL_STATUS = "G8_V1_FINAL_COMPARISON_COMPLETE"
GATE8_V1_FINAL_BRANCH = "agent/gate8-v1-final-comparison-execution-v0"
GATE8_V1_GEMMA_RESULT_HEAD = "1d48ecfd623a2fb9e3a2f846a4d1c49d20d8cadc"
GATE8_V1_POPULATION_RESULT_HEAD = "14636d219781381853f81036b96c691b7e6997ee"
GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD = "6bb89111a47713bea0a23bb1cae662ed5ec56b42"

GATE8_V1_POPULATION_SUMMARY_SHA256 = "6d30d773f11c1155df3346128385da9231610ea05e95937e5acccb5529fca3fe"
GATE8_V1_POPULATION_PER_WORLD_SHA256 = "45e36bda230440d4fa2342183154b474473498df51917e147a37e0baa81c3323"
GATE8_V1_POPULATION_MANIFEST_SHA256 = "8214aa82733a4fab9148a3ea210fd110b0a85f857483f14b15cb53d0f451255d"
GATE8_V1_REFERENCE_SUMMARY_SHA256 = "7e7f8002b41d25d6448ecaea6882fa84926b006d95c1aa024a94b774b0b305ab"
GATE8_V1_REFERENCE_PER_WORLD_SHA256 = "dda1009295378f4626b444b016b7aed2ff06c3468dc8b385d64809d4704a4706"
GATE8_V1_REFERENCE_PROMPT_INDEX_SHA256 = "e238d801743939acaa362455410f85edd6a67d50f189d68a09bf75ffb63c60ab"
GATE8_V1_REFERENCE_SQLITE_SHA256 = "7173853a236b777a02596bce3b61abecef3e61d52df661eb39422325fbb224a1"
GATE8_V1_REFERENCE_MANIFEST_SHA256 = "3fc0628c0c5fb56901160f35639c708a44cb2501540db9ea5022dce0e374b743"

GATE8_V1_BOOTSTRAP_SAMPLES = 20_000
GATE8_V1_BOOTSTRAP_CONFIDENCE = 0.95
GATE8_V1_BOOTSTRAP_QUANTILE_METHOD = "linear"
GATE8_V1_BOOTSTRAP_NAMESPACE = "gate8-v1-final-comparison-bootstrap-v0"
GATE8_V1_POOLED_COUPLING = (
    "independent_condition_streams_same_replicate_index_equal_condition_weight"
)
GATE8_V1_VALID_CONDITIONS = (
    (32, 4),
    (64, 4), (64, 8),
    (128, 4), (128, 8), (128, 16),
    (256, 4), (256, 8), (256, 16), (256, 32),
    (512, 4), (512, 8), (512, 16), (512, 32), (512, 64),
    (1024, 4), (1024, 8), (1024, 16), (1024, 32), (1024, 64), (1024, 128),
)


def _load_protocol():
    name = "gate8_v1_final_comparison_protocol_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen Gate8 v1 scientific protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("ascii")).digest()[:8], "big")


def _binary_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint8)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{label} must contain only binary correctness values")
    return array


def world_delta_vector(population_correctness: Any, reference_correctness: Any) -> np.ndarray:
    population = _binary_array(population_correctness, (3, 512), "population correctness")
    reference = _binary_array(reference_correctness, (512,), "reference correctness")
    return population.astype(np.float64).mean(axis=0) - reference.astype(np.float64)


def empirical_bootstrap_values(world_values: Any, namespace: str) -> np.ndarray:
    vector = np.asarray(world_values, dtype=np.float64)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Gate8 final-comparison world vector must be finite with 512 values")
    unique, counts = np.unique(vector, return_counts=True)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed(namespace)))
    sampled = rng.multinomial(
        512,
        counts.astype(np.float64) / 512.0,
        size=GATE8_V1_BOOTSTRAP_SAMPLES,
    )
    return (sampled @ unique) / 512.0


def quantile_interval(values: Any) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (GATE8_V1_BOOTSTRAP_SAMPLES,) or not np.isfinite(array).all():
        raise ValueError("Gate8 final-comparison bootstrap vector is invalid")
    low, high = np.quantile(
        array,
        (0.025, 0.975),
        method=GATE8_V1_BOOTSTRAP_QUANTILE_METHOD,
    )
    return float(low), float(high)


def condition_comparison(
    *,
    population: int,
    depth: int,
    population_correctness: Any,
    reference_correctness: Any,
    maximum_reference_input_tokens: int,
) -> tuple[dict[str, Any], np.ndarray]:
    if (population, depth) not in GATE8_V1_VALID_CONDITIONS:
        raise ValueError("Gate8 final-comparison condition is outside the frozen matrix")
    if not 0 < maximum_reference_input_tokens <= 24_576:
        raise ValueError("Gate8 final-comparison reference token count is invalid")
    pop = _binary_array(population_correctness, (3, 512), "population correctness")
    ref = _binary_array(reference_correctness, (512,), "reference correctness")
    world_values = world_delta_vector(pop, ref)
    samples = empirical_bootstrap_values(
        world_values,
        f"{GATE8_V1_BOOTSTRAP_NAMESPACE}:condition:{population}:{depth}",
    )
    low, high = quantile_interval(samples)
    seed_accuracies = tuple(float(pop[seed].mean()) for seed in range(3))
    population_accuracy = float(sum(seed_accuracies) / 3.0)
    reference_accuracy = float(ref.mean())
    delta = population_accuracy - reference_accuracy
    row = {
        "population": population,
        "depth": depth,
        "population_accuracy": population_accuracy,
        "population_seed_accuracies": list(seed_accuracies),
        "reference_accuracy": reference_accuracy,
        "population_minus_reference_delta": delta,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "maximum_reference_input_tokens": maximum_reference_input_tokens,
        "population_correctness_matrix_sha256": hashlib.sha256(pop.tobytes(order="C")).hexdigest(),
        "reference_correctness_vector_sha256": hashlib.sha256(ref.tobytes(order="C")).hexdigest(),
        "paired_world_delta_vector_sha256": hashlib.sha256(world_values.tobytes(order="C")).hexdigest(),
    }
    return row, world_values


def pooled_comparison(condition_world_values: Mapping[tuple[int, int], Any]) -> dict[str, Any]:
    if tuple(condition_world_values) != GATE8_V1_VALID_CONDITIONS:
        raise ValueError("Gate8 pooled comparison must cover the exact ordered 21-condition matrix")
    pooled_samples = np.zeros(GATE8_V1_BOOTSTRAP_SAMPLES, dtype=np.float64)
    points = []
    for population, depth in GATE8_V1_VALID_CONDITIONS:
        vector = np.asarray(condition_world_values[(population, depth)], dtype=np.float64)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("Gate8 pooled condition vector is invalid")
        points.append(float(vector.mean()))
        pooled_samples += empirical_bootstrap_values(
            vector,
            f"{GATE8_V1_BOOTSTRAP_NAMESPACE}:pooled:{population}:{depth}",
        ) / len(GATE8_V1_VALID_CONDITIONS)
    low, high = quantile_interval(pooled_samples)
    return {
        "population_minus_reference_delta": float(sum(points) / len(points)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "conditions": len(points),
        "condition_weight": 1.0 / len(points),
        "coupling": GATE8_V1_POOLED_COUPLING,
        "bootstrap_replicate_vector_sha256": hashlib.sha256(
            pooled_samples.tobytes(order="C")
        ).hexdigest(),
    }


def classify(condition_rows: tuple[dict[str, Any], ...], pooled: dict[str, Any]) -> str:
    if tuple((row["population"], row["depth"]) for row in condition_rows) != GATE8_V1_VALID_CONDITIONS:
        raise ValueError("Gate8 classification rows must cover the exact condition matrix")
    protocol = _load_protocol()
    evidence = tuple(
        protocol.Gate8V1ReferenceEvidence(
            population=int(row["population"]),
            depth=int(row["depth"]),
            population_accuracy=float(row["population_accuracy"]),
            reference_accuracy=float(row["reference_accuracy"]),
            population_seed_accuracies=tuple(float(value) for value in row["population_seed_accuracies"]),
            population_minus_reference_delta=float(row["population_minus_reference_delta"]),
            bootstrap_ci_low=float(row["bootstrap_ci_low"]),
            bootstrap_ci_high=float(row["bootstrap_ci_high"]),
            maximum_reference_input_tokens=int(row["maximum_reference_input_tokens"]),
        )
        for row in condition_rows
    )
    pooled_row = protocol.capability.Gate8ReferencePooledRow(
        population_minus_reference_delta=float(pooled["population_minus_reference_delta"]),
        bootstrap_ci_low=float(pooled["bootstrap_ci_low"]),
        bootstrap_ci_high=float(pooled["bootstrap_ci_high"]),
    )
    return protocol.classify_gate8_v1_reference_comparison(
        conditions=evidence,
        pooled=pooled_row,
    )


def final_comparison_plan() -> dict[str, Any]:
    return {
        "version": GATE8_V1_FINAL_COMPARISON_VERSION,
        "branch": GATE8_V1_FINAL_BRANCH,
        "gemma_result_head": GATE8_V1_GEMMA_RESULT_HEAD,
        "population_result_head": GATE8_V1_POPULATION_RESULT_HEAD,
        "scientific_protocol_head": GATE8_V1_SCIENTIFIC_PROTOCOL_HEAD,
        "conditions": [list(row) for row in GATE8_V1_VALID_CONDITIONS],
        "worlds_per_condition": 512,
        "population_seeds": [0, 1, 2],
        "bootstrap_samples": GATE8_V1_BOOTSTRAP_SAMPLES,
        "bootstrap_confidence": GATE8_V1_BOOTSTRAP_CONFIDENCE,
        "bootstrap_namespace": GATE8_V1_BOOTSTRAP_NAMESPACE,
        "bootstrap_quantile_method": GATE8_V1_BOOTSTRAP_QUANTILE_METHOD,
        "pooled_coupling": GATE8_V1_POOLED_COUPLING,
        "model_loading_admitted": False,
        "world_generation_admitted": False,
        "population_execution_admitted": False,
        "reference_inference_admitted": False,
        "training_admitted": False,
    }
