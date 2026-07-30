"""Independent CPU auditor for Gate-7 information-ceiling precision confirmation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_HEAD = "8d7865ab01b4b04b875ed2ca627b68a6c33c81f7"
BASE_RESULT_HEAD = "4eb3e50a3ca7898ff81aebebddb7b049ff855df3"
BASE_RESULT_SHA256 = "71a383ced44419f84022738448c460d79a3fb21746f436649e5f14399704f731"
BASE_AUDIT_SHA256 = "86a7dbb774119cca9bcd697978081e0872b41e4e61a3f8b08538e0cc89c8397d"
BASE_RECOVERY_SHA256 = "ccd4bbd353aba09b8a2d38d155bb9f883b862123bf196693889b515d5452324b"
BASE_MANIFEST_SHA256 = "026f75a76888efe020c57da9d719140169eedd5e024555db20da9590cfea2b45"
BASE_OUTCOME = "G7_INFORMATION_CEILING_INCONCLUSIVE"
EXPERIMENT_VERSION = "gate7-information-ceiling-precision-confirmation-v0"
SCIENTIFIC_STATUS = "FRESH_GATE7_INFORMATION_CEILING_PRECISION_CONFIRMATION_EVIDENCE"
POPULATIONS = (16_384, 32_768, 65_536, 131_072)
CHECKPOINT_INDICES = (0, 1, 2)
CHECKPOINTS = {
    0: (
        "be1b18e22a1f713b48de8934aca1c0302f1255342684158c6ae31a7c3618a719",
        "0be5c26d1056da1bf12a53be5ba5e6d1cadb7815eb117e2a77db269391c1c5aa",
    ),
    1: (
        "a9b712736f440168ccd86ee070b8d301be0eb834b9d17418d40c0175207cadfb",
        "b9685382992cb2f94454b6faa3675d458f236e7b0c7d8399bea256bedcb02e46",
    ),
    2: (
        "cd7869511d07679cb1ad430743cad031299a575413e8a853ef9116d545c7480a",
        "1bca7012c7350c3b3fe8c9790a65c36eea8fcd8bef3e123034817ff78673a7bb",
    ),
}
PARAMETER_COUNT = 19_649
WORLD_COUNT = 2_048
BATCH_SIZE = 64
BATCH_COUNT = 32
BOOTSTRAP_SAMPLES = 20_000
PRIMARY_ATTEMPTS = 128
ATTEMPTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024)
HINT_RELIABILITY = 0.70
MARGIN = 0.02
LEARNED = "learned_score_rank"
BAYES = "bayes_hint_likelihood_rank"
HASH = "public_hash_rank"
RANKERS = (LEARNED, BAYES, HASH)
COMPARISONS = ("learned_vs_bayes", "learned_vs_hash", "bayes_vs_hash")
HIDDEN_NAMESPACE = "gate7-information-ceiling-precision-confirmation-hidden-v0"
HINT_NAMESPACE = "gate7-information-ceiling-precision-confirmation-hints-v0"
RUNTIME_NAMESPACE = "gate7-information-ceiling-precision-confirmation-runtime-v0"
TIE_NAMESPACE = "gate7-information-ceiling-precision-confirmation-public-tie-v0"
HASH_NAMESPACE = "gate7-information-ceiling-precision-confirmation-public-hash-v0"
BOOTSTRAP_NAMESPACE = "gate7-information-ceiling-precision-confirmation-clustered-bootstrap-v0"
OUTCOME_DOMINANT = "G7_PRECISION_INFORMATION_CEILING_DOMINANT"
OUTCOME_GAP = "G7_PRECISION_SCORER_REPRESENTATION_GAP"
OUTCOME_MIXED = "G7_PRECISION_INFORMATION_AND_SCORER_GAP_MIXED"
OUTCOME_INCONCLUSIVE = "G7_PRECISION_INCONCLUSIVE"
VALID_OUTCOMES = {
    OUTCOME_DOMINANT,
    OUTCOME_GAP,
    OUTCOME_MIXED,
    OUTCOME_INCONCLUSIVE,
}
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)


@dataclass(frozen=True, slots=True)
class Gate7PrecisionAudit:
    artifact_valid: bool
    scientific_status: str
    campaign_outcome: str | None
    primary_coverage_by_population_checkpoint: dict[str, dict[str, float]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in parts).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def world_seed_from_parts(*parts: object) -> int:
    return seed_from_parts(*parts) & ((1 << 63) - 1)


def generate_world(population: int, world_index: int) -> dict[str, Any]:
    depth = population.bit_length() - 1
    task_depth = depth + 1
    hidden_rng = random.Random(
        world_seed_from_parts(
            HIDDEN_NAMESPACE, population, world_index, task_depth
        )
    )
    hidden = tuple(hidden_rng.randrange(2) for _ in range(task_depth))
    hint_rng = random.Random(
        world_seed_from_parts(HINT_NAMESPACE, population, world_index, task_depth)
    )
    hints = tuple(
        bit if hint_rng.random() < HINT_RELIABILITY else 1 - bit
        for bit in hidden
    )
    parent = 0
    hint_prefix = 0
    for hidden_bit, hint_bit in zip(
        hidden[:depth], hints[:depth], strict=True
    ):
        parent = parent * 2 + hidden_bit
        hint_prefix = hint_prefix * 2 + hint_bit

    def permutation(namespace: str) -> tuple[int, int]:
        rng = random.Random(
            world_seed_from_parts(namespace, population, world_index, depth)
        )
        return rng.randrange(1, population, 2), rng.randrange(population)

    tie_multiplier, tie_offset = permutation(TIE_NAMESPACE)
    hash_multiplier, hash_offset = permutation(HASH_NAMESPACE)
    return {
        "runtime_seed": world_seed_from_parts(
            RUNTIME_NAMESPACE, population, world_index, depth
        ),
        "parent": parent,
        "hint_prefix": hint_prefix,
        "tie_multiplier": tie_multiplier,
        "tie_offset": tie_offset,
        "hash_multiplier": hash_multiplier,
        "hash_offset": hash_offset,
    }


def affine_priority(
    candidate: int, multiplier: int, offset: int, population: int
) -> int:
    return (candidate * multiplier + offset) & (population - 1)


def exact_public_ranks(
    population: int, world_index: int
) -> tuple[int, int, int]:
    world = generate_world(population, world_index)
    parent = int(world["parent"])
    hint_prefix = int(world["hint_prefix"])
    hidden_distance = (parent ^ hint_prefix).bit_count()
    depth = population.bit_length() - 1
    lower_shells = sum(
        math.comb(depth, distance) for distance in range(hidden_distance)
    )
    hidden_tie = affine_priority(
        parent,
        int(world["tie_multiplier"]),
        int(world["tie_offset"]),
        population,
    )
    boundary_before = 0
    for flipped_positions in itertools.combinations(
        range(depth), hidden_distance
    ):
        candidate = hint_prefix
        for bit_position in flipped_positions:
            candidate ^= 1 << bit_position
        if affine_priority(
            candidate,
            int(world["tie_multiplier"]),
            int(world["tie_offset"]),
            population,
        ) < hidden_tie:
            boundary_before += 1
    bayes_rank = 1 + lower_shells + boundary_before
    hash_rank = 1 + affine_priority(
        parent,
        int(world["hash_multiplier"]),
        int(world["hash_offset"]),
        population,
    )
    return int(world["runtime_seed"]), bayes_rank, hash_rank


def bayes_expected(population: int, attempts: int) -> float:
    depth = population.bit_length() - 1
    remaining = attempts
    expected = 0.0
    error = 1.0 - HINT_RELIABILITY
    for distance in range(depth + 1):
        shell = math.comb(depth, distance)
        probability = (
            shell
            * (error**distance)
            * (HINT_RELIABILITY ** (depth - distance))
        )
        admitted = min(remaining, shell)
        expected += probability * admitted / shell
        remaining -= admitted
        if remaining == 0:
            break
    return expected


def float_equal(
    left: object, right: object, tolerance: float = 1e-12
) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def expected_rank_summary(ranks: list[int]) -> dict[str, Any]:
    ordered = sorted(ranks)

    def quantile(probability: float) -> int:
        return ordered[int(math.floor(probability * (len(ordered) - 1)))]

    return {
        "coverage_by_attempt": {
            str(attempt): sum(int(rank <= attempt) for rank in ranks) / len(ranks)
            for attempt in ATTEMPTS
        },
        "mean_rank": sum(ranks) / len(ranks),
        "rank_quantiles": {
            "p25": quantile(0.25),
            "p50": quantile(0.50),
            "p75": quantile(0.75),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
        },
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks)
        / len(ranks),
        "mean_log2_rank_plus_one": sum(
            math.log2(rank + 1) for rank in ranks
        )
        / len(ranks),
        "rank_checksum": sum(ranks),
    }


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (values + _SPLITMIX_GAMMA) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _SPLITMIX_M1) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _SPLITMIX_M2) & _UINT64_MASK
        return (z ^ (z >> np.uint64(31))) & _UINT64_MASK


def bootstrap_estimates(values: np.ndarray, seed: int) -> np.ndarray:
    estimates = np.empty(
        (BOOTSTRAP_SAMPLES, values.shape[1]), dtype=np.float64
    )
    draws = np.arange(WORLD_COUNT, dtype=np.uint64)[None, :]
    for start in range(0, BOOTSTRAP_SAMPLES, 128):
        stop = min(start + 128, BOOTSTRAP_SAMPLES)
        replicates = np.arange(start, stop, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):
            counters = (
                np.uint64(seed)
                + replicates * np.uint64(WORLD_COUNT)
                + draws
            )
        indices = (
            _splitmix64(counters) % np.uint64(WORLD_COUNT)
        ).astype(np.int64)
        estimates[start:stop] = values[indices].mean(axis=1)
    return estimates


def intervals(estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(estimates, axis=0)
    low = ordered[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))]
    high = ordered[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))]
    return low, high


def difference_matrix(ranks: dict[str, list[int]]) -> np.ndarray:
    learned = np.asarray(ranks[LEARNED], dtype=np.int64) <= PRIMARY_ATTEMPTS
    bayes = np.asarray(ranks[BAYES], dtype=np.int64) <= PRIMARY_ATTEMPTS
    public_hash = (
        np.asarray(ranks[HASH], dtype=np.int64) <= PRIMARY_ATTEMPTS
    )
    return np.column_stack(
        (
            learned.astype(np.int8) - bayes.astype(np.int8),
            learned.astype(np.int8) - public_hash.astype(np.int8),
            bayes.astype(np.int8) - public_hash.astype(np.int8),
        )
    ).astype(np.float64)


def summary_triplet(
    scope: str,
    population: int | None,
    checkpoint: int | None,
    matrix: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    point = matrix.mean(axis=0)
    estimates = bootstrap_estimates(matrix, seed)
    low, high = intervals(estimates)
    rows = [
        {
            "scope": scope,
            "population": population,
            "checkpoint_index": checkpoint,
            "comparison": comparison,
            "attempts": PRIMARY_ATTEMPTS,
            "coverage_delta": float(point[index]),
            "bootstrap_ci_low": float(low[index]),
            "bootstrap_ci_high": float(high[index]),
        }
        for index, comparison in enumerate(COMPARISONS)
    ]
    by_name = {row["comparison"]: row for row in rows}
    learned_bayes = by_name["learned_vs_bayes"]
    comparison = {
        "learned_minus_bayes_delta": learned_bayes["coverage_delta"],
        "learned_minus_bayes_ci_low": learned_bayes["bootstrap_ci_low"],
        "learned_minus_bayes_ci_high": learned_bayes["bootstrap_ci_high"],
        "learned_minus_hash_ci_low": by_name["learned_vs_hash"][
            "bootstrap_ci_low"
        ],
        "bayes_minus_hash_ci_low": by_name["bayes_vs_hash"][
            "bootstrap_ci_low"
        ],
    }
    if population is not None:
        comparison["population"] = population
    if checkpoint is not None:
        comparison["checkpoint_index"] = checkpoint
    return rows, comparison


def compare_payload(
    observed: object, expected: object, path: str, errors: list[str]
) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            errors.append(f"{path} is not an object")
            return
        if set(observed) != set(expected):
            errors.append(f"{path} keys changed")
            return
        for key, value in expected.items():
            compare_payload(observed[key], value, f"{path}.{key}", errors)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            errors.append(f"{path} list changed")
            return
        for index, value in enumerate(expected):
            compare_payload(observed[index], value, f"{path}[{index}]", errors)
        return
    if isinstance(expected, float):
        if not float_equal(observed, expected):
            errors.append(f"{path} changed")
        return
    if observed != expected:
        errors.append(f"{path} changed")


def classify(
    cells: list[dict[str, Any]],
    populations: list[dict[str, Any]],
    pooled: dict[str, Any],
) -> str:
    pooled_gap = (
        float(pooled["learned_minus_bayes_ci_high"]) < -MARGIN
        and float(pooled["bayes_minus_hash_ci_low"]) > 0.0
    )
    local_gap = any(
        float(row["learned_minus_bayes_ci_high"]) < -MARGIN
        and float(row["bayes_minus_hash_ci_low"]) > 0.0
        for row in cells + populations
    )
    controls = (
        float(pooled["learned_minus_hash_ci_low"]) > 0.0
        and float(pooled["bayes_minus_hash_ci_low"]) > 0.0
    )
    near = float(pooled["learned_minus_bayes_ci_low"]) > -MARGIN and controls
    points = all(
        float(row["learned_minus_bayes_delta"]) > -MARGIN
        for row in populations
    )
    if pooled_gap:
        return OUTCOME_GAP
    if near and points and not local_gap:
        return OUTCOME_DOMINANT
    if local_gap:
        return OUTCOME_MIXED
    return OUTCOME_INCONCLUSIVE


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("precision artifact must be one JSON object")
    return payload


def audit_gate7_precision_confirmation(path: Path) -> Gate7PrecisionAudit:
    errors: list[str] = []
    primary: dict[str, dict[str, float]] = {}
    try:
        payload = load_payload(path)
    except Exception as exc:  # noqa: BLE001
        return Gate7PrecisionAudit(
            False, "INVALID_ARTIFACT", None, {}, (str(exc),)
        )

    exact = {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_head": PROTOCOL_HEAD,
        "base_result_head": BASE_RESULT_HEAD,
        "base_decomposition_result_sha256": BASE_RESULT_SHA256,
        "base_decomposition_recovered_audit_sha256": BASE_AUDIT_SHA256,
        "base_decomposition_recovery_record_sha256": BASE_RECOVERY_SHA256,
        "base_decomposition_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_decomposition_outcome": BASE_OUTCOME,
        "populations": list(POPULATIONS),
        "checkpoint_indices": list(CHECKPOINT_INDICES),
        "rankers": list(RANKERS),
        "attempt_ladder": list(ATTEMPTS),
        "primary_attempts": PRIMARY_ATTEMPTS,
        "world_count_per_checkpoint_population": WORLD_COUNT,
        "evaluation_batch_size": BATCH_SIZE,
        "physical_batch_count": BATCH_COUNT,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_unit": (
            "world_index_clustered_within_population_across_T0_T1_T2"
        ),
        "pooled_weighting": "equal_population_then_equal_checkpoint",
        "bootstrap_generator": "splitmix64_counter_v0",
        "learned_parameter_count": PARAMETER_COUNT,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            errors.append(f"{key} mismatch")
    for key in ("execution_admitted", "execution_opened"):
        if payload.get(key) is not True:
            errors.append(f"{key} must be true")
    for key in (
        "result_opened",
        "training_performed",
        "checkpoint_selection_performed",
        "communication_intervention_performed",
        "recycling_intervention_performed",
        "specialization_intervention_performed",
        "topology_intervention_performed",
        "adaptive_attempt_exposure_performed",
        "prior_worlds_reused",
        "compiler_enabled",
        "cuda_graphs_enabled",
        "mixed_precision_enabled",
    ):
        if payload.get(key) is not False:
            errors.append(f"{key} must be false")
    if not float_equal(payload.get("hint_reliability"), HINT_RELIABILITY):
        errors.append("hint reliability changed")
    if not float_equal(payload.get("near_ceiling_margin"), MARGIN):
        errors.append("near-ceiling margin changed")
    expected_analytic = {
        str(population): bayes_expected(population, PRIMARY_ATTEMPTS)
        for population in POPULATIONS
    }
    observed_analytic = payload.get("analytic_primary_ceiling_by_population")
    if not isinstance(observed_analytic, dict):
        errors.append("analytic ceiling is not an object")
    else:
        for key, expected in expected_analytic.items():
            if key not in observed_analytic or not float_equal(
                observed_analytic[key], expected, 1e-15
            ):
                errors.append(f"analytic ceiling changed for N{key}")

    checkpoint_rows = payload.get("transition_checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
        errors.append("artifact must bind exactly three checkpoints")
    else:
        for row in checkpoint_rows:
            checkpoint = (
                row.get("checkpoint_index") if isinstance(row, dict) else None
            )
            if checkpoint not in CHECKPOINTS:
                errors.append("invalid checkpoint identity")
                continue
            sha, fingerprint = CHECKPOINTS[int(checkpoint)]
            if str(row.get("checkpoint_sha256", "")).lower() != sha:
                errors.append(f"checkpoint {checkpoint} SHA changed")
            if row.get("parameter_fingerprint") != fingerprint:
                errors.append(f"checkpoint {checkpoint} fingerprint changed")
            if row.get("learned_parameter_count") != PARAMETER_COUNT:
                errors.append(f"checkpoint {checkpoint} parameter count changed")

    tiers = payload.get("tiers")
    checkpoint_matrices: dict[int, list[np.ndarray]] = {}
    expected_cells: list[dict[str, Any]] = []
    expected_populations: list[dict[str, Any]] = []
    if not isinstance(tiers, list) or len(tiers) != len(POPULATIONS):
        errors.append("artifact must contain the complete population ladder")
        tiers = []
    for tier_index, population in enumerate(POPULATIONS):
        if tier_index >= len(tiers) or not isinstance(tiers[tier_index], dict):
            continue
        tier = tiers[tier_index]
        if (
            tier.get("population") != population
            or tier.get("tier_status") != "COMPLETE"
        ):
            errors.append(f"N{population} tier identity/status changed")
        if (
            tier.get("world_indices") != list(range(WORLD_COUNT))
            or tier.get("world_count") != WORLD_COUNT
        ):
            errors.append(f"N{population} world coverage changed")
        checkpoint_results = tier.get("checkpoint_results")
        if not isinstance(checkpoint_results, list) or len(checkpoint_results) != 3:
            errors.append(f"N{population} checkpoint matrix changed")
            continue
        public_expected = [
            exact_public_ranks(population, index) for index in range(WORLD_COUNT)
        ]
        runtime_expected = [row[0] for row in public_expected]
        bayes_expected_ranks = [row[1] for row in public_expected]
        hash_expected_ranks = [row[2] for row in public_expected]
        matrices: list[np.ndarray] = []
        for checkpoint_index, checkpoint_row in enumerate(checkpoint_results):
            if not isinstance(checkpoint_row, dict):
                errors.append(
                    f"N{population} C{checkpoint_index} is not an object"
                )
                continue
            if checkpoint_row.get("checkpoint_index") != checkpoint_index:
                errors.append(
                    f"N{population} C{checkpoint_index} identity changed"
                )
            if checkpoint_row.get("world_indices") != list(range(WORLD_COUNT)):
                errors.append(
                    f"N{population} C{checkpoint_index} world indices changed"
                )
            if checkpoint_row.get("runtime_seeds") != runtime_expected:
                errors.append(
                    f"N{population} C{checkpoint_index} runtime seeds changed"
                )
            if checkpoint_row.get("batch_count") != BATCH_COUNT:
                errors.append(
                    f"N{population} C{checkpoint_index} batch count changed"
                )
            ranks = checkpoint_row.get("ranks_by_ranker")
            if not isinstance(ranks, dict) or set(ranks) != set(RANKERS):
                errors.append(
                    f"N{population} C{checkpoint_index} ranker matrix changed"
                )
                continue
            if (
                ranks.get(BAYES) != bayes_expected_ranks
                or ranks.get(HASH) != hash_expected_ranks
            ):
                errors.append(
                    f"N{population} C{checkpoint_index} public ranks changed"
                )
            learned_ranks = ranks.get(LEARNED)
            if (
                not isinstance(learned_ranks, list)
                or len(learned_ranks) != WORLD_COUNT
                or any(
                    not isinstance(rank, int) or not 1 <= rank <= population
                    for rank in learned_ranks
                )
            ):
                errors.append(
                    f"N{population} C{checkpoint_index} learned ranks invalid"
                )
                continue
            matrix = difference_matrix(
                {ranker: list(ranks[ranker]) for ranker in RANKERS}
            )
            matrices.append(matrix)
            summary_rows = checkpoint_row.get("rank_summaries")
            if not isinstance(summary_rows, list) or len(summary_rows) != 3:
                errors.append(
                    f"N{population} C{checkpoint_index} rank summaries changed"
                )
            else:
                for ranker, observed_summary in zip(
                    RANKERS, summary_rows, strict=True
                ):
                    expected_summary = {
                        "checkpoint_index": checkpoint_index,
                        "population": population,
                        "ranker": ranker,
                        **expected_rank_summary(list(ranks[ranker])),
                    }
                    compare_payload(
                        observed_summary,
                        expected_summary,
                        f"N{population}.C{checkpoint_index}.{ranker}",
                        errors,
                    )
            cell_rows, cell_comparison = summary_triplet(
                "cell",
                population,
                checkpoint_index,
                matrix,
                seed_from_parts(
                    BOOTSTRAP_NAMESPACE,
                    "cell",
                    population,
                    checkpoint_index,
                ),
            )
            compare_payload(
                checkpoint_row.get("cell_bootstrap_summaries"),
                cell_rows,
                f"N{population}.C{checkpoint_index}.cell_bootstrap",
                errors,
            )
            compare_payload(
                checkpoint_row.get("cell_comparison"),
                cell_comparison,
                f"N{population}.C{checkpoint_index}.cell_comparison",
                errors,
            )
            expected_cells.append(cell_comparison)
            primary[f"{population}:C{checkpoint_index}"] = {
                ranker: sum(
                    int(rank <= PRIMARY_ATTEMPTS) for rank in ranks[ranker]
                )
                / WORLD_COUNT
                for ranker in RANKERS
            }
        if len(matrices) == 3:
            checkpoint_matrices[population] = matrices
            clustered = np.stack(matrices, axis=1).mean(axis=1)
            population_rows, population_comparison = summary_triplet(
                "population",
                population,
                None,
                clustered,
                seed_from_parts(
                    BOOTSTRAP_NAMESPACE, "population", population
                ),
            )
            compare_payload(
                tier.get("population_bootstrap_summaries"),
                population_rows,
                f"N{population}.population_bootstrap",
                errors,
            )
            compare_payload(
                tier.get("population_comparison"),
                population_comparison,
                f"N{population}.population_comparison",
                errors,
            )
            expected_populations.append(population_comparison)

    pooled_comparison: dict[str, Any] | None = None
    if tuple(checkpoint_matrices) == POPULATIONS:
        population_points = []
        population_estimates = []
        for population in POPULATIONS:
            clustered = np.stack(
                checkpoint_matrices[population], axis=1
            ).mean(axis=1)
            population_points.append(clustered.mean(axis=0))
            population_estimates.append(
                bootstrap_estimates(
                    clustered,
                    seed_from_parts(
                        BOOTSTRAP_NAMESPACE, "pooled", population
                    ),
                )
            )
        point = np.stack(population_points).mean(axis=0)
        estimates = np.stack(population_estimates).mean(axis=0)
        low, high = intervals(estimates)
        pooled_rows = [
            {
                "scope": "pooled",
                "population": None,
                "checkpoint_index": None,
                "comparison": comparison,
                "attempts": PRIMARY_ATTEMPTS,
                "coverage_delta": float(point[index]),
                "bootstrap_ci_low": float(low[index]),
                "bootstrap_ci_high": float(high[index]),
            }
            for index, comparison in enumerate(COMPARISONS)
        ]
        by_name = {row["comparison"]: row for row in pooled_rows}
        learned_bayes = by_name["learned_vs_bayes"]
        pooled_comparison = {
            "learned_minus_bayes_delta": learned_bayes["coverage_delta"],
            "learned_minus_bayes_ci_low": learned_bayes["bootstrap_ci_low"],
            "learned_minus_bayes_ci_high": learned_bayes[
                "bootstrap_ci_high"
            ],
            "learned_minus_hash_ci_low": by_name["learned_vs_hash"][
                "bootstrap_ci_low"
            ],
            "bayes_minus_hash_ci_low": by_name["bayes_vs_hash"][
                "bootstrap_ci_low"
            ],
        }
        compare_payload(
            payload.get("pooled_bootstrap_summaries"),
            pooled_rows,
            "pooled_bootstrap",
            errors,
        )
        compare_payload(
            payload.get("pooled_comparison"),
            pooled_comparison,
            "pooled_comparison",
            errors,
        )

    compare_payload(
        payload.get("cell_comparisons"),
        expected_cells,
        "cell_comparisons",
        errors,
    )
    compare_payload(
        payload.get("population_comparisons"),
        expected_populations,
        "population_comparisons",
        errors,
    )
    observed_outcome = payload.get("campaign_outcome")
    expected_outcome = None
    if (
        pooled_comparison is not None
        and len(expected_cells) == 12
        and len(expected_populations) == 4
    ):
        expected_outcome = classify(
            expected_cells, expected_populations, pooled_comparison
        )
        if observed_outcome != expected_outcome:
            errors.append("campaign outcome changed")
    if observed_outcome not in VALID_OUTCOMES:
        errors.append("campaign outcome is not frozen")

    return Gate7PrecisionAudit(
        artifact_valid=not errors,
        scientific_status=SCIENTIFIC_STATUS if not errors else "INVALID_ARTIFACT",
        campaign_outcome=(
            expected_outcome if not errors else observed_outcome
        ),
        primary_coverage_by_population_checkpoint=(
            primary if not errors else {}
        ),
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate7_precision_confirmation(args.result)
    args.output.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
