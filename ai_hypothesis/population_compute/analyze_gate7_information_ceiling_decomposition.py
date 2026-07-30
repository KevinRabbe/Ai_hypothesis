"""Independent standard-library auditor for Gate-7 information-ceiling artifacts."""

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

PROTOCOL_HEAD = "3640699f1727886c9ad2e954269fad660dc34370"
BASE_RESULT_HEAD = "4591dae55cada819e848ae7f929d5e8f2b8805d6"
BASE_RESULT_SHA256 = "4921ea99b44156f08271d6fb2b2e0bcba98ef6a646ed0aaf040762d47aa03b36"
BASE_AUDIT_SHA256 = "92f52a9e7fad3cb5d8962a9127a0cd7140656a0a8f03cfba08fe7cd5376a03fd"
BASE_MANIFEST_SHA256 = "ee9dcefbaf5efe9a75b20d407cb1a4f47ff0b04bbdce4613f2539b76af2c8cca"
SCIENTIFIC_STATUS = "FRESH_GATE7_INFORMATION_CEILING_DECOMPOSITION_EVIDENCE"
EXPERIMENT_VERSION = "gate7-information-ceiling-decomposition-v0"
POPULATIONS = (16_384, 32_768, 65_536, 131_072)
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
CHECKPOINT_INDICES = (0, 1, 2)
PARAMETER_COUNT = 19_649
WORLD_COUNT = 512
BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 10_000
HINT_RELIABILITY = 0.70
MARGIN = 0.02
ATTEMPTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024)
PRIMARY_ATTEMPTS = 128
LEARNED = "learned_score_rank"
BAYES = "bayes_hint_likelihood_rank"
HASH = "public_hash_rank"
RANKERS = (LEARNED, BAYES, HASH)
HIDDEN_NAMESPACE = "gate7-information-ceiling-decomposition-hidden-v0"
HINT_NAMESPACE = "gate7-information-ceiling-decomposition-hints-v0"
RUNTIME_NAMESPACE = "gate7-information-ceiling-decomposition-runtime-v0"
TIE_NAMESPACE = "gate7-information-ceiling-decomposition-public-tie-v0"
HASH_NAMESPACE = "public_hash_rank-v0"
BOOTSTRAP_NAMESPACE = "gate7-information-ceiling-decomposition-bootstrap-v0"
OUTCOME_DOMINANT = "G7_INFORMATION_CEILING_DOMINANT"
OUTCOME_GAP = "G7_SCORER_REPRESENTATION_GAP"
OUTCOME_MIXED = "G7_INFORMATION_AND_SCORER_GAP_MIXED"
OUTCOME_INCONCLUSIVE = "G7_INFORMATION_CEILING_INCONCLUSIVE"
VALID_OUTCOMES = {OUTCOME_DOMINANT, OUTCOME_GAP, OUTCOME_MIXED, OUTCOME_INCONCLUSIVE}


@dataclass(frozen=True, slots=True)
class Gate7InformationCeilingAudit:
    artifact_valid: bool
    scientific_status: str
    campaign_outcome: str | None
    primary_coverage_by_population_checkpoint: dict[str, dict[str, float]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def generate_world(population: int, world_index: int) -> dict[str, Any]:
    depth = population.bit_length() - 1
    task_depth = depth + 1
    hidden_rng = random.Random(seed_from_parts(HIDDEN_NAMESPACE, population, world_index, task_depth))
    hidden = tuple(hidden_rng.randrange(2) for _ in range(task_depth))
    hint_rng = random.Random(seed_from_parts(HINT_NAMESPACE, population, world_index, task_depth))
    hints = tuple(
        bit if hint_rng.random() < HINT_RELIABILITY else 1 - bit for bit in hidden
    )
    parent = 0
    for bit in hidden[:depth]:
        parent = parent * 2 + bit
    hint_prefix = 0
    for bit in hints[:depth]:
        hint_prefix = hint_prefix * 2 + bit

    def permutation(namespace: str) -> tuple[int, int]:
        rng = random.Random(seed_from_parts(namespace, population, world_index, depth))
        return rng.randrange(1, population, 2), rng.randrange(population)

    tie_multiplier, tie_offset = permutation(TIE_NAMESPACE)
    hash_multiplier, hash_offset = permutation(HASH_NAMESPACE)
    return {
        "runtime_seed": seed_from_parts(RUNTIME_NAMESPACE, population, world_index, depth),
        "parent": parent,
        "hint_prefix": hint_prefix,
        "tie_multiplier": tie_multiplier,
        "tie_offset": tie_offset,
        "hash_multiplier": hash_multiplier,
        "hash_offset": hash_offset,
    }


def affine_priority(candidate: int, multiplier: int, offset: int, population: int) -> int:
    return (candidate * multiplier + offset) & (population - 1)


def exact_public_ranks(population: int, world_index: int) -> tuple[int, int, int]:
    world = generate_world(population, world_index)
    parent = int(world["parent"])
    hint_prefix = int(world["hint_prefix"])
    hidden_distance = (parent ^ hint_prefix).bit_count()
    lower_shells = sum(math.comb(population.bit_length() - 1, distance) for distance in range(hidden_distance))
    hidden_tie = affine_priority(
        parent,
        int(world["tie_multiplier"]),
        int(world["tie_offset"]),
        population,
    )
    boundary_before = 0
    depth = population.bit_length() - 1
    for flipped_positions in itertools.combinations(range(depth), hidden_distance):
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
        probability = shell * (error**distance) * (HINT_RELIABILITY ** (depth - distance))
        admitted = min(remaining, shell)
        expected += probability * admitted / shell
        remaining -= admitted
        if remaining == 0:
            break
    return expected


def float_equal(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[int(math.floor(probability * (len(ordered) - 1)))]


def expected_rank_summary(ranks: list[int]) -> dict[str, Any]:
    return {
        "coverage_by_attempt": {
            str(attempt): sum(int(rank <= attempt) for rank in ranks) / len(ranks)
            for attempt in ATTEMPTS
        },
        "mean_rank": sum(ranks) / len(ranks),
        "rank_quantiles": {
            "p25": quantile(ranks, 0.25),
            "p50": quantile(ranks, 0.50),
            "p75": quantile(ranks, 0.75),
            "p90": quantile(ranks, 0.90),
            "p95": quantile(ranks, 0.95),
            "p99": quantile(ranks, 0.99),
        },
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_log2_rank_plus_one": sum(math.log2(rank + 1) for rank in ranks) / len(ranks),
        "rank_checksum": sum(ranks),
    }


def bootstrap_summary(
    *, population: int, checkpoint: int, comparison: str, left: list[int], right: list[int]
) -> tuple[float, float, float]:
    differences = tuple(
        int(left_rank <= PRIMARY_ATTEMPTS) - int(right_rank <= PRIMARY_ATTEMPTS)
        for left_rank, right_rank in zip(left, right, strict=True)
    )
    rng = random.Random(
        seed_from_parts(BOOTSTRAP_NAMESPACE, population, checkpoint, PRIMARY_ATTEMPTS, comparison)
    )
    estimates = [
        sum(differences[rng.randrange(WORLD_COUNT)] for _ in range(WORLD_COUNT)) / WORLD_COUNT
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    estimates.sort()
    low = estimates[int(math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1)))]
    high = estimates[int(math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1)))]
    return sum(differences) / WORLD_COUNT, low, high


def classify(rows: list[dict[str, Any]]) -> str:
    near = [
        float(row["learned_minus_bayes_ci_low"]) > -MARGIN
        and float(row["learned_minus_hash_ci_low"]) > 0.0
        and float(row["bayes_minus_hash_ci_low"]) > 0.0
        for row in rows
    ]
    gaps = [
        float(row["learned_minus_bayes_ci_high"]) < -MARGIN
        and float(row["bayes_minus_hash_ci_low"]) > 0.0
        for row in rows
    ]
    if all(near):
        return OUTCOME_DOMINANT
    if all(gaps):
        return OUTCOME_GAP
    if any(gaps):
        return OUTCOME_MIXED
    return OUTCOME_INCONCLUSIVE


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("information-ceiling artifact must be one JSON object")
    return payload


def audit_gate7_information_ceiling_decomposition(path: Path) -> Gate7InformationCeilingAudit:
    errors: list[str] = []
    primary: dict[str, dict[str, float]] = {}
    try:
        payload = load_payload(path)
    except Exception as exc:  # noqa: BLE001
        return Gate7InformationCeilingAudit(False, "INVALID_ARTIFACT", None, {}, (str(exc),))

    exact = {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "protocol_head": PROTOCOL_HEAD,
        "base_result_head": BASE_RESULT_HEAD,
        "base_result_sha256": BASE_RESULT_SHA256,
        "base_audit_sha256": BASE_AUDIT_SHA256,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "populations": list(POPULATIONS),
        "rankers": list(RANKERS),
        "attempt_ladder": list(ATTEMPTS),
        "primary_attempts": PRIMARY_ATTEMPTS,
        "world_count_per_checkpoint_population": WORLD_COUNT,
        "evaluation_batch_size": BATCH_SIZE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            errors.append(f"{key} mismatch")
    for key in (
        "execution_admitted",
        "execution_opened",
    ):
        if payload.get(key) is not True:
            errors.append(f"{key} must be true")
    for key in (
        "result_opened",
        "training_performed",
        "checkpoint_selection_performed",
        "communication_intervention_performed",
        "adaptive_attempt_exposure_performed",
        "continuation_worlds_reused",
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
    expected_analytic = {str(pop): bayes_expected(pop, PRIMARY_ATTEMPTS) for pop in POPULATIONS}
    observed_analytic = payload.get("analytic_primary_ceiling_by_population")
    if not isinstance(observed_analytic, dict) or any(
        key not in observed_analytic or not float_equal(observed_analytic[key], value, 1e-15)
        for key, value in expected_analytic.items()
    ):
        errors.append("analytic primary ceiling changed")

    checkpoint_rows = payload.get("transition_checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
        errors.append("artifact must bind exactly three checkpoints")
    else:
        for row in checkpoint_rows:
            if not isinstance(row, dict) or row.get("checkpoint_index") not in CHECKPOINTS:
                errors.append("invalid checkpoint identity")
                continue
            checkpoint = int(row["checkpoint_index"])
            sha, fingerprint = CHECKPOINTS[checkpoint]
            if str(row.get("checkpoint_sha256", "")).lower() != sha:
                errors.append(f"checkpoint {checkpoint} SHA mismatch")
            if row.get("parameter_fingerprint") != fingerprint:
                errors.append(f"checkpoint {checkpoint} fingerprint mismatch")
            if row.get("learned_parameter_count") != PARAMETER_COUNT:
                errors.append(f"checkpoint {checkpoint} parameter count mismatch")

    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or [tier.get("population") for tier in tiers if isinstance(tier, dict)] != list(POPULATIONS):
        errors.append("tiers must cover the exact population ladder")
        tiers = []

    classifier_rows: list[dict[str, Any]] = []
    for tier in tiers:
        population = int(tier.get("population", -1))
        if tier.get("world_indices") != list(range(WORLD_COUNT)):
            errors.append(f"N{population} world indices changed")
        if tier.get("world_count") != WORLD_COUNT or tier.get("evaluation_batch_size") != BATCH_SIZE:
            errors.append(f"N{population} world geometry changed")
        if not float_equal(tier.get("analytic_primary_ceiling"), bayes_expected(population, PRIMARY_ATTEMPTS), 1e-15):
            errors.append(f"N{population} analytic ceiling changed")
        checkpoint_results = tier.get("checkpoint_results")
        if not isinstance(checkpoint_results, list) or [row.get("checkpoint_index") for row in checkpoint_results if isinstance(row, dict)] != list(CHECKPOINT_INDICES):
            errors.append(f"N{population} checkpoint matrix changed")
            continue
        public_vectors: dict[str, list[int]] | None = None
        generated = [exact_public_ranks(population, world_index) for world_index in range(WORLD_COUNT)]
        expected_runtime = [row[0] for row in generated]
        expected_bayes = [row[1] for row in generated]
        expected_hash = [row[2] for row in generated]

        for checkpoint_result in checkpoint_results:
            checkpoint = int(checkpoint_result["checkpoint_index"])
            if checkpoint_result.get("world_indices") != list(range(WORLD_COUNT)):
                errors.append(f"N{population} C{checkpoint} world indices changed")
            if checkpoint_result.get("runtime_seeds") != expected_runtime:
                errors.append(f"N{population} C{checkpoint} runtime seeds changed")
            ranks_by_ranker = checkpoint_result.get("ranks_by_ranker")
            if not isinstance(ranks_by_ranker, dict) or tuple(ranks_by_ranker) != RANKERS:
                errors.append(f"N{population} C{checkpoint} ranker matrix changed")
                continue
            learned = ranks_by_ranker.get(LEARNED)
            bayes = ranks_by_ranker.get(BAYES)
            hash_ranks = ranks_by_ranker.get(HASH)
            if not all(isinstance(values, list) and len(values) == WORLD_COUNT for values in (learned, bayes, hash_ranks)):
                errors.append(f"N{population} C{checkpoint} rank vector length changed")
                continue
            if any(type(rank) is not int or not 1 <= rank <= population for values in (learned, bayes, hash_ranks) for rank in values):
                errors.append(f"N{population} C{checkpoint} rank bounds changed")
            if bayes != expected_bayes:
                errors.append(f"N{population} C{checkpoint} Bayes ranks mismatch")
            if hash_ranks != expected_hash:
                errors.append(f"N{population} C{checkpoint} hash ranks mismatch")
            current_public = {BAYES: bayes, HASH: hash_ranks}
            if public_vectors is None:
                public_vectors = current_public
            elif current_public != public_vectors:
                errors.append(f"N{population} public ranks changed across checkpoints")

            summaries = checkpoint_result.get("rank_summaries")
            if not isinstance(summaries, list) or [row.get("ranker") for row in summaries if isinstance(row, dict)] != list(RANKERS):
                errors.append(f"N{population} C{checkpoint} rank summaries changed")
            else:
                for summary, ranker in zip(summaries, RANKERS, strict=True):
                    expected_summary = expected_rank_summary(ranks_by_ranker[ranker])
                    for key in ("coverage_by_attempt", "rank_quantiles", "rank_checksum"):
                        if summary.get(key) != expected_summary[key]:
                            errors.append(f"N{population} C{checkpoint} {ranker} {key} mismatch")
                    for key in ("mean_rank", "mean_reciprocal_rank", "mean_log2_rank_plus_one"):
                        if not float_equal(summary.get(key), expected_summary[key]):
                            errors.append(f"N{population} C{checkpoint} {ranker} {key} mismatch")

            expected_pairs = []
            for comparison, left_name, right_name in (
                (f"c{checkpoint}_learned_vs_bayes_m128", LEARNED, BAYES),
                (f"c{checkpoint}_learned_vs_hash_m128", LEARNED, HASH),
                (f"c{checkpoint}_bayes_vs_hash_m128", BAYES, HASH),
            ):
                delta, low, high = bootstrap_summary(
                    population=population,
                    checkpoint=checkpoint,
                    comparison=comparison,
                    left=ranks_by_ranker[left_name],
                    right=ranks_by_ranker[right_name],
                )
                expected_pairs.append((comparison, left_name, right_name, delta, low, high))
            observed_pairs = checkpoint_result.get("paired_summaries")
            if not isinstance(observed_pairs, list) or len(observed_pairs) != 3:
                errors.append(f"N{population} C{checkpoint} paired summaries changed")
                continue
            pair_index = {row.get("comparison"): row for row in observed_pairs if isinstance(row, dict)}
            for comparison, left_name, right_name, delta, low, high in expected_pairs:
                row = pair_index.get(comparison)
                if not isinstance(row, dict):
                    errors.append(f"N{population} C{checkpoint} missing {comparison}")
                    continue
                if row.get("treatment_ranker") != left_name or row.get("reference_ranker") != right_name:
                    errors.append(f"N{population} C{checkpoint} {comparison} identity mismatch")
                for key, expected_value in (
                    ("coverage_delta", delta),
                    ("bootstrap_ci_low", low),
                    ("bootstrap_ci_high", high),
                ):
                    if not float_equal(row.get(key), expected_value):
                        errors.append(f"N{population} C{checkpoint} {comparison} {key} mismatch")
            l_b = pair_index.get(f"c{checkpoint}_learned_vs_bayes_m128", {})
            l_h = pair_index.get(f"c{checkpoint}_learned_vs_hash_m128", {})
            b_h = pair_index.get(f"c{checkpoint}_bayes_vs_hash_m128", {})
            classifier = {
                "checkpoint_index": checkpoint,
                "population": population,
                "learned_minus_bayes_ci_low": l_b.get("bootstrap_ci_low"),
                "learned_minus_bayes_ci_high": l_b.get("bootstrap_ci_high"),
                "learned_minus_hash_ci_low": l_h.get("bootstrap_ci_low"),
                "bayes_minus_hash_ci_low": b_h.get("bootstrap_ci_low"),
            }
            classifier_rows.append(classifier)
            observed_classifier = checkpoint_result.get("classifier_comparison")
            if not isinstance(observed_classifier, dict):
                errors.append(f"N{population} C{checkpoint} classifier row missing")
            else:
                for key, expected_value in classifier.items():
                    if not float_equal(observed_classifier.get(key), expected_value) if isinstance(expected_value, float) else observed_classifier.get(key) != expected_value:
                        errors.append(f"N{population} C{checkpoint} classifier {key} mismatch")
            primary[f"{population}:C{checkpoint}"] = {
                LEARNED: sum(int(rank <= PRIMARY_ATTEMPTS) for rank in learned) / WORLD_COUNT,
                BAYES: sum(int(rank <= PRIMARY_ATTEMPTS) for rank in bayes) / WORLD_COUNT,
                HASH: sum(int(rank <= PRIMARY_ATTEMPTS) for rank in hash_ranks) / WORLD_COUNT,
            }

    expected_order = [(population, checkpoint) for population in POPULATIONS for checkpoint in CHECKPOINT_INDICES]
    if [(row.get("population"), row.get("checkpoint_index")) for row in classifier_rows] != expected_order:
        errors.append("classifier rows are not population-major T0/T1/T2")
    observed_campaign = payload.get("campaign_outcome")
    expected_campaign = classify(classifier_rows) if len(classifier_rows) == 12 else None
    if observed_campaign not in VALID_OUTCOMES:
        errors.append("unknown campaign outcome")
    if expected_campaign is not None and observed_campaign != expected_campaign:
        errors.append("campaign outcome mismatch")

    return Gate7InformationCeilingAudit(
        artifact_valid=not errors,
        scientific_status=SCIENTIFIC_STATUS if not errors else "INVALID_ARTIFACT",
        campaign_outcome=observed_campaign if isinstance(observed_campaign, str) else None,
        primary_coverage_by_population_checkpoint=primary,
        errors=tuple(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_gate7_information_ceiling_decomposition(args.artifact)
    args.output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if audit.artifact_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
