"""Run the admitted Gate-7 information-ceiling precision-confirmation campaign."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from .gate7_information_ceiling_decomposition_protocol import (
    expected_primary_ceiling_by_population,
)
from .gate7_information_ceiling_precision_confirmation import (
    GATE7_PRECISION_ATTEMPT_LADDER,
    GATE7_PRECISION_BASE_RESULT_HEAD,
    GATE7_PRECISION_BOOTSTRAP_SAMPLES,
    GATE7_PRECISION_CHECKPOINT_INDICES,
    GATE7_PRECISION_DECOMPOSITION_MANIFEST_SHA256,
    GATE7_PRECISION_DECOMPOSITION_OUTCOME,
    GATE7_PRECISION_DECOMPOSITION_RECOVERED_AUDIT_SHA256,
    GATE7_PRECISION_DECOMPOSITION_RECOVERY_RECORD_SHA256,
    GATE7_PRECISION_DECOMPOSITION_RESULT_SHA256,
    GATE7_PRECISION_EVALUATION_BATCH_SIZE,
    GATE7_PRECISION_EXECUTION_ADMITTED,
    GATE7_PRECISION_EXECUTION_SCIENTIFIC_STATUS,
    GATE7_PRECISION_HINT_RELIABILITY,
    GATE7_PRECISION_LEARNED_PARAMETER_COUNT,
    GATE7_PRECISION_NEAR_CEILING_MARGIN,
    GATE7_PRECISION_PHYSICAL_BATCH_COUNT,
    GATE7_PRECISION_POPULATIONS,
    GATE7_PRECISION_PRIMARY_ATTEMPTS,
    GATE7_PRECISION_PROTOCOL_HEAD,
    GATE7_PRECISION_RANKERS,
    GATE7_PRECISION_VERSION,
    GATE7_PRECISION_WORLD_COUNT,
    aggregate_gate7_precision_rank_batches,
    evaluate_gate7_precision_rank_batch,
    gate7_precision_cell_statistics,
    gate7_precision_pooled_statistics,
    gate7_precision_population_statistics,
    load_verified_gate7_precision_checkpoint,
    precision_world_batch,
    summarize_gate7_precision_ranks,
)
from .gate7_information_ceiling_precision_confirmation_protocol import (
    Gate7PrecisionCellComparison,
    Gate7PrecisionPopulationComparison,
    classify_gate7_precision_confirmation,
)
from .gate7_scale_neutral_transition_bridge import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return (
        args.transition_checkpoint0,
        args.transition_checkpoint1,
        args.transition_checkpoint2,
    )


def run_gate7_information_ceiling_precision_confirmation(
    *, output_root: Path, transition_checkpoint_paths: tuple[Path, Path, Path]
) -> int:
    if output_root.exists():
        raise FileExistsError(
            f"precision-confirmation output already exists: {output_root}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("admitted precision-confirmation execution requires CUDA")
    if not GATE7_PRECISION_EXECUTION_ADMITTED:
        raise RuntimeError("precision-confirmation execution is not admitted")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    result_path = output_root / "gate7-information-ceiling-precision-confirmation.json"
    runtime_path = output_root / "runtime.json"
    started = time.monotonic()

    checkpoint_identities = []
    for checkpoint_index, checkpoint_path in zip(
        GATE7_PRECISION_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        strict=True,
    ):
        model, identity = load_verified_gate7_precision_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=checkpoint_path,
            device="cpu",
        )
        checkpoint_identities.append(identity)
        del model

    result: dict[str, Any] = {
        "experiment_version": GATE7_PRECISION_VERSION,
        "scientific_status": GATE7_PRECISION_EXECUTION_SCIENTIFIC_STATUS,
        "execution_admitted": True,
        "execution_opened": True,
        "result_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "communication_intervention_performed": False,
        "recycling_intervention_performed": False,
        "specialization_intervention_performed": False,
        "topology_intervention_performed": False,
        "adaptive_attempt_exposure_performed": False,
        "prior_worlds_reused": False,
        "protocol_head": GATE7_PRECISION_PROTOCOL_HEAD,
        "base_result_head": GATE7_PRECISION_BASE_RESULT_HEAD,
        "base_decomposition_result_sha256": (
            GATE7_PRECISION_DECOMPOSITION_RESULT_SHA256
        ),
        "base_decomposition_recovered_audit_sha256": (
            GATE7_PRECISION_DECOMPOSITION_RECOVERED_AUDIT_SHA256
        ),
        "base_decomposition_recovery_record_sha256": (
            GATE7_PRECISION_DECOMPOSITION_RECOVERY_RECORD_SHA256
        ),
        "base_decomposition_manifest_sha256": (
            GATE7_PRECISION_DECOMPOSITION_MANIFEST_SHA256
        ),
        "base_decomposition_outcome": GATE7_PRECISION_DECOMPOSITION_OUTCOME,
        "hint_reliability": GATE7_PRECISION_HINT_RELIABILITY,
        "near_ceiling_margin": GATE7_PRECISION_NEAR_CEILING_MARGIN,
        "learned_parameter_count": GATE7_PRECISION_LEARNED_PARAMETER_COUNT,
        "populations": list(GATE7_PRECISION_POPULATIONS),
        "checkpoint_indices": list(GATE7_PRECISION_CHECKPOINT_INDICES),
        "rankers": list(GATE7_PRECISION_RANKERS),
        "attempt_ladder": list(GATE7_PRECISION_ATTEMPT_LADDER),
        "primary_attempts": GATE7_PRECISION_PRIMARY_ATTEMPTS,
        "world_count_per_checkpoint_population": GATE7_PRECISION_WORLD_COUNT,
        "evaluation_batch_size": GATE7_PRECISION_EVALUATION_BATCH_SIZE,
        "physical_batch_count": GATE7_PRECISION_PHYSICAL_BATCH_COUNT,
        "bootstrap_samples": GATE7_PRECISION_BOOTSTRAP_SAMPLES,
        "bootstrap_unit": (
            "world_index_clustered_within_population_across_T0_T1_T2"
        ),
        "pooled_weighting": "equal_population_then_equal_checkpoint",
        "bootstrap_generator": "splitmix64_counter_v0",
        "analytic_primary_ceiling_by_population": {
            str(population): value
            for population, value in expected_primary_ceiling_by_population().items()
            if population in GATE7_PRECISION_POPULATIONS
        },
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "transition_checkpoints": [
            identity.to_dict() for identity in checkpoint_identities
        ],
        "tiers": [],
        "cell_comparisons": [],
        "population_comparisons": [],
        "pooled_comparison": None,
        "campaign_outcome": "RUNNING",
    }
    _write_json(result_path, result)

    print(
        "Gate-7 information-ceiling precision confirmation — FRESH SCIENTIFIC EVIDENCE",
        flush=True,
    )
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Protocol head: {GATE7_PRECISION_PROTOCOL_HEAD}", flush=True)
    print(f"Populations: {GATE7_PRECISION_POPULATIONS}", flush=True)
    print(
        f"Worlds: {GATE7_PRECISION_WORLD_COUNT} per checkpoint/population",
        flush=True,
    )
    print(f"Rankers: {GATE7_PRECISION_RANKERS}", flush=True)
    print(
        "Training/checkpoint selection/communication intervention: NONE",
        flush=True,
    )
    print("Prior-world reuse: NONE", flush=True)

    checkpoints_by_population: dict[int, tuple[Any, ...]] = {}
    tiers: list[dict[str, Any]] = []
    for population in GATE7_PRECISION_POPULATIONS:
        print(f"\nN={population} precision rank matrix...", flush=True)
        checkpoint_objects = []
        tier: dict[str, Any] = {
            "population": population,
            "analytic_primary_ceiling": (
                expected_primary_ceiling_by_population()[population]
            ),
            "world_indices": list(range(GATE7_PRECISION_WORLD_COUNT)),
            "world_count": GATE7_PRECISION_WORLD_COUNT,
            "evaluation_batch_size": GATE7_PRECISION_EVALUATION_BATCH_SIZE,
            "checkpoint_results": [],
            "population_bootstrap_summaries": [],
            "population_comparison": None,
            "tier_status": "RUNNING",
        }
        public_reference_ranks: dict[str, tuple[int, ...]] | None = None
        for checkpoint_index, checkpoint_path in zip(
            GATE7_PRECISION_CHECKPOINT_INDICES,
            transition_checkpoint_paths,
            strict=True,
        ):
            model, identity = load_verified_gate7_precision_checkpoint(
                checkpoint_index=checkpoint_index,
                checkpoint_path=checkpoint_path,
                device="cuda",
            )
            batch_rows = []
            print(
                f"  C{checkpoint_index}: "
                f"{GATE7_PRECISION_PHYSICAL_BATCH_COUNT} B64 complete frontiers",
                flush=True,
            )
            for batch_number, batch_start in enumerate(
                range(
                    0,
                    GATE7_PRECISION_WORLD_COUNT,
                    GATE7_PRECISION_EVALUATION_BATCH_SIZE,
                ),
                start=1,
            ):
                worlds = precision_world_batch(
                    population=population, batch_start=batch_start
                )
                batch_rows.append(
                    evaluate_gate7_precision_rank_batch(
                        model,
                        checkpoint_index=checkpoint_index,
                        worlds=worlds,
                        device="cuda",
                    )
                )
                print(
                    f"    batch {batch_number}/"
                    f"{GATE7_PRECISION_PHYSICAL_BATCH_COUNT} complete "
                    f"({batch_start:04d}..{batch_start + 63:04d})",
                    flush=True,
                )
            checkpoint = aggregate_gate7_precision_rank_batches(
                tuple(batch_rows)
            )
            if checkpoint.parameter_fingerprint != identity.parameter_fingerprint:
                raise RuntimeError(
                    "precision rank result checkpoint fingerprint changed"
                )
            current_public = {
                ranker: checkpoint.ranks_by_ranker[ranker]
                for ranker in GATE7_PRECISION_RANKERS[1:]
            }
            if public_reference_ranks is None:
                public_reference_ranks = current_public
            elif current_public != public_reference_ranks:
                raise RuntimeError(
                    "precision Bayes/hash ranks changed across checkpoints"
                )

            rank_summaries = tuple(
                summarize_gate7_precision_ranks(
                    checkpoint=checkpoint, ranker=ranker
                )
                for ranker in GATE7_PRECISION_RANKERS
            )
            cell_bootstrap, cell_comparison = gate7_precision_cell_statistics(
                checkpoint
            )
            summary_by_ranker = {row.ranker: row for row in rank_summaries}
            tier["checkpoint_results"].append(
                {
                    **checkpoint.to_dict(),
                    "checkpoint_identity": identity.to_dict(),
                    "rank_summaries": [
                        row.to_dict() for row in rank_summaries
                    ],
                    "cell_bootstrap_summaries": [
                        row.to_dict() for row in cell_bootstrap
                    ],
                    "cell_comparison": cell_comparison.to_dict(),
                }
            )
            result["cell_comparisons"].append(cell_comparison.to_dict())
            checkpoint_objects.append(checkpoint)
            print(
                f"  C{checkpoint_index} M128 learned="
                f"{summary_by_ranker[GATE7_PRECISION_RANKERS[0]].coverage_by_attempt['128']:.4f} "
                f"bayes="
                f"{summary_by_ranker[GATE7_PRECISION_RANKERS[1]].coverage_by_attempt['128']:.4f} "
                f"hash="
                f"{summary_by_ranker[GATE7_PRECISION_RANKERS[2]].coverage_by_attempt['128']:.4f} "
                f"L-B CI=[{cell_comparison.learned_minus_bayes_ci_low:+.4f},"
                f"{cell_comparison.learned_minus_bayes_ci_high:+.4f}]",
                flush=True,
            )
            del model, batch_rows
            _release_cuda()
            result["tiers"] = tiers + [tier]
            _write_json(result_path, result)

        checkpoints = tuple(checkpoint_objects)
        checkpoints_by_population[population] = checkpoints
        population_bootstrap, population_comparison = (
            gate7_precision_population_statistics(checkpoints)
        )
        tier["population_bootstrap_summaries"] = [
            row.to_dict() for row in population_bootstrap
        ]
        tier["population_comparison"] = population_comparison.to_dict()
        tier["tier_status"] = "COMPLETE"
        result["population_comparisons"].append(
            population_comparison.to_dict()
        )
        tiers.append(tier)
        result["tiers"] = tiers
        _write_json(result_path, result)

    pooled_bootstrap, pooled_comparison = gate7_precision_pooled_statistics(
        checkpoints_by_population
    )
    cells = tuple(
        Gate7PrecisionCellComparison(**row)
        for row in result["cell_comparisons"]
    )
    populations = tuple(
        Gate7PrecisionPopulationComparison(**row)
        for row in result["population_comparisons"]
    )
    campaign_outcome = classify_gate7_precision_confirmation(
        cells=cells,
        populations=populations,
        pooled=pooled_comparison,
    )
    result["pooled_bootstrap_summaries"] = [
        row.to_dict() for row in pooled_bootstrap
    ]
    result["pooled_comparison"] = pooled_comparison.to_dict()
    result["campaign_outcome"] = campaign_outcome
    _write_json(result_path, result)

    runtime = {
        "scientific_status": GATE7_PRECISION_EXECUTION_SCIENTIFIC_STATUS,
        "campaign_outcome": campaign_outcome,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "communication_intervention_performed": False,
        "prior_worlds_reused": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "result_sha256": sha256_file(result_path),
        "transition_checkpoint_paths": [
            str(path.resolve()) for path in transition_checkpoint_paths
        ],
    }
    _write_json(runtime_path, runtime)
    print(
        json.dumps(
            {
                "status": (
                    "GATE7_INFORMATION_CEILING_PRECISION_CONFIRMATION_COMPLETE"
                ),
                "campaign_outcome": campaign_outcome,
                "result": str(result_path),
                "result_sha256": runtime["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-checkpoint0", type=Path, required=True)
    parser.add_argument("--transition-checkpoint1", type=Path, required=True)
    parser.add_argument("--transition-checkpoint2", type=Path, required=True)
    args = parser.parse_args()
    return run_gate7_information_ceiling_precision_confirmation(
        output_root=args.output_root,
        transition_checkpoint_paths=_checkpoint_paths(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
