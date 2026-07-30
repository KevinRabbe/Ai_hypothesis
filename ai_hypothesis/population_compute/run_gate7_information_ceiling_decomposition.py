"""Run the one admitted Gate-7 information-ceiling decomposition campaign."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from .gate7_information_ceiling_decomposition import (
    GATE7_INFORMATION_CEILING_ATTEMPT_LADDER,
    GATE7_INFORMATION_CEILING_BASE_RESULT_HEAD,
    GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES,
    GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
    GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
    GATE7_INFORMATION_CEILING_EXECUTION_ADMITTED,
    GATE7_INFORMATION_CEILING_HINT_RELIABILITY,
    GATE7_INFORMATION_CEILING_NEAR_CEILING_MARGIN,
    GATE7_INFORMATION_CEILING_POPULATIONS,
    GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS,
    GATE7_INFORMATION_CEILING_PROTOCOL_HEAD,
    GATE7_INFORMATION_CEILING_RANKERS,
    GATE7_INFORMATION_CEILING_SCIENTIFIC_STATUS,
    GATE7_INFORMATION_CEILING_WORLD_COUNT,
    aggregate_information_ceiling_rank_batches,
    classify_from_paired_summaries,
    comparison_for_frozen_classifier,
    evaluate_information_ceiling_rank_batch,
    expected_primary_ceiling_by_population,
    information_ceiling_world_batch,
    load_verified_gate7_information_ceiling_checkpoint,
    paired_information_ceiling_summary,
    summarize_information_ceiling_ranks,
)
from .gate7_information_ceiling_decomposition_protocol import (
    GATE7_INFORMATION_CEILING_AUDIT_SHA256,
    GATE7_INFORMATION_CEILING_MANIFEST_SHA256,
    GATE7_INFORMATION_CEILING_RESULT_SHA256,
)
from .gate7_scale_neutral_transition_bridge import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def run_gate7_information_ceiling_decomposition(
    *,
    output_root: Path,
    transition_checkpoint_paths: tuple[Path, Path, Path],
) -> int:
    if output_root.exists():
        raise FileExistsError(f"information-ceiling output already exists: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted information-ceiling execution requires CUDA")
    if not GATE7_INFORMATION_CEILING_EXECUTION_ADMITTED:
        raise RuntimeError("information-ceiling execution is not admitted")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True)
    result_path = output_root / "gate7-information-ceiling-decomposition.json"
    runtime_path = output_root / "runtime.json"
    started = time.monotonic()

    checkpoint_identities = []
    for checkpoint_index, checkpoint_path in zip(
        GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
        transition_checkpoint_paths,
        strict=True,
    ):
        model, identity = load_verified_gate7_information_ceiling_checkpoint(
            checkpoint_index=checkpoint_index,
            checkpoint_path=checkpoint_path,
            device="cpu",
        )
        checkpoint_identities.append(identity)
        del model

    result: dict[str, Any] = {
        "experiment_version": "gate7-information-ceiling-decomposition-v0",
        "scientific_status": GATE7_INFORMATION_CEILING_SCIENTIFIC_STATUS,
        "execution_admitted": True,
        "execution_opened": True,
        "result_opened": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "communication_intervention_performed": False,
        "adaptive_attempt_exposure_performed": False,
        "continuation_worlds_reused": False,
        "protocol_head": GATE7_INFORMATION_CEILING_PROTOCOL_HEAD,
        "base_result_head": GATE7_INFORMATION_CEILING_BASE_RESULT_HEAD,
        "base_result_sha256": GATE7_INFORMATION_CEILING_RESULT_SHA256,
        "base_audit_sha256": GATE7_INFORMATION_CEILING_AUDIT_SHA256,
        "base_manifest_sha256": GATE7_INFORMATION_CEILING_MANIFEST_SHA256,
        "hint_reliability": GATE7_INFORMATION_CEILING_HINT_RELIABILITY,
        "near_ceiling_margin": GATE7_INFORMATION_CEILING_NEAR_CEILING_MARGIN,
        "populations": list(GATE7_INFORMATION_CEILING_POPULATIONS),
        "rankers": list(GATE7_INFORMATION_CEILING_RANKERS),
        "attempt_ladder": list(GATE7_INFORMATION_CEILING_ATTEMPT_LADDER),
        "primary_attempts": GATE7_INFORMATION_CEILING_PRIMARY_ATTEMPTS,
        "world_count_per_checkpoint_population": GATE7_INFORMATION_CEILING_WORLD_COUNT,
        "evaluation_batch_size": GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
        "bootstrap_samples": GATE7_INFORMATION_CEILING_BOOTSTRAP_SAMPLES,
        "analytic_primary_ceiling_by_population": {
            str(population): value
            for population, value in expected_primary_ceiling_by_population().items()
        },
        "compiler_enabled": False,
        "cuda_graphs_enabled": False,
        "mixed_precision_enabled": False,
        "transition_checkpoints": [identity.to_dict() for identity in checkpoint_identities],
        "tiers": [],
        "campaign_outcome": "RUNNING",
    }
    _write_json(result_path, result)

    print("Gate-7 information-ceiling decomposition — FRESH SCIENTIFIC EVIDENCE", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Protocol head: {GATE7_INFORMATION_CEILING_PROTOCOL_HEAD}", flush=True)
    print(f"Populations: {GATE7_INFORMATION_CEILING_POPULATIONS}", flush=True)
    print(f"Rankers: {GATE7_INFORMATION_CEILING_RANKERS}", flush=True)
    print(f"Attempt curve: {GATE7_INFORMATION_CEILING_ATTEMPT_LADDER}", flush=True)
    print("Training/checkpoint selection/communication intervention: NONE", flush=True)
    print("Continuation-world reuse: NONE", flush=True)

    classifier_rows = []
    tiers: list[dict[str, Any]] = []
    for population in GATE7_INFORMATION_CEILING_POPULATIONS:
        print(f"\nN={population} information-ceiling rank matrix...", flush=True)
        tier: dict[str, Any] = {
            "population": population,
            "analytic_primary_ceiling": expected_primary_ceiling_by_population()[population],
            "world_indices": list(range(GATE7_INFORMATION_CEILING_WORLD_COUNT)),
            "world_count": GATE7_INFORMATION_CEILING_WORLD_COUNT,
            "evaluation_batch_size": GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
            "checkpoint_results": [],
            "tier_status": "RUNNING",
        }
        public_reference_ranks: dict[str, tuple[int, ...]] | None = None

        for checkpoint_index, checkpoint_path in zip(
            GATE7_INFORMATION_CEILING_CHECKPOINT_INDICES,
            transition_checkpoint_paths,
            strict=True,
        ):
            model, identity = load_verified_gate7_information_ceiling_checkpoint(
                checkpoint_index=checkpoint_index,
                checkpoint_path=checkpoint_path,
                device="cuda",
            )
            batch_rows = []
            print(f"  C{checkpoint_index}: eight B64 complete frontiers", flush=True)
            for batch_number, batch_start in enumerate(
                range(
                    0,
                    GATE7_INFORMATION_CEILING_WORLD_COUNT,
                    GATE7_INFORMATION_CEILING_EVALUATION_BATCH_SIZE,
                ),
                start=1,
            ):
                worlds = information_ceiling_world_batch(
                    population=population,
                    batch_start=batch_start,
                )
                row = evaluate_information_ceiling_rank_batch(
                    model,
                    checkpoint_index=checkpoint_index,
                    worlds=worlds,
                    device="cuda",
                )
                batch_rows.append(row)
                print(
                    f"    batch {batch_number}/8 complete "
                    f"({batch_start:03d}..{batch_start + 63:03d})",
                    flush=True,
                )
            checkpoint = aggregate_information_ceiling_rank_batches(tuple(batch_rows))
            if checkpoint.parameter_fingerprint != identity.parameter_fingerprint:
                raise RuntimeError("rank result checkpoint fingerprint changed")

            current_public = {
                ranker: checkpoint.ranks_by_ranker[ranker]
                for ranker in GATE7_INFORMATION_CEILING_RANKERS[1:]
            }
            if public_reference_ranks is None:
                public_reference_ranks = current_public
            elif current_public != public_reference_ranks:
                raise RuntimeError("Bayes/hash ranks changed across checkpoints")

            rank_summaries = [
                summarize_information_ceiling_ranks(checkpoint=checkpoint, ranker=ranker)
                for ranker in GATE7_INFORMATION_CEILING_RANKERS
            ]
            learned_vs_bayes = paired_information_ceiling_summary(
                comparison=f"c{checkpoint_index}_learned_vs_bayes_m128",
                checkpoint=checkpoint,
                treatment_ranker=GATE7_INFORMATION_CEILING_RANKERS[0],
                reference_ranker=GATE7_INFORMATION_CEILING_RANKERS[1],
            )
            learned_vs_hash = paired_information_ceiling_summary(
                comparison=f"c{checkpoint_index}_learned_vs_hash_m128",
                checkpoint=checkpoint,
                treatment_ranker=GATE7_INFORMATION_CEILING_RANKERS[0],
                reference_ranker=GATE7_INFORMATION_CEILING_RANKERS[2],
            )
            bayes_vs_hash = paired_information_ceiling_summary(
                comparison=f"c{checkpoint_index}_bayes_vs_hash_m128",
                checkpoint=checkpoint,
                treatment_ranker=GATE7_INFORMATION_CEILING_RANKERS[1],
                reference_ranker=GATE7_INFORMATION_CEILING_RANKERS[2],
            )
            classifier_row = comparison_for_frozen_classifier(
                learned_vs_bayes=learned_vs_bayes,
                learned_vs_hash=learned_vs_hash,
                bayes_vs_hash=bayes_vs_hash,
            )
            classifier_rows.append(classifier_row)
            summary_index = {summary.ranker: summary for summary in rank_summaries}
            checkpoint_payload = {
                **checkpoint.to_dict(),
                "checkpoint_identity": identity.to_dict(),
                "rank_summaries": [summary.to_dict() for summary in rank_summaries],
                "paired_summaries": [
                    learned_vs_bayes.to_dict(),
                    learned_vs_hash.to_dict(),
                    bayes_vs_hash.to_dict(),
                ],
                "classifier_comparison": {
                    "checkpoint_index": classifier_row.checkpoint_index,
                    "population": classifier_row.population,
                    "learned_minus_bayes_ci_low": classifier_row.learned_minus_bayes_ci_low,
                    "learned_minus_bayes_ci_high": classifier_row.learned_minus_bayes_ci_high,
                    "learned_minus_hash_ci_low": classifier_row.learned_minus_hash_ci_low,
                    "bayes_minus_hash_ci_low": classifier_row.bayes_minus_hash_ci_low,
                    "near_ceiling": classifier_row.near_ceiling(),
                    "clear_scorer_gap": classifier_row.clear_scorer_gap(),
                },
            }
            tier["checkpoint_results"].append(checkpoint_payload)
            print(
                f"  C{checkpoint_index} M128 learned="
                f"{summary_index[GATE7_INFORMATION_CEILING_RANKERS[0]].coverage_by_attempt['128']:.4f} "
                f"bayes={summary_index[GATE7_INFORMATION_CEILING_RANKERS[1]].coverage_by_attempt['128']:.4f} "
                f"hash={summary_index[GATE7_INFORMATION_CEILING_RANKERS[2]].coverage_by_attempt['128']:.4f} "
                f"L-B CI=[{learned_vs_bayes.bootstrap_ci_low:+.4f},"
                f"{learned_vs_bayes.bootstrap_ci_high:+.4f}]",
                flush=True,
            )
            del model, batch_rows, checkpoint
            _release_cuda()
            tier["tier_status"] = "CHECKPOINT_COMPLETE"
            result["tiers"] = tiers + [tier]
            _write_json(result_path, result)

        tier["tier_status"] = "COMPLETE"
        tiers.append(tier)
        result["tiers"] = tiers
        _write_json(result_path, result)

    campaign_outcome = classify_from_paired_summaries(tuple(classifier_rows))
    result["tiers"] = tiers
    result["campaign_outcome"] = campaign_outcome
    result["classifier_rows"] = [
        {
            "checkpoint_index": row.checkpoint_index,
            "population": row.population,
            "learned_minus_bayes_ci_low": row.learned_minus_bayes_ci_low,
            "learned_minus_bayes_ci_high": row.learned_minus_bayes_ci_high,
            "learned_minus_hash_ci_low": row.learned_minus_hash_ci_low,
            "bayes_minus_hash_ci_low": row.bayes_minus_hash_ci_low,
            "near_ceiling": row.near_ceiling(),
            "clear_scorer_gap": row.clear_scorer_gap(),
        }
        for row in classifier_rows
    ]
    _write_json(result_path, result)

    runtime = {
        "scientific_status": GATE7_INFORMATION_CEILING_SCIENTIFIC_STATUS,
        "campaign_outcome": campaign_outcome,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "communication_intervention_performed": False,
        "continuation_worlds_reused": False,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - started,
        "result_sha256": sha256_file(result_path),
        "transition_checkpoint_paths": [str(path.resolve()) for path in transition_checkpoint_paths],
    }
    _write_json(runtime_path, runtime)
    print(
        json.dumps(
            {
                "status": "GATE7_INFORMATION_CEILING_DECOMPOSITION_COMPLETE",
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
    return run_gate7_information_ceiling_decomposition(
        output_root=args.output_root,
        transition_checkpoint_paths=_checkpoint_paths(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
