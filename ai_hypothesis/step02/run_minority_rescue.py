"""Validation-only diagnostic for recoverable minority evidence in Step 2.

This experiment does not alter the v0 reducer and never opens the frozen test set.
It asks whether the strongest protected non-primary label can be accepted safely by
an inference-visible gate when population mean evidence chooses the wrong class.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ai_hypothesis.step01.model import LABEL_TO_INDEX
from ai_hypothesis.step01.torch_data import make_loader
from .evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from .population import HomogeneousWorkerBank
from .rescue import (
    MINORITY_RESCUE_FEATURE_NAMES,
    build_minority_candidates,
    fit_rescue_gate,
    rescue_threshold_metrics,
    score_rescue_gate,
    select_rescue_threshold,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether strong minority evidence is distinguishable from noisy "
            "minority evidence using validation data only."
        )
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("vmap", "loop"), default="vmap")
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic development/confirmation split seed.",
    )
    parser.add_argument("--aggregation-config", default=None)
    parser.add_argument("--max-harm-rate", type=float, default=0.001)
    parser.add_argument("--fit-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="results/step02/minority_rescue/validation_result.json",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _load_aggregation_config(path: str | None) -> AggregationConfig:
    if path is None:
        return AggregationConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = AggregationConfig(**payload)
    config.validate()
    return config


def _development_confirmation_masks(
    count: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reproducible random half-split masks over answerable validation rows.

    The benchmark cycles task/difficulty pairs deterministically by row index, so an
    even/odd split can systematically separate task/difficulty compositions. A fixed
    random permutation keeps the validation-only leakage boundary while avoiding that
    structural partition.
    """

    if count < 2:
        raise ValueError("at least two answerable rows are required for splitting")

    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    development_count = count // 2

    development_mask = torch.zeros(count, dtype=torch.bool)
    development_mask[order[:development_count]] = True
    confirmation_mask = ~development_mask
    return development_mask, confirmation_mask


def _candidate_summary(
    primary_correct: torch.Tensor,
    candidate_correct: torch.Tensor,
    candidate_exists: torch.Tensor,
) -> dict[str, float | int]:
    primary_ok = primary_correct.to(dtype=torch.bool)
    candidate_ok = candidate_correct.to(dtype=torch.bool)
    exists = candidate_exists.to(dtype=torch.bool)
    primary_errors = ~primary_ok
    opportunities = primary_errors & exists & candidate_ok
    total = int(primary_ok.numel())
    primary_error_count = int(primary_errors.sum().item())
    candidate_count = int(exists.sum().item())
    opportunity_count = int(opportunities.sum().item())
    return {
        "answerable_count": total,
        "primary_accuracy": float(primary_ok.to(torch.float32).mean().item()),
        "primary_error_count": primary_error_count,
        "candidate_count": candidate_count,
        "candidate_rate": candidate_count / total if total else 0.0,
        "recoverable_primary_error_count": opportunity_count,
        "recoverable_primary_error_rate": (
            opportunity_count / primary_error_count if primary_error_count else 0.0
        ),
    }


def _collect_validation_rows(
    bank: HomogeneousWorkerBank,
    loader: torch.utils.data.DataLoader,
    aggregation_config: AggregationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_chunks: list[torch.Tensor] = []
    candidate_exists_chunks: list[torch.Tensor] = []
    primary_correct_chunks: list[torch.Tensor] = []
    candidate_correct_chunks: list[torch.Tensor] = []

    for raw_batch in loader:
        samples = raw_batch["samples"]
        tasks = tuple(sample.task for sample in samples)
        output = bank(raw_batch["features"], raw_batch["mask"])
        evidence = build_evidence_matrix(output, tasks, aggregation_config)
        summary, decision = aggregate_evidence(evidence, aggregation_config)
        candidates = build_minority_candidates(evidence, summary, decision)

        truth_indices = torch.tensor(
            [LABEL_TO_INDEX.get(sample.label, -1) for sample in samples],
            device=candidates.features.device,
            dtype=torch.long,
        )
        answerable = truth_indices >= 0
        if not bool(answerable.any()):
            continue

        truths = truth_indices[answerable]
        primaries = candidates.primary_label_indices[answerable]
        candidate_indices = candidates.candidate_label_indices[answerable]
        feature_chunks.append(candidates.features[answerable].detach().cpu())
        candidate_exists_chunks.append(candidates.candidate_exists[answerable].detach().cpu())
        primary_correct_chunks.append((primaries == truths).detach().cpu())
        candidate_correct_chunks.append((candidate_indices == truths).detach().cpu())

    if not feature_chunks:
        raise ValueError("validation loader produced no answerable samples")

    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(candidate_exists_chunks, dim=0),
        torch.cat(primary_correct_chunks, dim=0),
        torch.cat(candidate_correct_chunks, dim=0),
    )


def main() -> None:
    args = _parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.fit_steps <= 0:
        raise ValueError("fit-steps must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= args.max_harm_rate <= 1.0:
        raise ValueError("max-harm-rate must be in [0, 1]")

    aggregation_config = _load_aggregation_config(args.aggregation_config)
    device = _resolve_device(args.device)
    bank = HomogeneousWorkerBank.from_checkpoints(
        args.checkpoints,
        device=device,
        execution_backend=args.backend,
    )
    loader = make_loader(
        split="validation",
        count=args.count,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=0,
    )

    features, candidate_exists, primary_correct, candidate_correct = _collect_validation_rows(
        bank,
        loader,
        aggregation_config,
    )

    development_mask, confirmation_mask = _development_confirmation_masks(
        features.shape[0],
        seed=args.seed,
    )
    improvement_targets = (~primary_correct & candidate_correct).to(torch.float32)

    gate = fit_rescue_gate(
        features[development_mask],
        improvement_targets[development_mask],
        candidate_exists[development_mask],
        steps=args.fit_steps,
        learning_rate=args.learning_rate,
    )
    development_scores = score_rescue_gate(
        features[development_mask],
        candidate_exists[development_mask],
        gate,
    )
    selection = select_rescue_threshold(
        development_scores,
        candidate_exists[development_mask],
        primary_correct[development_mask],
        candidate_correct[development_mask],
        max_harm_rate=args.max_harm_rate,
    )
    selected_threshold = float(selection["selected"]["threshold"])
    gate = replace(gate, threshold=selected_threshold)

    confirmation_scores = score_rescue_gate(
        features[confirmation_mask],
        candidate_exists[confirmation_mask],
        gate,
    )
    confirmation_threshold_metrics = rescue_threshold_metrics(
        confirmation_scores,
        candidate_exists[confirmation_mask],
        primary_correct[confirmation_mask],
        candidate_correct[confirmation_mask],
        threshold=gate.threshold,
    )

    result = {
        "experiment_version": "step02-minority-rescue-v0",
        "split": "validation",
        "test_set_accessed": False,
        "purpose": (
            "Determine whether inference-visible evidence can distinguish genuinely "
            "useful minority candidates from noisy alternatives."
        ),
        "device": device,
        "backend": args.backend,
        "count_requested": args.count,
        "answerable_count": int(features.shape[0]),
        "split_seed": args.seed,
        "development_rule": "seeded random half of answerable validation rows",
        "confirmation_rule": "complementary seeded random half of answerable validation rows",
        "aggregation_config": asdict(aggregation_config),
        "feature_names": list(MINORITY_RESCUE_FEATURE_NAMES),
        "gate": asdict(gate),
        "development": {
            "candidate_summary": _candidate_summary(
                primary_correct[development_mask],
                candidate_correct[development_mask],
                candidate_exists[development_mask],
            ),
            "threshold_selection": selection,
        },
        "confirmation": {
            "candidate_summary": _candidate_summary(
                primary_correct[confirmation_mask],
                candidate_correct[confirmation_mask],
                candidate_exists[confirmation_mask],
            ),
            "selected_threshold_metrics": confirmation_threshold_metrics,
        },
        "checkpoints": [asdict(checkpoint) for checkpoint in bank.checkpoints],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "minority_rescue_validation_complete",
                "population_width": bank.population_width,
                "selected_threshold": gate.threshold,
                "confirmation_accuracy_delta": confirmation_threshold_metrics[
                    "accuracy_delta"
                ],
                "confirmation_gain_count": confirmation_threshold_metrics["gain_count"],
                "confirmation_harm_count": confirmation_threshold_metrics["harm_count"],
                "result_path": str(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
