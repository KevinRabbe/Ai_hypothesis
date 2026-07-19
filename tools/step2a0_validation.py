"""Run the Step 2A.0 validation-only population readiness experiment."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from ai_hypothesis.step01.model import (
    LABEL_TO_INDEX,
    NON_UNCERTAIN_LABELS,
    Step01Output,
    decode_predictions,
)
from ai_hypothesis.step01.torch_data import make_loader
from ai_hypothesis.step02.evidence import (
    AggregationConfig,
    aggregate_evidence,
    build_evidence_matrix,
)
from ai_hypothesis.step02.evaluation import (
    _majority_prediction_indices,
    _mean_probability_predictions,
    _worker_prediction_indices,
)
from ai_hypothesis.step02.population import HomogeneousWorkerBank


LABELS_WITH_UNCERTAIN = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")
WIDTHS = (1, 2, 4, 5)


def checkpoint_paths() -> list[Path]:
    root = Path("results/step01/checkpoint_50k_extended_15k")
    return [root / f"seed_{seed}" / "best.pt" for seed in range(1, 6)]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def worker_predictions(output, threshold: float = 0.5) -> torch.Tensor:
    return _worker_prediction_indices(output, uncertainty_threshold=threshold)


def evidence_predictions(output, samples, config: AggregationConfig) -> tuple[str, ...]:
    tasks = tuple(sample.task for sample in samples)
    evidence = build_evidence_matrix(output, tasks, config)
    _, decision = aggregate_evidence(evidence, config)
    return decision.predictions


def run_equivalence(
    *,
    paths: list[Path],
    device: str,
    output_dir: Path,
    count: int,
    batch_size: int,
    seed: int,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    rows = []
    for width in WIDTHS:
        selected = paths[:width]
        loader = make_loader(
            split="validation",
            count=count,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=0,
        )
        loop = HomogeneousWorkerBank.from_checkpoints(
            selected, device=device, execution_backend="loop"
        )
        vmap = HomogeneousWorkerBank.from_checkpoints(
            selected, device=device, execution_backend="vmap"
        )

        label_max = 0.0
        label_mean_sum = 0.0
        label_count = 0
        unc_max = 0.0
        unc_mean_sum = 0.0
        unc_count = 0
        worker_agree = 0
        worker_total = 0
        evidence_agree = 0
        evidence_total = 0
        label_allclose = True
        uncertainty_allclose = True
        captured_warnings: list[str] = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for batch in loader:
                with torch.inference_mode():
                    loop_output = loop(batch["features"], batch["mask"])
                    vmap_output = vmap(batch["features"], batch["mask"])

                label_diff = (loop_output.label_logits - vmap_output.label_logits).abs()
                unc_diff = (
                    loop_output.uncertainty_logits
                    - vmap_output.uncertainty_logits
                ).abs()
                label_max = max(label_max, float(label_diff.max().cpu()))
                unc_max = max(unc_max, float(unc_diff.max().cpu()))
                label_mean_sum += float(label_diff.sum().cpu())
                unc_mean_sum += float(unc_diff.sum().cpu())
                label_count += label_diff.numel()
                unc_count += unc_diff.numel()
                label_allclose = label_allclose and bool(
                    torch.allclose(
                        loop_output.label_logits,
                        vmap_output.label_logits,
                        atol=atol,
                        rtol=rtol,
                    )
                )
                uncertainty_allclose = uncertainty_allclose and bool(
                    torch.allclose(
                        loop_output.uncertainty_logits,
                        vmap_output.uncertainty_logits,
                        atol=atol,
                        rtol=rtol,
                    )
                )
                loop_worker = worker_predictions(loop_output)
                vmap_worker = worker_predictions(vmap_output)
                worker_agree += int((loop_worker == vmap_worker).sum().cpu())
                worker_total += loop_worker.numel()
                loop_evidence = evidence_predictions(
                    loop_output, batch["samples"], AggregationConfig()
                )
                vmap_evidence = evidence_predictions(
                    vmap_output, batch["samples"], AggregationConfig()
                )
                evidence_agree += sum(
                    int(a == b) for a, b in zip(loop_evidence, vmap_evidence)
                )
                evidence_total += len(loop_evidence)
            captured_warnings = sorted({str(item.message) for item in caught})

        rows.append(
            {
                "width": width,
                "split": "validation",
                "count": count,
                "batch_size": batch_size,
                "device": device,
                "atol": atol,
                "rtol": rtol,
                "label_max_abs_diff": label_max,
                "label_mean_abs_diff": label_mean_sum / label_count,
                "uncertainty_max_abs_diff": unc_max,
                "uncertainty_mean_abs_diff": unc_mean_sum / unc_count,
                "label_allclose": label_allclose,
                "uncertainty_allclose": uncertainty_allclose,
                "worker_prediction_agreement_rate": worker_agree / worker_total,
                "evidence_prediction_agreement_rate": evidence_agree / evidence_total,
                "warnings": captured_warnings,
            }
        )
    json_dump(output_dir / "phase_a_equivalence.json", rows)
    return rows


def run_performance(
    *,
    paths: list[Path],
    device: str,
    output_dir: Path,
    count: int,
    batch_size: int,
    seed: int,
    warmup_batches: int,
) -> list[dict[str, Any]]:
    rows = []
    for backend in ("loop", "vmap"):
        bank = HomogeneousWorkerBank.from_checkpoints(
            paths[:5], device=device, execution_backend=backend
        )
        loader = make_loader(
            split="validation",
            count=count,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=0,
        )
        warmup_loader = make_loader(
            split="validation",
            count=batch_size * warmup_batches,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for batch in warmup_loader:
                _ = bank(batch["features"], batch["mask"])
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            started = time.perf_counter()
            sample_count = 0
            for batch in loader:
                _ = bank(batch["features"], batch["mask"])
                sample_count += len(batch["samples"])
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
            else:
                peak = None
            elapsed = time.perf_counter() - started
            captured_warnings = sorted({str(item.message) for item in caught})

        rows.append(
            {
                "backend": backend,
                "device": device,
                "width": 5,
                "split": "validation",
                "count": sample_count,
                "batch_size": batch_size,
                "warmup_batches": warmup_batches,
                "total_wall_time_seconds": elapsed,
                "samples_per_second": sample_count / elapsed,
                "worker_evaluations_per_second": sample_count * 5 / elapsed,
                "peak_gpu_memory_bytes": peak,
                "warnings": captured_warnings,
            }
        )
    json_dump(output_dir / "phase_b_backend_performance.json", rows)
    return rows


def evaluate_detailed(
    bank: HomogeneousWorkerBank,
    *,
    count: int,
    batch_size: int,
    seed: int,
    config: AggregationConfig,
) -> dict[str, Any]:
    loader = make_loader(
        split="validation",
        count=count,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=0,
    )
    total = 0
    evidence_correct = 0
    majority_correct = 0
    mean_logit_correct = 0
    mean_probability_correct = 0
    oracle_any_correct = 0
    all_wrong = 0
    utilization_failures = 0
    majority_failures = 0
    reducer_rescues = 0
    minority_rescue_opportunities = 0
    minority_rescues = 0
    strong_correct_minority_cases = 0
    minority_suppressions = 0
    majority_harm_cases = 0
    disagreement_entropy_sum = 0.0
    mean_uncertainty_sum = 0.0
    mean_invalid_mass_sum = 0.0
    worker_correct = [0 for _ in range(bank.population_width)]
    by_task: dict[str, dict[str, int]] = {}

    with torch.inference_mode():
        for batch in loader:
            output = bank(batch["features"], batch["mask"])
            tasks = tuple(sample.task for sample in batch["samples"])
            evidence = build_evidence_matrix(output, tasks, config)
            summary, decision = aggregate_evidence(evidence, config)
            worker_idx = worker_predictions(output)
            majority_idx = _majority_prediction_indices(worker_idx)
            majority_predictions = tuple(
                LABELS_WITH_UNCERTAIN[int(index)] for index in majority_idx
            )
            mean_logit_output = Step01Output(
                label_logits=output.label_logits.mean(dim=0),
                uncertainty_logits=output.uncertainty_logits.mean(dim=0),
            )
            mean_logit_predictions = tuple(decode_predictions(mean_logit_output))
            mean_probability_predictions = _mean_probability_predictions(output, uncertainty_threshold=0.5)

            for sample_index, sample in enumerate(batch["samples"]):
                total += 1
                truth = sample.label
                evidence_prediction = decision.predictions[sample_index]
                majority_prediction = majority_predictions[sample_index]
                worker_predictions_for_sample = [
                    LABELS_WITH_UNCERTAIN[int(worker_idx[i, sample_index])]
                    for i in range(bank.population_width)
                ]
                worker_correct_flags = [
                    prediction == truth for prediction in worker_predictions_for_sample
                ]
                for worker_index, is_correct in enumerate(worker_correct_flags):
                    worker_correct[worker_index] += int(is_correct)
                any_correct = any(worker_correct_flags)
                evidence_is_correct = evidence_prediction == truth
                majority_is_correct = majority_prediction == truth
                evidence_correct += int(evidence_is_correct)
                majority_correct += int(majority_is_correct)
                mean_logit_correct += int(mean_logit_predictions[sample_index] == truth)
                mean_probability_correct += int(
                    mean_probability_predictions[sample_index] == truth
                )
                oracle_any_correct += int(any_correct)
                all_wrong += int(not any_correct)
                utilization_failures += int(any_correct and not evidence_is_correct)
                majority_failures += int((not majority_is_correct) and any_correct)
                reducer_rescues += int(
                    (not majority_is_correct) and any_correct and evidence_is_correct
                )
                rescue_opportunity = (not majority_is_correct) and any_correct
                minority_rescue_opportunities += int(rescue_opportunity)
                minority_rescues += int(rescue_opportunity and evidence_is_correct)

                strong_correct_minority = False
                if truth != "UNCERTAIN" and truth in LABEL_TO_INDEX:
                    truth_index = LABEL_TO_INDEX[truth]
                    strong_correct_evidence = bool(
                        (
                            evidence.evidence_scores[:, sample_index, truth_index]
                            >= config.strong_evidence_threshold
                        ).any()
                    )
                    strong_correct_minority = (
                        strong_correct_evidence and majority_prediction != truth
                    )
                strong_correct_minority_cases += int(strong_correct_minority)
                minority_suppressions += int(
                    strong_correct_minority and not evidence_is_correct
                )
                majority_harm_cases += int(
                    majority_is_correct and not evidence_is_correct
                )

                stats = by_task.setdefault(
                    sample.task.value,
                    {
                        "count": 0,
                        "evidence_correct": 0,
                        "majority_correct": 0,
                        "mean_logit_correct": 0,
                        "mean_probability_correct": 0,
                        "oracle_any_correct": 0,
                    },
                )
                stats["count"] += 1
                stats["evidence_correct"] += int(evidence_is_correct)
                stats["majority_correct"] += int(majority_is_correct)
                stats["mean_logit_correct"] += int(
                    mean_logit_predictions[sample_index] == truth
                )
                stats["mean_probability_correct"] += int(
                    mean_probability_predictions[sample_index] == truth
                )
                stats["oracle_any_correct"] += int(any_correct)

            disagreement_entropy_sum += float(summary.disagreement_entropy.sum().cpu())
            mean_uncertainty_sum += float(summary.mean_uncertainty.sum().cpu())
            mean_invalid_mass_sum += float(summary.mean_invalid_label_mass.sum().cpu())

    worker_accuracies = [correct / total for correct in worker_correct]
    minority_rescue_rate = (
        minority_rescues / minority_rescue_opportunities
        if minority_rescue_opportunities
        else 0.0
    )
    minority_suppression_rate = (
        minority_suppressions / strong_correct_minority_cases
        if strong_correct_minority_cases
        else 0.0
    )
    return {
        "count": total,
        "population_width": bank.population_width,
        "execution_backend": bank.execution_backend,
        "unit_config": asdict(bank.unit_config),
        "aggregation_config": asdict(config),
        "evidence_reducer_accuracy": evidence_correct / total,
        "majority_vote_accuracy": majority_correct / total,
        "mean_logit_accuracy": mean_logit_correct / total,
        "mean_probability_accuracy": mean_probability_correct / total,
        "oracle_any_correct_coverage": oracle_any_correct / total,
        "all_wrong_rate": all_wrong / total,
        "discovery_failure_rate": all_wrong / total,
        "utilization_failure_rate": utilization_failures / total,
        "majority_failure_with_oracle_rate": majority_failures / total,
        "reducer_rescue_rate_total": reducer_rescues / total,
        "minority_rescue_opportunity_rate": minority_rescue_opportunities / total,
        "minority_rescue_rate": minority_rescue_rate,
        "minority_suppression_rate": minority_suppression_rate,
        "strong_correct_minority_case_rate": strong_correct_minority_cases / total,
        "majority_harm_rate": majority_harm_cases / total,
        "evidence_utilization_gap": (oracle_any_correct / total)
        - (evidence_correct / total),
        "mean_disagreement_entropy": disagreement_entropy_sum / total,
        "mean_population_uncertainty": mean_uncertainty_sum / total,
        "mean_invalid_label_mass": mean_invalid_mass_sum / total,
        "single_worker_accuracy": {
            "values": worker_accuracies,
            "min": min(worker_accuracies),
            "max": max(worker_accuracies),
            "mean": sum(worker_accuracies) / len(worker_accuracies),
        },
        "by_task": {
            task: {
                "count": stats["count"],
                "evidence_accuracy": stats["evidence_correct"] / stats["count"],
                "majority_vote_accuracy": stats["majority_correct"] / stats["count"],
                "mean_logit_accuracy": stats["mean_logit_correct"] / stats["count"],
                "mean_probability_accuracy": (
                    stats["mean_probability_correct"] / stats["count"]
                ),
                "oracle_any_correct_coverage": (
                    stats["oracle_any_correct"] / stats["count"]
                ),
            }
            for task, stats in sorted(by_task.items())
        },
    }


def run_validation(
    *,
    paths: list[Path],
    device: str,
    backend: str,
    output_dir: Path,
    count: int,
    batch_size: int,
    seed: int,
    config: AggregationConfig,
    tag: str,
) -> list[dict[str, Any]]:
    rows = []
    for width in WIDTHS:
        bank = HomogeneousWorkerBank.from_checkpoints(
            paths[:width], device=device, execution_backend=backend
        )
        metrics = evaluate_detailed(
            bank,
            count=count,
            batch_size=batch_size,
            seed=seed,
            config=config,
        )
        result = {
            "runtime_version": "step02-population-runtime-v0",
            "evidence_contract_version": "step02-evidence-v0",
            "experiment_phase": "step2a0_validation_only",
            "calibration_tag": tag,
            "split": "validation",
            "device": device,
            "backend": backend,
            "count": count,
            "batch_size": batch_size,
            "seed": seed,
            "checkpoints": [str(path) for path in paths[:width]],
            "metrics": metrics,
        }
        json_dump(output_dir / tag / f"width_{width}" / "result.json", result)
        rows.append(metrics)
    json_dump(output_dir / f"{tag}_summary.json", rows)
    return rows


def calibration_configs() -> list[tuple[str, AggregationConfig]]:
    base = AggregationConfig()
    return [
        ("baseline_defaults", base),
        ("no_margin_gate", replace(base, min_primary_margin=0.0)),
        ("low_margin_gate", replace(base, min_primary_margin=0.05)),
        ("medium_margin_gate", replace(base, min_primary_margin=0.10)),
        ("no_protected_conflict_gap", replace(base, protected_conflict_mean_gap=0.0)),
        (
            "weaker_strong_evidence",
            replace(base, strong_evidence_threshold=1.0),
        ),
        (
            "stronger_strong_evidence",
            replace(base, strong_evidence_threshold=3.0),
        ),
        (
            "low_margin_no_conflict_gap",
            replace(base, min_primary_margin=0.05, protected_conflict_mean_gap=0.0),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/step02/step2a0_50k_validation")
    parser.add_argument("--equivalence-count", type=int, default=512)
    parser.add_argument("--performance-count", type=int, default=4096)
    parser.add_argument("--validation-count", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    paths = checkpoint_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "phase": "step2a0_validation_only",
        "split": "validation",
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "checkpoints": [str(path) for path in paths],
        "widths": WIDTHS,
        "aggregation_parameters": {
            field: {
                "default": value,
                "semantics": {
                    "eps": "probability/log numerical floor",
                    "evidence_clip": "absolute clip for centered log-support evidence",
                    "top_k": "per-label strongest evidence entries retained",
                    "support_threshold": "evidence threshold for support-count diagnostics",
                    "strong_evidence_threshold": "threshold for protected minority candidate evidence",
                    "min_primary_margin": "minimum mean-evidence gap before choosing primary label",
                    "max_mean_uncertainty": "maximum population mean uncertainty before abstention",
                    "max_mean_invalid_mass": "maximum mean task-invalid probability mass before abstention",
                    "protected_conflict_mean_gap": "mean-evidence gap below which protected competitor causes UNCERTAIN",
                }[field],
            }
            for field, value in asdict(AggregationConfig()).items()
        },
    }
    json_dump(output_dir / "manifest.json", meta)

    equivalence = run_equivalence(
        paths=paths,
        device=args.device,
        output_dir=output_dir,
        count=args.equivalence_count,
        batch_size=args.batch_size,
        seed=args.seed,
        atol=1e-5,
        rtol=1e-5,
    )
    if not all(row["label_allclose"] and row["uncertainty_allclose"] for row in equivalence):
        raise RuntimeError("loop/vmap equivalence failed; stopping before validation")

    performance = run_performance(
        paths=paths,
        device=args.device,
        output_dir=output_dir,
        count=args.performance_count,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_batches=2,
    )
    loop_perf = next(row for row in performance if row["backend"] == "loop")
    vmap_perf = next(row for row in performance if row["backend"] == "vmap")
    if vmap_perf["samples_per_second"] > loop_perf["samples_per_second"] * 1.05:
        chosen_backend = "vmap"
        reason = "vmap was at least 5% faster in the CUDA performance check"
    else:
        chosen_backend = "loop"
        reason = "vmap was not materially faster; using simpler reference backend"
    json_dump(
        output_dir / "phase_b_backend_choice.json",
        {"chosen_backend": chosen_backend, "reason": reason},
    )

    baseline = run_validation(
        paths=paths,
        device=args.device,
        backend=chosen_backend,
        output_dir=output_dir,
        count=args.validation_count,
        batch_size=args.batch_size,
        seed=args.seed,
        config=AggregationConfig(),
        tag="baseline_uncalibrated",
    )

    calibration_rows = []
    for tag, config in calibration_configs():
        rows = run_validation(
            paths=paths,
            device=args.device,
            backend=chosen_backend,
            output_dir=output_dir,
            count=args.validation_count,
            batch_size=args.batch_size,
            seed=args.seed,
            config=config,
            tag=f"calibration_{tag}",
        )
        width5 = next(row for row in rows if row["population_width"] == 5)
        calibration_rows.append(
            {
                "tag": tag,
                "config": asdict(config),
                "width5_evidence_reducer_accuracy": width5[
                    "evidence_reducer_accuracy"
                ],
                "width5_evidence_utilization_gap": width5[
                    "evidence_utilization_gap"
                ],
                "width5_minority_rescue_rate": width5["minority_rescue_rate"],
                "width5_minority_suppression_rate": width5[
                    "minority_suppression_rate"
                ],
                "width5_majority_harm_rate": width5["majority_harm_rate"],
            }
        )
    calibration_rows.sort(
        key=lambda row: (
            row["width5_evidence_reducer_accuracy"],
            -row["width5_minority_suppression_rate"],
            -row["width5_evidence_utilization_gap"],
        ),
        reverse=True,
    )
    candidate = calibration_rows[0]
    json_dump(output_dir / "phase_g_calibration_summary.json", calibration_rows)
    json_dump(
        output_dir / "phase_h_candidate_frozen_from_validation.json",
        {
            "status": "CANDIDATE_FROZEN_FROM_VALIDATION",
            "selected_tag": candidate["tag"],
            "selection_basis": (
                "highest width-5 validation evidence reducer accuracy, with "
                "minority suppression and utilization gap retained as diagnostics"
            ),
            "config": candidate["config"],
        },
    )
    json_dump(
        output_dir / "final_summary.json",
        {
            "equivalence": equivalence,
            "performance": performance,
            "backend_choice": {"chosen_backend": chosen_backend, "reason": reason},
            "baseline": baseline,
            "calibration_summary": calibration_rows,
            "candidate": candidate,
        },
    )
    print(json.dumps({"event": "step2a0_complete", "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
