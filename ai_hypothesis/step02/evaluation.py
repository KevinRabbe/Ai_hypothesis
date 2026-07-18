"""Streaming evaluation for Step 2 population experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ai_hypothesis.step01.model import (
    LABEL_TO_INDEX,
    NON_UNCERTAIN_LABELS,
    Step01Output,
    decode_predictions,
)
from .evidence import AggregationConfig, aggregate_evidence, build_evidence_matrix
from .population import HomogeneousWorkerBank, PopulationOutput


_ALL_DECODED_LABELS: tuple[str, ...] = (*NON_UNCERTAIN_LABELS, "UNCERTAIN")


def _worker_prediction_indices(
    output: PopulationOutput,
    *,
    uncertainty_threshold: float,
) -> torch.Tensor:
    uncertain = torch.sigmoid(output.uncertainty_logits) >= uncertainty_threshold
    label_indices = output.label_logits.argmax(dim=-1)
    uncertain_index = len(NON_UNCERTAIN_LABELS)
    return torch.where(
        uncertain,
        torch.full_like(label_indices, uncertain_index),
        label_indices,
    )


def _majority_prediction_indices(worker_predictions: torch.Tensor) -> torch.Tensor:
    counts = F.one_hot(
        worker_predictions,
        num_classes=len(_ALL_DECODED_LABELS),
    ).sum(dim=0)
    return counts.argmax(dim=-1)


def _mean_probability_predictions(
    output: PopulationOutput,
    *,
    uncertainty_threshold: float,
) -> tuple[str, ...]:
    mean_probabilities = torch.softmax(output.label_logits, dim=-1).mean(dim=0)
    mean_uncertainty = torch.sigmoid(output.uncertainty_logits).mean(dim=0)
    labels = mean_probabilities.argmax(dim=-1)
    return tuple(
        "UNCERTAIN"
        if bool(mean_uncertainty[index] >= uncertainty_threshold)
        else NON_UNCERTAIN_LABELS[int(labels[index])]
        for index in range(labels.shape[0])
    )


def evaluate_population(
    bank: HomogeneousWorkerBank,
    loader: torch.utils.data.DataLoader,
    *,
    aggregation_config: AggregationConfig = AggregationConfig(),
    uncertainty_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one fixed homogeneous population over a benchmark loader.

    The function streams batches and keeps only aggregate diagnostics, avoiding a
    full dataset x worker x label evidence tensor in host memory.
    """

    if not 0.0 <= uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold must be in [0, 1]")
    aggregation_config.validate()

    total = 0
    evidence_correct = 0
    majority_correct = 0
    mean_logit_correct = 0
    mean_probability_correct = 0
    oracle_any_correct = 0
    all_wrong = 0
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

    for raw_batch in loader:
        features = raw_batch["features"]
        mask = raw_batch["mask"]
        samples = raw_batch["samples"]
        tasks = tuple(sample.task for sample in samples)

        output = bank(features, mask)
        evidence = build_evidence_matrix(output, tasks, aggregation_config)
        summary, decision = aggregate_evidence(evidence, aggregation_config)

        worker_prediction_indices = _worker_prediction_indices(
            output,
            uncertainty_threshold=uncertainty_threshold,
        )
        majority_indices = _majority_prediction_indices(worker_prediction_indices)
        majority_predictions = tuple(
            _ALL_DECODED_LABELS[int(index)] for index in majority_indices
        )

        mean_logit_output = Step01Output(
            label_logits=output.label_logits.mean(dim=0),
            uncertainty_logits=output.uncertainty_logits.mean(dim=0),
        )
        mean_logit_predictions = tuple(
            decode_predictions(
                mean_logit_output,
                uncertainty_threshold=uncertainty_threshold,
            )
        )
        mean_probability_predictions = _mean_probability_predictions(
            output,
            uncertainty_threshold=uncertainty_threshold,
        )

        for sample_index, sample in enumerate(samples):
            total += 1
            truth = sample.label
            evidence_prediction = decision.predictions[sample_index]
            majority_prediction = majority_predictions[sample_index]
            mean_logit_prediction = mean_logit_predictions[sample_index]
            mean_probability_prediction = mean_probability_predictions[sample_index]

            evidence_is_correct = evidence_prediction == truth
            majority_is_correct = majority_prediction == truth
            evidence_correct += int(evidence_is_correct)
            majority_correct += int(majority_is_correct)
            mean_logit_correct += int(mean_logit_prediction == truth)
            mean_probability_correct += int(mean_probability_prediction == truth)

            worker_predictions = [
                _ALL_DECODED_LABELS[
                    int(worker_prediction_indices[worker_index, sample_index])
                ]
                for worker_index in range(bank.population_width)
            ]
            worker_correct_flags = [
                prediction == truth for prediction in worker_predictions
            ]
            for worker_index, is_correct in enumerate(worker_correct_flags):
                worker_correct[worker_index] += int(is_correct)

            any_correct = any(worker_correct_flags)
            oracle_any_correct += int(any_correct)
            all_wrong += int(not any_correct)

            rescue_opportunity = (not majority_is_correct) and any_correct
            minority_rescue_opportunities += int(rescue_opportunity)
            minority_rescues += int(rescue_opportunity and evidence_is_correct)

            strong_correct_minority = False
            if truth != "UNCERTAIN" and truth in LABEL_TO_INDEX:
                truth_index = LABEL_TO_INDEX[truth]
                strong_correct_evidence = bool(
                    (
                        evidence.evidence_scores[:, sample_index, truth_index]
                        >= aggregation_config.strong_evidence_threshold
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

            task_stats = by_task.setdefault(
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
            task_stats["count"] += 1
            task_stats["evidence_correct"] += int(evidence_is_correct)
            task_stats["majority_correct"] += int(majority_is_correct)
            task_stats["mean_logit_correct"] += int(mean_logit_prediction == truth)
            task_stats["mean_probability_correct"] += int(
                mean_probability_prediction == truth
            )
            task_stats["oracle_any_correct"] += int(any_correct)

        disagreement_entropy_sum += float(summary.disagreement_entropy.sum().cpu())
        mean_uncertainty_sum += float(summary.mean_uncertainty.sum().cpu())
        mean_invalid_mass_sum += float(summary.mean_invalid_label_mass.sum().cpu())

    if total == 0:
        raise ValueError("evaluation loader produced no samples")

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

    task_metrics = {
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
    }

    evidence_accuracy = evidence_correct / total
    oracle_coverage = oracle_any_correct / total
    return {
        "count": total,
        "population_width": bank.population_width,
        "execution_backend": bank.execution_backend,
        "unit_config": {
            "d_model": bank.unit_config.d_model,
            "block_count": bank.unit_config.block_count,
            "attention_heads": bank.unit_config.attention_heads,
            "feed_forward_width": bank.unit_config.feed_forward_width,
            "dropout": bank.unit_config.dropout,
            "sequence_length": bank.unit_config.sequence_length,
            "feature_width": bank.unit_config.feature_width,
        },
        "evidence_reducer_accuracy": evidence_accuracy,
        "majority_vote_accuracy": majority_correct / total,
        "mean_logit_accuracy": mean_logit_correct / total,
        "mean_probability_accuracy": mean_probability_correct / total,
        "oracle_any_correct_coverage": oracle_coverage,
        "all_wrong_rate": all_wrong / total,
        "minority_rescue_opportunity_rate": minority_rescue_opportunities / total,
        "minority_rescue_rate": minority_rescue_rate,
        "minority_suppression_rate": minority_suppression_rate,
        "majority_harm_rate": majority_harm_cases / total,
        "evidence_utilization_gap": oracle_coverage - evidence_accuracy,
        "mean_disagreement_entropy": disagreement_entropy_sum / total,
        "mean_population_uncertainty": mean_uncertainty_sum / total,
        "mean_invalid_label_mass": mean_invalid_mass_sum / total,
        "single_worker_accuracy": {
            "values": worker_accuracies,
            "min": min(worker_accuracies),
            "max": max(worker_accuracies),
            "mean": sum(worker_accuracies) / len(worker_accuracies),
        },
        "by_task": task_metrics,
    }
