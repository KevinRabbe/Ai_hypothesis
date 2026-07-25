"""Adapter from the Step 2 homogeneous neural bank to the persistent runtime.

The adapter does not make a population-level decision. Each selected worker emits one
structured local evidence contribution. Aggregation/integration remains downstream.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ai_hypothesis.runtime import AttemptResult, AttemptStatus, EvidenceContribution
from ai_hypothesis.runtime.worker_runtime import AttemptRequest
from ai_hypothesis.step01.model import NON_UNCERTAIN_LABELS, Step01Output
from ai_hypothesis.step01.schema import (
    FEATURE_WIDTH,
    SEQUENCE_LENGTH,
    TaskFamily,
)
from .evidence import AggregationConfig, build_evidence_matrix
from .population import HomogeneousWorkerBank, PopulationOutput


class Step02RuntimeWorkerBank:
    """Expose independently weighted Step 2 workers through ``WorkerBank`` semantics."""

    def __init__(
        self,
        bank: HomogeneousWorkerBank,
        *,
        aggregation_config: AggregationConfig = AggregationConfig(),
        worker_ids: Sequence[str] | None = None,
    ) -> None:
        aggregation_config.validate()
        if worker_ids is None:
            resolved_ids = tuple(f"worker-{index}" for index in range(bank.population_width))
        else:
            resolved_ids = tuple(worker_ids)
        if len(resolved_ids) != bank.population_width:
            raise ValueError("worker_ids must contain one ID per checkpoint")
        if any(not worker_id or not worker_id.strip() for worker_id in resolved_ids):
            raise ValueError("worker IDs must be non-empty")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("worker IDs must be unique")

        self.bank = bank
        self.aggregation_config = aggregation_config
        self.worker_ids = resolved_ids
        self._worker_indices = {
            worker_id: index for index, worker_id in enumerate(self.worker_ids)
        }

    def execute_batch(
        self,
        requests: Sequence[AttemptRequest],
    ) -> tuple[AttemptResult, ...]:
        if not requests:
            return ()

        results: list[AttemptResult | None] = [None] * len(requests)
        valid_positions: list[int] = []
        valid_requests: list[AttemptRequest] = []
        worker_indices: list[int] = []
        features: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        tasks: list[TaskFamily] = []

        for request_position, request in enumerate(requests):
            try:
                worker_index = self._worker_indices[request.worker_id]
                item = request.work_item
                feature = self._feature_tensor(item.context.get("features"))
                mask = self._mask_tensor(item.context.get("mask"))
                task = self._task(item.context.get("task"))
            except (KeyError, TypeError, ValueError) as error:
                results[request_position] = self._failed_result(
                    request,
                    code="STEP02_WORK_ITEM_INVALID",
                    followup="repair Step 2 Work Item context or worker assignment",
                    error=error,
                    neural_worker_evaluations=0,
                )
                continue

            valid_positions.append(request_position)
            valid_requests.append(request)
            worker_indices.append(worker_index)
            features.append(feature)
            masks.append(mask)
            tasks.append(task)

        if valid_requests:
            valid_results = self._execute_valid_requests(
                valid_requests,
                worker_indices,
                features,
                masks,
                tasks,
            )
            for request_position, result in zip(
                valid_positions,
                valid_results,
                strict=True,
            ):
                results[request_position] = result

        if any(result is None for result in results):
            raise RuntimeError("Step 2 runtime bank failed to produce one result per request")
        return tuple(result for result in results if result is not None)

    def _execute_valid_requests(
        self,
        requests: Sequence[AttemptRequest],
        worker_indices: Sequence[int],
        features: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        tasks: Sequence[TaskFamily],
    ) -> tuple[AttemptResult, ...]:
        feature_batch = torch.stack(tuple(features), dim=0)
        mask_batch = torch.stack(tuple(masks), dim=0)
        output = self.bank.forward_selected(worker_indices, feature_batch, mask_batch)
        evidence = build_evidence_matrix(
            PopulationOutput(
                label_logits=output.label_logits.unsqueeze(0),
                uncertainty_logits=output.uncertainty_logits.unsqueeze(0),
            ),
            tasks,
            self.aggregation_config,
        )

        results: list[AttemptResult] = []
        for sample_index, request in enumerate(requests):
            try:
                result = self._successful_result(
                    request,
                    worker_index=worker_indices[sample_index],
                    task=tasks[sample_index],
                    sample_index=sample_index,
                    output=output,
                    evidence=evidence,
                )
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                result = self._failed_result(
                    request,
                    code="STEP02_WORKER_OUTPUT_INVALID",
                    followup="inspect the selected worker checkpoint and numerical output",
                    error=error,
                    neural_worker_evaluations=1,
                )
            results.append(result)

        return tuple(results)

    def _successful_result(
        self,
        request: AttemptRequest,
        *,
        worker_index: int,
        task: TaskFamily,
        sample_index: int,
        output: Step01Output,
        evidence,
    ) -> AttemptResult:
        sample_logits = output.label_logits[sample_index]
        sample_uncertainty_logit = output.uncertainty_logits[sample_index]
        if not bool(torch.isfinite(sample_logits).all()):
            raise ValueError("worker label logits must contain only finite values")
        if not bool(torch.isfinite(sample_uncertainty_logit)):
            raise ValueError("worker uncertainty logit must be finite")

        checkpoint = self.bank.checkpoints[worker_index]
        label_index = int(evidence.top_valid_label_indices[0, sample_index].item())
        label = NON_UNCERTAIN_LABELS[label_index]
        strength = float(evidence.evidence_scores[0, sample_index, label_index].item())
        uncertainty = float(evidence.uncertainty_probability[0, sample_index].item())
        reliability = float(evidence.reliability[0, sample_index].item())
        invalid_mass = float(evidence.invalid_label_mass[0, sample_index].item())
        top_margin = float(evidence.top_margin[0, sample_index].item())

        contribution = EvidenceContribution(
            evidence_id=f"{request.attempt_id}:local-evidence",
            kind="STEP02_LOCAL_CLASS_EVIDENCE",
            summary=f"{task.value}: strongest valid label {label}",
            reference_ids=request.work_item.reference_ids,
            strength=strength,
            uncertainty=uncertainty,
            data={
                "task": task.value,
                "top_label": label,
                "top_label_index": label_index,
                "reliability": reliability,
                "invalid_label_mass": invalid_mass,
                "top_margin": top_margin,
                "label_logits": [float(value) for value in sample_logits.tolist()],
                "uncertainty_logit": float(sample_uncertainty_logit.item()),
                "label_probabilities": [
                    float(value)
                    for value in evidence.label_probabilities_all[0, sample_index].tolist()
                ],
                "valid_label_probabilities": [
                    float(value)
                    for value in evidence.valid_label_probabilities[0, sample_index].tolist()
                ],
                "evidence_scores": [
                    float(value)
                    for value in evidence.evidence_scores[0, sample_index].tolist()
                ],
                "worker_index": worker_index,
                "checkpoint_path": checkpoint.path,
                "checkpoint_step": checkpoint.step,
                "checkpoint_validation_score": checkpoint.validation_score,
                "selected_execution_backend": self.bank.selected_execution_backend,
                "population_execution_backend": self.bank.execution_backend,
                "device": str(self.bank.device),
            },
        )
        contribution.validate()
        result = AttemptResult(
            attempt_id=request.attempt_id,
            work_item_id=request.work_item.work_item_id,
            thread_id=request.work_item.thread_id,
            worker_id=request.worker_id,
            status=AttemptStatus.COMPLETED,
            evidence=(contribution,),
            progress_made=True,
            resource_usage={
                "neural_worker_evaluations": 1,
                "selected_execution_backend": self.bank.selected_execution_backend,
                "population_execution_backend": self.bank.execution_backend,
                "device": str(self.bank.device),
            },
        )
        result.validate()
        return result

    @staticmethod
    def _failed_result(
        request: AttemptRequest,
        *,
        code: str,
        followup: str,
        error: Exception,
        neural_worker_evaluations: int,
    ) -> AttemptResult:
        result = AttemptResult(
            attempt_id=request.attempt_id,
            work_item_id=request.work_item.work_item_id,
            thread_id=request.work_item.thread_id,
            worker_id=request.worker_id,
            status=AttemptStatus.FAILED,
            observations=(f"{code}: {type(error).__name__}: {error}",),
            requested_followups=(followup,),
            progress_made=False,
            resource_usage={
                "neural_worker_evaluations": neural_worker_evaluations,
                "failure_code": code,
                "error_type": type(error).__name__,
            },
        )
        result.validate()
        return result

    @staticmethod
    def _feature_tensor(value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("WorkItem context 'features' must be a torch.Tensor")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        expected_shape = (SEQUENCE_LENGTH, FEATURE_WIDTH)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"WorkItem features must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        value = value.to(dtype=torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("WorkItem features must contain only finite values")
        return value

    @staticmethod
    def _mask_tensor(value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("WorkItem context 'mask' must be a torch.Tensor")
        if value.ndim == 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        expected_shape = (SEQUENCE_LENGTH,)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"WorkItem mask must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        value = value.to(dtype=torch.bool)
        if not bool(value.any()):
            raise ValueError("WorkItem mask must contain at least one valid row")
        return value

    @staticmethod
    def _task(value: object) -> TaskFamily:
        if isinstance(value, TaskFamily):
            return value
        if isinstance(value, str):
            try:
                return TaskFamily(value)
            except ValueError as error:
                raise ValueError(f"unknown Step 2 task {value!r}") from error
        raise TypeError("WorkItem context 'task' must be a TaskFamily or task value string")
