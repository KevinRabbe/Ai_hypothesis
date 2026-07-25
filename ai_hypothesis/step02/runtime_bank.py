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
from ai_hypothesis.step01.schema import TaskFamily
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

        worker_indices: list[int] = []
        features: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        tasks: list[TaskFamily] = []

        for request in requests:
            try:
                worker_index = self._worker_indices[request.worker_id]
            except KeyError as error:
                raise KeyError(f"unknown worker_id {request.worker_id!r}") from error
            worker_indices.append(worker_index)

            item = request.work_item
            features.append(self._feature_tensor(item.context.get("features")))
            masks.append(self._mask_tensor(item.context.get("mask")))
            tasks.append(self._task(item.context.get("task")))

        feature_batch = torch.stack(features, dim=0)
        mask_batch = torch.stack(masks, dim=0)
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
                summary=f"{tasks[sample_index].value}: strongest valid label {label}",
                reference_ids=request.work_item.reference_ids,
                strength=strength,
                uncertainty=uncertainty,
                data={
                    "task": tasks[sample_index].value,
                    "top_label": label,
                    "top_label_index": label_index,
                    "reliability": reliability,
                    "invalid_label_mass": invalid_mass,
                    "top_margin": top_margin,
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
                },
            )
            contribution.validate()
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    evidence=(contribution,),
                    progress_made=True,
                    resource_usage={"neural_worker_evaluations": 1},
                )
            )

        return tuple(results)

    @staticmethod
    def _feature_tensor(value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("WorkItem context 'features' must be a torch.Tensor")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.ndim != 2:
            raise ValueError("WorkItem features must have shape [sequence, feature]")
        return value

    @staticmethod
    def _mask_tensor(value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("WorkItem context 'mask' must be a torch.Tensor")
        if value.ndim == 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.ndim != 1:
            raise ValueError("WorkItem mask must have shape [sequence]")
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
