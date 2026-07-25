"""Persistent-runtime bridge for the large-scope relevance benchmark.

The generic runtime remains unaware of tensors, relevance labels, or benchmark layout.
This adapter converts one benchmark window into one scoped Work Item and converts the
frozen selected-worker output back into ordinary EvidenceContribution records.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Sequence

import torch

from ai_hypothesis.runtime import (
    AttemptResult,
    AttemptStatus,
    EvidenceContribution,
    ProjectedState,
    SchedulerAction,
    SchedulerDecision,
    WorkPreparation,
    WorkPreparationBatch,
    WorkPurpose,
)
from ai_hypothesis.step01.model import LABEL_TO_INDEX, NON_UNCERTAIN_LABELS, Step01Output
from ai_hypothesis.step01.schema import TaskFamily
from ai_hypothesis.step02.evidence import AggregationConfig, build_evidence_matrix
from ai_hypothesis.step02.population import PopulationOutput

from .evaluate import ScopeWorkerMode, SelectedWorkerBank
from .relevance import (
    LARGE_SCOPE_BENCHMARK_VERSION,
    LargeScopeRelevanceSample,
    diverse_worker_indices,
    inspection_prefix,
    same_worker_indices,
)


_RUNTIME_EVIDENCE_KIND = "LARGE_SCOPE_RELEVANCE_WINDOW"


def large_scope_region_id(sample: LargeScopeRelevanceSample, window_index: int) -> str:
    """Return stable opaque identity for one source window inside one benchmark world."""

    sample.validate()
    if not 0 <= window_index < sample.config.window_count:
        raise IndexError("window_index is outside the large-scope world")
    payload = "|".join(
        (
            LARGE_SCOPE_BENCHMARK_VERSION,
            sample.split,
            str(sample.seed),
            str(window_index),
            str(sample.window_seeds[window_index]),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"scope-{digest}"


def large_scope_worker_id(
    worker_index: int,
    *,
    checkpoint_id: str | None = None,
) -> str:
    if worker_index < 0:
        raise ValueError("worker_index must be non-negative")
    if checkpoint_id is None:
        # Fallback for protocol-compatible fake/test banks that do not expose durable
        # checkpoint identity. Real HomogeneousWorkerBank instances do expose it.
        return f"large-scope-worker-{worker_index}"
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("checkpoint_id must be non-empty text")
    digest = hashlib.sha256(checkpoint_id.encode("utf-8")).hexdigest()
    return f"large-scope-worker-{digest}"


@dataclass(frozen=True, slots=True)
class LargeScopeRuntimeContextProvider:
    sample: LargeScopeRelevanceSample
    mode: ScopeWorkerMode | str

    def __call__(
        self,
        _state: ProjectedState,
        decision: SchedulerDecision,
    ) -> WorkPreparation | WorkPreparationBatch:
        self.sample.validate()
        mode = ScopeWorkerMode(self.mode)
        window_indices = inspection_prefix(self.sample, decision.width)
        items: list[WorkPreparation] = []
        for window_index in window_indices:
            window = self.sample.windows[window_index]
            region_id = large_scope_region_id(self.sample, window_index)
            items.append(
                WorkPreparation(
                    context={
                        "large_scope_features": window.features,
                        "large_scope_mask": window.mask,
                        "large_scope_split": self.sample.split,
                        "large_scope_world_seed": self.sample.seed,
                        "large_scope_window_index": window_index,
                        "large_scope_window_seed": self.sample.window_seeds[window_index],
                        "large_scope_mode": mode.value,
                    },
                    reference_ids=(region_id,),
                    scope_region_ids=(region_id,),
                )
            )
        if decision.width == 1:
            return items[0]
        return WorkPreparationBatch(items=tuple(items))


class PlannedScopeWorkerSelector:
    """Exact benchmark worker plan, including intentional checkpoint reuse.

    This selector is benchmark-specific. The generic WorkerSelectorV0 can continue to
    prefer distinct workers; the same-worker control needs repeated use of one checkpoint
    across different source regions without redefining generic runtime semantics.
    """

    def __init__(self, planned_worker_ids: Sequence[str]) -> None:
        self._planned_worker_ids = tuple(planned_worker_ids)
        if not self._planned_worker_ids:
            raise ValueError("planned_worker_ids must not be empty")
        if any(not worker_id or not worker_id.strip() for worker_id in self._planned_worker_ids):
            raise ValueError("planned worker IDs must be non-empty")

    @property
    def population_width(self) -> int:
        return len(self._planned_worker_ids)

    def choose(
        self,
        action: SchedulerAction,
        *,
        previous_worker_id: str | None,
    ) -> str:
        return self.choose_many(
            action,
            previous_worker_id=previous_worker_id,
            count=1,
        )[0]

    def choose_many(
        self,
        _action: SchedulerAction,
        *,
        previous_worker_id: str | None,
        count: int,
    ) -> tuple[str, ...]:
        del previous_worker_id
        if count <= 0:
            raise ValueError("worker selection count must be positive")
        if count > len(self._planned_worker_ids):
            raise ValueError("requested width exceeds the fixed benchmark worker plan")
        return self._planned_worker_ids[:count]


class FixedScopeScheduler:
    """Exact-width benchmark control, not an adaptive scheduler policy."""

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width

    def choose(
        self,
        candidates,
        *,
        integration_backpressure: bool = False,
        max_width: int = 1,
    ) -> SchedulerDecision:
        if integration_backpressure:
            raise ValueError("fixed-scope benchmark does not run under integration backpressure")
        if len(candidates) != 1:
            raise ValueError("fixed-scope benchmark requires exactly one schedulable thread")
        if self.width > max_width:
            raise ValueError("fixed-scope width exceeds available benchmark worker plan")
        state = candidates[0].state
        decision = SchedulerDecision(
            decision_id=uuid.uuid4().hex,
            thread_id=state.thread_id,
            action=(SchedulerAction.CONTINUE if self.width == 1 else SchedulerAction.ADD_WIDTH),
            purpose=WorkPurpose.EXPLORE,
            width=self.width,
            reason_codes=("FIXED_SCOPE_BENCHMARK",),
            projection_revision=state.revision,
        )
        decision.validate()
        return decision


class LargeScopeRuntimeWorkerBank:
    """Adapt HomogeneousWorkerBank.forward_selected to generic WorkerRuntime batches."""

    def __init__(
        self,
        bank: SelectedWorkerBank,
        evidence_config: AggregationConfig = AggregationConfig(),
    ) -> None:
        if bank.population_width <= 0:
            raise ValueError("selected worker bank must contain at least one worker")
        evidence_config.validate()
        self.bank = bank
        self.evidence_config = evidence_config

        raw_checkpoint_ids = getattr(bank, "checkpoint_ids", None)
        if raw_checkpoint_ids is None:
            self.checkpoint_ids: tuple[str, ...] | None = None
        else:
            checkpoint_ids = tuple(raw_checkpoint_ids)
            if len(checkpoint_ids) != bank.population_width:
                raise ValueError("checkpoint identity count must match population width")
            if any(
                not isinstance(checkpoint_id, str) or not checkpoint_id.strip()
                for checkpoint_id in checkpoint_ids
            ):
                raise ValueError("checkpoint identities must be non-empty strings")
            if len(set(checkpoint_ids)) != len(checkpoint_ids):
                raise ValueError("checkpoint identities must be unique")
            self.checkpoint_ids = checkpoint_ids

        self.worker_ids = tuple(
            large_scope_worker_id(
                index,
                checkpoint_id=(
                    self.checkpoint_ids[index]
                    if self.checkpoint_ids is not None
                    else None
                ),
            )
            for index in range(bank.population_width)
        )
        self.worker_bank_id = "worker-bank-sha256-" + hashlib.sha256(
            "\n".join(self.worker_ids).encode("utf-8")
        ).hexdigest()
        self._worker_index = {
            worker_id: index for index, worker_id in enumerate(self.worker_ids)
        }

    def worker_id_for_index(self, worker_index: int) -> str:
        if not 0 <= worker_index < len(self.worker_ids):
            raise IndexError("worker index is outside the selected bank")
        return self.worker_ids[worker_index]

    def execute_batch(self, requests) -> tuple[AttemptResult, ...]:
        requests = tuple(requests)
        if not requests:
            return ()

        worker_indices: list[int] = []
        features: list[object] = []
        masks: list[object] = []
        metadata: list[dict[str, object]] = []
        for request in requests:
            try:
                worker_index = self._worker_index[request.worker_id]
            except KeyError as error:
                raise ValueError(f"unknown large-scope worker ID {request.worker_id!r}") from error
            item = request.work_item
            if len(item.scope_region_ids) != 1:
                raise ValueError("large-scope runtime Work Item must own exactly one region")
            context = item.context
            feature_rows = context.get("large_scope_features")
            mask_rows = context.get("large_scope_mask")
            if feature_rows is None or mask_rows is None:
                raise ValueError("large-scope Work Item is missing feature or mask context")
            split = context.get("large_scope_split")
            world_seed = context.get("large_scope_world_seed")
            window_index = context.get("large_scope_window_index")
            window_seed = context.get("large_scope_window_seed")
            mode = context.get("large_scope_mode")
            if not isinstance(split, str) or not split:
                raise ValueError("large-scope Work Item split must be text")
            if isinstance(world_seed, bool) or not isinstance(world_seed, int) or world_seed < 0:
                raise ValueError("large-scope world seed must be a non-negative integer")
            if isinstance(window_index, bool) or not isinstance(window_index, int) or window_index < 0:
                raise ValueError("large-scope window index must be a non-negative integer")
            if isinstance(window_seed, bool) or not isinstance(window_seed, int) or window_seed < 0:
                raise ValueError("large-scope window seed must be a non-negative integer")
            if mode not in {member.value for member in ScopeWorkerMode}:
                raise ValueError("large-scope worker mode is invalid")

            worker_indices.append(worker_index)
            features.append(feature_rows)
            masks.append(mask_rows)
            meta: dict[str, object] = {
                "region_id": item.scope_region_ids[0],
                "split": split,
                "world_seed": world_seed,
                "window_index": window_index,
                "window_seed": window_seed,
                "mode": mode,
                "worker_index": worker_index,
                "worker_bank_id": self.worker_bank_id,
            }
            if self.checkpoint_ids is not None:
                meta["checkpoint_id"] = self.checkpoint_ids[worker_index]
            metadata.append(meta)

        feature_tensor = torch.tensor(features, dtype=torch.float32)
        mask_tensor = torch.tensor(masks, dtype=torch.bool)
        raw: Step01Output = self.bank.forward_selected(
            worker_indices,
            feature_tensor,
            mask_tensor,
        )
        batch_size = len(requests)
        if raw.label_logits.shape[0] != batch_size or raw.uncertainty_logits.shape[0] != batch_size:
            raise ValueError("selected worker bank returned a mismatched batch size")

        evidence = build_evidence_matrix(
            PopulationOutput(
                label_logits=raw.label_logits.unsqueeze(0),
                uncertainty_logits=raw.uncertainty_logits.unsqueeze(0),
            ),
            [TaskFamily.RELEVANCE] * batch_size,
            self.evidence_config,
        )
        relevant_index = LABEL_TO_INDEX["RELEVANT"]
        not_relevant_index = LABEL_TO_INDEX["NOT_RELEVANT"]

        results: list[AttemptResult] = []
        for batch_index, request in enumerate(requests):
            uncertainty = float(evidence.uncertainty_probability[0, batch_index].item())
            local_top_index = int(evidence.top_valid_label_indices[0, batch_index].item())
            local_label = (
                "UNCERTAIN"
                if uncertainty >= 0.5
                else NON_UNCERTAIN_LABELS[local_top_index]
            )
            relevant_score = float(
                evidence.evidence_scores[0, batch_index, relevant_index].item()
            )
            not_relevant_score = float(
                evidence.evidence_scores[0, batch_index, not_relevant_index].item()
            )
            invalid_mass = float(evidence.invalid_label_mass[0, batch_index].item())
            top_margin = float(evidence.top_margin[0, batch_index].item())
            meta = metadata[batch_index]
            evidence_id = f"scope-evidence-{request.attempt_id}"
            contribution = EvidenceContribution(
                evidence_id=evidence_id,
                kind=_RUNTIME_EVIDENCE_KIND,
                summary=f"Large-scope relevance evidence for window {meta['window_index']}",
                reference_ids=(str(meta["region_id"]),),
                strength=relevant_score,
                uncertainty=uncertainty,
                data={
                    "benchmark_version": LARGE_SCOPE_BENCHMARK_VERSION,
                    **meta,
                    "local_label": local_label,
                    "relevant_evidence": relevant_score,
                    "not_relevant_evidence": not_relevant_score,
                    "invalid_label_mass": invalid_mass,
                    "top_margin": top_margin,
                },
            )
            results.append(
                AttemptResult(
                    attempt_id=request.attempt_id,
                    work_item_id=request.work_item.work_item_id,
                    thread_id=request.work_item.thread_id,
                    worker_id=request.worker_id,
                    status=AttemptStatus.COMPLETED,
                    evidence=(contribution,),
                    progress_made=True,
                    resource_usage={"local_window_evaluations": 1},
                )
            )
        return tuple(results)


def planned_scope_worker_selector(
    runtime_bank: LargeScopeRuntimeWorkerBank,
    sample: LargeScopeRelevanceSample,
    *,
    mode: ScopeWorkerMode | str,
    max_width: int,
) -> PlannedScopeWorkerSelector:
    sample.validate()
    mode = ScopeWorkerMode(mode)
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    if max_width > sample.config.window_count:
        raise ValueError("max_width cannot exceed world window_count")
    if mode is ScopeWorkerMode.SAME_WORKER:
        indices = same_worker_indices(
            seed=sample.seed,
            width=max_width,
            population_width=runtime_bank.bank.population_width,
            split=sample.split,
        )
    else:
        indices = diverse_worker_indices(
            seed=sample.seed,
            width=max_width,
            population_width=runtime_bank.bank.population_width,
            split=sample.split,
        )
    return PlannedScopeWorkerSelector(
        tuple(runtime_bank.worker_id_for_index(index) for index in indices)
    )
