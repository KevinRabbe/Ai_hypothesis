"""Batch many persistent large-scope worlds through one homogeneous neural call per round."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from typing import Sequence

from ai_hypothesis.runtime import (
    RuntimeControlLoop,
    SchedulerAction,
    SchedulerDecision,
    SchedulerSignals,
    SQLiteResearchLedger,
    ThreadStateProjector,
    TracingScheduler,
)
from ai_hypothesis.step02.evidence import AggregationConfig

from .coverage_planner import CoverageAwareScopePlanner
from .evaluate import ScopeWorkerMode, SelectedWorkerBank
from .persistent_experiment import (
    PersistentScopeEvaluation,
    PersistentScopeEvaluationProjector,
    PersistentScopeWorkerSelector,
)
from .relevance import LARGE_SCOPE_BENCHMARK_VERSION, LargeScopeRelevanceSample
from .runtime_bridge import LargeScopeRuntimeWorkerBank


class FixedScopeRoundScheduler:
    """Give each candidate one exact-width benchmark decision in snapshot order."""

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
            raise ValueError("persistent benchmark round does not run under backpressure")
        if not candidates:
            raise ValueError("fixed scope round requires at least one candidate")
        if self.width > max_width:
            raise ValueError("fixed scope round width exceeds remaining attempt budget")
        state = candidates[0].state
        decision = SchedulerDecision(
            decision_id=uuid.uuid4().hex,
            thread_id=state.thread_id,
            action=(
                SchedulerAction.CONTINUE
                if self.width == 1
                else SchedulerAction.ADD_WIDTH
            ),
            purpose=state.purpose,
            width=self.width,
            reason_codes=("FIXED_SCOPE_ROUND",),
            projection_revision=state.revision,
        )
        decision.validate()
        return decision


class PersistentScopeMultiWorldWorkerSelector:
    """Route each Work Thread through its own restart-safe checkpoint sequence."""

    def __init__(
        self,
        runtime_bank: LargeScopeRuntimeWorkerBank,
        samples_by_thread: dict[str, LargeScopeRelevanceSample],
        mode: ScopeWorkerMode | str,
    ) -> None:
        if not samples_by_thread:
            raise ValueError("samples_by_thread must not be empty")
        self.mode = ScopeWorkerMode(mode)
        self._selectors = {
            thread_id: PersistentScopeWorkerSelector(
                runtime_bank,
                sample,
                self.mode,
            )
            for thread_id, sample in samples_by_thread.items()
        }
        self._population_width = min(
            selector.population_width for selector in self._selectors.values()
        )

    @property
    def population_width(self) -> int:
        return self._population_width

    def choose_many_for_thread(
        self,
        thread_id: str,
        action: SchedulerAction,
        *,
        previous_worker_id: str | None,
        count: int,
    ) -> tuple[str, ...]:
        try:
            selector = self._selectors[thread_id]
        except KeyError as error:
            raise ValueError(f"unknown persistent benchmark thread {thread_id!r}") from error
        return selector.choose_many(
            action,
            previous_worker_id=previous_worker_id,
            count=count,
        )

    def choose_many(
        self,
        action: SchedulerAction,
        *,
        previous_worker_id: str | None,
        count: int,
    ) -> tuple[str, ...]:
        del action, previous_worker_id, count
        raise RuntimeError("multi-world worker selection requires thread identity")


@dataclass(frozen=True, slots=True)
class PersistentScopeWorldBatchEvaluation:
    rounds: int
    step_width: int
    world_count: int
    local_window_evaluations: int
    worlds: tuple[PersistentScopeEvaluation, ...]


class PersistentScopeWorldBatchExperiment:
    """Run equal-width persistent rounds for several worlds in one WorkerBank batch."""

    def __init__(
        self,
        *,
        ledger: SQLiteResearchLedger,
        samples: Sequence[LargeScopeRelevanceSample],
        bank: SelectedWorkerBank,
        mode: ScopeWorkerMode | str,
        step_width: int,
        evidence_config: AggregationConfig = AggregationConfig(),
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        if step_width <= 0:
            raise ValueError("step_width must be positive")
        evidence_config.validate()

        resolved_samples = tuple(samples)
        for sample in resolved_samples:
            sample.validate()
            if step_width > sample.config.window_count:
                raise ValueError("step_width cannot exceed any world window_count")
        sample_keys = tuple((sample.split, sample.seed) for sample in resolved_samples)
        if len(set(sample_keys)) != len(sample_keys):
            raise ValueError("persistent world batch requires unique split/seed worlds")

        self.ledger = ledger
        self.samples = resolved_samples
        self.mode = ScopeWorkerMode(mode)
        self.step_width = step_width
        self.evidence_config = evidence_config
        self.runtime_bank = LargeScopeRuntimeWorkerBank(bank, evidence_config)
        self.thread_ids = tuple(
            persistent_scope_world_thread_id(sample, self.mode)
            for sample in self.samples
        )
        self.samples_by_thread = dict(zip(self.thread_ids, self.samples, strict=True))
        self.planners = {
            thread_id: CoverageAwareScopePlanner(
                ledger,
                sample,
                self.mode,
            )
            for thread_id, sample in self.samples_by_thread.items()
        }
        self.worker_selector = PersistentScopeMultiWorldWorkerSelector(
            self.runtime_bank,
            self.samples_by_thread,
            self.mode,
        )
        if step_width > self.worker_selector.population_width:
            raise ValueError("step_width exceeds a world's persistent worker allocation limit")
        self.loop = RuntimeControlLoop(
            ledger=ledger,
            scheduler=TracingScheduler(
                ledger,
                FixedScopeRoundScheduler(step_width),
            ),
            worker_bank=self.runtime_bank,
            worker_ids=self.runtime_bank.worker_ids,
            worker_selector=self.worker_selector,
        )
        self._ensure_threads()

    def run_rounds(self, round_count: int) -> PersistentScopeWorldBatchEvaluation:
        if round_count <= 0:
            raise ValueError("round_count must be positive")
        for _ in range(round_count):
            batch = self.loop.run_many(
                signal_provider=self._signal_provider,
                context_provider=self._context_provider,
                max_threads=len(self.samples),
                max_attempts=len(self.samples) * self.step_width,
            )
            if len(batch.steps) != len(self.samples):
                raise RuntimeError("persistent world round did not schedule every world")
            if batch.neural_attempt_count != len(self.samples) * self.step_width:
                raise RuntimeError("persistent world round used the wrong neural budget")
        return self.evaluate()

    def evaluate(self) -> PersistentScopeWorldBatchEvaluation:
        events = self.ledger.read_all_events()
        worlds = tuple(
            PersistentScopeEvaluationProjector(
                sample,
                self.mode,
                expected_worker_bank_id=self.runtime_bank.worker_bank_id,
            ).project(
                events,
                thread_id=thread_id,
                step_width=self.step_width,
            )
            for thread_id, sample in zip(self.thread_ids, self.samples, strict=True)
        )
        round_counts = {world.step_count for world in worlds}
        if len(round_counts) != 1:
            raise ValueError("persistent world batch contains unequal completed round counts")
        rounds = next(iter(round_counts))
        return PersistentScopeWorldBatchEvaluation(
            rounds=rounds,
            step_width=self.step_width,
            world_count=len(worlds),
            local_window_evaluations=sum(world.attempt_count for world in worlds),
            worlds=worlds,
        )

    def _signal_provider(self, state) -> SchedulerSignals:
        planner = self._planner_for(state.thread_id)
        return planner.augment_signals(
            state,
            SchedulerSignals(
                importance=1.0,
                recent_progress=1.0,
            ),
        )

    def _context_provider(self, state, decision):
        return self._planner_for(state.thread_id)(state, decision)

    def _planner_for(self, thread_id: str) -> CoverageAwareScopePlanner:
        try:
            return self.planners[thread_id]
        except KeyError as error:
            raise ValueError(f"unknown persistent benchmark thread {thread_id!r}") from error

    def _ensure_threads(self) -> None:
        states = ThreadStateProjector().project_all(self.ledger.read_all_events())
        by_id = {state.thread_id: state for state in states}
        expected_ids = set(self.thread_ids)
        unexpected_active = [
            state.thread_id
            for state in states
            if state.thread_id not in expected_ids and state.status != "COMPLETE"
        ]
        if unexpected_active:
            raise ValueError(
                "persistent world batch requires a dedicated ledger or no unrelated active threads"
            )

        for thread_id, sample in zip(self.thread_ids, self.samples, strict=True):
            metadata = _persistent_world_metadata(
                sample,
                self.mode,
                self.step_width,
                self.runtime_bank,
                self.evidence_config,
            )
            existing = by_id.get(thread_id)
            if existing is None:
                self.loop.create_thread(
                    thread_id=thread_id,
                    objective="Evaluate persistent large-scope relevance",
                    metadata=metadata,
                )
                continue
            if existing.status == "COMPLETE" or existing.merged_into_thread_id is not None:
                raise ValueError(f"persistent world thread {thread_id!r} is terminal")
            for key, expected in metadata.items():
                if existing.metadata.get(key) != expected:
                    raise ValueError(
                        f"existing persistent world metadata mismatch for {key!r}"
                    )


def persistent_scope_world_thread_id(
    sample: LargeScopeRelevanceSample,
    mode: ScopeWorkerMode | str,
) -> str:
    sample.validate()
    mode = ScopeWorkerMode(mode)
    payload = "|".join(
        (
            LARGE_SCOPE_BENCHMARK_VERSION,
            sample.split,
            str(sample.seed),
            mode.value,
            str(sample.config.window_count),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"scope-world-{digest}"


def _persistent_world_metadata(
    sample: LargeScopeRelevanceSample,
    mode: ScopeWorkerMode,
    step_width: int,
    runtime_bank: LargeScopeRuntimeWorkerBank,
    evidence_config: AggregationConfig,
) -> dict[str, object]:
    return {
        "benchmark_version": LARGE_SCOPE_BENCHMARK_VERSION,
        "large_scope_split": sample.split,
        "large_scope_world_seed": sample.seed,
        "large_scope_mode": mode.value,
        "large_scope_window_count": sample.config.window_count,
        "large_scope_step_width": step_width,
        "large_scope_worker_bank_id": runtime_bank.worker_bank_id,
        "large_scope_population_width": runtime_bank.bank.population_width,
        "large_scope_evidence_config": asdict(evidence_config),
    }
