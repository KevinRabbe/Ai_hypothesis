"""Simple inspectable scheduler baseline for persistent Work Threads.

The scoring policy is intentionally provisional. The durable architecture contract
is the SchedulerDecision boundary; numeric weights can be replaced later without
changing Work Items, ledger events, or projected thread state.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Sequence

from .contracts import ProjectedState, SchedulerAction, SchedulerDecision, WorkPurpose


@dataclass(frozen=True, slots=True)
class SchedulerSignals:
    """Bounded inference-visible metadata used by Scheduler v0."""

    importance: float = 0.5
    uncertainty: float = 0.0
    contradiction_severity: float = 0.0
    missing_coverage: float = 0.0
    novelty: float = 0.0
    dependency_impact: float = 0.0
    recent_progress: float = 0.0
    verification_need: float = 0.0
    starvation: float = 0.0
    estimated_cost: float = 0.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    exploration_probability: float = 0.20
    challenge_threshold: float = 0.65
    verification_threshold: float = 0.65
    stagnation_threshold: float = 0.05

    def validate(self) -> None:
        for name, value in (
            ("exploration_probability", self.exploration_probability),
            ("challenge_threshold", self.challenge_threshold),
            ("verification_threshold", self.verification_threshold),
            ("stagnation_threshold", self.stagnation_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SchedulableThread:
    state: ProjectedState
    signals: SchedulerSignals

    def validate(self) -> None:
        self.state.validate()
        self.signals.validate()


class SchedulerV0:
    """Deterministic priority scheduling plus structured-random exploration."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.config.validate()
        self._rng = rng or random.Random()

    def choose(
        self,
        candidates: Sequence[SchedulableThread],
        *,
        integration_backpressure: bool = False,
    ) -> SchedulerDecision:
        active = [candidate for candidate in candidates if candidate.state.status != "COMPLETE"]
        if not active:
            raise ValueError("scheduler requires at least one non-complete thread")
        for candidate in active:
            candidate.validate()

        exploring = (
            not integration_backpressure
            and self.config.exploration_probability > 0.0
            and self._rng.random() < self.config.exploration_probability
        )

        selected = (
            self._choose_exploration(active)
            if exploring
            else max(active, key=self._priority_score)
        )
        action, purpose, reason_codes = self._choose_action(
            selected,
            exploring=exploring,
            integration_backpressure=integration_backpressure,
        )

        decision = SchedulerDecision(
            decision_id=uuid.uuid4().hex,
            thread_id=selected.state.thread_id,
            action=action,
            purpose=purpose,
            reason_codes=reason_codes,
            projection_revision=selected.state.revision,
        )
        decision.validate()
        return decision

    def _priority_score(self, candidate: SchedulableThread) -> float:
        signals = candidate.signals
        return (
            2.0 * signals.importance
            + signals.uncertainty
            + signals.contradiction_severity
            + signals.missing_coverage
            + signals.novelty
            + signals.dependency_impact
            + signals.recent_progress
            + signals.verification_need
            + signals.starvation
            - signals.estimated_cost
        )

    def _choose_exploration(
        self,
        candidates: Sequence[SchedulableThread],
    ) -> SchedulableThread:
        # Structured exploration: randomize among threads in proportion to signals
        # that indicate under-explored or unresolved work, rather than uniform random.
        weights = [
            0.01
            + candidate.signals.missing_coverage
            + candidate.signals.novelty
            + candidate.signals.uncertainty
            for candidate in candidates
        ]
        return self._rng.choices(list(candidates), weights=weights, k=1)[0]

    def _choose_action(
        self,
        candidate: SchedulableThread,
        *,
        exploring: bool,
        integration_backpressure: bool,
    ) -> tuple[SchedulerAction, WorkPurpose, tuple[str, ...]]:
        signals = candidate.signals

        if integration_backpressure:
            if signals.verification_need >= self.config.verification_threshold:
                return SchedulerAction.VERIFY, WorkPurpose.VERIFY, ("BACKPRESSURE", "VERIFY")
            return SchedulerAction.SYNTHESIZE, WorkPurpose.SYNTHESIZE, ("BACKPRESSURE",)

        if exploring:
            return SchedulerAction.ADD_WIDTH, WorkPurpose.EXPLORE, ("STRUCTURED_EXPLORATION",)

        if signals.verification_need >= self.config.verification_threshold:
            return SchedulerAction.VERIFY, WorkPurpose.VERIFY, ("VERIFICATION_NEEDED",)

        if signals.contradiction_severity >= self.config.challenge_threshold:
            return SchedulerAction.CHALLENGE, WorkPurpose.CHALLENGE, ("CONTRADICTION",)

        if candidate.state.status == "PAUSED":
            return SchedulerAction.CONTINUE, candidate.state.purpose, ("RESUME",)

        if signals.recent_progress <= self.config.stagnation_threshold:
            return SchedulerAction.ROTATE_WORKER, candidate.state.purpose, ("STAGNATION",)

        return SchedulerAction.CONTINUE, candidate.state.purpose, ("PROGRESS",)
