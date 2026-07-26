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
    integration_backlog: float = 0.0
    starvation: float = 0.0
    estimated_cost: float = 0.0
    # Generic demand for bounded synthesis/organization work. The scheduler deliberately
    # does not know whether the source is thread consolidation, final synthesis, or another
    # future projection. Appended to preserve positional compatibility.
    synthesis_need: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("importance", self.importance),
            ("uncertainty", self.uncertainty),
            ("contradiction_severity", self.contradiction_severity),
            ("missing_coverage", self.missing_coverage),
            ("novelty", self.novelty),
            ("dependency_impact", self.dependency_impact),
            ("recent_progress", self.recent_progress),
            ("verification_need", self.verification_need),
            ("integration_backlog", self.integration_backlog),
            ("starvation", self.starvation),
            ("estimated_cost", self.estimated_cost),
            ("synthesis_need", self.synthesis_need),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    exploration_probability: float = 0.20
    exploration_width: int = 2
    challenge_threshold: float = 0.65
    verification_threshold: float = 0.65
    stagnation_threshold: float = 0.05
    # Keep a small permanent discovery lane even while integration is overloaded.
    # Appended to preserve positional compatibility with the original config fields.
    backpressure_exploration_probability: float = 0.05
    # Generic synthesis threshold. Appended after all existing config fields so old
    # positional construction retains exactly the same meaning.
    synthesis_threshold: float = 0.65

    def validate(self) -> None:
        if self.exploration_width <= 0:
            raise ValueError("exploration_width must be positive")
        for name, value in (
            ("exploration_probability", self.exploration_probability),
            ("challenge_threshold", self.challenge_threshold),
            ("verification_threshold", self.verification_threshold),
            ("stagnation_threshold", self.stagnation_threshold),
            ("synthesis_threshold", self.synthesis_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < self.backpressure_exploration_probability < 1.0:
            raise ValueError("backpressure_exploration_probability must be in (0, 1)")


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
        max_width: int = 1,
    ) -> SchedulerDecision:
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        active = [candidate for candidate in candidates if candidate.state.status != "COMPLETE"]
        if not active:
            raise ValueError("scheduler requires at least one non-complete thread")
        for candidate in active:
            candidate.validate()

        backpressure_exploring = (
            integration_backpressure
            and self._rng.random() < self.config.backpressure_exploration_probability
        )
        normal_exploring = (
            not integration_backpressure
            and self.config.exploration_probability > 0.0
            and self._rng.random() < self.config.exploration_probability
        )
        exploring = backpressure_exploring or normal_exploring
        servicing_backpressure = integration_backpressure and not backpressure_exploring

        if servicing_backpressure:
            selected = max(
                active,
                key=lambda candidate: (
                    candidate.signals.integration_backlog,
                    self._priority_score(candidate),
                ),
            )
        elif exploring:
            selected = self._choose_exploration(active)
        else:
            selected = max(active, key=self._priority_score)

        action, purpose, reason_codes = self._choose_action(
            selected,
            exploring=exploring,
            integration_backpressure=servicing_backpressure,
            backpressure_exploring=backpressure_exploring,
        )
        width = (
            min(self.config.exploration_width, max_width)
            if action is SchedulerAction.ADD_WIDTH
            else 1
        )

        decision = SchedulerDecision(
            decision_id=uuid.uuid4().hex,
            thread_id=selected.state.thread_id,
            action=action,
            purpose=purpose,
            width=width,
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
            + signals.synthesis_need
            + signals.starvation
            - signals.estimated_cost
        )

    def _choose_exploration(
        self,
        candidates: Sequence[SchedulableThread],
    ) -> SchedulableThread:
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
        backpressure_exploring: bool,
    ) -> tuple[SchedulerAction, WorkPurpose, tuple[str, ...]]:
        signals = candidate.signals

        if integration_backpressure:
            if signals.verification_need >= self.config.verification_threshold:
                return SchedulerAction.VERIFY, WorkPurpose.VERIFY, ("BACKPRESSURE", "VERIFY")
            return SchedulerAction.SYNTHESIZE, WorkPurpose.SYNTHESIZE, ("BACKPRESSURE",)

        if exploring:
            reasons = ["STRUCTURED_EXPLORATION"]
            if backpressure_exploring:
                reasons.append("BACKPRESSURE_EXPLORATION")
            return SchedulerAction.ADD_WIDTH, WorkPurpose.EXPLORE, tuple(reasons)

        if signals.verification_need >= self.config.verification_threshold:
            return SchedulerAction.VERIFY, WorkPurpose.VERIFY, ("VERIFICATION_NEEDED",)

        if signals.contradiction_severity >= self.config.challenge_threshold:
            return SchedulerAction.CHALLENGE, WorkPurpose.CHALLENGE, ("CONTRADICTION",)

        if signals.synthesis_need >= self.config.synthesis_threshold:
            return SchedulerAction.SYNTHESIZE, WorkPurpose.SYNTHESIZE, ("SYNTHESIS_NEEDED",)

        if candidate.state.status == "PAUSED":
            return SchedulerAction.CONTINUE, candidate.state.purpose, ("RESUME",)

        if signals.recent_progress <= self.config.stagnation_threshold:
            return SchedulerAction.ROTATE_WORKER, candidate.state.purpose, ("STAGNATION",)

        return SchedulerAction.CONTINUE, candidate.state.purpose, ("PROGRESS",)
