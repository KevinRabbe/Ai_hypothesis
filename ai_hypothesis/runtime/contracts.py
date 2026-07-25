"""Scale-invariant semantic contracts for the population runtime.

These types intentionally describe *what* crosses component boundaries, not how
storage, scheduling, batching, or integration are implemented. Construction can
therefore continue under provisional policies without coupling the architecture
to any one experiment result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class WorkPurpose(str, Enum):
    EXPLORE = "EXPLORE"
    PROGRESS = "PROGRESS"
    CHALLENGE = "CHALLENGE"
    VERIFY = "VERIFY"
    SYNTHESIZE = "SYNTHESIZE"


class AttemptStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class SchedulerAction(str, Enum):
    CONTINUE = "CONTINUE"
    ROTATE_WORKER = "ROTATE_WORKER"
    ADD_WIDTH = "ADD_WIDTH"
    FORK = "FORK"
    CHALLENGE = "CHALLENGE"
    VERIFY = "VERIFY"
    SYNTHESIZE = "SYNTHESIZE"
    PAUSE = "PAUSE"
    COMPLETE = "COMPLETE"


class EvidenceDispositionKind(str, Enum):
    INTEGRATED = "INTEGRATED"
    DUPLICATE = "DUPLICATE"
    IRRELEVANT = "IRRELEVANT"
    INVALID = "INVALID"
    LOCAL_ONLY = "LOCAL_ONLY"


class KnowledgeAssessmentKind(str, Enum):
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    RETRACTED = "RETRACTED"


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class WorkItem:
    work_item_id: str
    thread_id: str
    objective: str
    purpose: WorkPurpose
    projection_revision: int
    scheduler_decision_id: str | None = None
    scope_region_ids: tuple[str, ...] = ()
    reference_ids: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    resource_budget: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text("work_item_id", self.work_item_id)
        _require_text("thread_id", self.thread_id)
        _require_text("objective", self.objective)
        _require_non_negative("projection_revision", self.projection_revision)
        if self.scheduler_decision_id is not None:
            _require_text("scheduler_decision_id", self.scheduler_decision_id)
        for name, values in (
            ("scope_region_ids", self.scope_region_ids),
            ("reference_ids", self.reference_ids),
            ("parent_ids", self.parent_ids),
        ):
            if any(not value or not value.strip() for value in values):
                raise ValueError(f"{name} must not contain empty IDs")
        if len(set(self.scope_region_ids)) != len(self.scope_region_ids):
            raise ValueError("scope_region_ids must be unique inside one Work Item")


@dataclass(frozen=True, slots=True)
class EvidenceContribution:
    evidence_id: str
    kind: str
    summary: str
    reference_ids: tuple[str, ...] = ()
    strength: float | None = None
    uncertainty: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text("evidence_id", self.evidence_id)
        _require_text("kind", self.kind)
        _require_text("summary", self.summary)
        if any(not value for value in self.reference_ids):
            raise ValueError("reference_ids must not contain empty IDs")
        if self.strength is not None:
            _require_finite("strength", self.strength)
        if self.uncertainty is not None:
            _require_finite("uncertainty", self.uncertainty)
            if not 0.0 <= self.uncertainty <= 1.0:
                raise ValueError("uncertainty must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EvidenceDisposition:
    evidence_ids: tuple[str, ...]
    disposition: EvidenceDispositionKind
    reason: str | None = None

    def validate(self) -> None:
        if not self.evidence_ids:
            raise ValueError("evidence disposition requires at least one evidence ID")
        if any(not evidence_id or not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("evidence IDs must be non-empty")
        if self.reason is not None:
            _require_text("reason", self.reason)


@dataclass(frozen=True, slots=True)
class KnowledgeAssessment:
    """Verification/challenge outcome for existing knowledge deltas."""

    delta_ids: tuple[str, ...]
    assessment: KnowledgeAssessmentKind
    reason: str | None = None

    def validate(self) -> None:
        if not self.delta_ids:
            raise ValueError("knowledge assessment requires at least one delta ID")
        if any(not delta_id or not delta_id.strip() for delta_id in self.delta_ids):
            raise ValueError("delta IDs must be non-empty")
        if self.reason is not None:
            _require_text("reason", self.reason)


@dataclass(frozen=True, slots=True)
class AttemptResult:
    attempt_id: str
    work_item_id: str
    thread_id: str
    worker_id: str
    status: AttemptStatus
    observations: tuple[str, ...] = ()
    evidence: tuple[EvidenceContribution, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    knowledge_deltas: tuple[KnowledgeDelta, ...] = ()
    knowledge_assessments: tuple[KnowledgeAssessment, ...] = ()
    evidence_dispositions: tuple[EvidenceDisposition, ...] = ()
    hypotheses_proposed: tuple[str, ...] = ()
    hypotheses_strengthened: tuple[str, ...] = ()
    hypotheses_weakened: tuple[str, ...] = ()
    hypotheses_rejected: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    possibilities_eliminated: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    requested_followups: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    progress_made: bool = False
    resource_usage: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("work_item_id", self.work_item_id),
            ("thread_id", self.thread_id),
            ("worker_id", self.worker_id),
        ):
            _require_text(name, value)
        for contribution in self.evidence:
            contribution.validate()
        for delta in self.knowledge_deltas:
            delta.validate()
        for assessment in self.knowledge_assessments:
            assessment.validate()
        for disposition in self.evidence_dispositions:
            disposition.validate()
        if any(not value for value in self.evidence_refs):
            raise ValueError("evidence_refs must not contain empty IDs")


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    event_id: str
    event_type: str
    sequence: int
    payload_schema: str
    thread_id: str | None = None
    attempt_id: str | None = None
    reference_ids: tuple[str, ...] = ()
    parent_event_ids: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text("event_id", self.event_id)
        _require_text("event_type", self.event_type)
        _require_text("payload_schema", self.payload_schema)
        _require_non_negative("sequence", self.sequence)
        if self.thread_id is not None:
            _require_text("thread_id", self.thread_id)
        if self.attempt_id is not None:
            _require_text("attempt_id", self.attempt_id)


@dataclass(frozen=True, slots=True)
class ProjectedState:
    revision: int
    thread_id: str
    objective: str
    status: str
    purpose: WorkPurpose
    reference_ids: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    dependency_thread_ids: tuple[str, ...] = ()
    parent_thread_ids: tuple[str, ...] = ()
    child_thread_ids: tuple[str, ...] = ()
    merged_from_thread_ids: tuple[str, ...] = ()
    merged_into_thread_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_negative("revision", self.revision)
        _require_text("thread_id", self.thread_id)
        _require_text("objective", self.objective)
        _require_text("status", self.status)
        for name, values in (
            ("dependency_thread_ids", self.dependency_thread_ids),
            ("parent_thread_ids", self.parent_thread_ids),
            ("child_thread_ids", self.child_thread_ids),
            ("merged_from_thread_ids", self.merged_from_thread_ids),
        ):
            if any(not value or not value.strip() for value in values):
                raise ValueError(f"{name} must not contain empty IDs")
            if self.thread_id in values:
                raise ValueError(f"{name} must not contain the thread itself")
        if self.merged_into_thread_id is not None:
            _require_text("merged_into_thread_id", self.merged_into_thread_id)
            if self.merged_into_thread_id == self.thread_id:
                raise ValueError("thread cannot be merged into itself")


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    decision_id: str
    thread_id: str
    action: SchedulerAction
    purpose: WorkPurpose | None = None
    width: int = 1
    work_item_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    projection_revision: int = 0

    def validate(self) -> None:
        _require_text("decision_id", self.decision_id)
        _require_text("thread_id", self.thread_id)
        if self.width <= 0:
            raise ValueError("scheduler decision width must be positive")
        _require_non_negative("projection_revision", self.projection_revision)
        if any(not work_item_id for work_item_id in self.work_item_ids):
            raise ValueError("work_item_ids must not contain empty IDs")
        if self.work_item_ids and len(self.work_item_ids) != self.width:
            raise ValueError("work_item_ids must match allocated scheduler width")


@dataclass(frozen=True, slots=True)
class KnowledgeDelta:
    delta_id: str
    kind: str
    summary: str
    reference_ids: tuple[str, ...]
    causal_event_ids: tuple[str, ...] = ()
    thread_id: str | None = None

    def validate(self) -> None:
        _require_text("delta_id", self.delta_id)
        _require_text("kind", self.kind)
        _require_text("summary", self.summary)
        if not self.reference_ids:
            raise ValueError("knowledge deltas must retain at least one underlying reference")
        if any(not value for value in self.reference_ids):
            raise ValueError("reference_ids must not contain empty IDs")
        if self.thread_id is not None:
            _require_text("thread_id", self.thread_id)
