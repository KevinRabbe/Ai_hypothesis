"""Read-only outcomes for backpressure integration allocations.

This projection joins durable scheduler decisions to their worker attempts and integration
outputs. It measures what an allocated width actually produced without inventing a reward
function or requiring additional runtime logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .contracts import LedgerEvent


_TERMINAL_ATTEMPT_TYPES = frozenset(
    {
        "ATTEMPT_COMPLETED",
        "ATTEMPT_PARTIAL",
        "ATTEMPT_FAILED",
        "ATTEMPT_CRASHED",
        "ATTEMPT_INVALID_RESULT",
    }
)


@dataclass(frozen=True, slots=True)
class IntegrationAttemptOutcome:
    attempt_id: str
    worker_id: str
    input_evidence_ids: tuple[str, ...]
    non_evidence_input_reference_count: int
    terminal_event_type: str | None
    progress_made: bool | None
    dispositioned_input_evidence_ids: tuple[str, ...]
    disposition_reference_count: int
    out_of_input_disposition_reference_count: int
    knowledge_delta_ids: tuple[str, ...]
    knowledge_referenced_input_evidence_ids: tuple[str, ...]

    @property
    def terminal(self) -> bool:
        return self.terminal_event_type is not None


@dataclass(frozen=True, slots=True)
class IntegrationAllocationOutcome:
    decision_id: str
    thread_id: str
    width: int
    projection_revision: int
    reason_codes: tuple[str, ...]
    attempts: tuple[IntegrationAttemptOutcome, ...]

    @property
    def partitioned(self) -> bool:
        return "PARTITIONED_INTEGRATION" in self.reason_codes

    @property
    def started_attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def terminal_attempt_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.terminal)

    @property
    def width_utilization(self) -> float:
        return self.started_attempt_count / self.width

    @property
    def input_reference_count(self) -> int:
        return sum(len(attempt.input_evidence_ids) for attempt in self.attempts)

    @property
    def unique_input_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                evidence_id
                for attempt in self.attempts
                for evidence_id in attempt.input_evidence_ids
            )
        )

    @property
    def unique_input_evidence_count(self) -> int:
        return len(self.unique_input_evidence_ids)

    @property
    def duplicate_input_authority_count(self) -> int:
        return self.input_reference_count - self.unique_input_evidence_count

    @property
    def disposition_reference_count(self) -> int:
        return sum(attempt.disposition_reference_count for attempt in self.attempts)

    @property
    def unique_dispositioned_input_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                evidence_id
                for attempt in self.attempts
                for evidence_id in attempt.dispositioned_input_evidence_ids
            )
        )

    @property
    def unique_dispositioned_input_evidence_count(self) -> int:
        return len(self.unique_dispositioned_input_evidence_ids)

    @property
    def duplicate_disposition_reference_count(self) -> int:
        return max(
            0,
            self.disposition_reference_count
            - self.unique_dispositioned_input_evidence_count,
        )

    @property
    def input_absorption_fraction(self) -> float | None:
        if self.unique_input_evidence_count == 0:
            return None
        return (
            self.unique_dispositioned_input_evidence_count
            / self.unique_input_evidence_count
        )

    @property
    def knowledge_delta_count(self) -> int:
        return sum(len(attempt.knowledge_delta_ids) for attempt in self.attempts)

    @property
    def knowledge_referenced_input_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                evidence_id
                for attempt in self.attempts
                for evidence_id in attempt.knowledge_referenced_input_evidence_ids
            )
        )

    @property
    def knowledge_referenced_input_evidence_count(self) -> int:
        return len(self.knowledge_referenced_input_evidence_ids)


@dataclass(frozen=True, slots=True)
class IntegrationWidthSummary:
    width: int
    allocation_count: int
    partitioned_allocation_count: int
    started_attempt_count: int
    terminal_attempt_count: int
    unique_input_evidence_total: int
    duplicate_input_authority_total: int
    unique_dispositioned_input_evidence_total: int
    disposition_reference_total: int
    duplicate_disposition_reference_total: int
    knowledge_delta_total: int
    knowledge_referenced_input_evidence_total: int
    mean_input_absorption_fraction: float | None

    @property
    def terminal_attempt_fraction(self) -> float | None:
        if self.started_attempt_count == 0:
            return None
        return self.terminal_attempt_count / self.started_attempt_count


@dataclass(slots=True)
class _MutableAttempt:
    attempt_id: str
    worker_id: str
    start_sequence: int
    input_evidence_ids: tuple[str, ...]
    non_evidence_input_reference_count: int
    terminal_event_type: str | None = None
    terminal_sequence: int | None = None
    progress_made: bool | None = None
    dispositioned_input_evidence_ids: list[str] = field(default_factory=list)
    disposition_reference_count: int = 0
    out_of_input_disposition_reference_count: int = 0
    knowledge_delta_ids: list[str] = field(default_factory=list)
    knowledge_referenced_input_evidence_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _DecisionRecord:
    decision_id: str
    thread_id: str
    width: int
    projection_revision: int
    reason_codes: tuple[str, ...]
    sequence: int


class IntegrationAllocationOutcomeProjector:
    """Join traced integration decisions to attempts, dispositions and knowledge deltas."""

    def project(
        self,
        events: Sequence[LedgerEvent],
    ) -> tuple[IntegrationAllocationOutcome, ...]:
        evidence_sequences, decisions = self._discover_inputs(events)
        attempts, decision_attempt_order = self._index_attempt_starts(
            events,
            evidence_sequences=evidence_sequences,
            decisions=decisions,
        )
        self._fold_attempt_outputs(events, attempts)

        outcomes: list[IntegrationAllocationOutcome] = []
        for decision in sorted(decisions.values(), key=lambda item: item.sequence):
            frozen_attempts = tuple(
                self._freeze_attempt(attempts[attempt_id])
                for attempt_id in decision_attempt_order[decision.decision_id]
            )
            outcomes.append(
                IntegrationAllocationOutcome(
                    decision_id=decision.decision_id,
                    thread_id=decision.thread_id,
                    width=decision.width,
                    projection_revision=decision.projection_revision,
                    reason_codes=decision.reason_codes,
                    attempts=frozen_attempts,
                )
            )
        return tuple(outcomes)

    def _discover_inputs(
        self,
        events: Sequence[LedgerEvent],
    ) -> tuple[dict[str, int], dict[str, _DecisionRecord]]:
        previous_sequence = -1
        evidence_sequences: dict[str, int] = {}
        decisions: dict[str, _DecisionRecord] = {}

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be in strictly increasing sequence order")
            previous_sequence = event.sequence

            if event.event_type == "EVIDENCE_ADDED":
                evidence_id = event.payload.get("evidence_id")
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise ValueError("EVIDENCE_ADDED is missing evidence_id")
                if evidence_id in evidence_sequences:
                    raise ValueError(f"duplicate durable evidence ID {evidence_id!r}")
                evidence_sequences[evidence_id] = event.sequence
            elif event.event_type == "SCHEDULER_DECISION_RECORDED":
                decision = self._integration_decision(event)
                if decision is None:
                    continue
                if decision.decision_id in decisions:
                    raise ValueError("scheduler decision ID was recorded more than once")
                decisions[decision.decision_id] = decision

        return evidence_sequences, decisions

    def _index_attempt_starts(
        self,
        events: Sequence[LedgerEvent],
        *,
        evidence_sequences: dict[str, int],
        decisions: dict[str, _DecisionRecord],
    ) -> tuple[dict[str, _MutableAttempt], dict[str, list[str]]]:
        attempts: dict[str, _MutableAttempt] = {}
        decision_attempt_order: dict[str, list[str]] = {
            decision_id: [] for decision_id in decisions
        }

        for event in events:
            if event.event_type != "ATTEMPT_STARTED":
                continue
            decision_id = event.payload.get("scheduler_decision_id")
            if not isinstance(decision_id, str) or decision_id not in decisions:
                continue
            if event.attempt_id is None:
                raise ValueError("traced integration ATTEMPT_STARTED is missing attempt_id")
            if event.attempt_id in attempts:
                raise ValueError("integration attempt ID was started more than once")

            decision = decisions[decision_id]
            if event.sequence <= decision.sequence:
                raise ValueError("integration ATTEMPT_STARTED precedes scheduler decision")
            if event.thread_id != decision.thread_id:
                raise ValueError("integration attempt thread does not match scheduler decision")

            worker_id = event.payload.get("worker_id")
            if not isinstance(worker_id, str) or not worker_id:
                raise ValueError("integration ATTEMPT_STARTED is missing worker_id")

            input_evidence: list[str] = []
            non_evidence_count = 0
            for reference_id in event.reference_ids:
                evidence_sequence = evidence_sequences.get(reference_id)
                if evidence_sequence is None:
                    non_evidence_count += 1
                    continue
                if evidence_sequence >= event.sequence:
                    raise ValueError(
                        "integration ATTEMPT_STARTED references evidence created after attempt start"
                    )
                input_evidence.append(reference_id)

            attempts[event.attempt_id] = _MutableAttempt(
                attempt_id=event.attempt_id,
                worker_id=worker_id,
                start_sequence=event.sequence,
                input_evidence_ids=tuple(input_evidence),
                non_evidence_input_reference_count=non_evidence_count,
            )
            decision_attempt_order[decision_id].append(event.attempt_id)
            if len(decision_attempt_order[decision_id]) > decision.width:
                raise ValueError(
                    "integration scheduler decision started more attempts than allocated width"
                )

        return attempts, decision_attempt_order

    def _fold_attempt_outputs(
        self,
        events: Sequence[LedgerEvent],
        attempts: dict[str, _MutableAttempt],
    ) -> None:
        for event in events:
            attempt_id = event.attempt_id
            if attempt_id is None or attempt_id not in attempts:
                continue
            if event.event_type == "ATTEMPT_STARTED":
                continue

            attempt = attempts[attempt_id]
            if event.sequence <= attempt.start_sequence:
                raise ValueError("integration attempt output precedes ATTEMPT_STARTED")

            is_relevant_output = (
                event.event_type == "INTEGRATION_DISPOSITION_RECORDED"
                or event.event_type == "KNOWLEDGE_DELTA_RECORDED"
                or event.event_type in _TERMINAL_ATTEMPT_TYPES
            )
            if not is_relevant_output:
                continue

            if attempt.terminal_sequence is not None:
                if event.event_type in _TERMINAL_ATTEMPT_TYPES:
                    raise ValueError("integration attempt has more than one terminal event")
                raise ValueError("integration attempt produced output after terminal event")

            input_set = set(attempt.input_evidence_ids)
            if event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
                for reference_id in event.reference_ids:
                    if reference_id in input_set:
                        attempt.disposition_reference_count += 1
                        attempt.dispositioned_input_evidence_ids.append(reference_id)
                    else:
                        attempt.out_of_input_disposition_reference_count += 1
            elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
                delta_id = event.payload.get("delta_id")
                if not isinstance(delta_id, str) or not delta_id:
                    raise ValueError("KNOWLEDGE_DELTA_RECORDED is missing delta_id")
                attempt.knowledge_delta_ids.append(delta_id)
                for source_id in self._knowledge_source_ids(event):
                    if source_id in input_set:
                        attempt.knowledge_referenced_input_evidence_ids.append(source_id)
            else:
                attempt.terminal_event_type = event.event_type
                attempt.terminal_sequence = event.sequence
                raw_progress = event.payload.get("progress_made")
                attempt.progress_made = raw_progress if isinstance(raw_progress, bool) else None

    @staticmethod
    def _integration_decision(event: LedgerEvent) -> _DecisionRecord | None:
        action = event.payload.get("action")
        raw_reasons = event.payload.get("reason_codes")
        if action != "SYNTHESIZE" or not isinstance(raw_reasons, list):
            return None
        if any(not isinstance(reason, str) for reason in raw_reasons):
            raise ValueError("scheduler reason_codes must be a string list")
        reasons = tuple(raw_reasons)
        if "BACKPRESSURE" not in reasons:
            return None

        decision_id = event.payload.get("decision_id")
        width = event.payload.get("width")
        projection_revision = event.payload.get("projection_revision")
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("integration scheduler trace is missing decision_id")
        if event.thread_id is None:
            raise ValueError("integration scheduler trace is missing thread_id")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("integration scheduler trace has invalid width")
        if (
            isinstance(projection_revision, bool)
            or not isinstance(projection_revision, int)
            or projection_revision < 0
        ):
            raise ValueError("integration scheduler trace has invalid projection_revision")
        return _DecisionRecord(
            decision_id=decision_id,
            thread_id=event.thread_id,
            width=width,
            projection_revision=projection_revision,
            reason_codes=reasons,
            sequence=event.sequence,
        )

    @staticmethod
    def _knowledge_source_ids(event: LedgerEvent) -> tuple[str, ...]:
        raw_sources = event.payload.get("source_reference_ids")
        if isinstance(raw_sources, list) and all(
            isinstance(value, str) for value in raw_sources
        ):
            return tuple(raw_sources)
        delta_id = event.payload.get("delta_id")
        return tuple(
            reference_id
            for reference_id in event.reference_ids
            if reference_id != delta_id
        )

    @staticmethod
    def _freeze_attempt(attempt: _MutableAttempt) -> IntegrationAttemptOutcome:
        return IntegrationAttemptOutcome(
            attempt_id=attempt.attempt_id,
            worker_id=attempt.worker_id,
            input_evidence_ids=attempt.input_evidence_ids,
            non_evidence_input_reference_count=attempt.non_evidence_input_reference_count,
            terminal_event_type=attempt.terminal_event_type,
            progress_made=attempt.progress_made,
            dispositioned_input_evidence_ids=tuple(attempt.dispositioned_input_evidence_ids),
            disposition_reference_count=attempt.disposition_reference_count,
            out_of_input_disposition_reference_count=(
                attempt.out_of_input_disposition_reference_count
            ),
            knowledge_delta_ids=tuple(attempt.knowledge_delta_ids),
            knowledge_referenced_input_evidence_ids=tuple(
                attempt.knowledge_referenced_input_evidence_ids
            ),
        )


def summarize_integration_allocations_by_width(
    outcomes: Sequence[IntegrationAllocationOutcome],
) -> tuple[IntegrationWidthSummary, ...]:
    grouped: dict[int, list[IntegrationAllocationOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.width, []).append(outcome)

    summaries: list[IntegrationWidthSummary] = []
    for width in sorted(grouped):
        allocations = grouped[width]
        absorption_values = [
            value
            for outcome in allocations
            if (value := outcome.input_absorption_fraction) is not None
        ]
        summaries.append(
            IntegrationWidthSummary(
                width=width,
                allocation_count=len(allocations),
                partitioned_allocation_count=sum(
                    1 for outcome in allocations if outcome.partitioned
                ),
                started_attempt_count=sum(
                    outcome.started_attempt_count for outcome in allocations
                ),
                terminal_attempt_count=sum(
                    outcome.terminal_attempt_count for outcome in allocations
                ),
                unique_input_evidence_total=sum(
                    outcome.unique_input_evidence_count for outcome in allocations
                ),
                duplicate_input_authority_total=sum(
                    outcome.duplicate_input_authority_count for outcome in allocations
                ),
                unique_dispositioned_input_evidence_total=sum(
                    outcome.unique_dispositioned_input_evidence_count
                    for outcome in allocations
                ),
                disposition_reference_total=sum(
                    outcome.disposition_reference_count for outcome in allocations
                ),
                duplicate_disposition_reference_total=sum(
                    outcome.duplicate_disposition_reference_count
                    for outcome in allocations
                ),
                knowledge_delta_total=sum(
                    outcome.knowledge_delta_count for outcome in allocations
                ),
                knowledge_referenced_input_evidence_total=sum(
                    outcome.knowledge_referenced_input_evidence_count
                    for outcome in allocations
                ),
                mean_input_absorption_fraction=(
                    sum(absorption_values) / len(absorption_values)
                    if absorption_values
                    else None
                ),
            )
        )
    return tuple(summaries)
