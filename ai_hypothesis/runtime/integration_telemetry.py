"""Read-only knowledge-integration telemetry over Research Ledger history.

This module measures information-flow state without changing integration policy. Durable
replay uses ledger sequence distance for latency because LedgerEvent intentionally has no
wall-clock timestamp. Real rates are derived only when a caller supplies an observed
wall-clock interval between two snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import EvidenceDispositionKind, LedgerEvent


@dataclass(frozen=True, slots=True)
class IntegrationTelemetrySnapshot:
    revision: int
    evidence_count: int
    dispositioned_evidence_count: int
    disposition_reference_count: int
    backlog_count: int
    knowledge_delta_count: int
    knowledge_referenced_evidence_count: int
    knowledge_source_reference_count: int
    disposition_counts: Mapping[str, int]
    redisposition_count: int
    unknown_disposition_reference_count: int
    unknown_knowledge_reference_count: int
    mean_disposition_latency_sequences: float | None
    max_disposition_latency_sequences: int | None
    mean_backlog_age_sequences: float | None
    oldest_backlog_age_sequences: int

    @property
    def disposition_fraction(self) -> float:
        return (
            self.dispositioned_evidence_count / self.evidence_count
            if self.evidence_count
            else 0.0
        )

    @property
    def knowledge_reference_fraction(self) -> float:
        return (
            self.knowledge_referenced_evidence_count / self.evidence_count
            if self.evidence_count
            else 0.0
        )

    @property
    def evidence_per_knowledge_delta(self) -> float | None:
        if self.knowledge_delta_count <= 0:
            return None
        return self.knowledge_source_reference_count / self.knowledge_delta_count


@dataclass(frozen=True, slots=True)
class IntegrationBandwidthWindow:
    """Observed wall-clock rates between two durable telemetry snapshots."""

    elapsed_seconds: float
    evidence_generated: int
    disposition_references_recorded: int
    knowledge_deltas_recorded: int
    backlog_delta: int
    evidence_per_second: float
    disposition_references_per_second: float
    knowledge_deltas_per_second: float
    backlog_growth_per_second: float
    absorption_ratio: float | None

    @classmethod
    def between(
        cls,
        previous: IntegrationTelemetrySnapshot,
        current: IntegrationTelemetrySnapshot,
        *,
        elapsed_seconds: float,
    ) -> "IntegrationBandwidthWindow":
        if elapsed_seconds <= 0.0:
            raise ValueError("elapsed_seconds must be positive")
        if current.revision < previous.revision:
            raise ValueError("current telemetry revision cannot precede previous revision")

        evidence_generated = current.evidence_count - previous.evidence_count
        disposition_references = (
            current.disposition_reference_count - previous.disposition_reference_count
        )
        knowledge_deltas = current.knowledge_delta_count - previous.knowledge_delta_count
        for name, value in (
            ("evidence_count", evidence_generated),
            ("disposition_reference_count", disposition_references),
            ("knowledge_delta_count", knowledge_deltas),
        ):
            if value < 0:
                raise ValueError(f"{name} regressed between telemetry snapshots")

        backlog_delta = current.backlog_count - previous.backlog_count
        absorption_ratio = (
            disposition_references / evidence_generated
            if evidence_generated > 0
            else None
        )
        return cls(
            elapsed_seconds=elapsed_seconds,
            evidence_generated=evidence_generated,
            disposition_references_recorded=disposition_references,
            knowledge_deltas_recorded=knowledge_deltas,
            backlog_delta=backlog_delta,
            evidence_per_second=evidence_generated / elapsed_seconds,
            disposition_references_per_second=disposition_references / elapsed_seconds,
            knowledge_deltas_per_second=knowledge_deltas / elapsed_seconds,
            backlog_growth_per_second=backlog_delta / elapsed_seconds,
            absorption_ratio=absorption_ratio,
        )


class IntegrationTelemetryProjector:
    """Project policy-free integration-flow diagnostics from append-only events."""

    def project(
        self,
        events: Sequence[LedgerEvent],
        *,
        thread_id: str | None = None,
    ) -> IntegrationTelemetrySnapshot:
        if thread_id is not None and (not thread_id or not thread_id.strip()):
            raise ValueError("thread_id must be non-empty when supplied")

        evidence_sequences: dict[str, int] = {}
        evidence_threads: dict[str, str | None] = {}
        disposition_events: list[tuple[int, EvidenceDispositionKind, tuple[str, ...]]] = []
        knowledge_events: list[LedgerEvent] = []
        revision = 0
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError("events must be in strictly increasing sequence order")
            previous_sequence = event.sequence
            revision = event.sequence

            if event.event_type == "EVIDENCE_ADDED":
                evidence_id = event.payload.get("evidence_id")
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise ValueError("EVIDENCE_ADDED is missing evidence_id")
                if evidence_id in evidence_sequences:
                    raise ValueError(f"duplicate durable evidence ID {evidence_id!r}")
                evidence_sequences[evidence_id] = event.sequence
                evidence_threads[evidence_id] = event.thread_id
            elif event.event_type == "INTEGRATION_DISPOSITION_RECORDED":
                raw = event.payload.get("disposition")
                try:
                    disposition = EvidenceDispositionKind(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError("invalid durable evidence disposition") from error
                disposition_events.append(
                    (event.sequence, disposition, tuple(event.reference_ids))
                )
            elif event.event_type == "KNOWLEDGE_DELTA_RECORDED":
                knowledge_events.append(event)

        selected_evidence = {
            evidence_id
            for evidence_id, source_thread_id in evidence_threads.items()
            if thread_id is None or source_thread_id == thread_id
        }

        dispositions_by_evidence: dict[
            str, list[tuple[int, EvidenceDispositionKind]]
        ] = {evidence_id: [] for evidence_id in selected_evidence}
        unknown_disposition_reference_count = 0
        disposition_reference_count = 0
        for sequence, disposition, reference_ids in disposition_events:
            for evidence_id in reference_ids:
                if evidence_id not in evidence_sequences:
                    if thread_id is None:
                        unknown_disposition_reference_count += 1
                    continue
                if evidence_id not in selected_evidence:
                    continue
                if sequence < evidence_sequences[evidence_id]:
                    raise ValueError("evidence disposition precedes evidence creation")
                disposition_reference_count += 1
                dispositions_by_evidence[evidence_id].append((sequence, disposition))

        disposition_counts = {kind.value: 0 for kind in EvidenceDispositionKind}
        disposition_latencies: list[int] = []
        backlog_ages: list[int] = []
        redisposition_count = 0
        dispositioned_evidence_count = 0
        for evidence_id in selected_evidence:
            records = dispositions_by_evidence[evidence_id]
            if not records:
                backlog_ages.append(revision - evidence_sequences[evidence_id])
                continue
            dispositioned_evidence_count += 1
            records.sort(key=lambda value: value[0])
            first_sequence, first_disposition = records[0]
            disposition_counts[first_disposition.value] += 1
            disposition_latencies.append(first_sequence - evidence_sequences[evidence_id])
            redisposition_count += max(0, len(records) - 1)

        knowledge_delta_count = 0
        knowledge_referenced_ids: set[str] = set()
        knowledge_source_reference_count = 0
        unknown_knowledge_reference_count = 0
        for event in knowledge_events:
            if thread_id is not None and event.thread_id != thread_id:
                # Deltas targeting another thread do not count toward this thread's delta
                # production, but their use of this thread's evidence still matters below.
                pass
            else:
                knowledge_delta_count += 1

            raw_sources = event.payload.get("source_reference_ids")
            if isinstance(raw_sources, list) and all(
                isinstance(value, str) for value in raw_sources
            ):
                source_ids = tuple(raw_sources)
            else:
                delta_id = event.payload.get("delta_id")
                source_ids = tuple(
                    reference_id
                    for reference_id in event.reference_ids
                    if reference_id != delta_id
                )

            for reference_id in source_ids:
                if reference_id not in evidence_sequences:
                    if thread_id is None:
                        unknown_knowledge_reference_count += 1
                    continue
                if reference_id not in selected_evidence:
                    continue
                knowledge_source_reference_count += 1
                knowledge_referenced_ids.add(reference_id)

        return IntegrationTelemetrySnapshot(
            revision=revision,
            evidence_count=len(selected_evidence),
            dispositioned_evidence_count=dispositioned_evidence_count,
            disposition_reference_count=disposition_reference_count,
            backlog_count=len(backlog_ages),
            knowledge_delta_count=knowledge_delta_count,
            knowledge_referenced_evidence_count=len(knowledge_referenced_ids),
            knowledge_source_reference_count=knowledge_source_reference_count,
            disposition_counts=disposition_counts,
            redisposition_count=redisposition_count,
            unknown_disposition_reference_count=unknown_disposition_reference_count,
            unknown_knowledge_reference_count=unknown_knowledge_reference_count,
            mean_disposition_latency_sequences=(
                sum(disposition_latencies) / len(disposition_latencies)
                if disposition_latencies
                else None
            ),
            max_disposition_latency_sequences=(
                max(disposition_latencies) if disposition_latencies else None
            ),
            mean_backlog_age_sequences=(
                sum(backlog_ages) / len(backlog_ages) if backlog_ages else None
            ),
            oldest_backlog_age_sequences=max(backlog_ages) if backlog_ages else 0,
        )
