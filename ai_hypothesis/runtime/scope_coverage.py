"""Project source-region coverage from ordinary worker-attempt ledger history.

Coverage is derived state, not a separate durable subsystem. Work Items declare the
source regions they inspect; ATTEMPT_STARTED preserves that allocation, and terminal
attempt events determine whether an inspection resolved or aborted.

The projector deliberately does not decide that a region is "sufficiently covered".
It exposes observable counts so scheduler/domain policy can define that later.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .contracts import LedgerEvent


_TERMINAL_ATTEMPT_EVENTS = {
    "ATTEMPT_COMPLETED": "completed",
    "ATTEMPT_PARTIAL": "partial",
    "ATTEMPT_FAILED": "failed",
    "ATTEMPT_CRASHED": "crashed",
    "ATTEMPT_INVALID_RESULT": "invalid",
}


@dataclass(frozen=True, slots=True)
class ScopeRegionCoverage:
    region_id: str
    started_attempt_count: int
    completed_attempt_count: int
    partial_attempt_count: int
    failed_attempt_count: int
    crashed_attempt_count: int
    invalid_attempt_count: int
    worker_ids: tuple[str, ...]
    first_sequence: int
    last_sequence: int

    @property
    def resolved_attempt_count(self) -> int:
        """Attempts that produced a valid terminal worker result, including failure."""

        return (
            self.completed_attempt_count
            + self.partial_attempt_count
            + self.failed_attempt_count
        )

    @property
    def aborted_attempt_count(self) -> int:
        return self.crashed_attempt_count + self.invalid_attempt_count

    @property
    def distinct_worker_count(self) -> int:
        return len(self.worker_ids)

    @property
    def has_resolved_inspection(self) -> bool:
        return self.resolved_attempt_count > 0


@dataclass(frozen=True, slots=True)
class ThreadScopeCoverage:
    thread_id: str
    regions: tuple[ScopeRegionCoverage, ...]

    @property
    def attempted_region_ids(self) -> tuple[str, ...]:
        return tuple(region.region_id for region in self.regions)

    @property
    def resolved_region_ids(self) -> tuple[str, ...]:
        return tuple(
            region.region_id for region in self.regions if region.has_resolved_inspection
        )

    def missing_region_ids(
        self,
        expected_region_ids: Sequence[str],
        *,
        require_resolved: bool = True,
    ) -> tuple[str, ...]:
        expected = _validated_expected_regions(expected_region_ids)
        present = set(
            self.resolved_region_ids if require_resolved else self.attempted_region_ids
        )
        return tuple(region_id for region_id in expected if region_id not in present)

    def coverage_fraction(
        self,
        expected_region_ids: Sequence[str],
        *,
        require_resolved: bool = True,
    ) -> float:
        expected = _validated_expected_regions(expected_region_ids)
        if not expected:
            return 1.0
        missing = self.missing_region_ids(
            expected,
            require_resolved=require_resolved,
        )
        return (len(expected) - len(missing)) / len(expected)


@dataclass(slots=True)
class _MutableRegion:
    region_id: str
    first_sequence: int
    last_sequence: int
    started_attempt_count: int = 0
    completed_attempt_count: int = 0
    partial_attempt_count: int = 0
    failed_attempt_count: int = 0
    crashed_attempt_count: int = 0
    invalid_attempt_count: int = 0
    worker_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _AttemptScope:
    thread_id: str
    worker_id: str
    region_ids: tuple[str, ...]
    terminal_event_type: str | None = None


class ScopeCoverageProjector:
    """Derive per-thread region coverage from append-only attempt history."""

    def project(
        self,
        events: Iterable[LedgerEvent],
        *,
        thread_id: str | None = None,
    ) -> tuple[ThreadScopeCoverage, ...]:
        if thread_id is not None and (not thread_id or not thread_id.strip()):
            raise ValueError("thread_id must be non-empty when supplied")

        attempts: dict[str, _AttemptScope | None] = {}
        regions_by_thread: dict[str, dict[str, _MutableRegion]] = {}
        thread_order: list[str] = []
        previous_sequence = -1

        for event in events:
            event.validate()
            if event.sequence <= previous_sequence:
                raise ValueError(
                    "events must be supplied in strictly increasing sequence order"
                )
            previous_sequence = event.sequence

            if event.event_type == "ATTEMPT_STARTED":
                self._apply_started(
                    event,
                    attempts=attempts,
                    regions_by_thread=regions_by_thread,
                    thread_order=thread_order,
                )
                continue

            terminal_kind = _TERMINAL_ATTEMPT_EVENTS.get(event.event_type)
            if terminal_kind is None or event.attempt_id is None:
                continue
            attempt = attempts.get(event.attempt_id)
            if attempt is None:
                # Unscoped attempts and pre-coverage history remain valid but do not
                # participate in this derived projection.
                continue
            if attempt.terminal_event_type is not None:
                raise ValueError(
                    f"scoped attempt {event.attempt_id!r} has multiple terminal events"
                )
            if event.thread_id is not None and event.thread_id != attempt.thread_id:
                raise ValueError("scoped attempt terminal event changed thread identity")
            attempt.terminal_event_type = event.event_type

            for region_id in attempt.region_ids:
                region = regions_by_thread[attempt.thread_id][region_id]
                setattr(region, f"{terminal_kind}_attempt_count", getattr(region, f"{terminal_kind}_attempt_count") + 1)
                region.last_sequence = event.sequence

        snapshots: list[ThreadScopeCoverage] = []
        for current_thread_id in thread_order:
            if thread_id is not None and current_thread_id != thread_id:
                continue
            regions = tuple(
                self._freeze_region(region)
                for region in regions_by_thread[current_thread_id].values()
            )
            snapshots.append(ThreadScopeCoverage(current_thread_id, regions))
        return tuple(snapshots)

    def for_thread(
        self,
        events: Iterable[LedgerEvent],
        thread_id: str,
    ) -> ThreadScopeCoverage:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        snapshots = self.project(events, thread_id=thread_id)
        if snapshots:
            return snapshots[0]
        return ThreadScopeCoverage(thread_id=thread_id, regions=())

    @staticmethod
    def _apply_started(
        event: LedgerEvent,
        *,
        attempts: dict[str, _AttemptScope | None],
        regions_by_thread: dict[str, dict[str, _MutableRegion]],
        thread_order: list[str],
    ) -> None:
        if event.attempt_id is None:
            raise ValueError("ATTEMPT_STARTED requires attempt_id")
        if event.attempt_id in attempts:
            raise ValueError(f"attempt {event.attempt_id!r} was started more than once")

        raw_region_ids = event.payload.get("scope_region_ids")
        if raw_region_ids is None:
            attempts[event.attempt_id] = None
            return
        if not isinstance(raw_region_ids, list):
            raise ValueError("ATTEMPT_STARTED scope_region_ids must be a list")
        if not raw_region_ids:
            attempts[event.attempt_id] = None
            return
        if any(not isinstance(value, str) or not value.strip() for value in raw_region_ids):
            raise ValueError("scope_region_ids must contain non-empty strings")
        region_ids = tuple(raw_region_ids)
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("scope_region_ids must be unique inside one Work Item")
        if event.thread_id is None:
            raise ValueError("scoped ATTEMPT_STARTED requires thread_id")
        worker_id = event.payload.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("scoped ATTEMPT_STARTED requires worker_id")

        attempts[event.attempt_id] = _AttemptScope(
            thread_id=event.thread_id,
            worker_id=worker_id,
            region_ids=region_ids,
        )
        thread_regions = regions_by_thread.get(event.thread_id)
        if thread_regions is None:
            thread_regions = {}
            regions_by_thread[event.thread_id] = thread_regions
            thread_order.append(event.thread_id)

        for region_id in region_ids:
            region = thread_regions.get(region_id)
            if region is None:
                region = _MutableRegion(
                    region_id=region_id,
                    first_sequence=event.sequence,
                    last_sequence=event.sequence,
                )
                thread_regions[region_id] = region
            region.started_attempt_count += 1
            region.last_sequence = event.sequence
            if worker_id not in region.worker_ids:
                region.worker_ids.append(worker_id)

    @staticmethod
    def _freeze_region(region: _MutableRegion) -> ScopeRegionCoverage:
        return ScopeRegionCoverage(
            region_id=region.region_id,
            started_attempt_count=region.started_attempt_count,
            completed_attempt_count=region.completed_attempt_count,
            partial_attempt_count=region.partial_attempt_count,
            failed_attempt_count=region.failed_attempt_count,
            crashed_attempt_count=region.crashed_attempt_count,
            invalid_attempt_count=region.invalid_attempt_count,
            worker_ids=tuple(region.worker_ids),
            first_sequence=region.first_sequence,
            last_sequence=region.last_sequence,
        )


def _validated_expected_regions(values: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(values)
    if any(not value or not value.strip() for value in resolved):
        raise ValueError("expected_region_ids must contain non-empty IDs")
    if len(set(resolved)) != len(resolved):
        raise ValueError("expected_region_ids must be unique")
    return resolved
