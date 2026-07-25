"""Tests corruption rejection in persistent large-scope projections."""

from __future__ import annotations

import unittest

from ai_hypothesis.large_scope import (
    PersistentScopeEvaluationProjector,
    ScopeWorkerMode,
    generate_large_scope_relevance,
    large_scope_region_id,
)
from ai_hypothesis.runtime import LedgerEvent


def _event(
    sequence: int,
    event_type: str,
    *,
    thread_id: str = "thread-1",
    attempt_id: str | None = None,
    payload: dict | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=f"event-{sequence}",
        event_type=event_type,
        sequence=sequence,
        payload_schema="v1",
        thread_id=thread_id,
        attempt_id=attempt_id,
        payload=payload or {},
    )


class PersistentScopeIntegrityTests(unittest.TestCase):
    def test_rejects_budget_matched_but_mispartitioned_attempt_history(self) -> None:
        sample = generate_large_scope_relevance(92)
        regions = tuple(large_scope_region_id(sample, index) for index in range(4))
        events = (
            _event(
                1,
                "SCHEDULER_DECISION_RECORDED",
                payload={"decision_id": "decision-a", "width": 2},
            ),
            _event(
                2,
                "SCHEDULER_DECISION_RECORDED",
                payload={"decision_id": "decision-b", "width": 2},
            ),
            _event(
                3,
                "ATTEMPT_STARTED",
                attempt_id="a1",
                payload={
                    "worker_id": "worker-1",
                    "scheduler_decision_id": "decision-a",
                    "scope_region_ids": [regions[0]],
                },
            ),
            _event(
                4,
                "ATTEMPT_STARTED",
                attempt_id="a2",
                payload={
                    "worker_id": "worker-2",
                    "scheduler_decision_id": "decision-a",
                    "scope_region_ids": [regions[1]],
                },
            ),
            _event(
                5,
                "ATTEMPT_STARTED",
                attempt_id="a3",
                payload={
                    "worker_id": "worker-3",
                    "scheduler_decision_id": "decision-a",
                    "scope_region_ids": [regions[2]],
                },
            ),
            _event(
                6,
                "ATTEMPT_STARTED",
                attempt_id="b1",
                payload={
                    "worker_id": "worker-4",
                    "scheduler_decision_id": "decision-b",
                    "scope_region_ids": [regions[3]],
                },
            ),
        )

        projector = PersistentScopeEvaluationProjector(
            sample,
            ScopeWorkerMode.DIVERSE_WORKERS,
            expected_worker_bank_id="worker-bank-sha256-test",
        )
        with self.assertRaisesRegex(
            ValueError,
            "does not own exactly step_width attempts",
        ):
            projector.project(events, thread_id="thread-1", step_width=2)

    def test_rejects_attempt_linked_to_unknown_scheduler_decision(self) -> None:
        sample = generate_large_scope_relevance(94)
        region = large_scope_region_id(sample, 0)
        events = (
            _event(
                1,
                "SCHEDULER_DECISION_RECORDED",
                payload={"decision_id": "decision-a", "width": 1},
            ),
            _event(
                2,
                "ATTEMPT_STARTED",
                attempt_id="a1",
                payload={
                    "worker_id": "worker-1",
                    "scheduler_decision_id": "missing-decision",
                    "scope_region_ids": [region],
                },
            ),
        )
        projector = PersistentScopeEvaluationProjector(
            sample,
            ScopeWorkerMode.DIVERSE_WORKERS,
            expected_worker_bank_id="worker-bank-sha256-test",
        )
        with self.assertRaisesRegex(ValueError, "unknown scheduler decision"):
            projector.project(events, thread_id="thread-1", step_width=1)


if __name__ == "__main__":
    unittest.main()
