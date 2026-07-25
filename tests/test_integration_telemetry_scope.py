from __future__ import annotations

import unittest

from ai_hypothesis.runtime.contracts import LedgerEvent
from ai_hypothesis.runtime.integration_telemetry import (
    IntegrationBandwidthWindow,
    IntegrationTelemetryProjector,
)


class IntegrationTelemetryScopeTests(unittest.TestCase):
    def test_bandwidth_window_rejects_cross_scope_comparison(self) -> None:
        events = (
            LedgerEvent(
                event_id="event-1",
                event_type="EVIDENCE_ADDED",
                sequence=1,
                payload_schema="runtime-event-v0",
                thread_id="thread-a",
                reference_ids=("evidence-a",),
                payload={"evidence_id": "evidence-a"},
            ),
        )
        projector = IntegrationTelemetryProjector()
        global_snapshot = projector.project(events)
        thread_snapshot = projector.project(events, thread_id="thread-a")

        self.assertIsNone(global_snapshot.scope_thread_id)
        self.assertEqual(thread_snapshot.scope_thread_id, "thread-a")
        with self.assertRaisesRegex(ValueError, "same thread scope"):
            IntegrationBandwidthWindow.between(
                global_snapshot,
                thread_snapshot,
                elapsed_seconds=1.0,
            )


if __name__ == "__main__":
    unittest.main()
