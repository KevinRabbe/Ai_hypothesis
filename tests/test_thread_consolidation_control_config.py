from __future__ import annotations

import unittest

from ai_hypothesis.runtime import SchedulerSignals, SQLiteResearchLedger, WorkPreparation
from ai_hypothesis.runtime.thread_consolidation import (
    ThreadConsolidationConfig,
    ThreadConsolidationPlanner,
)
from ai_hypothesis.runtime.thread_consolidation_control import (
    ThreadConsolidationControlAdapter,
    ThreadConsolidationPressureConfig,
    ThreadConsolidationPressureProjector,
)


class ThreadConsolidationControlConfigTests(unittest.TestCase):
    def test_pressure_and_planner_readiness_thresholds_must_match(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        planner = ThreadConsolidationPlanner(
            ThreadConsolidationConfig(selection_limit=8, minimum_source_deltas=3)
        )
        pressure = ThreadConsolidationPressureProjector(
            ThreadConsolidationPressureConfig(
                full_pressure_count=8,
                minimum_source_deltas=2,
            )
        )

        with self.assertRaisesRegex(ValueError, "minimum_source_deltas must match"):
            ThreadConsolidationControlAdapter(
                ledger=ledger,
                signal_fallback=lambda state: SchedulerSignals(),
                context_fallback=lambda state, decision: WorkPreparation(),
                pressure_projector=pressure,
                planner=planner,
            )

    def test_default_pressure_inherits_custom_planner_minimum(self) -> None:
        ledger = SQLiteResearchLedger(":memory:")
        self.addCleanup(ledger.close)
        planner = ThreadConsolidationPlanner(
            ThreadConsolidationConfig(selection_limit=8, minimum_source_deltas=4)
        )
        control = ThreadConsolidationControlAdapter(
            ledger=ledger,
            signal_fallback=lambda state: SchedulerSignals(),
            context_fallback=lambda state, decision: WorkPreparation(),
            planner=planner,
        )

        self.assertEqual(
            control.pressure_projector.config.minimum_source_deltas,
            4,
        )
        self.assertGreaterEqual(
            control.pressure_projector.config.full_pressure_count,
            4,
        )


if __name__ == "__main__":
    unittest.main()
