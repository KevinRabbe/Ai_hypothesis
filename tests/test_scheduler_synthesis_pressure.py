from __future__ import annotations

import random
import unittest

from ai_hypothesis.runtime import (
    ProjectedState,
    SchedulerAction,
    SchedulerConfig,
    SchedulerSignals,
    SchedulerV0,
    SchedulableThread,
    WorkPurpose,
)


def _candidate(thread_id: str, **signals: float) -> SchedulableThread:
    return SchedulableThread(
        state=ProjectedState(
            revision=1,
            thread_id=thread_id,
            objective=f"objective {thread_id}",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        ),
        signals=SchedulerSignals(recent_progress=1.0, **signals),
    )


class SchedulerSynthesisPressureTests(unittest.TestCase):
    def test_synthesis_need_produces_generic_synthesis_decision(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(exploration_probability=0.0, synthesis_threshold=0.6)
        )
        decision = scheduler.choose(
            (_candidate("thread-a", importance=1.0, synthesis_need=0.9),)
        )

        self.assertEqual(decision.action, SchedulerAction.SYNTHESIZE)
        self.assertEqual(decision.purpose, WorkPurpose.SYNTHESIZE)
        self.assertEqual(decision.width, 1)
        self.assertEqual(decision.reason_codes, ("SYNTHESIS_NEEDED",))

    def test_synthesis_need_participates_in_thread_priority(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(exploration_probability=0.0, synthesis_threshold=0.6)
        )
        decision = scheduler.choose(
            (
                _candidate("ordinary", importance=0.6),
                _candidate("synthesis", importance=0.3, synthesis_need=1.0),
            )
        )

        self.assertEqual(decision.thread_id, "synthesis")
        self.assertEqual(decision.action, SchedulerAction.SYNTHESIZE)

    def test_contradiction_challenge_precedes_generic_synthesis(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(
                exploration_probability=0.0,
                challenge_threshold=0.6,
                synthesis_threshold=0.6,
            )
        )
        decision = scheduler.choose(
            (
                _candidate(
                    "thread-a",
                    importance=1.0,
                    contradiction_severity=0.9,
                    synthesis_need=1.0,
                ),
            )
        )

        self.assertEqual(decision.action, SchedulerAction.CHALLENGE)
        self.assertEqual(decision.purpose, WorkPurpose.CHALLENGE)

    def test_verification_precedes_generic_synthesis(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(
                exploration_probability=0.0,
                verification_threshold=0.6,
                synthesis_threshold=0.6,
            )
        )
        decision = scheduler.choose(
            (
                _candidate(
                    "thread-a",
                    importance=1.0,
                    verification_need=0.9,
                    synthesis_need=1.0,
                ),
            )
        )

        self.assertEqual(decision.action, SchedulerAction.VERIFY)
        self.assertEqual(decision.purpose, WorkPurpose.VERIFY)

    def test_raw_integration_backpressure_precedes_generic_synthesis(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(
                exploration_probability=0.0,
                synthesis_threshold=0.6,
                backpressure_exploration_probability=0.01,
            ),
            rng=random.Random(7),
        )
        decision = scheduler.choose(
            (
                _candidate(
                    "thread-a",
                    importance=1.0,
                    integration_backlog=1.0,
                    synthesis_need=1.0,
                ),
            ),
            integration_backpressure=True,
        )

        self.assertEqual(decision.action, SchedulerAction.SYNTHESIZE)
        self.assertIn("BACKPRESSURE", decision.reason_codes)
        self.assertNotIn("SYNTHESIS_NEEDED", decision.reason_codes)

    def test_structured_exploration_still_precedes_generic_synthesis_when_drawn(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(exploration_probability=1.0, synthesis_threshold=0.0),
            rng=random.Random(7),
        )
        decision = scheduler.choose(
            (_candidate("thread-a", synthesis_need=1.0, missing_coverage=1.0),),
            max_width=4,
        )

        self.assertEqual(decision.action, SchedulerAction.ADD_WIDTH)
        self.assertEqual(decision.purpose, WorkPurpose.EXPLORE)
        self.assertIn("STRUCTURED_EXPLORATION", decision.reason_codes)

    def test_synthesis_fields_are_appended_to_positional_contracts(self) -> None:
        signals = SchedulerSignals(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2)
        self.assertEqual(signals.estimated_cost, 0.2)
        self.assertEqual(signals.synthesis_need, 0.0)

        config = SchedulerConfig(0.1, 3, 0.6, 0.7, 0.02, 0.05)
        self.assertEqual(config.backpressure_exploration_probability, 0.05)
        self.assertEqual(config.synthesis_threshold, 0.65)

    def test_synthesis_values_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "synthesis_need"):
            SchedulerSignals(synthesis_need=1.1).validate()
        with self.assertRaisesRegex(ValueError, "synthesis_threshold"):
            SchedulerV0(SchedulerConfig(synthesis_threshold=-0.1))


if __name__ == "__main__":
    unittest.main()
