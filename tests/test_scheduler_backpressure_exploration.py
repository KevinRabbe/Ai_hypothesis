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


def _candidate(
    thread_id: str,
    *,
    integration_backlog: float = 0.0,
    missing_coverage: float = 0.0,
    verification_need: float = 0.0,
) -> SchedulableThread:
    return SchedulableThread(
        state=ProjectedState(
            revision=1,
            thread_id=thread_id,
            objective=f"objective {thread_id}",
            status="ACTIVE",
            purpose=WorkPurpose.PROGRESS,
        ),
        signals=SchedulerSignals(
            importance=1.0,
            recent_progress=1.0,
            integration_backlog=integration_backlog,
            missing_coverage=missing_coverage,
            verification_need=verification_need,
        ),
    )


class SchedulerBackpressureExplorationTests(unittest.TestCase):
    def test_backpressure_normally_services_integration(self) -> None:
        scheduler = SchedulerV0(rng=random.Random(7))
        decision = scheduler.choose(
            (_candidate("thread-a", integration_backlog=1.0),),
            integration_backpressure=True,
            max_width=4,
        )

        self.assertEqual(decision.action, SchedulerAction.SYNTHESIZE)
        self.assertEqual(decision.purpose, WorkPurpose.SYNTHESIZE)
        self.assertEqual(decision.width, 1)
        self.assertIn("BACKPRESSURE", decision.reason_codes)
        self.assertNotIn("BACKPRESSURE_EXPLORATION", decision.reason_codes)

    def test_backpressure_keeps_a_structured_exploration_lane(self) -> None:
        scheduler = SchedulerV0(rng=random.Random(31))
        decision = scheduler.choose(
            (_candidate("thread-a", integration_backlog=1.0, missing_coverage=1.0),),
            integration_backpressure=True,
            max_width=4,
        )

        self.assertEqual(decision.action, SchedulerAction.ADD_WIDTH)
        self.assertEqual(decision.purpose, WorkPurpose.EXPLORE)
        self.assertEqual(decision.width, 2)
        self.assertIn("STRUCTURED_EXPLORATION", decision.reason_codes)
        self.assertIn("BACKPRESSURE_EXPLORATION", decision.reason_codes)

    def test_backpressure_verification_still_dominates_when_servicing_backlog(self) -> None:
        scheduler = SchedulerV0(rng=random.Random(7))
        decision = scheduler.choose(
            (
                _candidate(
                    "thread-a",
                    integration_backlog=1.0,
                    verification_need=1.0,
                ),
            ),
            integration_backpressure=True,
        )

        self.assertEqual(decision.action, SchedulerAction.VERIFY)
        self.assertEqual(decision.purpose, WorkPurpose.VERIFY)
        self.assertIn("BACKPRESSURE", decision.reason_codes)
        self.assertIn("VERIFY", decision.reason_codes)

    def test_backpressure_probability_cannot_be_zero(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "backpressure_exploration_probability",
        ):
            SchedulerV0(
                SchedulerConfig(backpressure_exploration_probability=0.0)
            )

    def test_backpressure_probability_is_appended_to_config_contract(self) -> None:
        # Existing positional fields keep their original meaning.
        config = SchedulerConfig(0.1, 3, 0.6, 0.7, 0.02)
        self.assertEqual(config.exploration_probability, 0.1)
        self.assertEqual(config.exploration_width, 3)
        self.assertEqual(config.challenge_threshold, 0.6)
        self.assertEqual(config.verification_threshold, 0.7)
        self.assertEqual(config.stagnation_threshold, 0.02)
        self.assertEqual(config.backpressure_exploration_probability, 0.05)


if __name__ == "__main__":
    unittest.main()
