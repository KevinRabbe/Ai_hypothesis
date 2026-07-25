"""Tests for the simple inspectable Scheduler v0 policy."""

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


def candidate(
    thread_id: str,
    *,
    status: str = "ACTIVE",
    purpose: WorkPurpose = WorkPurpose.PROGRESS,
    **signals: float,
) -> SchedulableThread:
    return SchedulableThread(
        state=ProjectedState(
            revision=1,
            thread_id=thread_id,
            objective=f"Objective {thread_id}",
            status=status,
            purpose=purpose,
        ),
        signals=SchedulerSignals(**signals),
    )


class SchedulerV0Tests(unittest.TestCase):
    def test_exploitation_prefers_higher_priority_thread(self) -> None:
        scheduler = SchedulerV0(SchedulerConfig(exploration_probability=0.0))
        decision = scheduler.choose(
            (
                candidate("low", importance=0.1, recent_progress=0.5),
                candidate("high", importance=0.9, recent_progress=0.5),
            )
        )

        self.assertEqual(decision.thread_id, "high")
        self.assertEqual(decision.action, SchedulerAction.CONTINUE)

    def test_verification_and_challenge_are_targeted_actions(self) -> None:
        scheduler = SchedulerV0(SchedulerConfig(exploration_probability=0.0))

        verify = scheduler.choose(
            (candidate("verify", importance=1.0, verification_need=0.9),)
        )
        self.assertEqual(verify.action, SchedulerAction.VERIFY)
        self.assertEqual(verify.purpose, WorkPurpose.VERIFY)

        challenge = scheduler.choose(
            (candidate("challenge", importance=1.0, contradiction_severity=0.9),)
        )
        self.assertEqual(challenge.action, SchedulerAction.CHALLENGE)
        self.assertEqual(challenge.purpose, WorkPurpose.CHALLENGE)

    def test_stagnation_rotates_worker_but_paused_thread_resumes(self) -> None:
        scheduler = SchedulerV0(SchedulerConfig(exploration_probability=0.0))

        stagnant = scheduler.choose((candidate("stagnant", importance=1.0),))
        self.assertEqual(stagnant.action, SchedulerAction.ROTATE_WORKER)

        paused = scheduler.choose((candidate("paused", status="PAUSED", importance=1.0),))
        self.assertEqual(paused.action, SchedulerAction.CONTINUE)
        self.assertIn("RESUME", paused.reason_codes)

    def test_structured_exploration_produces_exploration_work(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(exploration_probability=1.0),
            rng=random.Random(7),
        )
        decision = scheduler.choose(
            (
                candidate("known", importance=1.0, recent_progress=0.5),
                candidate(
                    "under-covered",
                    importance=0.1,
                    missing_coverage=1.0,
                    novelty=1.0,
                ),
            )
        )

        self.assertEqual(decision.action, SchedulerAction.ADD_WIDTH)
        self.assertEqual(decision.purpose, WorkPurpose.EXPLORE)
        self.assertIn("STRUCTURED_EXPLORATION", decision.reason_codes)

    def test_backpressure_redirects_discovery_to_integration_work(self) -> None:
        scheduler = SchedulerV0(
            SchedulerConfig(exploration_probability=1.0),
            rng=random.Random(7),
        )
        decision = scheduler.choose(
            (candidate("thread-1", importance=1.0, missing_coverage=1.0),),
            integration_backpressure=True,
        )

        self.assertEqual(decision.action, SchedulerAction.SYNTHESIZE)
        self.assertEqual(decision.purpose, WorkPurpose.SYNTHESIZE)
        self.assertIn("BACKPRESSURE", decision.reason_codes)

    def test_complete_threads_are_not_scheduled(self) -> None:
        scheduler = SchedulerV0(SchedulerConfig(exploration_probability=0.0))
        decision = scheduler.choose(
            (
                candidate("done", status="COMPLETE", importance=1.0),
                candidate("open", importance=0.1, recent_progress=0.5),
            )
        )
        self.assertEqual(decision.thread_id, "open")

    def test_signals_are_bounded(self) -> None:
        scheduler = SchedulerV0(SchedulerConfig(exploration_probability=0.0))
        with self.assertRaises(ValueError):
            scheduler.choose((candidate("bad", importance=1.1),))


if __name__ == "__main__":
    unittest.main()
