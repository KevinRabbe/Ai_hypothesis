from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from ai_hypothesis.population_compute.gate4_adaptive_activation import (
    GATE4_DEPTH,
    GATE4_RESERVE_CAPACITY,
    GATE4_SCHEDULED_SLOTS,
    GATE4_STATIC_TERMINAL_COUNT,
    GATE4_TOTAL_LEARNED_UPDATES,
    Gate4EvaluationWorld,
    Gate4SchedulerMode,
    run_gate4_world_batch,
)


class _FastFakeScorer:
    """Small deterministic scorer implementing the Gate3V1Scorer runtime surface."""

    def to(self, device: torch.device | str) -> "_FastFakeScorer":
        self.device = torch.device(device)
        return self

    def initial_state(self, count: int, *, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((count, 1), dtype=torch.float32, device=device)

    def advance(self, state: torch.Tensor, phase_input: torch.Tensor, *, repeats: int) -> torch.Tensor:
        signal = phase_input.sum(dim=1, keepdim=True)
        return state + signal * float(repeats)

    def score(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, 0]

    def trainable_parameter_count(self) -> int:
        return 19_649

    def parameter_fingerprint(self) -> str:
        return "synthetic-gate4-test-fingerprint"


def _synthetic_world(seed: int) -> Gate4EvaluationWorld:
    world = Gate4EvaluationWorld(
        world_index=0,
        public=Gate3V1PublicWorld(
            seed=seed,
            depth=GATE4_DEPTH,
            noisy_hints=(0, 1, 0, 1, 1, 0, 1, 0),
        ),
        hidden_path=(0, 1, 0, 1, 0, 1, 0, 1),
    )
    world.validate()
    return world


class Gate4AdaptiveActivationTests(unittest.TestCase):
    def test_frozen_work_identity(self) -> None:
        self.assertEqual(GATE4_RESERVE_CAPACITY, 256)
        self.assertEqual(GATE4_SCHEDULED_SLOTS, 159)
        self.assertEqual(GATE4_TOTAL_LEARNED_UPDATES, 2544)

    def test_all_scheduler_modes_preserve_work_and_population_bounds(self) -> None:
        scorer = _FastFakeScorer()
        worlds = (_synthetic_world(101), _synthetic_world(202))
        for mode in Gate4SchedulerMode:
            with self.subTest(mode=mode.value):
                results = run_gate4_world_batch(scorer, worlds, mode=mode, device="cpu")
                self.assertEqual(len(results), len(worlds))
                for result in results:
                    telemetry = result.telemetry
                    self.assertEqual(telemetry.productive_slots + telemetry.sink_slots, 159)
                    self.assertEqual(telemetry.total_learned_updates, 2544)
                    self.assertEqual(len(telemetry.live_nonterminal_population_by_slot), 159)
                    self.assertEqual(len(telemetry.activated_parent_depth_by_slot), 159)
                    self.assertLessEqual(telemetry.max_live_nonterminal_population, 256)
                    self.assertEqual(
                        sum(telemetry.productive_activations_by_parent_depth),
                        telemetry.productive_slots,
                    )
                    self.assertEqual(
                        len(telemetry.terminal_generation_slot_indices),
                        telemetry.generated_terminal_count,
                    )

    def test_static_generation_schedule_is_exact(self) -> None:
        scorer = _FastFakeScorer()
        result = run_gate4_world_batch(
            scorer,
            (_synthetic_world(303),),
            mode=Gate4SchedulerMode.STATIC_GENERATION,
            device="cpu",
        )[0]
        telemetry = result.telemetry
        self.assertEqual(telemetry.productive_slots, 159)
        self.assertEqual(telemetry.sink_slots, 0)
        self.assertEqual(
            telemetry.productive_activations_by_parent_depth,
            (1, 2, 4, 8, 16, 32, 64, 32),
        )
        self.assertEqual(telemetry.generated_terminal_count, GATE4_STATIC_TERMINAL_COUNT)
        self.assertEqual(telemetry.unique_generated_terminal_count, GATE4_STATIC_TERMINAL_COUNT)
        self.assertTrue(all(slot >= 127 for slot in telemetry.terminal_generation_slot_indices))

    def test_adaptive_modes_are_sparse_one_parent_per_slot(self) -> None:
        scorer = _FastFakeScorer()
        world = (_synthetic_world(404),)
        for mode in (Gate4SchedulerMode.ADAPTIVE_SCORE, Gate4SchedulerMode.ADAPTIVE_HASH):
            with self.subTest(mode=mode.value):
                telemetry = run_gate4_world_batch(scorer, world, mode=mode, device="cpu")[0].telemetry
                self.assertEqual(telemetry.productive_slots, 159)
                self.assertEqual(telemetry.sink_slots, 0)
                self.assertEqual(len(telemetry.activated_parent_depth_by_slot), 159)
                self.assertTrue(all(0 <= depth < GATE4_DEPTH for depth in telemetry.activated_parent_depth_by_slot))


if __name__ == "__main__":
    unittest.main()
