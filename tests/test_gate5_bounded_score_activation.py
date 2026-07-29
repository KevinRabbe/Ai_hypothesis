from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from ai_hypothesis.population_compute.gate5_bounded_score_activation import (
    GATE5_STAGE_A_FRONTIER,
    GATE5_STAGE_B_SLOTS,
    GATE5_TOTAL_LEARNED_UPDATES,
    Gate5EvaluationWorld,
    Gate5SchedulerMode,
    _bounded_indices,
    _bounded_visible_candidates,
)
from ai_hypothesis.population_compute.gate5_bounded_score_batch import run_gate5_strict_world_batch
from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1NeuralCandidate


class Gate5BoundedScoreActivationTests(unittest.TestCase):
    def _world(self) -> Gate5EvaluationWorld:
        world = Gate5EvaluationWorld(
            world_index=0,
            public=Gate3V1PublicWorld(seed=12345, depth=8, noisy_hints=(0, 1, 0, 1, 0, 1, 0, 1)),
            hidden_path=(0, 1, 0, 1, 0, 1, 0, 1),
        )
        world.validate()
        return world

    def _model(self) -> Gate3V1Scorer:
        torch.manual_seed(7)
        return Gate3V1Scorer()

    def test_bounded_indices_are_unique_and_deterministic(self) -> None:
        first = _bounded_indices(count=64, k=16, world_seed=9, slot_index=70, group="k16")
        second = _bounded_indices(count=64, k=16, world_seed=9, slot_index=70, group="k16")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len(set(first)), 16)
        self.assertTrue(all(0 <= index < 64 for index in first))

    def test_k16_score_and_hash_share_sampler_on_identical_reserve(self) -> None:
        model = self._model()
        state = model.initial_state(20, device="cpu")
        reserve = tuple(
            Gate3V1NeuralCandidate(path=(index // 2, index % 2), state=state[index], score=float(index))
            for index in range(4)
        )
        # Use eight valid binary paths instead of relying on decimal index bits.
        paths = (
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
        )
        reserve = tuple(
            Gate3V1NeuralCandidate(path=path, state=state[index], score=float(index))
            for index, path in enumerate(paths)
        )
        score_visible = _bounded_visible_candidates(
            reserve,
            mode=Gate5SchedulerMode.BOUNDED_SCORE_K16,
            world_seed=88,
            slot_index=73,
        )
        hash_visible = _bounded_visible_candidates(
            reserve,
            mode=Gate5SchedulerMode.BOUNDED_HASH_K16,
            world_seed=88,
            slot_index=73,
        )
        self.assertEqual(
            tuple(candidate.path for candidate in score_visible),
            tuple(candidate.path for candidate in hash_visible),
        )

    def test_strict_runtime_preserves_work_and_visibility(self) -> None:
        world = self._world()
        for mode in (
            Gate5SchedulerMode.GLOBAL_SCORE,
            Gate5SchedulerMode.BOUNDED_SCORE_K16,
            Gate5SchedulerMode.BOUNDED_HASH_K16,
        ):
            result = run_gate5_strict_world_batch(
                self._model(),
                (world,),
                mode=mode,
                device="cpu",
            )[0]
            telemetry = result.telemetry
            self.assertEqual(telemetry.productive_slots, 159)
            self.assertEqual(telemetry.sink_slots, 0)
            self.assertEqual(telemetry.total_learned_updates, GATE5_TOTAL_LEARNED_UPDATES)
            self.assertEqual(telemetry.stage_a_frontier_width, GATE5_STAGE_A_FRONTIER)
            self.assertEqual(len(telemetry.stage_b_live_population_by_slot), GATE5_STAGE_B_SLOTS)
            self.assertEqual(len(telemetry.selected_parent_paths_by_slot), GATE5_STAGE_B_SLOTS)
            if mode is Gate5SchedulerMode.GLOBAL_SCORE:
                self.assertEqual(
                    telemetry.stage_b_visible_candidate_count_by_slot,
                    telemetry.stage_b_live_population_by_slot,
                )
                self.assertEqual(
                    telemetry.stage_b_score_observation_count_by_slot,
                    telemetry.stage_b_live_population_by_slot,
                )
            elif mode is Gate5SchedulerMode.BOUNDED_SCORE_K16:
                self.assertTrue(
                    all(
                        visible == min(16, live)
                        for visible, live in zip(
                            telemetry.stage_b_visible_candidate_count_by_slot,
                            telemetry.stage_b_live_population_by_slot,
                            strict=True,
                        )
                    )
                )
                self.assertEqual(
                    telemetry.stage_b_score_observation_count_by_slot,
                    telemetry.stage_b_visible_candidate_count_by_slot,
                )
            else:
                self.assertTrue(
                    all(
                        visible == min(16, live)
                        for visible, live in zip(
                            telemetry.stage_b_visible_candidate_count_by_slot,
                            telemetry.stage_b_live_population_by_slot,
                            strict=True,
                        )
                    )
                )
                self.assertTrue(
                    all(value == 0 for value in telemetry.stage_b_score_observation_count_by_slot)
                )


if __name__ == "__main__":
    unittest.main()
