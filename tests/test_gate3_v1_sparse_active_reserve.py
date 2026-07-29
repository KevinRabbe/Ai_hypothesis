from __future__ import annotations

import inspect
import unittest

from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import (
    GATE3_V1_ACTIVE_CHILD_LANES,
    GATE3_V1_CONFIRMATION_WORLD_START,
    GATE3_V1_DEPTHS,
    GATE3_V1_DEVELOPMENT_WORLD_START,
    GATE3_V1_HINT_RELIABILITY,
    GATE3_V1_RECURRENT_UPDATES_PER_CHILD,
    GATE3_V1_RESERVE_CAPACITIES,
    GATE3_V1_SCORE_QUANTIZATION,
    GATE3_V1_SEARCH_ROUNDS,
    Gate3V1Candidate,
    Gate3V1ControlMode,
    apply_gate3_v1_reserve_control,
    build_gate3_v1_condition_plan,
    generate_gate3_v1_world,
    make_gate3_v1_accounting,
    quantize_gate3_v1_score,
    score_generated_solution,
)


class Gate3V1SparseActiveReserveTests(unittest.TestCase):
    def test_frozen_ladders_and_work_budgets(self) -> None:
        self.assertEqual(GATE3_V1_DEPTHS, (6, 8, 10))
        self.assertEqual(
            GATE3_V1_RESERVE_CAPACITIES,
            {
                6: (1, 4, 16),
                8: (1, 4, 16, 64),
                10: (1, 4, 16, 64, 256),
            },
        )
        self.assertEqual(GATE3_V1_SEARCH_ROUNDS, {6: 16, 8: 64, 10: 256})
        self.assertEqual(GATE3_V1_ACTIVE_CHILD_LANES, 2)
        self.assertEqual(GATE3_V1_RECURRENT_UPDATES_PER_CHILD, 8)
        self.assertEqual(GATE3_V1_HINT_RELIABILITY, 0.70)
        self.assertEqual(GATE3_V1_SCORE_QUANTIZATION, 1e-3)
        self.assertEqual(GATE3_V1_DEVELOPMENT_WORLD_START, 1 << 30)
        self.assertEqual(GATE3_V1_CONFIRMATION_WORLD_START, 1 << 31)

        expected_totals = {6: 256, 8: 1024, 10: 4096}
        for depth, expected_total in expected_totals.items():
            signatures = set()
            for capacity in GATE3_V1_RESERVE_CAPACITIES[depth]:
                for mode in Gate3V1ControlMode:
                    plan = build_gate3_v1_condition_plan(
                        depth=depth,
                        reserve_capacity=capacity,
                        mode=mode,
                    )
                    self.assertEqual(plan.total_learned_updates, expected_total)
                    signatures.add(plan.mechanical_signature())
            self.assertEqual(len(signatures), 1)

    def test_world_generation_is_deterministic_and_runtime_public_view_has_no_answer(self) -> None:
        first = generate_gate3_v1_world(seed=12345, depth=10)
        second = generate_gate3_v1_world(seed=12345, depth=10)
        other = generate_gate3_v1_world(seed=12346, depth=10)
        self.assertEqual(first, second)
        self.assertNotEqual(first.hidden_path, other.hidden_path)
        self.assertEqual(len(first.public.noisy_hints), 10)
        self.assertFalse(hasattr(first.public, "hidden_path"))

    def test_runtime_control_function_cannot_accept_hidden_answer(self) -> None:
        parameters = inspect.signature(apply_gate3_v1_reserve_control).parameters
        self.assertNotIn("hidden_path", parameters)
        self.assertNotIn("answer", parameters)

    def test_score_quantization_is_frozen_to_milliscale(self) -> None:
        self.assertEqual(quantize_gate3_v1_score(0.12349), 123)
        self.assertEqual(quantize_gate3_v1_score(0.12351), 124)
        with self.assertRaises(ValueError):
            quantize_gate3_v1_score(float("nan"))

    def test_l1_controls_are_exact_structural_identities(self) -> None:
        candidate = Gate3V1Candidate(path=(1, 0), score=0.42, state_token="state-a")
        outputs = {
            mode: apply_gate3_v1_reserve_control(
                (candidate,),
                reserve_capacity=1,
                mode=mode,
                world_seed=91,
                expansion_index=3,
                world_depth=8,
            )
            for mode in Gate3V1ControlMode
        }
        stable = outputs[Gate3V1ControlMode.STABLE_RESERVE]
        for output in outputs.values():
            self.assertEqual(output, stable)

    def test_collapsed_control_removes_distinct_alternatives_without_changing_slot_count(self) -> None:
        candidates = (
            Gate3V1Candidate(path=(0,), score=0.9, state_token="s0"),
            Gate3V1Candidate(path=(1,), score=0.8, state_token="s1"),
            Gate3V1Candidate(path=(0, 0), score=0.7, state_token="s2"),
            Gate3V1Candidate(path=(0, 1), score=0.6, state_token="s3"),
        )
        stable = apply_gate3_v1_reserve_control(
            candidates,
            reserve_capacity=4,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            world_seed=4,
            expansion_index=7,
            world_depth=8,
        )
        collapsed = apply_gate3_v1_reserve_control(
            candidates,
            reserve_capacity=4,
            mode=Gate3V1ControlMode.COLLAPSED_DIVERSITY,
            world_seed=4,
            expansion_index=7,
            world_depth=8,
        )
        self.assertEqual(len(stable), len(collapsed))
        self.assertGreater(len({candidate.path for candidate in stable}), 1)
        self.assertEqual(len({candidate.path for candidate in collapsed}), 1)
        self.assertEqual(len({candidate.state_token for candidate in collapsed}), 1)

    def test_reshuffled_control_preserves_candidate_paths_but_reassigns_histories(self) -> None:
        candidates = tuple(
            Gate3V1Candidate(path=(bit, index % 2), score=0.9 - index * 0.1, state_token=f"s{index}")
            for index, bit in enumerate((0, 1, 0, 1))
        )
        stable = apply_gate3_v1_reserve_control(
            candidates,
            reserve_capacity=4,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            world_seed=111,
            expansion_index=5,
            world_depth=10,
        )
        reshuffled = apply_gate3_v1_reserve_control(
            candidates,
            reserve_capacity=4,
            mode=Gate3V1ControlMode.RESHUFFLED_CONTINUITY,
            world_seed=111,
            expansion_index=5,
            world_depth=10,
        )
        self.assertEqual(tuple(row.path for row in stable), tuple(row.path for row in reshuffled))
        self.assertEqual(
            sorted((row.score, row.state_token) for row in stable),
            sorted((row.score, row.state_token) for row in reshuffled),
        )

    def test_productive_and_sink_work_always_sum_to_frozen_total(self) -> None:
        accounting = make_gate3_v1_accounting(
            depth=10,
            reserve_capacity=1,
            mode=Gate3V1ControlMode.STABLE_RESERVE,
            productive_rounds=10,
        )
        self.assertEqual(accounting.scheduled_rounds, 256)
        self.assertEqual(accounting.productive_rounds, 10)
        self.assertEqual(accounting.sink_rounds, 246)
        self.assertEqual(accounting.productive_learned_updates, 160)
        self.assertEqual(accounting.sink_learned_updates, 3936)
        self.assertEqual(accounting.total_learned_updates, 4096)
        accounting.validate()

    def test_exact_search_coverage_is_evaluation_only_membership(self) -> None:
        hidden = (1, 0, 1, 1, 0, 0)
        generated = ((0, 0, 0, 0, 0, 0), hidden)
        self.assertTrue(score_generated_solution(hidden_path=hidden, generated_terminal_paths=generated))
        self.assertFalse(
            score_generated_solution(
                hidden_path=hidden,
                generated_terminal_paths=((1, 0, 1, 1, 0, 1),),
            )
        )


if __name__ == "__main__":
    unittest.main()
