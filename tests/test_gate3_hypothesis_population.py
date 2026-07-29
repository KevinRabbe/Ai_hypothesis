from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate3_hypothesis_population import (
    GATE3_CONFIRMATION_WORLD_START,
    GATE3_DEPTHS,
    GATE3_DEVELOPMENT_WORLD_START,
    GATE3_HINT_RELIABILITY,
    GATE3_WIDTHS_BY_DEPTH,
    Gate3ControlMode,
    Gate3ObservationKind,
    build_gate3_condition_plan,
    deterministic_tie_break,
    generate_gate3_world,
    reshuffled_state_permutation,
)


class Gate3HypothesisPopulationTests(unittest.TestCase):
    def test_world_generation_is_deterministic(self) -> None:
        a = generate_gate3_world(seed=12345, depth=8)
        b = generate_gate3_world(seed=12345, depth=8)
        self.assertEqual(a, b)
        self.assertEqual(a.observation_signature(), b.observation_signature())

    def test_development_and_confirmation_domains_are_disjoint(self) -> None:
        self.assertNotEqual(GATE3_DEVELOPMENT_WORLD_START, GATE3_CONFIRMATION_WORLD_START)
        self.assertLess(GATE3_DEVELOPMENT_WORLD_START, GATE3_CONFIRMATION_WORLD_START)

    def test_world_contains_noisy_hints_then_exact_reveals(self) -> None:
        for depth in GATE3_DEPTHS:
            world = generate_gate3_world(seed=91, depth=depth)
            world.validate()
            self.assertEqual(len(world.observations), 2 * depth)
            for index, observation in enumerate(world.observations[:depth]):
                self.assertIs(observation.kind, Gate3ObservationKind.BRANCH_HINT)
                self.assertEqual(observation.bit_index, index)
                self.assertEqual(observation.observed_bit, world.noisy_hints[index])
            for index, observation in enumerate(world.observations[depth:]):
                self.assertIs(observation.kind, Gate3ObservationKind.DELAYED_REVEAL)
                self.assertEqual(observation.bit_index, index)
                self.assertEqual(observation.observed_bit, world.hidden_path[index])

    def test_hint_process_matches_frozen_reliability_over_large_deterministic_sample(self) -> None:
        matched = 0
        total = 0
        for seed in range(2_000):
            world = generate_gate3_world(seed=seed, depth=8)
            matched += sum(int(hint == hidden) for hint, hidden in zip(world.noisy_hints, world.hidden_path, strict=True))
            total += world.depth
        observed = matched / total
        self.assertGreater(observed, GATE3_HINT_RELIABILITY - 0.02)
        self.assertLess(observed, GATE3_HINT_RELIABILITY + 0.02)

    def test_all_widths_and_controls_receive_identical_world_information(self) -> None:
        for depth in GATE3_DEPTHS:
            world = generate_gate3_world(seed=777, depth=depth)
            plans = [
                build_gate3_condition_plan(world, width=width, mode=mode)
                for width in GATE3_WIDTHS_BY_DEPTH[depth]
                for mode in Gate3ControlMode
            ]
            signatures = {plan.observation_signature for plan in plans}
            observation_counts = {plan.unique_world_observation_count for plan in plans}
            self.assertEqual(len(signatures), 1)
            self.assertEqual(observation_counts, {2 * depth})

    def test_learned_update_budget_is_exactly_width_and_mode_independent(self) -> None:
        expected_totals = {4: 128, 6: 768, 8: 4096}
        for depth in GATE3_DEPTHS:
            world = generate_gate3_world(seed=12, depth=depth)
            totals = set()
            for width in GATE3_WIDTHS_BY_DEPTH[depth]:
                for mode in Gate3ControlMode:
                    plan = build_gate3_condition_plan(world, width=width, mode=mode)
                    plan.validate()
                    totals.add(plan.learned_update_count)
                    self.assertTrue(all(phase.learned_updates_in_phase == (1 << depth) for phase in plan.phases))
            self.assertEqual(totals, {expected_totals[depth]})

    def test_narrow_width_gets_more_refinement_while_wide_width_gets_more_breadth(self) -> None:
        world = generate_gate3_world(seed=44, depth=8)
        narrow = build_gate3_condition_plan(world, width=1, mode=Gate3ControlMode.STABLE_DIVERSE)
        wide = build_gate3_condition_plan(world, width=256, mode=Gate3ControlMode.STABLE_DIVERSE)

        self.assertEqual(narrow.learned_update_count, wide.learned_update_count)
        self.assertEqual(narrow.phases[-1].evaluated_state_slots, 1)
        self.assertEqual(wide.phases[-1].evaluated_state_slots, 256)
        self.assertEqual(narrow.phases[-1].recurrent_updates_per_evaluated_state, 256)
        self.assertEqual(wide.phases[-1].recurrent_updates_per_evaluated_state, 1)

        # Final branching phase: W1 refines two children deeply; W256 evaluates all 256 unique slots once.
        self.assertEqual(narrow.phases[7].evaluated_state_slots, 2)
        self.assertEqual(narrow.phases[7].recurrent_updates_per_evaluated_state, 128)
        self.assertEqual(wide.phases[7].evaluated_state_slots, 256)
        self.assertEqual(wide.phases[7].recurrent_updates_per_evaluated_state, 1)

    def test_controls_are_mechanically_matched(self) -> None:
        for depth in GATE3_DEPTHS:
            world = generate_gate3_world(seed=404, depth=depth)
            for width in GATE3_WIDTHS_BY_DEPTH[depth]:
                plans = [build_gate3_condition_plan(world, width=width, mode=mode) for mode in Gate3ControlMode]
                signatures = {plan.mechanical_signature() for plan in plans}
                self.assertEqual(len(signatures), 1)

    def test_width_one_controls_are_structurally_identical_before_runtime_semantics(self) -> None:
        world = generate_gate3_world(seed=1_234, depth=8)
        stable = build_gate3_condition_plan(world, width=1, mode=Gate3ControlMode.STABLE_DIVERSE)
        collapsed = build_gate3_condition_plan(world, width=1, mode=Gate3ControlMode.COLLAPSED_DIVERSITY)
        reshuffled = build_gate3_condition_plan(world, width=1, mode=Gate3ControlMode.RESHUFFLED_CONTINUITY)
        self.assertEqual(stable.mechanical_signature(), collapsed.mechanical_signature())
        self.assertEqual(stable.mechanical_signature(), reshuffled.mechanical_signature())
        self.assertEqual(reshuffled_state_permutation(world_seed=world.seed, phase_index=3, state_count=1), (0,))

    def test_final_branching_phase_reaches_full_hypothesis_space_at_largest_width(self) -> None:
        for depth in GATE3_DEPTHS:
            width = 1 << depth
            world = generate_gate3_world(seed=8, depth=depth)
            plan = build_gate3_condition_plan(world, width=width, mode=Gate3ControlMode.STABLE_DIVERSE)
            final_branch = plan.phases[depth - 1]
            self.assertEqual(final_branch.evaluated_state_slots, width)
            self.assertEqual(final_branch.retained_state_slots_after, width)
            self.assertEqual(final_branch.recurrent_updates_per_evaluated_state, 1)

    def test_tie_break_is_deterministic_and_candidate_specific(self) -> None:
        key_a1 = deterministic_tie_break(world_seed=99, phase_index=4, candidate_path=(0, 1, 0, 1))
        key_a2 = deterministic_tie_break(world_seed=99, phase_index=4, candidate_path=(0, 1, 0, 1))
        key_b = deterministic_tie_break(world_seed=99, phase_index=4, candidate_path=(0, 1, 1, 1))
        self.assertEqual(key_a1, key_a2)
        self.assertNotEqual(key_a1, key_b)

    def test_reshuffle_is_a_permutation_and_deterministic(self) -> None:
        a = reshuffled_state_permutation(world_seed=888, phase_index=5, state_count=64)
        b = reshuffled_state_permutation(world_seed=888, phase_index=5, state_count=64)
        self.assertEqual(a, b)
        self.assertEqual(tuple(sorted(a)), tuple(range(64)))

    def test_invalid_width_is_rejected(self) -> None:
        world = generate_gate3_world(seed=3, depth=4)
        with self.assertRaises(ValueError):
            build_gate3_condition_plan(world, width=64, mode=Gate3ControlMode.STABLE_DIVERSE)


if __name__ == "__main__":
    unittest.main()
