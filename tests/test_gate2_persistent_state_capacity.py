from __future__ import annotations

import unittest

from ai_hypothesis.population_compute.gate2_persistent_state_capacity import (
    GATE2_ENTITY_COUNTS,
    GATE2_EVIDENCE_ROUNDS,
    GATE2_TOTAL_ROUNDS,
    Gate2ControlMode,
    build_gate2_condition_matrix,
    build_gate2_condition_plan,
    gate2_population_widths,
    generate_gate2_world,
)


class Gate2PersistentStateCapacityTests(unittest.TestCase):
    def test_world_generation_is_deterministic_and_domain_complete(self) -> None:
        for entity_count in GATE2_ENTITY_COUNTS:
            first = generate_gate2_world(seed=17, entity_count=entity_count)
            second = generate_gate2_world(seed=17, entity_count=entity_count)
            different = generate_gate2_world(seed=18, entity_count=entity_count)

            self.assertEqual(first, second)
            self.assertNotEqual(first.entity_keys, different.entity_keys)
            self.assertEqual(len(first.observations), GATE2_TOTAL_ROUNDS * entity_count)
            self.assertEqual(first.learned_update_count, GATE2_TOTAL_ROUNDS * entity_count)
            self.assertEqual(len(set(first.entity_keys)), entity_count)

            for observation in first.observations:
                if observation.round_index < GATE2_EVIDENCE_ROUNDS:
                    payload = first.payloads[observation.entity_index]
                    expected_bit = (payload >> observation.round_index) & 1
                    self.assertEqual(observation.evidence_bit_value, expected_bit)
                    self.assertIsNone(observation.interference_token)
                else:
                    self.assertIsNone(observation.evidence_bit_index)
                    self.assertIsNone(observation.evidence_bit_value)
                    self.assertIsNotNone(observation.interference_token)

    def test_all_widths_and_controls_preserve_information_and_learned_work(self) -> None:
        for entity_count in GATE2_ENTITY_COUNTS:
            world = generate_gate2_world(seed=101 + entity_count, entity_count=entity_count)
            plans = build_gate2_condition_matrix(world)
            expected_signature = world.observation_signature()
            expected_updates = GATE2_TOTAL_ROUNDS * entity_count
            expected_plan_count = len(gate2_population_widths(entity_count)) * len(Gate2ControlMode)

            self.assertEqual(len(plans), expected_plan_count)
            for plan in plans:
                plan.validate()
                self.assertEqual(plan.observation_signature, expected_signature)
                self.assertEqual(plan.learned_update_count, expected_updates)
                self.assertEqual(plan.observation_count, expected_updates)
                self.assertEqual(plan.inspected_entity_count, entity_count)
                self.assertEqual(plan.source_coverage, 1.0)

                for round_index in range(GATE2_TOTAL_ROUNDS):
                    loads = plan.slot_loads(round_index)
                    self.assertEqual(sum(loads), entity_count)
                    self.assertEqual(len(loads), plan.width)
                    self.assertLessEqual(max(loads) - min(loads), 1)
                    # Every frozen width divides every compatible frozen entity count.
                    self.assertEqual(set(loads), {entity_count // plan.width})

    def test_stable_and_reset_controls_share_exact_routing(self) -> None:
        for entity_count in GATE2_ENTITY_COUNTS:
            world = generate_gate2_world(seed=303 + entity_count, entity_count=entity_count)
            for width in gate2_population_widths(entity_count):
                stable = build_gate2_condition_plan(
                    world,
                    width=width,
                    mode=Gate2ControlMode.STABLE_PERSISTENT,
                )
                reset = build_gate2_condition_plan(
                    world,
                    width=width,
                    mode=Gate2ControlMode.RESET_STATE,
                )

                self.assertEqual(stable.slot_by_round_entity, reset.slot_by_round_entity)
                self.assertFalse(stable.reset_state_each_round)
                self.assertTrue(reset.reset_state_each_round)
                self.assertTrue(
                    all(
                        round_slots == stable.slot_by_round_entity[0]
                        for round_slots in stable.slot_by_round_entity
                    )
                )

    def test_width_one_reshuffled_locality_is_exact_identity_control(self) -> None:
        for entity_count in GATE2_ENTITY_COUNTS:
            world = generate_gate2_world(seed=707 + entity_count, entity_count=entity_count)
            stable = build_gate2_condition_plan(
                world,
                width=1,
                mode=Gate2ControlMode.STABLE_PERSISTENT,
            )
            reshuffled = build_gate2_condition_plan(
                world,
                width=1,
                mode=Gate2ControlMode.RESHUFFLED_LOCALITY,
            )

            self.assertEqual(stable.slot_by_round_entity, reshuffled.slot_by_round_entity)
            self.assertEqual(stable.observation_signature, reshuffled.observation_signature)
            self.assertEqual(stable.learned_update_count, reshuffled.learned_update_count)

    def test_reshuffled_control_breaks_stable_locality_without_changing_load(self) -> None:
        for entity_count in (64, 256):
            world = generate_gate2_world(seed=911 + entity_count, entity_count=entity_count)
            width = 16
            stable = build_gate2_condition_plan(
                world,
                width=width,
                mode=Gate2ControlMode.STABLE_PERSISTENT,
            )
            reshuffled = build_gate2_condition_plan(
                world,
                width=width,
                mode=Gate2ControlMode.RESHUFFLED_LOCALITY,
            )

            self.assertTrue(
                any(
                    reshuffled.slot_by_round_entity[round_index]
                    != stable.slot_by_round_entity[round_index]
                    for round_index in range(GATE2_TOTAL_ROUNDS)
                )
            )
            self.assertTrue(
                any(
                    reshuffled.slot_by_round_entity[round_index]
                    != reshuffled.slot_by_round_entity[0]
                    for round_index in range(1, GATE2_TOTAL_ROUNDS)
                )
            )
            for round_index in range(GATE2_TOTAL_ROUNDS):
                self.assertEqual(
                    stable.slot_loads(round_index),
                    reshuffled.slot_loads(round_index),
                )

    def test_target_slot_load_tracks_collision_capacity(self) -> None:
        world = generate_gate2_world(seed=1234, entity_count=256)
        expected_loads = {1: 256, 4: 64, 16: 16, 64: 4, 256: 1}

        for width, expected_load in expected_loads.items():
            plan = build_gate2_condition_plan(
                world,
                width=width,
                mode=Gate2ControlMode.STABLE_PERSISTENT,
            )
            for round_index in range(GATE2_TOTAL_ROUNDS):
                self.assertEqual(
                    plan.target_slot_load(world, round_index),
                    expected_load,
                )

    def test_invalid_matrix_points_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_gate2_world(seed=1, entity_count=32)

        world = generate_gate2_world(seed=1, entity_count=16)
        with self.assertRaises(ValueError):
            build_gate2_condition_plan(
                world,
                width=64,
                mode=Gate2ControlMode.STABLE_PERSISTENT,
            )
        with self.assertRaises(TypeError):
            build_gate2_condition_plan(world, width=4, mode="stable_persistent")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
