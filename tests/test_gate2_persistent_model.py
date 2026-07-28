from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from ai_hypothesis.population_compute.gate2_persistent_model import (
    GATE2_OBSERVATION_WIDTH,
    Gate2PersistentStateModel,
    build_gate2_tensor_batch,
    decode_gate2_payload_logits,
    parallel_persistent_forward,
    serial_persistent_forward,
)
from ai_hypothesis.population_compute.gate2_persistent_state_capacity import (
    GATE2_TOTAL_ROUNDS,
    Gate2ControlMode,
    generate_gate2_world,
)


class Gate2PersistentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = Gate2PersistentStateModel()

    def test_observation_encoding_is_width_and_control_independent(self) -> None:
        worlds = [generate_gate2_world(seed=seed, entity_count=16) for seed in (1, 2)]
        stable_w1 = build_gate2_tensor_batch(
            worlds,
            width=1,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
        )
        stable_w16 = build_gate2_tensor_batch(
            worlds,
            width=16,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
        )
        reshuffled = build_gate2_tensor_batch(
            worlds,
            width=16,
            mode=Gate2ControlMode.RESHUFFLED_LOCALITY,
        )
        reset = build_gate2_tensor_batch(
            worlds,
            width=16,
            mode=Gate2ControlMode.RESET_STATE,
        )

        self.assertTrue(torch.equal(stable_w1.observations, stable_w16.observations))
        self.assertTrue(torch.equal(stable_w1.observations, reshuffled.observations))
        self.assertTrue(torch.equal(stable_w1.observations, reset.observations))
        self.assertEqual(stable_w1.learned_updates_per_sample, 8 * 16)
        self.assertEqual(stable_w16.learned_updates_per_sample, 8 * 16)
        self.assertEqual(stable_w1.observations.shape[-1], GATE2_OBSERVATION_WIDTH)

    def test_query_identity_is_kept_outside_observation_stream(self) -> None:
        original = generate_gate2_world(seed=31, entity_count=16)
        alternate_query = (original.query_entity_index + 1) % original.entity_count
        changed = replace(original, query_entity_index=alternate_query)
        changed.validate()

        first = build_gate2_tensor_batch(
            [original],
            width=4,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
        )
        second = build_gate2_tensor_batch(
            [changed],
            width=4,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
        )

        self.assertTrue(torch.equal(first.observations, second.observations))
        self.assertNotEqual(original.query_key, changed.query_key)
        self.assertFalse(torch.equal(first.query_bits, second.query_bits))

    def test_parallel_and_serial_persistent_schedules_are_output_equivalent(self) -> None:
        worlds = [generate_gate2_world(seed=seed, entity_count=16) for seed in (41, 42, 43)]
        for mode in Gate2ControlMode:
            for width in (1, 4, 16):
                with self.subTest(mode=mode, width=width):
                    batch = build_gate2_tensor_batch(worlds, width=width, mode=mode)
                    parallel = parallel_persistent_forward(self.model, batch)
                    serial = serial_persistent_forward(self.model, batch)

                    self.assertTrue(
                        torch.allclose(parallel.logits, serial.logits, rtol=2e-5, atol=2e-5)
                    )
                    self.assertTrue(
                        torch.allclose(
                            parallel.final_states,
                            serial.final_states,
                            rtol=2e-5,
                            atol=2e-5,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            decode_gate2_payload_logits(parallel.logits),
                            decode_gate2_payload_logits(serial.logits),
                        )
                    )
                    self.assertEqual(
                        parallel.telemetry.learned_updates_per_sample,
                        GATE2_TOTAL_ROUNDS * 16,
                    )
                    self.assertEqual(
                        serial.telemetry.learned_updates_per_sample,
                        GATE2_TOTAL_ROUNDS * 16,
                    )
                    self.assertEqual(
                        parallel.telemetry.persistent_state_vectors_per_sample,
                        width,
                    )
                    self.assertEqual(
                        serial.telemetry.persistent_state_vectors_per_sample,
                        width,
                    )
                    self.assertEqual(parallel.telemetry.peak_simultaneous_updates_per_sample, width)
                    self.assertEqual(serial.telemetry.peak_simultaneous_updates_per_sample, 1)

    def test_width_one_stable_and_reshuffled_are_exact_execution_identity(self) -> None:
        worlds = [generate_gate2_world(seed=seed, entity_count=64) for seed in (51, 52)]
        stable = build_gate2_tensor_batch(
            worlds,
            width=1,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
        )
        reshuffled = build_gate2_tensor_batch(
            worlds,
            width=1,
            mode=Gate2ControlMode.RESHUFFLED_LOCALITY,
        )

        self.assertTrue(
            torch.equal(
                stable.entity_order_by_round_slot_lane,
                reshuffled.entity_order_by_round_slot_lane,
            )
        )
        stable_output = parallel_persistent_forward(self.model, stable)
        reshuffled_output = parallel_persistent_forward(self.model, reshuffled)
        self.assertTrue(torch.equal(stable_output.logits, reshuffled_output.logits))
        self.assertTrue(torch.equal(stable_output.final_states, reshuffled_output.final_states))

    def test_reset_control_keeps_width_and_work_but_changes_state_history(self) -> None:
        worlds = [generate_gate2_world(seed=61, entity_count=16)]
        stable = build_gate2_tensor_batch(
            worlds,
            width=16,
            mode=Gate2ControlMode.STABLE_PERSISTENT,
        )
        reset = build_gate2_tensor_batch(
            worlds,
            width=16,
            mode=Gate2ControlMode.RESET_STATE,
        )
        stable_output = parallel_persistent_forward(self.model, stable)
        reset_output = parallel_persistent_forward(self.model, reset)

        self.assertEqual(stable.learned_updates_per_sample, reset.learned_updates_per_sample)
        self.assertEqual(stable.width, reset.width)
        self.assertFalse(torch.equal(stable_output.final_states, reset_output.final_states))

    def test_parameter_count_is_runtime_width_independent(self) -> None:
        parameter_count = self.model.trainable_parameter_count()
        fingerprint = self.model.parameter_fingerprint()
        world = generate_gate2_world(seed=71, entity_count=256)

        for width in (1, 4, 16, 64, 256):
            batch = build_gate2_tensor_batch(
                [world],
                width=width,
                mode=Gate2ControlMode.STABLE_PERSISTENT,
            )
            self.assertEqual(self.model.trainable_parameter_count(), parameter_count)
            self.assertEqual(self.model.parameter_fingerprint(), fingerprint)
            self.assertEqual(batch.learned_updates_per_sample, GATE2_TOTAL_ROUNDS * 256)
            self.assertEqual(batch.collision_load, 256 // width)


if __name__ == "__main__":
    unittest.main()
