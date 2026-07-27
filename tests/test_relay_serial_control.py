from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    generate_relay_world,
)
from ai_hypothesis.population_compute.contract import DEVELOPMENT_POPULATION_SIZES
from ai_hypothesis.population_compute.relay_model import (
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
)
from ai_hypothesis.population_compute.relay_serial_control import (
    normalized_parallel_forward,
    normalized_serial_forward,
)


class RelaySerialControlTests(unittest.TestCase):
    def test_serial_schedule_matches_parallel_normalized_across_full_ladder(self) -> None:
        torch.manual_seed(17)
        model = RelayPopulationModel()
        fingerprint_before = model.parameter_fingerprint()

        for difficulty_index, difficulty in enumerate(RELAY_DIFFICULTIES):
            worlds = tuple(
                generate_relay_world(10_000 + difficulty_index * 100 + offset, difficulty)
                for offset in range(3)
            )
            for active_workers in DEVELOPMENT_POPULATION_SIZES:
                with self.subTest(
                    difficulty=difficulty.name,
                    active_workers=active_workers,
                ):
                    batch = build_relay_tensor_batch(
                        worlds,
                        active_workers=active_workers,
                    )
                    parallel = normalized_parallel_forward(model, batch)
                    serial = normalized_serial_forward(model, batch)

                    torch.testing.assert_close(
                        serial.final_shared,
                        parallel.final_shared,
                        rtol=2e-5,
                        atol=2e-5,
                    )
                    torch.testing.assert_close(
                        serial.logits,
                        parallel.logits,
                        rtol=2e-5,
                        atol=2e-5,
                    )
                    self.assertTrue(
                        torch.equal(
                            decode_node_logits(serial.logits),
                            decode_node_logits(parallel.logits),
                        )
                    )

                    expected_updates = active_workers * difficulty.hop_count
                    self.assertEqual(
                        parallel.telemetry.worker_updates_per_sample,
                        expected_updates,
                    )
                    self.assertEqual(
                        serial.telemetry.worker_updates_per_sample,
                        expected_updates,
                    )
                    self.assertEqual(
                        parallel.telemetry.candidate_evaluations_per_sample,
                        expected_updates,
                    )
                    self.assertEqual(
                        serial.telemetry.candidate_evaluations_per_sample,
                        expected_updates,
                    )
                    self.assertEqual(
                        parallel.telemetry.peak_active_neural_states_per_sample,
                        active_workers,
                    )
                    self.assertEqual(
                        serial.telemetry.peak_active_neural_states_per_sample,
                        1,
                    )
                    self.assertEqual(
                        serial.telemetry.inter_state_communicated_scalars_per_sample,
                        0,
                    )

        self.assertEqual(model.parameter_fingerprint(), fingerprint_before)

    def test_parallel_communication_accounting_scales_linearly(self) -> None:
        model = RelayPopulationModel()
        difficulty = RELAY_DIFFICULTIES[0]
        batch = build_relay_tensor_batch(
            (generate_relay_world(1234, difficulty),),
            active_workers=64,
        )
        output = normalized_parallel_forward(model, batch)

        self.assertEqual(output.telemetry.worker_updates_per_sample, 64 * 2)
        self.assertEqual(
            output.telemetry.inter_state_communicated_scalars_per_sample,
            2 * 64 * 2 * model.config.message_width,
        )

    def test_nonpositive_round_override_is_rejected(self) -> None:
        model = RelayPopulationModel()
        difficulty = RELAY_DIFFICULTIES[0]
        batch = build_relay_tensor_batch(
            (generate_relay_world(1235, difficulty),),
            active_workers=4,
        )

        with self.assertRaisesRegex(ValueError, "recurrent_rounds must be positive"):
            normalized_parallel_forward(model, batch, recurrent_rounds=0)
        with self.assertRaisesRegex(ValueError, "recurrent_rounds must be positive"):
            normalized_serial_forward(model, batch, recurrent_rounds=0)


if __name__ == "__main__":
    unittest.main()
