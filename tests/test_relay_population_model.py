from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute import (
    DEVELOPMENT_POPULATION_SIZES,
    RELAY_DIFFICULTIES,
    CommunicationMode,
    generate_relay_world,
)
from ai_hypothesis.population_compute.relay_model import (
    RelayPopulationModel,
    build_relay_tensor_batch,
    decode_node_logits,
    encode_node_bits,
)


class RelayPopulationModelTests(unittest.TestCase):
    def test_fixed_bit_encoding_round_trips_node_identities(self) -> None:
        node_ids = torch.tensor([0, 1, 17, 255, 4095], dtype=torch.int64)
        bits = encode_node_bits(node_ids)
        decoded = decode_node_logits(bits)
        self.assertTrue(torch.equal(decoded, node_ids))

    def test_tensor_batch_scope_truth_matches_declared_world_threshold(self) -> None:
        difficulty = RELAY_DIFFICULTIES[1]
        worlds = tuple(generate_relay_world(seed, difficulty) for seed in range(4))
        self.assertEqual(
            tuple(world.scope_threshold for world in worlds),
            (4, 16, 64, 256),
        )

        for active_workers in DEVELOPMENT_POPULATION_SIZES:
            batch = build_relay_tensor_batch(
                worlds,
                active_workers=active_workers,
            )
            expected = torch.tensor(
                [active_workers >= world.scope_threshold for world in worlds],
                dtype=torch.bool,
            )
            self.assertTrue(torch.equal(batch.information_complete.cpu(), expected))

    def test_same_model_identity_survives_population_width_changes(self) -> None:
        torch.manual_seed(3)
        model = RelayPopulationModel()
        difficulty = RELAY_DIFFICULTIES[0]
        worlds = tuple(generate_relay_world(seed, difficulty) for seed in range(2))
        parameter_count = model.trainable_parameter_count()
        fingerprint = model.parameter_fingerprint()

        for active_workers in DEVELOPMENT_POPULATION_SIZES:
            batch = build_relay_tensor_batch(
                worlds,
                active_workers=active_workers,
            )
            output = model(
                batch,
                communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            )
            self.assertEqual(output.logits.shape, (2, 12))
            self.assertEqual(model.trainable_parameter_count(), parameter_count)
            self.assertEqual(model.parameter_fingerprint(), fingerprint)
            self.assertEqual(
                output.telemetry.active_state_updates,
                2 * active_workers * difficulty.hop_count,
            )
            self.assertEqual(
                output.telemetry.messages_emitted,
                2 * active_workers * difficulty.hop_count,
            )

    def test_no_communication_keeps_population_message_accounting_zero(self) -> None:
        model = RelayPopulationModel()
        difficulty = RELAY_DIFFICULTIES[0]
        batch = build_relay_tensor_batch(
            (generate_relay_world(0, difficulty),),
            active_workers=16,
        )
        output = model(
            batch,
            communication_mode=CommunicationMode.NO_COMMUNICATION,
        )
        self.assertEqual(output.telemetry.messages_emitted, 0)
        self.assertEqual(output.telemetry.communicated_scalar_count, 0)


if __name__ == "__main__":
    unittest.main()
