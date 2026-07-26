from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.model import (
    SharedPopulationCell,
    SharedPopulationConfig,
)


class SharedPopulationCellTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.config = SharedPopulationConfig(
            local_input_width=6,
            state_width=12,
            message_width=4,
            output_width=5,
        )
        self.model = SharedPopulationCell(self.config)

    def test_parameter_identity_is_independent_of_runtime_population_size(self) -> None:
        before_count = self.model.trainable_parameter_count()
        before_fingerprint = self.model.parameter_fingerprint()

        for workers in (1, 4, 16, 64, 256):
            local_inputs = torch.randn(2, workers, self.config.local_input_width)
            mask = torch.ones(2, workers, dtype=torch.bool)
            output = self.model(
                local_inputs,
                mask,
                recurrent_rounds=2,
                communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            )
            self.assertEqual(output.logits.shape, (2, self.config.output_width))
            self.assertEqual(
                output.final_states.shape,
                (2, workers, self.config.state_width),
            )
            self.assertEqual(
                output.final_shared.shape,
                (2, self.config.message_width),
            )

        self.assertEqual(self.model.trainable_parameter_count(), before_count)
        self.assertEqual(self.model.parameter_fingerprint(), before_fingerprint)

    def test_no_communication_reports_zero_communication_and_preserves_seed(self) -> None:
        local_inputs = torch.randn(1, 4, self.config.local_input_width)
        mask = torch.ones(1, 4, dtype=torch.bool)
        seed = torch.randn(1, self.config.message_width)

        output = self.model(
            local_inputs,
            mask,
            recurrent_rounds=3,
            communication_mode=CommunicationMode.NO_COMMUNICATION,
            shared_seed=seed,
        )

        self.assertEqual(output.telemetry.active_state_updates, 12)
        self.assertEqual(output.telemetry.messages_emitted, 0)
        self.assertEqual(output.telemetry.communicated_scalar_count, 0)
        self.assertTrue(torch.equal(output.final_shared, seed))

    def test_sparse_shared_communication_is_linear_in_active_states(self) -> None:
        local_inputs = torch.randn(2, 4, self.config.local_input_width)
        mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
            ]
        )

        output = self.model(
            local_inputs,
            mask,
            recurrent_rounds=3,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
        )

        active_states = 5
        expected_messages = active_states * 3
        self.assertEqual(output.telemetry.active_state_updates, expected_messages)
        self.assertEqual(output.telemetry.messages_emitted, expected_messages)
        self.assertEqual(
            output.telemetry.communicated_scalar_count,
            expected_messages * self.config.message_width * 2,
        )
        self.assertTrue(torch.all(output.final_shared.abs() <= 1.0))

    def test_learned_hot_path_executes_only_active_states(self) -> None:
        local_inputs = torch.randn(2, 4, self.config.local_input_width)
        mask = torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
            ]
        )
        active_states = 4
        seen: dict[str, list[int]] = {
            "input_projection": [],
            "update": [],
            "message_gate": [],
            "message_projection": [],
        }

        def record(name: str):
            def hook(_module, args) -> None:
                seen[name].append(int(args[0].shape[0]))

            return hook

        handles = [
            self.model.input_projection.register_forward_pre_hook(record("input_projection")),
            self.model.update.register_forward_pre_hook(record("update")),
            self.model.message_gate.register_forward_pre_hook(record("message_gate")),
            self.model.message_projection.register_forward_pre_hook(record("message_projection")),
        ]
        try:
            self.model(
                local_inputs,
                mask,
                recurrent_rounds=2,
                communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(seen["input_projection"], [active_states])
        self.assertEqual(seen["update"], [active_states, active_states])
        self.assertEqual(seen["message_gate"], [active_states, active_states])
        self.assertEqual(seen["message_projection"], [active_states, active_states])

    def test_inactive_states_remain_zero(self) -> None:
        local_inputs = torch.randn(1, 4, self.config.local_input_width)
        mask = torch.tensor([[True, False, True, False]])

        output = self.model(
            local_inputs,
            mask,
            recurrent_rounds=2,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
        )

        self.assertTrue(torch.equal(output.final_states[0, 1], torch.zeros(12)))
        self.assertTrue(torch.equal(output.final_states[0, 3], torch.zeros(12)))

    def test_shared_seed_shape_is_checked(self) -> None:
        local_inputs = torch.randn(2, 4, self.config.local_input_width)
        mask = torch.ones(2, 4, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "shared_seed"):
            self.model(
                local_inputs,
                mask,
                recurrent_rounds=1,
                communication_mode=CommunicationMode.SPARSE_SHARED_V0,
                shared_seed=torch.randn(2, self.config.message_width + 1),
            )

    def test_every_sample_requires_an_active_state(self) -> None:
        local_inputs = torch.randn(2, 4, self.config.local_input_width)
        mask = torch.tensor(
            [
                [True, False, False, False],
                [False, False, False, False],
            ]
        )

        with self.assertRaisesRegex(ValueError, "at least one worker state"):
            self.model(
                local_inputs,
                mask,
                recurrent_rounds=1,
                communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            )


if __name__ == "__main__":
    unittest.main()
