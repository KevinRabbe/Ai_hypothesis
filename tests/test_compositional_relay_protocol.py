from __future__ import annotations

import unittest

import torch
from torch import nn

from ai_hypothesis.population_compute.collective_relay import (
    RELAY_DIFFICULTIES,
    generate_relay_world,
)
from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.model import (
    PopulationForwardOutput,
    PopulationTelemetry,
    SharedPopulationCell,
    SharedPopulationConfig,
)
from ai_hypothesis.population_compute.relay_model import (
    NODE_BIT_WIDTH,
    RelayPopulationModel,
    build_relay_tensor_batch,
)


class _RecordingCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_seed: torch.Tensor | None = None
        self.message_content: torch.Tensor | None = None

    def forward(
        self,
        local_inputs: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        recurrent_rounds: int,
        communication_mode: CommunicationMode,
        shared_seed: torch.Tensor | None = None,
        message_content: torch.Tensor | None = None,
    ) -> PopulationForwardOutput:
        del recurrent_rounds, communication_mode
        self.shared_seed = None if shared_seed is None else shared_seed.detach().clone()
        self.message_content = (
            None if message_content is None else message_content.detach().clone()
        )
        batch_size, worker_count, _ = local_inputs.shape
        if shared_seed is None:
            raise AssertionError("relay model must provide a shared query seed")
        return PopulationForwardOutput(
            logits=local_inputs.new_zeros(batch_size, NODE_BIT_WIDTH),
            final_states=local_inputs.new_zeros(batch_size, worker_count, 1),
            final_shared=shared_seed,
            telemetry=PopulationTelemetry(
                active_state_updates=int(active_mask.sum().item()),
                messages_emitted=0,
                communicated_scalar_count=0,
            ),
        )


class CompositionalRelayProtocolTests(unittest.TestCase):
    def test_external_message_content_controls_sparse_field_and_ignores_inactive_slots(self) -> None:
        torch.manual_seed(7)
        config = SharedPopulationConfig(
            local_input_width=6,
            state_width=8,
            message_width=4,
            output_width=3,
        )
        cell = SharedPopulationCell(config)
        with torch.no_grad():
            cell.message_gate.weight.zero_()
            cell.message_gate.bias.fill_(20.0)

        local_inputs = torch.zeros(1, 3, config.local_input_width)
        active_mask = torch.tensor([[True, False, True]])
        message_content = torch.tensor(
            [
                [
                    [0.10, -0.20, 0.30, -0.40],
                    [100.0, 100.0, 100.0, 100.0],
                    [-0.05, 0.15, -0.25, 0.35],
                ]
            ],
            dtype=torch.float32,
        )

        output = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=1,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            message_content=message_content,
        )

        gate = torch.sigmoid(torch.tensor(20.0))
        expected = torch.tanh(
            (message_content[:, 0, :] + message_content[:, 2, :]) * gate
        )
        self.assertTrue(torch.allclose(output.final_shared, expected, atol=1e-6))

    def test_external_message_content_shape_is_checked(self) -> None:
        config = SharedPopulationConfig(
            local_input_width=6,
            state_width=8,
            message_width=4,
            output_width=3,
        )
        cell = SharedPopulationCell(config)
        with self.assertRaisesRegex(ValueError, "message_content"):
            cell(
                torch.zeros(2, 3, 6),
                torch.ones(2, 3, dtype=torch.bool),
                recurrent_rounds=1,
                communication_mode=CommunicationMode.SPARSE_SHARED_V0,
                message_content=torch.zeros(2, 3, 5),
            )

    def test_relay_query_and_candidate_values_share_one_learned_node_projection(self) -> None:
        torch.manual_seed(11)
        model = RelayPopulationModel()
        world = generate_relay_world(17, RELAY_DIFFICULTIES[0])
        batch = build_relay_tensor_batch((world,), active_workers=16)
        expected_seed = torch.tanh(model.query_projection(batch.start_bits))
        value_bits = batch.local_inputs[..., NODE_BIT_WIDTH:]
        expected_content = torch.tanh(model.query_projection(value_bits))

        recorder = _RecordingCell()
        model.cell = recorder
        model(
            batch,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
        )

        self.assertIsNotNone(recorder.shared_seed)
        self.assertIsNotNone(recorder.message_content)
        assert recorder.shared_seed is not None
        assert recorder.message_content is not None
        self.assertTrue(torch.allclose(recorder.shared_seed, expected_seed, atol=1e-7))
        self.assertTrue(
            torch.allclose(recorder.message_content, expected_content, atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
