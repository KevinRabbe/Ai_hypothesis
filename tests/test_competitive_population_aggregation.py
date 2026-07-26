from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.model import SharedPopulationCell, SharedPopulationConfig


class CompetitivePopulationAggregationTests(unittest.TestCase):
    def test_competitive_active_weights_form_one_bounded_message_mixture(self) -> None:
        torch.manual_seed(91)
        config = SharedPopulationConfig(
            local_input_width=6,
            state_width=8,
            message_width=4,
            output_width=3,
        )
        cell = SharedPopulationCell(config)
        local_inputs = torch.randn(2, 4, config.local_input_width)
        active_mask = torch.tensor(
            [
                [True, True, False, True],
                [True, False, True, True],
            ]
        )
        active_content = torch.tensor([0.25, -0.5, 0.75, -0.125])
        message_content = active_content.view(1, 1, -1).expand(2, 4, -1).clone()
        message_content[:, 2, :] = 1000.0
        message_content[1, 1, :] = -1000.0

        output = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=1,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            message_content=message_content,
            competitive_message_weights=True,
        )

        expected = torch.tanh(active_content).view(1, -1).expand(2, -1)
        self.assertTrue(torch.allclose(output.final_shared, expected, atol=1e-6))

    def test_noncompetitive_path_remains_available(self) -> None:
        torch.manual_seed(92)
        config = SharedPopulationConfig(
            local_input_width=6,
            state_width=8,
            message_width=4,
            output_width=3,
        )
        cell = SharedPopulationCell(config)
        local_inputs = torch.randn(1, 3, config.local_input_width)
        active_mask = torch.ones(1, 3, dtype=torch.bool)
        content = torch.randn(1, 3, config.message_width)

        ordinary = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=1,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            message_content=content,
            competitive_message_weights=False,
        )
        competitive = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=1,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            message_content=content,
            competitive_message_weights=True,
        )

        self.assertFalse(torch.allclose(ordinary.final_shared, competitive.final_shared))


if __name__ == "__main__":
    unittest.main()
