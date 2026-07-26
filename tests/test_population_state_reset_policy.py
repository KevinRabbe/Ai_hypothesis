from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_compute.contract import CommunicationMode
from ai_hypothesis.population_compute.model import SharedPopulationCell, SharedPopulationConfig


class PopulationStateResetPolicyTests(unittest.TestCase):
    def test_two_reset_rounds_equal_two_one_round_calls_chained_only_by_shared_field(self) -> None:
        torch.manual_seed(123)
        config = SharedPopulationConfig(
            local_input_width=6,
            state_width=9,
            message_width=5,
            output_width=4,
        )
        cell = SharedPopulationCell(config)
        local_inputs = torch.randn(2, 4, config.local_input_width)
        active_mask = torch.tensor(
            [
                [True, True, False, True],
                [True, False, True, True],
            ]
        )
        shared_seed = torch.randn(2, config.message_width)
        message_content = torch.tanh(torch.randn(2, 4, config.message_width))

        combined = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=2,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            shared_seed=shared_seed,
            message_content=message_content,
            reset_state_each_round=True,
        )
        first = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=1,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            shared_seed=shared_seed,
            message_content=message_content,
        )
        second = cell(
            local_inputs,
            active_mask,
            recurrent_rounds=1,
            communication_mode=CommunicationMode.SPARSE_SHARED_V0,
            shared_seed=first.final_shared,
            message_content=message_content,
        )

        self.assertTrue(torch.allclose(combined.final_shared, second.final_shared, atol=1e-7))
        self.assertTrue(torch.allclose(combined.final_states, second.final_states, atol=1e-7))
        self.assertTrue(torch.allclose(combined.logits, second.logits, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
