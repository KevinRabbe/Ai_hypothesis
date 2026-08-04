from __future__ import annotations

from collections import OrderedDict
import copy
import unittest

import torch
from torch.nn import functional as F

from ai_hypothesis.population_language import l0_protocol
from ai_hypothesis.population_language.l0_models import PopulationLanguageOrganism
from ai_hypothesis.population_language.l0_reference_training import canonical_state_sha256
from ai_hypothesis.population_language.post_training_learning_l0_adapter import (
    AdapterConfig,
    BoundedPopulationAdapter,
    NAMES,
    parameter_count,
    raw_fp32_bytes,
    validate_adapter_contract,
)


def task_prefix() -> torch.Tensor:
    ids = l0_protocol.TOKEN_TO_ID
    return torch.tensor(
        [[ids["<bos>"], ids["<query>"], ids["dax"], ids["red"], ids["<answer>"]]],
        dtype=torch.long,
    )


def make_base(seed: int = 41) -> PopulationLanguageOrganism:
    torch.manual_seed(seed)
    return PopulationLanguageOrganism(communication_rounds=1, top_k=1)


class AdapterContracts(unittest.TestCase):
    def test_static_contract_and_invalid_configuration(self) -> None:
        self.assertTrue(validate_adapter_contract()["valid"])
        self.assertEqual(parameter_count(6), 180_176)
        self.assertEqual(raw_fp32_bytes(6), 720_704)
        with self.assertRaises(ValueError):
            AdapterConfig(rank=3).validate()
        with self.assertRaises(ValueError):
            AdapterConfig(rank=2, alpha=float("nan")).validate()

    def test_production_shape_noop_freeze_and_original_l0_bypass(self) -> None:
        base = make_base()
        adapter = BoundedPopulationAdapter(base, model_seed=120100)
        self.assertEqual(adapter.trainable_parameter_count(), 180_176)
        self.assertEqual(adapter.raw_fp32_tensor_bytes(), 720_704)
        self.assertEqual(tuple(adapter.declared_parameters()), NAMES)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))

        inputs = task_prefix()
        with torch.no_grad():
            expected = base(inputs, worker_count=1)
            observed = adapter(inputs, worker_count=1)
        self.assertTrue(torch.equal(expected, observed))

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected_bf16 = base(inputs, worker_count=1)
            observed_bf16 = adapter(inputs, worker_count=1)
        self.assertTrue(torch.equal(expected_bf16, observed_bf16))

        original = torch.tensor([l0_protocol.make_episode("train", 0).token_ids], dtype=torch.long)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(original, worker_count=1), adapter(original, worker_count=1)))

    def test_update_isolated_and_tensor_only_restart_is_exact(self) -> None:
        base = make_base()
        adapter = BoundedPopulationAdapter(base, model_seed=120100)
        base_before = canonical_state_sha256(base)
        optimizer = torch.optim.SGD(adapter.declared_parameters().values(), lr=0.1)
        target = torch.tensor([l0_protocol.TOKEN_TO_ID["blue"]])
        loss = F.cross_entropy(adapter(task_prefix(), worker_count=1)[:, -1], target)
        loss.backward()
        optimizer.step()
        self.assertEqual(canonical_state_sha256(base), base_before)
        self.assertTrue(any(torch.count_nonzero(value).item() for value in adapter.adaptation_state_dict().values()))

        state = adapter.adaptation_state_dict()
        restarted_base = make_base()
        self.assertEqual(canonical_state_sha256(restarted_base), base_before)
        restarted = BoundedPopulationAdapter(restarted_base, model_seed=120100)
        restarted.load_adaptation_state_dict(state)
        with torch.no_grad():
            self.assertTrue(torch.equal(
                adapter(task_prefix(), worker_count=1),
                restarted(task_prefix(), worker_count=1),
            ))

        wrong_order = OrderedDict(reversed(tuple(state.items())))
        with self.assertRaises(ValueError):
            restarted.load_adaptation_state_dict(wrong_order)
        wrong_dtype = copy.deepcopy(state)
        wrong_dtype["value_logit_bias"] = wrong_dtype["value_logit_bias"].to(torch.float64)
        with self.assertRaises(ValueError):
            restarted.load_adaptation_state_dict(wrong_dtype)
        nonfinite = copy.deepcopy(state)
        nonfinite["value_logit_bias"][0] = float("nan")
        with self.assertRaises(ValueError):
            restarted.load_adaptation_state_dict(nonfinite)

    def test_mixed_task_and_original_batch_is_rejected(self) -> None:
        adapter = BoundedPopulationAdapter(make_base(), model_seed=120100)
        task = task_prefix()[0]
        not_task = task.clone()
        not_task[1] = l0_protocol.TOKEN_TO_ID["<def>"]
        with self.assertRaises(ValueError):
            adapter(torch.stack((task, not_task)), worker_count=1)


if __name__ == "__main__":
    unittest.main()
