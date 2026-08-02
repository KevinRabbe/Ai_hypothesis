from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ai_hypothesis/population_compute/gate9d_learned_shared_router_hard_negative_v1.py"


def _load():
    name = "gate9d_router_hard_negative_contract"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load hard-negative router contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Gate9DHardNegativeRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load()

    def test_identity_and_parameter_count(self) -> None:
        self.assertEqual(self.module.VERSION, "gate9d-learned-shared-router-hard-negative-v1")
        self.assertEqual(self.module.BASE_HEAD, "d974277db6e270433876228b7534d44456ecee3e")
        self.assertEqual(self.module.v0._parameter_count(self.module.SharedRouter()), 1218)

    def test_exact_routing_strata(self) -> None:
        strata = self.module.routing_strata(torch.device("cpu"))
        self.assertEqual({name: values[0].numel() for name, values in strata.items()}, {
            "zero": 256,
            "selected_basis": 1024,
            "unselected_basis": 1024,
            "distractor": 63232,
        })

    def test_balanced_batch_and_finite_gradients(self) -> None:
        strata = self.module.routing_strata(torch.device("cpu"))
        generator = torch.Generator(device="cpu").manual_seed(1)
        worker, query, targets = self.module._sample_balanced(strata, generator, torch.device("cpu"))
        self.assertEqual(worker.numel(), 4096)
        self.assertEqual(query.numel(), 4096)
        self.assertEqual(tuple(targets.shape), (4096, 2))
        model = self.module.SharedRouter()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(worker, query), targets)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()))

    def test_source_boundaries(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "generate_gate9_test_world(",
            "scientific_assignment_key",
            "operator_identity_visible = True",
            "support_output_used_by_router = True",
            "end_to_end_answer_loss_used = True",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("selected_basis", source)
        self.assertIn("unselected_basis", source)
        self.assertIn("message_count_exact", source)


if __name__ == "__main__":
    unittest.main()
