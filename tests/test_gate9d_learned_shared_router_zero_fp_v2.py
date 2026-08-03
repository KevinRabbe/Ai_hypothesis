from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "ai_hypothesis/population_compute/gate9d_learned_shared_router_zero_fp_v2.py"


def _load():
    spec = importlib.util.spec_from_file_location("gate9d_router_zero_fp_test", MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load zero-FP router module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ZeroFalsePositiveRouterTests(unittest.TestCase):
    def test_threshold_requires_strict_separation(self) -> None:
        module = _load()
        model = module.SharedRouter()
        worker, query, targets = module.v0.exhaustive_router_domain(torch.device("cpu"))
        with torch.no_grad():
            logits = torch.empty((worker.numel(), 2))
            logits[:, 0] = torch.where(targets[:, 0].bool(), 3.0, -2.0)
            logits[:, 1] = torch.where(targets[:, 1].bool(), 4.0, -1.0)

        class Fixed(torch.nn.Module):
            def forward(self, worker_input, query_input):
                index = worker_input * 256 + query_input
                return logits[index]

        calibration = module.calibrate_thresholds(Fixed(), torch.device("cpu"))
        self.assertTrue(calibration["separable"])
        self.assertGreater(calibration["gates"]["bias"]["margin"], 0.0)
        self.assertGreater(calibration["gates"]["contribution"]["margin"], 0.0)
        predictions = module.threshold_predictions(logits, calibration)
        self.assertTrue(torch.equal(predictions, targets.bool()))

    def test_overlap_fails_closed(self) -> None:
        module = _load()

        class Flat(torch.nn.Module):
            def forward(self, worker_input, query_input):
                return torch.zeros((worker_input.numel(), 2))

        calibration = module.calibrate_thresholds(Flat(), torch.device("cpu"))
        self.assertFalse(calibration["separable"])
        with self.assertRaisesRegex(RuntimeError, "not strictly separable"):
            module.threshold_predictions(torch.zeros((1, 2)), calibration)

    def test_contract_and_boundaries(self) -> None:
        module = _load()
        self.assertEqual(module.VERSION, "gate9d-learned-shared-router-zero-fp-v2")
        self.assertEqual(module.BASE_HEAD, "5e89fb42d6a84e32f163d3309abbb2294206f9a1")
        self.assertEqual(module.v0._parameter_count(module.SharedRouter()), 1218)
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("max_negative_logit", source)
        self.assertIn("min_positive_logit", source)
        for forbidden in (
            "generate_gate9_test_world(",
            "scientific_assignment_key",
            "classify_diagnostic(",
            "torch.save(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
