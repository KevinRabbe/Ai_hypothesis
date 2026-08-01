from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import unittest

import torch

ARCHITECTURE_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_worker_architecture.py"
)


def _load_architecture():
    name = "gate9_contextual_worker_architecture_test_module"
    spec = importlib.util.spec_from_file_location(name, ARCHITECTURE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 worker architecture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a = _load_architecture()


class Gate9ContextualWorkerArchitectureTests(unittest.TestCase):
    def _supports(self, batch: int = 4):
        rows = []
        for index in range(batch):
            rows.append(
                tuple(
                    (value, (value * (2 * index + 1) + 7 + index) % 256)
                    for value in a.GATE9_SUPPORT_ORDER
                )
            )
        return tuple(rows)

    def test_exact_parameter_count_breakdown_and_plan(self):
        torch.manual_seed(9009)
        model = a.Gate9ContextualWorker()
        self.assertEqual(model.learned_parameter_count(), 19_649)
        self.assertEqual(model.parameter_breakdown(), a.GATE9_PARAMETER_BREAKDOWN)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 19_649)
        self.assertEqual(len(tuple(model.parameters())), 17)
        plan = a.gate9_contextual_worker_architecture_plan()
        self.assertEqual(plan["graph_world_contract_head"], a.GATE9_GRAPH_WORLD_CONTRACT_HEAD)
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertEqual(plan["parameter_breakdown"], a.GATE9_PARAMETER_BREAKDOWN)
        self.assertTrue(plan["shared_across_workers"])
        self.assertTrue(plan["shared_across_populations"])
        self.assertTrue(plan["shared_across_rounds"])
        self.assertEqual(plan["per_operator_parameters"], 0)
        self.assertEqual(plan["padding_parameters"], 0)
        self.assertTrue(plan["contract_forward_backward_admitted"])
        for key in (
            "optimizer_admitted",
            "training_admitted",
            "checkpoint_serialization_admitted",
            "checkpoint_loading_admitted",
            "scientific_test_world_generation_admitted",
            "scientific_execution_admitted",
            "result_classification_admitted",
        ):
            self.assertFalse(plan[key])

    def test_forward_backward_uses_every_parameter_and_decodes_bytes(self):
        torch.manual_seed(9010)
        model = a.Gate9ContextualWorker()
        support_inputs, support_outputs, query = a.serialize_gate9_worker_batch(
            self._supports(8),
            (3, 5, 6, 7, 9, 10, 11, 12),
        )
        logits = model(support_inputs, support_outputs, query)
        self.assertEqual(tuple(logits.shape), (8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        decoded = model.decode_bytes(logits)
        self.assertEqual(tuple(decoded.shape), (8,))
        self.assertEqual(decoded.dtype, torch.long)
        self.assertTrue(torch.all((0 <= decoded) & (decoded <= 255)))

        targets = torch.tensor(
            [[(row >> bit) & 1 for bit in range(8)] for row in range(8)],
            dtype=torch.float32,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_serializer_exposes_only_support_pairs_and_query_in_fixed_order(self):
        signature = inspect.signature(a.serialize_gate9_worker_batch)
        self.assertEqual(
            tuple(signature.parameters),
            ("support_sets", "queries", "device"),
        )
        support_inputs, support_outputs, query = a.serialize_gate9_worker_batch(
            self._supports(2), (3, 255)
        )
        self.assertEqual(tuple(support_inputs.shape), (2, 9))
        self.assertEqual(tuple(support_outputs.shape), (2, 9))
        self.assertEqual(tuple(query.shape), (2,))
        self.assertEqual(tuple(support_inputs[0].tolist()), a.GATE9_SUPPORT_ORDER)
        for forbidden in (
            "operator_counter",
            "operator_key",
            "world_id",
            "worker_id",
            "node_id",
            "population",
            "depth",
            "round",
            "target_flag",
        ):
            self.assertNotIn(forbidden, signature.parameters)

    def test_serializer_and_forward_fail_closed_on_schema_order_dtype_and_range(self):
        supports = self._supports(1)
        reversed_support = (tuple(reversed(supports[0])),)
        with self.assertRaisesRegex(ValueError, "qualified order"):
            a.serialize_gate9_worker_batch(reversed_support, (3,))
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            a.serialize_gate9_worker_batch(supports, ())
        bad = list(supports[0])
        bad[0] = (bad[0][0], 256)
        with self.assertRaisesRegex(ValueError, "0..255"):
            a.serialize_gate9_worker_batch((tuple(bad),), (3,))

        model = a.Gate9ContextualWorker()
        inputs, outputs, query = a.serialize_gate9_worker_batch(supports, (3,))
        with self.assertRaisesRegex(ValueError, "torch.long"):
            model(inputs.float(), outputs, query)
        with self.assertRaisesRegex(ValueError, "qualified global order"):
            model(inputs.flip(1), outputs, query)
        with self.assertRaisesRegex(ValueError, "one byte per worker"):
            model(inputs, outputs, query.unsqueeze(1))
        with self.assertRaisesRegex(ValueError, r"shape \[batch,8\]"):
            model.decode_bytes(torch.zeros(1, 7))

    def test_support_context_and_query_both_affect_logits(self):
        torch.manual_seed(9011)
        model = a.Gate9ContextualWorker().eval()
        supports = self._supports(2)
        inputs, outputs, query = a.serialize_gate9_worker_batch(supports, (3, 3))
        logits = model(inputs, outputs, query)
        self.assertFalse(torch.equal(logits[0], logits[1]))

        inputs2, outputs2, query2 = a.serialize_gate9_worker_batch(
            (supports[0], supports[0]), (3, 5)
        )
        logits2 = model(inputs2, outputs2, query2)
        self.assertFalse(torch.equal(logits2[0], logits2[1]))

    def test_architecture_has_no_optimizer_training_checkpoint_or_science_surface(self):
        text = ARCHITECTURE_PATH.read_text(encoding="utf-8")
        forbidden = (
            "torch.optim",
            "torch.load(",
            "torch.save(",
            ".backward(",
            "generate_gate9_test_world(",
            "generate_gate9_contract_world(",
            "operator_from_counter(",
            "json.load",
            "from_pretrained(",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
