from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import random
import sys
import unittest

CONTRACT_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate9_contextual_operator_contract.py"
)


def _load_contract():
    name = "gate9_contextual_operator_contract_test_module"
    spec = importlib.util.spec_from_file_location(name, CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate9 operator contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = _load_contract()
p = c.protocol


class Gate9OperatorContractTests(unittest.TestCase):
    def test_splitmix64_mapping_is_exactly_invertible(self):
        fixed = (0, 1, 2, 2**32, 2**40, 2**48, 2**64 - 2, 2**64 - 1)
        rng = random.Random(9009)
        values = fixed + tuple(rng.getrandbits(64) for _ in range(20_000))
        keys = []
        for counter in values:
            key = c.splitmix64_bijection(counter)
            self.assertEqual(c.inverse_splitmix64_bijection(key), counter)
            keys.append(key)
        self.assertEqual(len(keys), len(set(keys)))

    def test_key_factorization_support_and_oracle_round_trip(self):
        rng = random.Random(9010)
        keys = (0, 1, 2**64 - 1) + tuple(
            rng.getrandbits(64) for _ in range(4_096)
        )
        seen_support_hashes = set()
        for key in keys:
            operator = c.operator_from_key(key)
            support = c.public_support_pairs(operator)
            recovered = c.reconstruct_operator_from_support(support)
            self.assertEqual(recovered, operator)
            for query in (3, 5, 127, 255):
                self.assertEqual(
                    c.apply_public_support_oracle(support, query),
                    operator.apply(query),
                )
            seen_support_hashes.add(
                hashlib.sha256(repr(support).encode("ascii")).hexdigest()
            )
        self.assertEqual(len(seen_support_hashes), len(keys))

    def test_every_operator_is_a_bijection_over_all_256_symbols(self):
        counters = (0, 1, 2, 2**32, 2**40, 2**48, 2**64 - 1)
        for counter in counters:
            operator = c.operator_from_counter(counter)
            outputs = tuple(operator.apply(value) for value in range(256))
            self.assertEqual(len(set(outputs)), 256)
            self.assertEqual(set(outputs), set(range(256)))

    def test_public_support_order_is_global_and_has_no_operator_side_channel(self):
        expected = c.GATE9_GLOBAL_SUPPORT_ORDER
        self.assertEqual(set(expected), set(p.GATE9_SUPPORT_INPUTS))
        self.assertEqual(len(expected), 9)
        orders = {
            tuple(source for source, _ in c.public_support_pairs(c.operator_from_counter(counter)))
            for counter in (0, 1, 2, 2**32, 2**40, 2**48)
        }
        self.assertEqual(orders, {expected})

    def test_support_and_matrix_validation_fail_closed(self):
        operator = c.operator_from_counter(123)
        support = list(c.public_support_pairs(operator))

        with self.assertRaisesRegex(ValueError, "exactly nine"):
            c.reconstruct_operator_from_support(support[:-1])

        duplicate = support.copy()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            c.reconstruct_operator_from_support(duplicate)

        wrong_basis = support.copy()
        wrong_basis[-1] = (3, wrong_basis[-1][1])
        with self.assertRaisesRegex(ValueError, "basis drifted"):
            c.reconstruct_operator_from_support(wrong_basis)

        with self.assertRaisesRegex(ValueError, "outside the support basis"):
            c.apply_public_support_oracle(support, 0)

        non_family = tuple(0 for _ in range(8))
        with self.assertRaisesRegex(ValueError, "unit LU"):
            c.decompose_unit_lu(non_family)

    def test_split_audit_binds_exact_counts_and_closed_boundaries(self):
        audit = c.gate9_operator_split_audit()
        self.assertEqual(audit["protocol_head"], c.GATE9_PROTOCOL_HEAD)
        self.assertEqual(
            [row["count"] for row in audit["ranges"]],
            [262_144, 32_768, 4_096, 2_629_632],
        )
        self.assertTrue(all(value == 0 for value in audit["intersections"].values()))
        self.assertTrue(audit["counter_mapping_bijective"])
        self.assertTrue(audit["operator_mapping_injective"])
        self.assertTrue(audit["support_order_operator_independent"])
        self.assertFalse(audit["operator_key_visible_to_model"])
        self.assertTrue(audit["operator_generation_admitted"])
        self.assertTrue(audit["public_support_generation_admitted"])
        self.assertTrue(audit["public_support_oracle_admitted"])
        for key in (
            "graph_world_generation_admitted",
            "architecture_admitted",
            "training_admitted",
            "checkpoint_loading_admitted",
            "scientific_test_admitted",
            "result_classification_admitted",
        ):
            self.assertFalse(audit[key])

    def test_contract_has_no_graph_model_training_or_artifact_surface(self):
        text = CONTRACT_PATH.read_text(encoding="utf-8")
        forbidden = (
            "import torch",
            "import numpy",
            "generate_gate9_world(",
            "torch.load(",
            "torch.optim",
            ".backward(",
            "json.load",
            "from_pretrained(",
            "scientific_test_admitted\": True",
            "training_admitted\": True",
        )
        for token in forbidden:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
