from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import unittest

import torch


MODULE_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_organism_architecture.py"
)


def _load_architecture():
    name = "gate8_organism_architecture_test_module"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 organism architecture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a = _load_architecture()


class Gate8OrganismArchitectureTests(unittest.TestCase):
    def test_exact_parameter_count_and_static_ledger(self) -> None:
        model = a.Gate8SharedWorkerCore()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            19_649,
        )
        ledger = a.gate8_architecture_parameter_ledger()
        self.assertEqual(sum(ledger.values()), 19_649)
        self.assertEqual(
            ledger,
            {
                "message_code_embedding": 3_072,
                "transform_embedding": 96,
                "root_symbol_embedding": 192,
                "role_embedding": 48,
                "initial_hidden_by_role": 128,
                "worker_update_weight_ih": 3_840,
                "worker_update_weight_hh": 3_072,
                "worker_update_biases": 192,
                "message_head": 8_448,
                "activity_head": 33,
                "answer_head": 528,
            },
        )

    def test_observed_parameter_shapes_match_the_frozen_architecture(self) -> None:
        model = a.Gate8SharedWorkerCore()
        observed = {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
        }
        self.assertEqual(
            observed,
            {
                "initial_hidden_by_role": (4, 32),
                "message_code_embedding.weight": (256, 12),
                "transform_embedding.weight": (8, 12),
                "root_symbol_embedding.weight": (16, 12),
                "role_embedding.weight": (4, 12),
                "worker_update.weight_ih": (96, 40),
                "worker_update.weight_hh": (96, 32),
                "worker_update.bias_ih": (96,),
                "worker_update.bias_hh": (96,),
                "message_head.weight": (256, 32),
                "message_head.bias": (256,),
                "activity_head.weight": (1, 32),
                "activity_head.bias": (1,),
                "answer_head.weight": (16, 32),
                "answer_head.bias": (16,),
            },
        )

    def test_role_ids_cover_exact_public_query_roles(self) -> None:
        source_is_root = torch.tensor([False, False, True, True])
        target_is_query = torch.tensor([False, True, False, True])
        observed = a.Gate8SharedWorkerCore.role_ids(
            source_is_root=source_is_root,
            target_is_query=target_is_query,
        )
        self.assertEqual(observed.tolist(), [0, 1, 2, 3])

    def test_one_shared_core_supports_small_and_large_worker_batches(self) -> None:
        torch.manual_seed(0)
        model = a.Gate8SharedWorkerCore().eval()
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        for population in (32, 1_024):
            source_is_root = torch.zeros(population, dtype=torch.bool)
            source_is_root[0] = True
            target_is_query = torch.zeros(population, dtype=torch.bool)
            target_is_query[-1] = True
            role_ids = model.role_ids(
                source_is_root=source_is_root,
                target_is_query=target_is_query,
            )
            hidden = model.initial_hidden(role_ids)
            output = model(
                inbox_code=torch.zeros(population, dtype=torch.long),
                transform_id=torch.arange(population, dtype=torch.long) % 8,
                root_symbol=torch.full(
                    (population,),
                    7,
                    dtype=torch.long,
                ),
                source_is_root=source_is_root,
                target_is_query=target_is_query,
                inbox_present=source_is_root,
                round_is_zero=torch.ones(population, dtype=torch.bool),
                hidden=hidden,
            )
            self.assertEqual(output.hidden.shape, (population, 32))
            self.assertEqual(output.message_logits.shape, (population, 256))
            self.assertEqual(output.activity_logit.shape, (population,))
            self.assertEqual(output.answer_logits.shape, (population, 16))
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                19_649,
            )
            self.assertEqual(
                tuple(id(parameter) for parameter in model.parameters()),
                parameter_ids,
            )

    def test_forward_surface_contains_no_node_worker_population_or_depth_identity(self) -> None:
        parameters = set(
            inspect.signature(a.Gate8SharedWorkerCore.forward).parameters
        )
        self.assertEqual(
            parameters,
            {
                "self",
                "inbox_code",
                "transform_id",
                "root_symbol",
                "source_is_root",
                "target_is_query",
                "inbox_present",
                "round_is_zero",
                "hidden",
            },
        )
        for forbidden in (
            "node_id",
            "source_node",
            "target_node",
            "worker_index",
            "population",
            "depth",
            "round_index",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_invalid_indices_and_shapes_fail_closed(self) -> None:
        model = a.Gate8SharedWorkerCore()
        source = torch.tensor([True])
        target = torch.tensor([False])
        hidden = model.initial_hidden(
            model.role_ids(
                source_is_root=source,
                target_is_query=target,
            )
        )
        base = dict(
            inbox_code=torch.tensor([0], dtype=torch.long),
            transform_id=torch.tensor([0], dtype=torch.long),
            root_symbol=torch.tensor([0], dtype=torch.long),
            source_is_root=source,
            target_is_query=target,
            inbox_present=torch.tensor([True]),
            round_is_zero=torch.tensor([True]),
            hidden=hidden,
        )
        invalid_transform = dict(base)
        invalid_transform["transform_id"] = torch.tensor([8], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "transform_id"):
            model(**invalid_transform)
        invalid_hidden = dict(base)
        invalid_hidden["hidden"] = torch.zeros(1, 31)
        with self.assertRaisesRegex(ValueError, "hidden state"):
            model(**invalid_hidden)
        invalid_flag = dict(base)
        invalid_flag["inbox_present"] = torch.tensor([1], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "inbox_present"):
            model(**invalid_flag)

    def test_architecture_plan_keeps_training_and_execution_closed(self) -> None:
        plan = a.gate8_organism_architecture_plan()
        self.assertEqual(
            plan["base_result_head"],
            "c7f5260189ef9ac1a1beb73596446316631090c7",
        )
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertTrue(plan["shared_across_workers"])
        self.assertTrue(plan["shared_across_populations"])
        self.assertTrue(plan["shared_across_rounds"])
        self.assertFalse(plan["runtime_topology_state_learned"])
        self.assertFalse(plan["node_identity_parameters"])
        self.assertFalse(plan["worker_identity_parameters"])
        self.assertFalse(plan["population_specific_parameters"])
        self.assertFalse(plan["depth_specific_parameters"])
        self.assertFalse(plan["graph_scheduler_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["checkpoint_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])
        self.assertFalse(plan["reference_model_admitted"])

    def test_architecture_module_has_no_training_data_or_reference_model_surface(self) -> None:
        text = MODULE_PATH.read_text(encoding="utf-8")
        for token in (
            "torch.optim",
            "optimizer.step",
            "backward(",
            "DataLoader",
            "generate_gate8_world",
            "split=\"test\"",
            "AutoModel",
            "AutoTokenizer",
            "from_pretrained",
            "snapshot_download",
            "model.safetensors",
            "checkpoint.pt",
        ):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
