from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch


WORLD_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
)
ARCHITECTURE_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_organism_architecture.py"
)
RUNTIME_PATH = pathlib.Path(
    "ai_hypothesis/population_compute/gate8_organism_runtime.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


w = _load(WORLD_PATH, "gate8_runtime_test_worlds")
a = _load(ARCHITECTURE_PATH, "gate8_runtime_test_architecture")
r = _load(RUNTIME_PATH, "gate8_runtime_test_contract")


def _constant_core(*, message_code: int = 23, answer_symbol: int = 9, active: bool = True):
    model = a.Gate8SharedWorkerCore()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.message_head.bias[message_code] = 1.0
        model.activity_head.bias.fill_(1.0 if active else -1.0)
        model.answer_head.bias[answer_symbol] = 1.0
    return model.eval()


def _contract_world(*, population: int = 32, depth: int = 4, world_index: int = 0):
    return w.generate_gate8_world(
        split="contract",
        seed=0,
        world_index=world_index,
        population=population,
        depth=depth,
    ).public


class Gate8OrganismRuntimeTests(unittest.TestCase):
    def test_full_runtime_reaches_target_synchronously(self) -> None:
        world = _contract_world(population=32, depth=4)
        result = r.run_gate8_contract_runtime(
            model=_constant_core(message_code=23, answer_symbol=9, active=True),
            world=world,
            mode=r.GATE8_RUNTIME_FULL,
        )
        self.assertTrue(result.target_reached)
        self.assertEqual(result.predicted_symbol, 9)
        self.assertEqual(result.rounds_executed, world.depth)
        self.assertEqual(result.rounds[0].mailbox_nodes_before, (world.query.root_node,))
        self.assertIn(
            result.target_worker_index,
            result.rounds[-1].scheduled_worker_indices,
        )
        self.assertEqual(result.communicated_bits, result.delivered_messages * 8)
        self.assertEqual(
            result.recurrent_updates,
            sum(len(row.scheduled_worker_indices) for row in result.rounds),
        )
        for row in result.rounds:
            self.assertEqual(row.recurrent_updates, len(row.scheduled_worker_indices))
            self.assertEqual(row.communicated_bits, len(row.delivered_messages) * 8)
            self.assertEqual(
                len({message.worker_index for message in row.delivered_messages}),
                len(row.delivered_messages),
            )
            self.assertLessEqual(
                len(row.delivered_messages),
                len(row.scheduled_worker_indices),
            )
            self.assertTrue(all(message.code == 23 for message in row.delivered_messages))

    def test_runtime_activation_gate_stops_propagation(self) -> None:
        world = _contract_world(population=32, depth=4, world_index=1)
        result = r.run_gate8_contract_runtime(
            model=_constant_core(active=False),
            world=world,
        )
        self.assertFalse(result.target_reached)
        self.assertIsNone(result.predicted_symbol)
        self.assertEqual(result.rounds_executed, 1)
        self.assertGreater(result.recurrent_updates, 0)
        self.assertEqual(result.activity_positive_workers, 0)
        self.assertEqual(result.delivered_messages, 0)
        self.assertEqual(result.communicated_bits, 0)

    def test_no_communication_ablation_suppresses_all_delivery(self) -> None:
        world = _contract_world(population=64, depth=8)
        result = r.run_gate8_contract_runtime(
            model=_constant_core(active=True),
            world=world,
            mode=r.GATE8_RUNTIME_NO_COMMUNICATION,
        )
        self.assertFalse(result.target_reached)
        self.assertEqual(result.rounds_executed, 1)
        self.assertGreater(result.activity_positive_workers, 0)
        self.assertEqual(result.delivered_messages, 0)
        self.assertEqual(result.communicated_bits, 0)
        self.assertTrue(
            all(not row.delivered_messages for row in result.rounds)
        )

    def test_shuffled_worker_ablation_is_deterministic_and_truth_independent(self) -> None:
        world = _contract_world(population=128, depth=16, world_index=7)
        model = _constant_core(active=True)
        first = r.run_gate8_contract_runtime(
            model=model,
            world=world,
            mode=r.GATE8_RUNTIME_SHUFFLED_WORKER,
        )
        second = r.run_gate8_contract_runtime(
            model=model,
            world=world,
            mode=r.GATE8_RUNTIME_SHUFFLED_WORKER,
        )
        identity = tuple(range(world.population))
        self.assertEqual(
            first.shuffled_worker_permutation,
            second.shuffled_worker_permutation,
        )
        self.assertNotEqual(first.shuffled_worker_permutation, identity)
        self.assertEqual(
            tuple(sorted(first.shuffled_worker_permutation)),
            identity,
        )
        self.assertEqual(
            sorted(first.effective_transform_ids),
            sorted(worker.transform_id for worker in world.workers),
        )
        self.assertEqual(first.effective_transform_ids, second.effective_transform_ids)
        self.assertEqual(first.rounds, second.rounds)
        self.assertTrue(first.target_reached)

    def test_full_and_shuffled_modes_preserve_topology_schedule_under_constant_policy(self) -> None:
        world = _contract_world(population=64, depth=8, world_index=4)
        model = _constant_core(active=True)
        full = r.run_gate8_contract_runtime(model=model, world=world)
        shuffled = r.run_gate8_contract_runtime(
            model=model,
            world=world,
            mode=r.GATE8_RUNTIME_SHUFFLED_WORKER,
        )
        self.assertEqual(
            tuple(row.scheduled_worker_indices for row in full.rounds),
            tuple(row.scheduled_worker_indices for row in shuffled.rounds),
        )
        self.assertEqual(full.recurrent_updates, shuffled.recurrent_updates)
        self.assertEqual(full.communicated_bits, shuffled.communicated_bits)
        self.assertEqual(full.target_worker_index, shuffled.target_worker_index)

    def test_contract_only_rejection_occurs_before_world_validation(self) -> None:
        class ForbiddenWorld:
            split = "test"

            def validate(self):
                raise AssertionError("validation must not run")

        with self.assertRaisesRegex(ValueError, "contract worlds only"):
            r.run_gate8_contract_runtime(
                model=_constant_core(),
                world=ForbiddenWorld(),
            )

    def test_runtime_requires_eval_mode_and_exact_parameter_count(self) -> None:
        world = _contract_world()
        training_model = _constant_core().train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            r.run_gate8_contract_runtime(model=training_model, world=world)

        class WrongCount(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def role_ids(self, **_kwargs):
                return torch.zeros(1, dtype=torch.long)

            def initial_hidden(self, _roles):
                return torch.zeros(1, 32)

        with self.assertRaisesRegex(ValueError, "19,649"):
            r.run_gate8_contract_runtime(
                model=WrongCount().eval(),
                world=world,
            )

    def test_runtime_plan_freezes_accounting_and_keeps_later_stages_closed(self) -> None:
        plan = r.gate8_organism_runtime_plan()
        self.assertEqual(
            plan["architecture_head"],
            "2afdcc9f13f138e97c7b3821cc2a5a77bd87cf0c",
        )
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertEqual(plan["admitted_split"], "contract")
        self.assertEqual(plan["root_seed_code"], 0)
        self.assertEqual(plan["root_seed_communicated_bits"], 0)
        self.assertEqual(plan["activity_threshold"], 0.0)
        self.assertEqual(plan["message_selection"], "argmax")
        self.assertEqual(plan["message_bits"], 8)
        self.assertTrue(plan["one_message_per_scheduled_worker_per_round"])
        self.assertFalse(plan["truth_used_by_runtime"])
        self.assertFalse(plan["model_outputs_used_to_construct_ablation"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["checkpoint_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])
        self.assertFalse(plan["reference_model_admitted"])

    def test_runtime_source_contains_no_oracle_training_or_reference_model_path(self) -> None:
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        for token in (
            "apply_gate8_transform",
            "gate8_exact_symbolic_oracle",
            ".truth",
            "generate_gate8_world",
            "torch.optim",
            "optimizer.step",
            "backward(",
            "DataLoader",
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
