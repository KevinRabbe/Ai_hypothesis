from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import unittest

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = (
    ROOT
    / "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
)
ARCHITECTURE_PATH = (
    ROOT
    / "ai_hypothesis/population_compute/gate8_factorized_organism_architecture.py"
)
RUNTIME_PATH = (
    ROOT
    / "ai_hypothesis/population_compute/gate8_factorized_organism_runtime.py"
)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 v1 test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORLDS = _load(WORLD_PATH, "gate8_v1_runtime_test_worlds")
ARCH = _load(ARCHITECTURE_PATH, "gate8_v1_runtime_test_architecture")
RUNTIME = _load(RUNTIME_PATH, "gate8_v1_runtime_test_runtime")


class Gate8FactorizedOrganismRuntimeTests(unittest.TestCase):
    @staticmethod
    def _world(*, population: int = 32, depth: int = 4, world_index: int = 0):
        return WORLDS.generate_gate8_world(
            split="contract",
            seed=0,
            world_index=world_index,
            population=population,
            depth=depth,
        )

    @staticmethod
    def _controlled_model(*, carrier: int = 5, symbol: int = 9):
        model = ARCH.Gate8V1SharedWorkerCore()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.carrier_head.bias[carrier] = 1.0
            model.symbol_head.bias[symbol] = 1.0
        return model.eval()

    def test_full_runtime_seeds_root_and_delivers_every_emission(self):
        generated = self._world()
        model = self._controlled_model(carrier=5, symbol=9)
        result = RUNTIME.run_gate8_v1_contract_runtime(
            model=model,
            world=generated.public,
        )

        self.assertEqual(result.mode, RUNTIME.GATE8_V1_RUNTIME_FULL)
        self.assertEqual(result.root_seed_code, generated.public.query.root_symbol)
        self.assertEqual(result.rounds_executed, generated.public.depth)
        self.assertTrue(result.target_reached)
        self.assertEqual(result.predicted_symbol, 9)
        self.assertEqual(result.target_message_code, (5 << 4) | 9)
        self.assertEqual(result.target_message_code & 0x0F, result.predicted_symbol)
        self.assertEqual(result.emitted_messages, result.recurrent_updates)
        self.assertEqual(result.delivered_messages, result.emitted_messages)
        self.assertEqual(result.communicated_bits, result.delivered_messages * 8)
        self.assertGreaterEqual(result.recurrent_updates, generated.public.depth)
        self.assertLessEqual(result.recurrent_updates, generated.public.population)

        first_round = result.rounds[0]
        self.assertTrue(first_round.inbox_codes)
        self.assertEqual(
            set(first_round.inbox_codes),
            {generated.public.query.root_symbol},
        )
        fixed_code = (5 << 4) | 9
        for row in result.rounds[1:]:
            self.assertEqual(set(row.inbox_codes), {fixed_code})
        for row in result.rounds:
            self.assertEqual(
                len(row.delivered_messages),
                len(row.scheduled_worker_indices),
            )
            self.assertEqual(
                tuple(message.code for message in row.delivered_messages),
                row.emitted_message_codes,
            )

    def test_no_communication_emits_but_does_not_deliver(self):
        generated = self._world(world_index=1)
        result = RUNTIME.run_gate8_v1_contract_runtime(
            model=self._controlled_model(),
            world=generated.public,
            mode=RUNTIME.GATE8_V1_RUNTIME_NO_COMMUNICATION,
        )

        self.assertEqual(result.rounds_executed, 1)
        self.assertFalse(result.target_reached)
        self.assertGreater(result.emitted_messages, 0)
        self.assertEqual(result.emitted_messages, result.recurrent_updates)
        self.assertEqual(result.delivered_messages, 0)
        self.assertEqual(result.communicated_bits, 0)
        self.assertFalse(result.rounds[0].delivered_messages)
        self.assertTrue(result.rounds[0].emitted_message_codes)

    def test_shuffled_worker_is_deterministic_and_preserves_marginals(self):
        generated = self._world(population=64, depth=8, world_index=2)
        model = self._controlled_model(carrier=7, symbol=3)
        first = RUNTIME.run_gate8_v1_contract_runtime(
            model=model,
            world=generated.public,
            mode=RUNTIME.GATE8_V1_RUNTIME_SHUFFLED_WORKER,
        )
        second = RUNTIME.run_gate8_v1_contract_runtime(
            model=model,
            world=generated.public,
            mode=RUNTIME.GATE8_V1_RUNTIME_SHUFFLED_WORKER,
        )
        full = RUNTIME.run_gate8_v1_contract_runtime(
            model=model,
            world=generated.public,
            mode=RUNTIME.GATE8_V1_RUNTIME_FULL,
        )

        identity = tuple(range(generated.public.population))
        original = tuple(worker.transform_id for worker in generated.public.workers)
        self.assertEqual(
            first.shuffled_worker_permutation,
            second.shuffled_worker_permutation,
        )
        self.assertNotEqual(first.shuffled_worker_permutation, identity)
        self.assertEqual(sorted(first.effective_transform_ids), sorted(original))
        self.assertNotEqual(first.effective_transform_ids, original)
        self.assertEqual(first.recurrent_updates, full.recurrent_updates)
        self.assertEqual(first.delivered_messages, full.delivered_messages)
        self.assertEqual(first.communicated_bits, full.communicated_bits)
        self.assertEqual(first.predicted_symbol, full.predicted_symbol)

    def test_maximum_condition_reaches_target_with_exact_accounting(self):
        generated = self._world(population=1_024, depth=128, world_index=3)
        result = RUNTIME.run_gate8_v1_contract_runtime(
            model=self._controlled_model(carrier=15, symbol=6),
            world=generated.public,
        )

        self.assertTrue(result.target_reached)
        self.assertEqual(result.rounds_executed, 128)
        self.assertEqual(result.predicted_symbol, 6)
        self.assertEqual(result.target_message_code, 246)
        self.assertEqual(result.emitted_messages, result.recurrent_updates)
        self.assertEqual(result.delivered_messages, result.recurrent_updates)
        self.assertEqual(result.communicated_bits, result.recurrent_updates * 8)
        self.assertGreaterEqual(result.recurrent_updates, 128)
        self.assertLessEqual(result.recurrent_updates, 1_024)

    def test_contract_split_is_rejected_before_world_validation(self):
        generated = self._world(world_index=4)
        counterfactual = dataclasses.replace(generated.public, split="test")
        with self.assertRaisesRegex(ValueError, "contract worlds only"):
            RUNTIME.run_gate8_v1_contract_runtime(
                model=self._controlled_model(),
                world=counterfactual,
            )

    def test_runtime_requires_eval_and_exact_parameter_count(self):
        generated = self._world(world_index=5)
        training_model = self._controlled_model().train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            RUNTIME.run_gate8_v1_contract_runtime(
                model=training_model,
                world=generated.public,
            )

        oversized = self._controlled_model()
        oversized.extra_parameter = nn.Parameter(torch.zeros(1))
        oversized.eval()
        with self.assertRaisesRegex(ValueError, "exactly 19,649"):
            RUNTIME.run_gate8_v1_contract_runtime(
                model=oversized,
                world=generated.public,
            )

    def test_result_contract_has_no_activity_or_duplicate_answer_surface(self):
        result_fields = {field.name for field in dataclasses.fields(RUNTIME.Gate8V1RuntimeResult)}
        round_fields = {field.name for field in dataclasses.fields(RUNTIME.Gate8V1RuntimeRound)}
        forbidden = {
            "activity_logit",
            "activity_positive_workers",
            "activity_positive_worker_indices",
            "answer_logits",
        }
        self.assertTrue(forbidden.isdisjoint(result_fields))
        self.assertTrue(forbidden.isdisjoint(round_fields))
        self.assertIn("target_message_code", result_fields)
        self.assertIn("symbol_logits", result_fields)
        self.assertIn("emitted_message_codes", round_fields)

    def test_runtime_source_reads_no_truth_or_symbolic_oracle(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn(".truth", source)
        self.assertNotIn("gate8_exact_symbolic_oracle", source)
        self.assertNotIn("apply_gate8_transform", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("torch.load", source)

    def test_runtime_plan_keeps_training_and_science_closed(self):
        plan = RUNTIME.gate8_v1_runtime_plan()
        self.assertEqual(
            plan["architecture_head"],
            "c3ab64008c816fa1eb6f9d6f8f0a1a99ed195ec8",
        )
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertEqual(plan["admitted_split"], "contract")
        self.assertTrue(plan["synchronous_delivery"])
        self.assertTrue(plan["deterministic_delivery_for_scheduled_workers"])
        self.assertTrue(plan["one_emission_per_recurrent_update"])
        self.assertTrue(plan["terminal_answer_equals_message_low_nibble"])
        self.assertFalse(plan["activity_gate"])
        self.assertTrue(plan["no_communication_ablation"])
        self.assertTrue(plan["shuffled_worker_ablation"])
        self.assertFalse(plan["reads_world_truth"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["checkpoint_loading_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])
        self.assertFalse(plan["reference_model_admitted"])


if __name__ == "__main__":
    unittest.main()
