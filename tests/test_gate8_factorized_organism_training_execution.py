from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path("ai_hypothesis/population_compute")
WORLD_PATH = ROOT / "gate8_distributed_transformation_worlds.py"
ARCHITECTURE_PATH = ROOT / "gate8_factorized_organism_architecture.py"
RUNTIME_PATH = ROOT / "gate8_factorized_organism_runtime.py"
PROTOCOL_PATH = ROOT / "gate8_factorized_organism_training_protocol.py"
TRAINING_PATH = ROOT / "gate8_factorized_organism_training.py"
DEVELOPMENT_RUNTIME_PATH = (
    ROOT / "gate8_factorized_organism_development_runtime.py"
)
RUNNER_PATH = pathlib.Path("scripts/train_gate8_factorized_organism.py")
WRAPPER_PATH = pathlib.Path("scripts/train_gate8_factorized_organism.ps1")


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Gate8 v1 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


w = _load(WORLD_PATH, "gate8_v1_training_execution_test_worlds")
a = _load(ARCHITECTURE_PATH, "gate8_v1_training_execution_test_architecture")
r = _load(RUNTIME_PATH, "gate8_v1_training_execution_test_runtime")
p = _load(PROTOCOL_PATH, "gate8_v1_training_execution_test_protocol")
t = _load(TRAINING_PATH, "gate8_v1_training_execution_test_mechanics")
d = _load(
    DEVELOPMENT_RUNTIME_PATH,
    "gate8_v1_training_execution_test_development",
)


def _contract_world(
    index: int = 0,
    population: int = 32,
    depth: int = 4,
):
    return w.generate_gate8_world(
        split="contract",
        seed=0,
        world_index=index,
        population=population,
        depth=depth,
    )


def _constant_core(*, carrier: int = 7, symbol: int = 5):
    model = a.Gate8V1SharedWorkerCore()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.carrier_head.bias[carrier] = 1.0
        model.symbol_head.bias[symbol] = 1.0
    return model.eval()


class Gate8V1FactorizedTrainingExecutionTests(unittest.TestCase):
    def test_local_labels_cover_every_edge_without_truth_input(self) -> None:
        generated = _contract_world()
        examples = t.gate8_v1_local_examples(
            world=generated.public,
            transform_permutations=w.GATE8_TRANSFORM_PERMUTATIONS,
            protocol=p,
        )
        self.assertEqual(len(examples), generated.public.population)
        self.assertEqual(
            tuple(example.worker_index for example in examples),
            tuple(range(generated.public.population)),
        )
        root_workers = tuple(
            worker
            for worker in generated.public.workers
            if worker.source_node == generated.public.query.root_node
        )
        self.assertTrue(root_workers)
        for worker in root_workers:
            self.assertEqual(
                examples[worker.worker_index].inbox_code,
                generated.public.query.root_symbol,
            )

        target_worker = next(
            worker.worker_index
            for worker in generated.public.workers
            if worker.target_node == generated.public.query.target_node
        )
        self.assertEqual(
            examples[target_worker].symbol_target,
            generated.truth.answer_symbol,
        )
        for example in examples:
            carrier, symbol = p.gate8_v1_decode_message_code(
                example.message_target
            )
            self.assertEqual(carrier, example.carrier_target)
            self.assertEqual(symbol, example.symbol_target)
            self.assertTrue(0 <= example.inbox_code < 256)

    def test_collated_batch_matches_factorized_model_surface(self) -> None:
        generated = (_contract_world(0), _contract_world(1))
        batch = t.collate_gate8_v1_local_batch(
            worlds=generated,
            transform_permutations=w.GATE8_TRANSFORM_PERMUTATIONS,
            protocol=p,
            device="cpu",
        )
        self.assertEqual(batch.world_count, 2)
        self.assertEqual(batch.edge_count, 64)
        self.assertTrue(
            torch.equal(
                batch.carrier_target * 16 + batch.symbol_target,
                batch.message_target,
            )
        )
        model = a.Gate8V1SharedWorkerCore().train()
        output = t.gate8_v1_local_forward(model, batch)
        losses = t.gate8_v1_local_loss(
            output=output,
            batch=batch,
            protocol=p,
        )
        self.assertTrue(torch.isfinite(losses.total))
        self.assertEqual(output.carrier_logits.shape, (64, 16))
        self.assertEqual(output.symbol_logits.shape, (64, 16))
        self.assertEqual(output.hidden.shape, (64, 65))
        self.assertFalse(hasattr(output, "activity_logit"))
        self.assertFalse(hasattr(output, "answer_logits"))
        self.assertFalse(hasattr(output, "message_logits"))

    def test_one_frozen_optimizer_step_updates_without_parameter_drift(self) -> None:
        torch.manual_seed(0)
        model = a.Gate8V1SharedWorkerCore().train()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        batch = t.collate_gate8_v1_local_batch(
            worlds=(_contract_world(2), _contract_world(3)),
            transform_permutations=w.GATE8_TRANSFORM_PERMUTATIONS,
            protocol=p,
            device="cpu",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=p.GATE8_V1_LEARNING_RATE,
            betas=p.GATE8_V1_ADAM_BETAS,
            eps=p.GATE8_V1_ADAM_EPSILON,
            weight_decay=p.GATE8_V1_WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)
        output = t.gate8_v1_local_forward(model, batch)
        losses = t.gate8_v1_local_loss(
            output=output,
            batch=batch,
            protocol=p,
        )
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            p.GATE8_V1_GRADIENT_CLIP_NORM,
        )
        self.assertTrue(torch.isfinite(gradient_norm))
        optimizer.step()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            19_649,
        )
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in model.named_parameters()
            )
        )

    def test_local_evaluation_merges_factorized_counts_and_coverage(self) -> None:
        model = _constant_core()
        rows = []
        for index in (4, 5):
            batch = t.collate_gate8_v1_local_batch(
                worlds=(_contract_world(index),),
                transform_permutations=w.GATE8_TRANSFORM_PERMUTATIONS,
                protocol=p,
                device="cpu",
            )
            rows.append(
                t.evaluate_gate8_v1_local_batch(
                    model=model,
                    batch=batch,
                    protocol=p,
                )
            )
        merged = t.merge_gate8_v1_local_evaluations(rows)
        self.assertEqual(merged.edge_count, 64)
        self.assertEqual(
            merged.exact_message_correct,
            sum(row.exact_message_correct for row in rows),
        )
        self.assertEqual(
            merged.carrier_correct,
            sum(row.carrier_correct for row in rows),
        )
        self.assertEqual(
            merged.symbol_correct,
            sum(row.symbol_correct for row in rows),
        )
        self.assertEqual(
            merged.inbox_codes,
            rows[0].inbox_codes | rows[1].inbox_codes,
        )
        self.assertEqual(
            merged.target_carriers,
            rows[0].target_carriers | rows[1].target_carriers,
        )

    def test_development_runtime_matches_qualified_v1_runtime(self) -> None:
        public = _contract_world(6, population=128, depth=16).public
        model = _constant_core(carrier=9, symbol=12)
        qualified = r.run_gate8_v1_contract_runtime(
            model=model,
            world=public,
            mode=r.GATE8_V1_RUNTIME_FULL,
        )
        development = d.run_gate8_v1_development_runtime(
            model=model,
            world=public,
        )
        self.assertEqual(development.target_reached, qualified.target_reached)
        self.assertEqual(
            development.predicted_symbol,
            qualified.predicted_symbol,
        )
        self.assertEqual(
            development.target_message_code,
            qualified.target_message_code,
        )
        self.assertEqual(
            development.root_seed_code,
            qualified.root_seed_code,
        )
        self.assertEqual(
            development.rounds_executed,
            qualified.rounds_executed,
        )
        self.assertEqual(
            development.recurrent_updates,
            qualified.recurrent_updates,
        )
        self.assertEqual(
            development.emitted_messages,
            qualified.emitted_messages,
        )
        self.assertEqual(
            development.delivered_messages,
            qualified.delivered_messages,
        )
        self.assertEqual(
            development.communicated_bits,
            qualified.communicated_bits,
        )
        self.assertEqual(
            development.target_worker_index,
            qualified.target_worker_index,
        )

    def test_test_and_demonstration_splits_fail_before_validation(self) -> None:
        class Forbidden:
            def __init__(self, split: str):
                self.split = split

            def validate(self):
                raise AssertionError(
                    "forbidden split validation must not execute"
                )

        for split in ("test", "demonstration"):
            with self.assertRaisesRegex(ValueError, "reject"):
                t.gate8_v1_local_examples(
                    world=Forbidden(split),
                    transform_permutations=w.GATE8_TRANSFORM_PERMUTATIONS,
                    protocol=p,
                )
            with self.assertRaisesRegex(ValueError, "reject"):
                d.run_gate8_v1_development_runtime(
                    model=_constant_core(),
                    world=Forbidden(split),
                )

    def test_batch_validation_rejects_component_message_disagreement(self) -> None:
        batch = t.collate_gate8_v1_local_batch(
            worlds=(_contract_world(7),),
            transform_permutations=w.GATE8_TRANSFORM_PERMUTATIONS,
            protocol=p,
            device="cpu",
        )
        bad = dataclasses.replace(
            batch,
            message_target=(batch.message_target + 1) % 256,
        )
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            bad.validate()

    def test_execution_sources_bind_exact_heads_and_seed0_boundary(self) -> None:
        self.assertEqual(
            t.GATE8_V1_TRAINING_EXECUTION_PROTOCOL_HEAD,
            "a33dc123d090268a531d112251ea3ab53cb50062",
        )
        self.assertEqual(
            d.GATE8_V1_DEVELOPMENT_RUNTIME_QUALIFIED_RUNTIME_HEAD,
            "333d88ac4fc52f1651741fba224e0b4605feedd3",
        )
        mechanics = TRAINING_PATH.read_text(encoding="utf-8")
        development = DEVELOPMENT_RUNTIME_PATH.read_text(encoding="utf-8")
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertNotIn(".truth", mechanics)
        self.assertNotIn(".truth", development)
        self.assertNotIn('split="test"', mechanics)
        self.assertNotIn('split="test"', development)
        self.assertIn('split="train"', runner)
        self.assertIn('split="validation"', runner)
        self.assertNotIn('split="test"', runner)
        self.assertIn("tuple(range(512, 1_024))", runner)
        self.assertIn("if seed != 0", runner)
        self.assertIn("choices=(0,)", runner)
        self.assertIn("[ValidateSet(0)]", wrapper)
        self.assertIn("GATE8_V1_TRAINING_WRAPPER_SMOKE", wrapper)
        for text in (mechanics, development, runner):
            for token in (
                "AutoModel",
                "AutoTokenizer",
                "from_pretrained",
                "snapshot_download",
                "model.safetensors",
            ):
                self.assertNotIn(token, text)

    def test_execution_plans_keep_scientific_boundaries_closed(self) -> None:
        mechanics = t.gate8_v1_training_mechanics_plan()
        development = d.gate8_v1_development_runtime_plan()
        self.assertEqual(
            mechanics["protocol_head"],
            "a33dc123d090268a531d112251ea3ab53cb50062",
        )
        self.assertEqual(
            mechanics["allowed_splits"],
            ["contract", "train", "validation"],
        )
        self.assertFalse(mechanics["scientific_test_split_allowed"])
        self.assertFalse(mechanics["demonstration_split_allowed"])
        self.assertFalse(mechanics["truth_field_read"])
        self.assertFalse(mechanics["activity_target"])
        self.assertFalse(mechanics["answer_target"])
        self.assertFalse(mechanics["joint_256_way_target"])
        self.assertEqual(
            development["allowed_splits"],
            ["contract", "train", "validation"],
        )
        self.assertFalse(development["scientific_test_allowed"])
        self.assertFalse(development["demonstration_allowed"])
        self.assertFalse(development["truth_read"])
        self.assertTrue(development["deterministic_delivery"])
        self.assertTrue(development["answer_equals_message_low_nibble"])


if __name__ == "__main__":
    unittest.main()
