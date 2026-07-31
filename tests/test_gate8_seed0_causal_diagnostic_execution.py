from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORLDS = _load(
    "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py",
    "gate8_seed0_diag_test_worlds",
)
ARCH = _load(
    "ai_hypothesis/population_compute/gate8_organism_architecture.py",
    "gate8_seed0_diag_test_arch",
)
DEVELOPMENT = _load(
    "ai_hypothesis/population_compute/gate8_organism_development_runtime.py",
    "gate8_seed0_diag_test_development",
)
PROTOCOL = _load(
    "ai_hypothesis/population_compute/gate8_seed0_causal_diagnostic_protocol.py",
    "gate8_seed0_diag_test_protocol",
)
RUNTIME = _load(
    "ai_hypothesis/population_compute/gate8_seed0_causal_diagnostic_runtime.py",
    "gate8_seed0_diag_test_runtime",
)
RUNNER = _load(
    "scripts/diagnose_gate8_seed0.py",
    "gate8_seed0_diag_test_runner",
)


class Gate8Seed0CausalDiagnosticExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.world = WORLDS.generate_gate8_world(
            split="contract",
            seed=0,
            world_index=0,
            population=32,
            depth=4,
        )

    def _controlled_model(
        self,
        *,
        activity_bias: float = 1.0,
        message_code: int = 37,
        answer_symbol: int = 7,
    ):
        model = ARCH.Gate8SharedWorkerCore()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.activity_head.bias.fill_(activity_bias)
            model.message_head.bias[message_code] = 1.0
            model.answer_head.bias[answer_symbol] = 1.0
        model.eval()
        return model

    def test_baseline_is_semantically_identical_to_qualified_development_runtime(self):
        model = self._controlled_model()
        qualified = DEVELOPMENT.run_gate8_development_runtime(
            model=model,
            world=self.world.public,
        )
        diagnostic = RUNTIME.run_gate8_seed0_diagnostic_runtime(
            model=model,
            world=self.world.public,
            probe="baseline",
        )
        self.assertEqual(diagnostic.target_reached, qualified.target_reached)
        self.assertEqual(diagnostic.predicted_symbol, qualified.predicted_symbol)
        self.assertEqual(diagnostic.rounds_executed, qualified.rounds_executed)
        self.assertEqual(diagnostic.recurrent_updates, qualified.recurrent_updates)
        self.assertEqual(diagnostic.delivered_messages, qualified.delivered_messages)
        self.assertEqual(diagnostic.communicated_bits, qualified.communicated_bits)

    def test_forced_activity_is_a_causal_runtime_intervention(self):
        model = self._controlled_model(activity_bias=-1.0)
        baseline = RUNTIME.run_gate8_seed0_diagnostic_runtime(
            model=model,
            world=self.world.public,
            probe="baseline",
        )
        forced = RUNTIME.run_gate8_seed0_diagnostic_runtime(
            model=model,
            world=self.world.public,
            probe="forced_active",
        )
        self.assertFalse(baseline.target_reached)
        self.assertTrue(forced.target_reached)
        self.assertGreater(forced.delivered_messages, baseline.delivered_messages)

    def test_message_low4_decode_changes_only_terminal_readout(self):
        model = self._controlled_model(
            activity_bias=1.0,
            message_code=37,
            answer_symbol=7,
        )
        answer = RUNTIME.run_gate8_seed0_diagnostic_runtime(
            model=model,
            world=self.world.public,
            probe="baseline",
        )
        message = RUNTIME.run_gate8_seed0_diagnostic_runtime(
            model=model,
            world=self.world.public,
            probe="message_low4_decode",
        )
        self.assertTrue(answer.target_reached)
        self.assertTrue(message.target_reached)
        self.assertEqual(answer.predicted_symbol, 7)
        self.assertEqual(message.predicted_symbol, 37 % 16)
        self.assertEqual(answer.target_message_code, message.target_message_code)
        self.assertEqual(answer.recurrent_updates, message.recurrent_updates)
        self.assertEqual(answer.delivered_messages, message.delivered_messages)

    def test_root_invariance_enumerates_exact_finite_table(self):
        model = self._controlled_model()
        result = RUNTIME.evaluate_gate8_nonroot_target_root_invariance(model=model)
        self.assertEqual(result.cases, 2_048)
        self.assertEqual(result.message_root_invariance, 1.0)
        self.assertEqual(result.answer_root_invariance, 1.0)
        self.assertEqual(result.activity_root_invariance, 1.0)

    def test_runtime_rejects_scientific_test_before_execution(self):
        test_world = WORLDS.generate_gate8_world(
            split="test",
            seed=0,
            world_index=0,
            population=32,
            depth=4,
        )
        with self.assertRaisesRegex(ValueError, "rejects this world split"):
            RUNTIME.run_gate8_seed0_diagnostic_runtime(
                model=self._controlled_model(),
                world=test_world.public,
                probe="baseline",
            )

    def test_runtime_requires_eval_and_exact_parameter_count(self):
        model = self._controlled_model()
        model.train()
        with self.assertRaisesRegex(ValueError, "requires model.eval"):
            RUNTIME.run_gate8_seed0_diagnostic_runtime(
                model=model,
                world=self.world.public,
                probe="baseline",
            )

        class WrongModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(1))

        wrong = WrongModel().eval()
        with self.assertRaisesRegex(ValueError, "19,649"):
            RUNTIME.run_gate8_seed0_diagnostic_runtime(
                model=wrong,
                world=self.world.public,
                probe="baseline",
            )

    def test_full_resume_learning_rate_has_exact_endpoints(self):
        self.assertEqual(
            RUNNER._full_resume_learning_rate(step=1, diagnostic_protocol=PROTOCOL),
            PROTOCOL.GATE8_FULL_RESUME_INITIAL_LEARNING_RATE,
        )
        self.assertEqual(
            RUNNER._full_resume_learning_rate(
                step=PROTOCOL.GATE8_FULL_RESUME_STEPS,
                diagnostic_protocol=PROTOCOL,
            ),
            PROTOCOL.GATE8_FULL_RESUME_MINIMUM_LEARNING_RATE,
        )
        previous = float("inf")
        for step in range(1, PROTOCOL.GATE8_FULL_RESUME_STEPS + 1):
            current = RUNNER._full_resume_learning_rate(
                step=step,
                diagnostic_protocol=PROTOCOL,
            )
            self.assertLessEqual(current, previous)
            previous = current

    def test_runtime_plan_keeps_forbidden_boundaries_closed(self):
        plan = RUNTIME.gate8_seed0_diagnostic_runtime_plan()
        self.assertEqual(plan["protocol_head"], "0fa9ec48c31b36c90d58da827139457fd812b98c")
        self.assertFalse(plan["truth_read"])
        self.assertFalse(plan["scientific_test_allowed"])
        self.assertFalse(plan["reference_model_allowed"])
        self.assertEqual(tuple(plan["runtime_probes"]), PROTOCOL.GATE8_RUNTIME_PROBES)


if __name__ == "__main__":
    unittest.main()
