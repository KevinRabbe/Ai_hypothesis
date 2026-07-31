from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "ai_hypothesis/population_compute/gate8_factorized_organism_architecture.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gate8_factorized_organism_architecture_test_module",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Gate8 v1 architecture module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = _load_module()


class Gate8FactorizedOrganismArchitectureTests(unittest.TestCase):
    def test_exact_no_padding_parameter_ledger(self):
        model = ARCH.Gate8V1SharedWorkerCore()
        static = ARCH.gate8_v1_architecture_parameter_ledger()
        observed = ARCH.gate8_v1_observed_parameter_ledger(model)

        self.assertEqual(sum(static.values()), 19_649)
        self.assertEqual(sum(observed.values()), 19_649)
        self.assertEqual(len(observed), 12)
        self.assertEqual(static["carrier_embedding"], 112)
        self.assertEqual(static["symbol_embedding"], 176)
        self.assertEqual(static["transform_embedding"], 24)
        self.assertEqual(static["initial_hidden_state"], 65)
        self.assertEqual(static["worker_update_weight_ih"], 4_095)
        self.assertEqual(static["worker_update_weight_hh"], 12_675)
        self.assertEqual(static["worker_update_biases"], 390)
        self.assertEqual(static["carrier_head"], 1_056)
        self.assertEqual(static["symbol_head"], 1_056)
        self.assertNotIn("padding", static)

    def test_all_256_codes_round_trip_through_two_nibbles(self):
        codes = torch.arange(256, dtype=torch.long)
        carrier, symbol = ARCH.Gate8V1SharedWorkerCore.split_message_code(codes)
        reconstructed = ARCH.Gate8V1SharedWorkerCore.compose_message_code(
            carrier=carrier,
            symbol=symbol,
        )

        self.assertTrue(torch.equal(codes, reconstructed))
        self.assertTrue(torch.equal(carrier, torch.arange(256) // 16))
        self.assertTrue(torch.equal(symbol, torch.arange(256) % 16))
        self.assertEqual(len(set(zip(carrier.tolist(), symbol.tolist()))), 256)

    def test_root_symbol_is_encoded_only_in_initial_message(self):
        symbols = torch.arange(16, dtype=torch.long)
        codes = ARCH.Gate8V1SharedWorkerCore.root_message_code(symbols)
        carrier, decoded_symbols = ARCH.Gate8V1SharedWorkerCore.split_message_code(codes)

        self.assertTrue(torch.equal(codes, symbols))
        self.assertTrue(torch.equal(carrier, torch.zeros(16, dtype=torch.long)))
        self.assertTrue(torch.equal(decoded_symbols, symbols))

    def test_forward_contract_has_only_message_transform_and_hidden(self):
        signature = inspect.signature(ARCH.Gate8V1SharedWorkerCore.forward)
        names = tuple(signature.parameters)
        self.assertEqual(names, ("self", "inbox_code", "transform_id", "hidden"))

        model = ARCH.Gate8V1SharedWorkerCore()
        forbidden = (
            "message_code_embedding",
            "root_symbol_embedding",
            "role_embedding",
            "initial_hidden_by_role",
            "message_head",
            "activity_head",
            "answer_head",
        )
        for name in forbidden:
            self.assertFalse(hasattr(model, name), name)

    def test_shared_core_executes_small_and_large_worker_batches(self):
        model = ARCH.Gate8V1SharedWorkerCore().eval()
        for batch in (1, 32, 1_024):
            inbox = torch.arange(batch, dtype=torch.long) % 256
            transform = torch.arange(batch, dtype=torch.long) % 8
            hidden = model.initial_hidden(batch)
            with torch.no_grad():
                output = model(
                    inbox_code=inbox,
                    transform_id=transform,
                    hidden=hidden,
                )
                message = model.predicted_message_code(output)
                symbol = model.predicted_symbol(output)

            self.assertEqual(output.hidden.shape, (batch, 65))
            self.assertEqual(output.carrier_logits.shape, (batch, 16))
            self.assertEqual(output.symbol_logits.shape, (batch, 16))
            self.assertEqual(message.shape, (batch,))
            self.assertEqual(symbol.shape, (batch,))
            self.assertTrue(torch.all((0 <= message) & (message < 256)))
            self.assertTrue(torch.all((0 <= symbol) & (symbol < 16)))

    def test_terminal_answer_is_exact_symbol_prediction(self):
        hidden = torch.zeros((2, 65))
        carrier_logits = torch.full((2, 16), -1.0)
        symbol_logits = torch.full((2, 16), -1.0)
        carrier_logits[0, 3] = 5.0
        carrier_logits[1, 12] = 5.0
        symbol_logits[0, 7] = 5.0
        symbol_logits[1, 4] = 5.0
        output = ARCH.Gate8V1WorkerStepOutput(
            hidden=hidden,
            carrier_logits=carrier_logits,
            symbol_logits=symbol_logits,
        )

        predicted_symbol = ARCH.Gate8V1SharedWorkerCore.predicted_symbol(output)
        predicted_message = ARCH.Gate8V1SharedWorkerCore.predicted_message_code(output)
        carrier, message_symbol = ARCH.Gate8V1SharedWorkerCore.split_message_code(
            predicted_message
        )

        self.assertEqual(predicted_symbol.tolist(), [7, 4])
        self.assertEqual(message_symbol.tolist(), [7, 4])
        self.assertEqual(carrier.tolist(), [3, 12])

    def test_initial_hidden_is_one_shared_learned_state(self):
        model = ARCH.Gate8V1SharedWorkerCore()
        hidden = model.initial_hidden(4)
        self.assertEqual(hidden.shape, (4, 65))
        for index in range(1, 4):
            self.assertTrue(torch.equal(hidden[0], hidden[index]))
        self.assertEqual(tuple(model.initial_hidden_state.shape), (65,))
        self.assertEqual(model.initial_hidden(0).shape, (0, 65))

    def test_inputs_fail_closed(self):
        model = ARCH.Gate8V1SharedWorkerCore()
        hidden = model.initial_hidden(2)

        bad_calls = (
            dict(
                inbox_code=torch.tensor([0, 256], dtype=torch.long),
                transform_id=torch.tensor([0, 1], dtype=torch.long),
                hidden=hidden,
            ),
            dict(
                inbox_code=torch.tensor([0, 1], dtype=torch.int32),
                transform_id=torch.tensor([0, 1], dtype=torch.long),
                hidden=hidden,
            ),
            dict(
                inbox_code=torch.tensor([0, 1], dtype=torch.long),
                transform_id=torch.tensor([0, 8], dtype=torch.long),
                hidden=hidden,
            ),
            dict(
                inbox_code=torch.tensor([0, 1], dtype=torch.long),
                transform_id=torch.tensor([0], dtype=torch.long),
                hidden=hidden,
            ),
            dict(
                inbox_code=torch.tensor([0, 1], dtype=torch.long),
                transform_id=torch.tensor([0, 1], dtype=torch.long),
                hidden=torch.zeros((2, 64)),
            ),
            dict(
                inbox_code=torch.tensor([0, 1], dtype=torch.long),
                transform_id=torch.tensor([0, 1], dtype=torch.long),
                hidden=torch.zeros((2, 65), dtype=torch.long),
            ),
        )
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    model(**kwargs)

        with self.assertRaises(ValueError):
            model.compose_message_code(
                carrier=torch.tensor([0, 16], dtype=torch.long),
                symbol=torch.tensor([0, 1], dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            model.compose_message_code(
                carrier=torch.tensor([0, 1], dtype=torch.long),
                symbol=torch.tensor([0], dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            model.root_message_code(torch.tensor([16], dtype=torch.long))
        with self.assertRaises(ValueError):
            model.initial_hidden(-1)
        with self.assertRaises(TypeError):
            model.initial_hidden(True)

    def test_plan_keeps_execution_training_and_test_closed(self):
        plan = ARCH.gate8_v1_architecture_plan()
        self.assertEqual(
            plan["base_result_head"],
            "ad54c8daa7617d54e15932da76da08212d0d1444",
        )
        self.assertEqual(plan["learned_parameter_count"], 19_649)
        self.assertTrue(plan["root_symbol_in_initial_message"])
        self.assertTrue(plan["factorized_message_heads"])
        self.assertTrue(plan["terminal_answer_is_symbol_head"])
        self.assertTrue(plan["deterministic_delivery_required_by_future_runtime"])
        self.assertFalse(plan["root_symbol_feature"])
        self.assertFalse(plan["role_feature"])
        self.assertFalse(plan["runtime_flag_feature"])
        self.assertFalse(plan["monolithic_message_head"])
        self.assertFalse(plan["duplicate_answer_head"])
        self.assertFalse(plan["activity_head"])
        self.assertEqual(plan["padding_parameters"], 0)
        self.assertFalse(plan["graph_scheduler_admitted"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["checkpoint_admitted"])
        self.assertFalse(plan["scientific_test_worlds_admitted"])
        self.assertFalse(plan["reference_model_admitted"])


if __name__ == "__main__":
    unittest.main()
