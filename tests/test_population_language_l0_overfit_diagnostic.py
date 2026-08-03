from __future__ import annotations

import tempfile
import unittest

import torch

from ai_hypothesis.population_language.l0_models import PopulationLanguageOrganism
from ai_hypothesis.population_language import l0_overfit_diagnostic as diagnostic


class PopulationLanguageL0OverfitContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(diagnostic.SEED)

    def test_binding_batch_has_shared_query_and_distinct_definitions(self) -> None:
        batch, first_answer_mask = diagnostic.binding_batch()
        self.assertEqual(tuple(batch.input_ids.shape), (4, 31))
        self.assertTrue(torch.all(batch.answer_mask.sum(dim=1) == 5))
        self.assertTrue(torch.all(first_answer_mask.sum(dim=1) == 1))
        query_slice = batch.input_ids[:, 13:21]
        self.assertTrue(torch.all(query_slice == query_slice[0]))
        self.assertFalse(torch.equal(batch.input_ids[0, 4:12], batch.input_ids[1, 4:12]))

    def test_definition_ablation_removes_only_semantic_values(self) -> None:
        batch, _ = diagnostic.binding_batch()
        ablated = diagnostic.ablate_definition_values(batch.input_ids)
        changed = ablated != batch.input_ids
        expected = torch.zeros_like(changed)
        expected[:, [4, 5, 10, 11]] = True
        self.assertTrue(torch.equal(changed, expected))

    def test_population_gradient_paths_are_live(self) -> None:
        batch, _ = diagnostic.binding_batch()
        model = PopulationLanguageOrganism(
            diagnostic.POPULATION_CONFIG,
            communication_rounds=diagnostic.COMMUNICATION_ROUNDS,
            top_k=diagnostic.TOP_K,
        )
        logits = model(batch.input_ids)
        loss = diagnostic.masked_loss(logits, batch.target_ids, batch.answer_mask)
        loss.backward()
        report = diagnostic._gradient_report(
            model, diagnostic._POPULATION_GRADIENT_PATHS
        )
        self.assertTrue(report["valid"])
        self.assertTrue(all(value is not None and value > 0 for value in report["norms"].values()))

    def test_classifier_is_closed(self) -> None:
        passing = [
            {"name": "transformer", "passed": True},
            {"name": "population", "passed": True},
        ]
        failing = [
            {"name": "transformer", "passed": True},
            {"name": "population", "passed": False},
        ]
        self.assertEqual(diagnostic.classify(passing), diagnostic.PASS)
        self.assertEqual(diagnostic.classify(failing), diagnostic.FAIL)
        with self.assertRaises(ValueError):
            diagnostic.classify(list(reversed(passing)))

    def test_existing_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                diagnostic.run(
                    __import__("pathlib").Path(directory), steps=1
                )


if __name__ == "__main__":
    unittest.main()
