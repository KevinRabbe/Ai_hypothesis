from __future__ import annotations

import gc
import unittest

import torch

from ai_hypothesis.population_language import l0_protocol as protocol
from ai_hypothesis.population_language.l0_data import materialize_batch
from ai_hypothesis.population_language.l0_models import (
    MatchedCausalTransformer,
    PopulationLanguageOrganism,
    count_parameters,
    deterministic_worker_coordinates,
)


class PopulationLanguageL0ModelContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(120100)

    def test_tensor_batch_and_answer_mask(self) -> None:
        batch = materialize_batch("train", (0, 1, 2))
        self.assertEqual(tuple(batch.input_ids.shape), (3, 31))
        self.assertEqual(batch.input_ids.shape, batch.target_ids.shape)
        self.assertEqual(batch.loss_mask.shape, batch.target_ids.shape)
        self.assertEqual(batch.answer_mask.shape, batch.target_ids.shape)
        self.assertTrue(torch.all(batch.answer_mask.sum(dim=1) == 5))
        self.assertFalse(torch.any(batch.answer_mask & ~batch.loss_mask))
        self.assertEqual(batch.ordinals, (0, 1, 2))

    def test_full_models_match_locked_parameter_counts(self) -> None:
        transformer = MatchedCausalTransformer()
        self.assertEqual(
            count_parameters(transformer), protocol.transformer_parameter_count()
        )
        del transformer
        gc.collect()

        population = PopulationLanguageOrganism()
        self.assertEqual(
            count_parameters(population), protocol.population_parameter_count()
        )
        del population
        gc.collect()

    def test_worker_count_changes_no_parameters(self) -> None:
        config = protocol.PopulationConfig(
            token_width=32,
            lexical_encoder_width=48,
            worker_width=16,
            worker_feed_forward=32,
            lexical_decoder_width=48,
            router_dim=8,
            training_workers=4,
        )
        model = PopulationLanguageOrganism(
            config, communication_rounds=2, top_k=2
        ).eval()
        parameter_count = count_parameters(model)
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            small = model.forward_with_state(inputs, worker_count=2)
            large = model.forward_with_state(inputs, worker_count=8)

        self.assertEqual(count_parameters(model), parameter_count)
        self.assertEqual(tuple(small.final_state.shape), (1, 2, 16))
        self.assertEqual(tuple(large.final_state.shape), (1, 8, 16))
        self.assertEqual(small.routed_messages, 1 * 4 * 2 * 2 * 2)
        self.assertEqual(large.routed_messages, 1 * 4 * 2 * 8 * 2)
        self.assertEqual(tuple(small.logits.shape), (1, 4, 64))
        self.assertEqual(tuple(large.logits.shape), (1, 4, 64))

    def test_transformer_is_causal(self) -> None:
        config = protocol.TransformerConfig(
            d_model=32, layers=2, feed_forward=64, heads=4
        )
        model = MatchedCausalTransformer(config).eval()
        inputs = torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 8, 9]], dtype=torch.long
        )
        with torch.no_grad():
            logits = model(inputs)
        self.assertTrue(torch.equal(logits[0, :3], logits[1, :3]))
        self.assertTrue(torch.isfinite(logits).all())

    def test_population_organism_is_causal(self) -> None:
        config = protocol.PopulationConfig(
            token_width=32,
            lexical_encoder_width=48,
            worker_width=16,
            worker_feed_forward=32,
            lexical_decoder_width=48,
            router_dim=8,
            training_workers=4,
        )
        model = PopulationLanguageOrganism(
            config, communication_rounds=2, top_k=2
        ).eval()
        inputs = torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 8, 9]], dtype=torch.long
        )
        with torch.no_grad():
            logits = model(inputs, worker_count=4)
        self.assertTrue(torch.equal(logits[0, :3], logits[1, :3]))
        self.assertTrue(torch.isfinite(logits).all())

    def test_coordinates_are_deterministic_and_nonlearned(self) -> None:
        first = deterministic_worker_coordinates(
            8, 16, device=torch.device("cpu"), dtype=torch.float32
        )
        second = deterministic_worker_coordinates(
            8, 16, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (8, 16))
        self.assertFalse(first.requires_grad)
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_invalid_inputs_fail_closed(self) -> None:
        transformer = MatchedCausalTransformer(
            protocol.TransformerConfig(
                d_model=16, layers=1, feed_forward=32, heads=4
            )
        )
        with self.assertRaises(ValueError):
            transformer(torch.tensor([1, 2, 3], dtype=torch.long))
        with self.assertRaises(ValueError):
            transformer(torch.tensor([[999]], dtype=torch.long))

        population = PopulationLanguageOrganism(
            protocol.PopulationConfig(
                token_width=16,
                lexical_encoder_width=24,
                worker_width=8,
                worker_feed_forward=16,
                lexical_decoder_width=24,
                router_dim=4,
                training_workers=2,
            ),
            communication_rounds=1,
            top_k=1,
        )
        with self.assertRaises(ValueError):
            population(torch.tensor([[1]], dtype=torch.long), worker_count=0)


if __name__ == "__main__":
    unittest.main()
