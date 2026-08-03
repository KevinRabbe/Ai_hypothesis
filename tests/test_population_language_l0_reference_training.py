from __future__ import annotations

import unittest

import torch

from ai_hypothesis.population_language import l0_protocol as protocol
from ai_hypothesis.population_language import l0_reference_training as training
from ai_hypothesis.population_language.l0_data import LanguageBatch
from ai_hypothesis.population_language.l0_models import PopulationLanguageOrganism


def _evaluation(
    split: str,
    episodes: int,
    checkpoint: str,
    *,
    answer_exact: float = 0.96,
    answer_nll: float = 0.2,
    worker: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "episodes": episodes,
        "checkpoint_sha256": checkpoint,
        "next_token_nll": 0.3,
        "perplexity": 1.35,
        "answer_span_nll": answer_nll,
        "answer_exact_accuracy": answer_exact,
        "color_token_accuracy": 0.98,
        "shape_token_accuracy": 0.98,
        "relation_token_accuracy": 0.99,
        "swapped_definition_answer_exact_accuracy": answer_exact,
        "definition_order_answer_token_agreement": 0.99,
        "estimated_forward_flops_per_episode": 1_000_000_000,
        "estimated_greedy_answer_flops_per_episode": 5_000_000_000,
        "answer_exact_accuracy_per_gigaflop": answer_exact / 5.0,
        "answer_exact_decoding": "AUTOREGRESSIVE_GREEDY_FIVE_TOKEN",
        "seconds": 1.0,
    }
    if worker is not None:
        row.update(
            {
                "worker_count": worker,
                "routed_messages": 1_000,
                "routed_messages_per_processed_token": 4.0,
                "routed_messages_per_episode": 124.0,
                "persistent_state_bytes_per_episode_bf16": worker * 256,
                "routing": {
                    "router_decisions": 100,
                    "selected_messages": 400,
                    "mean_router_entropy_nats": 1.0,
                    "normalized_router_entropy": 0.8,
                    "selected_sender_coverage": 1.0,
                    "effective_worker_utilization": 0.9,
                    "sender_selection_coefficient_of_variation": 0.2,
                },
            }
        )
    return row


def _valid_seed_rows(*, scaling: bool = False) -> list[dict[str, object]]:
    schedule = training.training_schedule_sha256()
    rows: list[dict[str, object]] = []
    for seed in protocol.INITIALIZATION_SEEDS:
        transformer_hash = f"{seed:064x}"[-64:]
        population_hash = f"{seed + 10:064x}"[-64:]
        transformer = {
            "parameter_count": protocol.transformer_parameter_count(),
            "optimizer_steps": training.OPTIMIZER_STEPS,
            "global_batch_size": training.GLOBAL_BATCH_SIZE,
            "microbatch": 8,
            "gradient_accumulation_steps": 32,
            "training_schedule_sha256": schedule,
            "training_tokens": (
                training.OPTIMIZER_STEPS
                * training.GLOBAL_BATCH_SIZE
                * training.EXPECTED_TARGET_TOKENS_PER_EPISODE
            ),
            "estimated_training_flops": 1,
            "seconds": 1.0,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 1,
            "canonical_checkpoint_sha256": transformer_hash,
            "checkpoint_file_sha256": "a" * 64,
            "validation": _evaluation(
                "validation",
                protocol.REFERENCE_VALIDATION_EPISODES,
                transformer_hash,
            ),
            "test": _evaluation(
                "test",
                protocol.REFERENCE_TEST_EPISODES,
                transformer_hash,
                answer_exact=0.97,
            ),
        }
        validation_by_workers: dict[str, object] = {}
        test_by_workers: dict[str, object] = {}
        for worker in protocol.EVAL_WORKERS:
            if scaling:
                progress = (worker - 16) / 240
                accuracy = 0.70 + 0.08 * progress
                nll = 0.4 - 0.08 * progress
            else:
                accuracy = 0.75
                nll = 0.3
            validation_by_workers[str(worker)] = _evaluation(
                "validation",
                protocol.REFERENCE_VALIDATION_EPISODES,
                population_hash,
                answer_exact=accuracy,
                answer_nll=nll,
                worker=worker,
            )
            test_by_workers[str(worker)] = _evaluation(
                "test",
                protocol.REFERENCE_TEST_EPISODES,
                population_hash,
                answer_exact=accuracy,
                answer_nll=nll,
                worker=worker,
            )
        population = {
            "parameter_count": protocol.population_parameter_count(),
            "optimizer_steps": training.OPTIMIZER_STEPS,
            "global_batch_size": training.GLOBAL_BATCH_SIZE,
            "microbatch": 8,
            "gradient_accumulation_steps": 32,
            "training_schedule_sha256": schedule,
            "training_tokens": (
                training.OPTIMIZER_STEPS
                * training.GLOBAL_BATCH_SIZE
                * training.EXPECTED_TARGET_TOKENS_PER_EPISODE
            ),
            "estimated_training_flops": 1,
            "seconds": 1.0,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 1,
            "canonical_checkpoint_sha256": population_hash,
            "checkpoint_file_sha256": "b" * 64,
            "validation_by_workers": validation_by_workers,
            "test_by_workers": test_by_workers,
        }
        rows.append({"seed": seed, "transformer": transformer, "population": population})
    return rows


class PopulationLanguageL0ReferenceTrainingContract(unittest.TestCase):
    def test_microbatch_contract_is_exact(self) -> None:
        contract = training.validate_contract(8, 16)
        self.assertEqual(contract.accumulation_steps, 32)
        with self.assertRaises(ValueError):
            training.validate_contract(7, 8)
        with self.assertRaises(ValueError):
            training.validate_contract(8, 0)

    def test_learning_rate_schedule_boundaries(self) -> None:
        self.assertAlmostEqual(
            training.learning_rate_for_step(0),
            training.PEAK_LEARNING_RATE / training.WARMUP_STEPS,
        )
        self.assertEqual(
            training.learning_rate_for_step(training.WARMUP_STEPS - 1),
            training.PEAK_LEARNING_RATE,
        )
        self.assertEqual(training.learning_rate_for_step(training.OPTIMIZER_STEPS - 1), 0.0)
        with self.assertRaises(ValueError):
            training.learning_rate_for_step(training.OPTIMIZER_STEPS)

    def test_training_schedule_is_bijective_per_epoch(self) -> None:
        steps_per_epoch = protocol.REFERENCE_TRAIN_EPISODES // training.GLOBAL_BATCH_SIZE
        first_epoch = [
            ordinal
            for step in range(steps_per_epoch)
            for ordinal in training.training_ordinals(step)
        ]
        self.assertEqual(len(first_epoch), protocol.REFERENCE_TRAIN_EPISODES)
        self.assertEqual(len(set(first_epoch)), protocol.REFERENCE_TRAIN_EPISODES)
        self.assertEqual(len(training.training_schedule_sha256()), 64)

    def test_definition_order_swap_changes_only_definition_blocks(self) -> None:
        values = torch.arange(62, dtype=torch.long).view(2, 31)
        swapped = training.swap_definition_order(values)
        self.assertTrue(torch.equal(swapped[:, 1:7], values[:, 7:13]))
        self.assertTrue(torch.equal(swapped[:, 7:13], values[:, 1:7]))
        self.assertTrue(torch.equal(swapped[:, :1], values[:, :1]))
        self.assertTrue(torch.equal(swapped[:, 13:], values[:, 13:]))

    def test_greedy_answer_generation_is_autoregressive(self) -> None:
        class IncrementModel(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros(
                    (*input_ids.shape, protocol.TransformerConfig().vocab_size),
                    dtype=torch.float32,
                )
                next_ids = (input_ids + 1) % protocol.TransformerConfig().vocab_size
                logits.scatter_(2, next_ids.unsqueeze(-1), 1.0)
                return logits

        input_ids = torch.zeros((2, 31), dtype=torch.long)
        input_ids[:, 21] = 5
        target_ids = torch.zeros_like(input_ids)
        loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
        answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        answer_mask[:, 21:26] = True
        batch = LanguageBatch(
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=loss_mask,
            answer_mask=answer_mask,
            ordinals=(0, 1),
        )
        prompt_length = training._answer_prompt_input_length(batch)
        generated = training.greedy_answer_tokens(
            model=IncrementModel(),
            name="transformer",
            batch=batch,
        )
        initial = batch.input_ids[:, prompt_length - 1]
        expected = torch.stack(
            [
                (initial + offset) % protocol.TransformerConfig().vocab_size
                for offset in range(1, training.ANSWER_TOKEN_COUNT + 1)
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(generated, expected))

        poisoned = batch.input_ids.clone()
        poisoned[:, prompt_length:] = 0
        regenerated = training.greedy_answer_tokens(
            model=IncrementModel(),
            name="transformer",
            batch=batch,
            input_ids=poisoned,
        )
        self.assertTrue(torch.equal(regenerated, generated))

    def test_active_flops_increase_with_worker_count(self) -> None:
        estimates = [
            training.population_forward_flops_per_episode(worker)
            for worker in protocol.EVAL_WORKERS
        ]
        self.assertEqual(estimates, sorted(estimates))
        self.assertGreater(estimates[-1], estimates[0])
        self.assertGreater(training.transformer_forward_flops_per_episode(), 0)

    def test_routing_probe_captures_exact_top_k_counts(self) -> None:
        config = protocol.PopulationConfig(
            token_width=32,
            lexical_encoder_width=64,
            worker_width=16,
            worker_feed_forward=32,
            lexical_decoder_width=64,
            router_dim=4,
            training_workers=4,
        )
        model = PopulationLanguageOrganism(
            config,
            communication_rounds=1,
            top_k=2,
        )
        values = torch.randn(2, 4, config.worker_width)
        with training.PopulationRoutingProbe(model, 4) as probe:
            model.router_query(values)
            model.router_key(values)
        summary = probe.summary()
        self.assertEqual(summary["router_decisions"], 8)
        self.assertEqual(summary["selected_messages"], 16)
        self.assertGreaterEqual(summary["effective_worker_utilization"], 0.0)
        self.assertLessEqual(summary["effective_worker_utilization"], 1.0)

    def test_classifier_fails_closed(self) -> None:
        rows = _valid_seed_rows()
        self.assertEqual(training.classify(rows), training.PASS)
        rows[0]["transformer"]["validation"]["answer_exact_accuracy"] = 0.94
        self.assertEqual(training.classify(rows), training.INVALID)

    def test_population_scaling_criterion_is_exact(self) -> None:
        supporting = training.population_scaling_summary(_valid_seed_rows(scaling=True))
        self.assertEqual(supporting["conclusion"], training.SCALING_SUPPORTS)
        flat = training.population_scaling_summary(_valid_seed_rows(scaling=False))
        self.assertEqual(flat["conclusion"], training.SCALING_DOES_NOT_SUPPORT)


if __name__ == "__main__":
    unittest.main()
