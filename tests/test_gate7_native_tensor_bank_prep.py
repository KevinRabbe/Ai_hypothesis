from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate7_native_tensor_bank_prep import (
    GATE7_NATIVE_K_LADDER,
    GATE7_NATIVE_MAX_STAGE_B_SLOTS,
    GATE7_NATIVE_POPULATION_LADDER,
    GATE7_NATIVE_TENSOR_BANK_PREPARATION_ONLY,
    Gate7NativeTensorBank,
    append_gate7_native_children,
    gate7_native_public_seed_tensor,
    prune_gate7_native_overflow,
    sample_gate7_native_positions,
    select_gate7_native_bounded_hash,
    select_gate7_native_bounded_score,
    swap_delete_gate7_native_parent,
)


class Gate7NativeTensorBankPreparationTests(unittest.TestCase):
    def test_preparation_only_boundary(self) -> None:
        self.assertTrue(GATE7_NATIVE_TENSOR_BANK_PREPARATION_ONLY)

    def test_sampler_returns_unique_live_positions_across_full_prepared_ladder(self) -> None:
        seeds = gate7_native_public_seed_tensor((123456789, 987654321), device="cpu")
        for population in GATE7_NATIVE_POPULATION_LADDER:
            live_counts = torch.tensor(
                [population, population - GATE7_NATIVE_MAX_STAGE_B_SLOTS],
                dtype=torch.int64,
            )
            for k in GATE7_NATIVE_K_LADDER:
                if k >= population:
                    continue
                for slot in (0, 63, 127):
                    sample = sample_gate7_native_positions(
                        population_capacity=population,
                        live_counts=live_counts,
                        k=k,
                        public_seeds=seeds,
                        slot_index=slot,
                        sampling_group_code=k,
                    )
                    self.assertEqual(sample.positions.shape, (2, k))
                    self.assertLessEqual(sample.metadata_candidates_examined, k + GATE7_NATIVE_MAX_STAGE_B_SLOTS)
                    for row in range(2):
                        positions = sample.positions[row]
                        self.assertEqual(torch.unique(positions).numel(), k)
                        self.assertTrue(bool((positions < live_counts[row]).all()))
                        self.assertTrue(bool((positions >= 0).all()))

    def test_sampler_work_does_not_scale_with_population_at_fixed_k(self) -> None:
        seeds = gate7_native_public_seed_tensor((42,), device="cpu")
        examined = []
        for population in (1024, 4096, 16384, 65536, 131072):
            sample = sample_gate7_native_positions(
                population_capacity=population,
                live_counts=torch.tensor([population - 128], dtype=torch.int64),
                k=64,
                public_seeds=seeds,
                slot_index=127,
                sampling_group_code=64,
            )
            examined.append(sample.metadata_candidates_examined)
        self.assertEqual(examined, [192, 192, 192, 192, 192])

    def test_score_and_hash_pair_receive_identical_visible_positions(self) -> None:
        population = 512
        batch = 2
        bank = Gate7NativeTensorBank(
            states=torch.zeros(batch, population + 1, 64),
            scores=torch.linspace(-1.0, 1.0, steps=population + 1).repeat(batch, 1),
            heap_ids=torch.arange(1, population + 2, dtype=torch.int64).repeat(batch, 1),
            live_counts=torch.tensor([512, 447], dtype=torch.int64),
            population_capacity=population,
        )
        bank.validate()
        seeds = gate7_native_public_seed_tensor((11, 22), device="cpu")
        score_parent, score_sample = select_gate7_native_bounded_score(
            bank,
            k=64,
            public_seeds=seeds,
            slot_index=17,
            sampling_group_code=64,
            tie_namespace_code=1,
        )
        hash_parent, hash_sample = select_gate7_native_bounded_hash(
            bank,
            k=64,
            public_seeds=seeds,
            slot_index=17,
            sampling_group_code=64,
            hash_namespace_code=2,
        )
        torch.testing.assert_close(score_sample.positions, hash_sample.positions, rtol=0.0, atol=0.0)
        self.assertEqual(score_parent.shape, hash_parent.shape)

    def test_swap_append_prune_preserves_dense_live_prefix_and_score_blind_victim(self) -> None:
        population = 512
        batch = 2
        base_heap = torch.arange(1, population + 2, dtype=torch.int64).repeat(batch, 1)
        base_states = torch.arange(batch * (population + 1) * 64, dtype=torch.float32).reshape(
            batch, population + 1, 64
        )
        counts = torch.tensor([512, 500], dtype=torch.int64)
        seeds = gate7_native_public_seed_tensor((101, 202), device="cpu")

        def make_bank(score_offset: float) -> Gate7NativeTensorBank:
            return Gate7NativeTensorBank(
                states=base_states.clone(),
                scores=torch.arange(batch * (population + 1), dtype=torch.float32).reshape(
                    batch, population + 1
                )
                + score_offset,
                heap_ids=base_heap.clone(),
                live_counts=counts.clone(),
                population_capacity=population,
            )

        selected = torch.tensor([5, 7], dtype=torch.int64)
        child_states = torch.full((batch, 2, 64), 3.25)
        child_ids = torch.tensor([[9001, 9002], [9101, 9102]], dtype=torch.int64)
        terminal = torch.tensor([False, True])

        outputs = []
        for score_offset in (0.0, 10000.0):
            bank = swap_delete_gate7_native_parent(make_bank(score_offset), selected_positions=selected)
            self.assertEqual(bank.live_counts.tolist(), [511, 499])
            child_scores = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32) + score_offset
            bank = append_gate7_native_children(
                bank,
                child_states=child_states,
                child_scores=child_scores,
                child_heap_ids=child_ids,
                terminal=terminal,
            )
            self.assertEqual(bank.live_counts.tolist(), [513, 499])
            bank, overflow = prune_gate7_native_overflow(
                bank,
                public_seeds=seeds,
                slot_index=9,
            )
            self.assertEqual(bank.live_counts.tolist(), [512, 499])
            self.assertEqual(overflow.tolist(), [1, 0])
            outputs.append(bank.heap_ids.clone())

        # Changing every neural score leaves answer-blind victim selection and heap membership unchanged.
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)

    def test_native_hot_path_avoids_population_wide_ordering_and_cuda_scalar_extraction(self) -> None:
        from ai_hypothesis.population_compute import gate7_native_tensor_bank_prep as module

        hot_functions = (
            module.sample_gate7_native_positions,
            module.gate7_native_priority,
            module.select_gate7_native_bounded_score,
            module.select_gate7_native_bounded_hash,
            module.gate7_native_victim_positions,
            module.swap_delete_gate7_native_parent,
            module.append_gate7_native_children,
            module.prune_gate7_native_overflow,
        )
        for function in hot_functions:
            source = inspect.getsource(function)
            for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "argsort(", "sorted("):
                self.assertNotIn(forbidden, source, (function.__name__, forbidden))

        # These three operations must remain true slot mutations. A full-bank clone would make their
        # real cost O(N) even though the logical data-structure operation is O(1).
        for function in (
            module.swap_delete_gate7_native_parent,
            module.append_gate7_native_children,
            module.prune_gate7_native_overflow,
        ):
            self.assertNotIn(".clone(", inspect.getsource(function), function.__name__)

        sampler_parameters = set(inspect.signature(module.sample_gate7_native_positions).parameters)
        self.assertNotIn("scores", sampler_parameters)
        self.assertNotIn("states", sampler_parameters)
        victim_parameters = set(inspect.signature(module.gate7_native_victim_positions).parameters)
        self.assertNotIn("scores", victim_parameters)
        self.assertNotIn("states", victim_parameters)


if __name__ == "__main__":
    unittest.main()
