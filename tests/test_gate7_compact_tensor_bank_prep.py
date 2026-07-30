from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1Scorer
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from ai_hypothesis.population_compute.gate6_fixed_k_population_scaling import Gate6EvaluationWorld
from ai_hypothesis.population_compute.gate7_compact_tensor_bank_prep import (
    GATE7_COMPACT_TENSOR_BANK_PREPARATION_ONLY,
    run_gate6_compatible_compact_stage_b_tensor,
)
from ai_hypothesis.population_compute.gate7_dynamic_tensor_bank_prep import (
    run_gate6_compatible_dynamic_stage_b_tensor,
)
from ai_hypothesis.population_compute.gate7_tensor_engine_prep import build_complete_tensor_frontier


def _world(index: int, seed: int, hints: tuple[int, ...]) -> Gate6EvaluationWorld:
    public = Gate3V1PublicWorld(seed=seed, depth=10, noisy_hints=hints)
    public.validate()
    world = Gate6EvaluationWorld(
        world_index=index,
        public=public,
        hidden_path=(1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    )
    world.validate()
    return world


class Gate7CompactTensorBankPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(271828)
        cls.model = Gate3V1Scorer()
        cls.worlds = (
            _world(0, 3333, (0, 0, 1, 1, 0, 1, 0, 1, 1, 0)),
            _world(1, 4444, (1, 1, 0, 0, 1, 0, 1, 0, 0, 1)),
        )
        cls.frontier = build_complete_tensor_frontier(
            cls.model,
            tuple(world.public for world in cls.worlds),
            frontier_depth=8,
            device="cpu",
        )

    def test_preparation_only_boundary(self) -> None:
        self.assertTrue(GATE7_COMPACT_TENSOR_BANK_PREPARATION_ONLY)

    def _assert_matches_dynamic(self, *, population: int, slots: int, mode: str, k: int, group: str) -> None:
        reference = run_gate6_compatible_dynamic_stage_b_tensor(
            self.model,
            self.worlds,
            self.frontier,
            population_capacity=population,
            stage_b_slots=slots,
            mode=mode,
            k=k,
            sampling_group=group,
            device="cpu",
        )
        compact = run_gate6_compatible_compact_stage_b_tensor(
            self.model,
            self.worlds,
            self.frontier,
            population_capacity=population,
            stage_b_slots=slots,
            mode=mode,
            k=k,
            sampling_group=group,
            device="cpu",
        )

        torch.testing.assert_close(compact.selected_heap_ids, reference.selected_heap_ids, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            compact.terminal_child_heap_ids,
            reference.terminal_child_heap_ids,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            compact.overflow_pruned_count,
            reference.overflow_pruned_count,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(compact.final_bank.heap_ids, reference.final_bank.heap_ids, rtol=0.0, atol=0.0)
        torch.testing.assert_close(compact.final_bank.live_mask, reference.final_bank.live_mask, rtol=0.0, atol=0.0)
        torch.testing.assert_close(compact.final_bank.states, reference.final_bank.states, rtol=0.0, atol=0.0)
        torch.testing.assert_close(compact.final_bank.scores, reference.final_bank.scores, rtol=0.0, atol=0.0)

    def test_compact_global_matches_qualified_dynamic_reference(self) -> None:
        self._assert_matches_dynamic(population=128, slots=24, mode="global_score", k=16, group="k16")

    def test_compact_k16_score_matches_qualified_dynamic_reference(self) -> None:
        self._assert_matches_dynamic(population=64, slots=32, mode="bounded_score", k=16, group="k16")

    def test_compact_k32_score_matches_qualified_dynamic_reference(self) -> None:
        self._assert_matches_dynamic(population=128, slots=24, mode="bounded_score", k=32, group="k32")

    def test_compact_k64_hash_matches_qualified_dynamic_reference(self) -> None:
        self._assert_matches_dynamic(population=256, slots=16, mode="bounded_hash", k=64, group="k64")

    def test_compact_hot_path_contains_no_full_sort_or_topk(self) -> None:
        from ai_hypothesis.population_compute import gate7_compact_tensor_bank_prep as module

        for function in (
            module._select_global_compact,
            module._compact_sample_positions,
            module._select_bounded_score_compact,
            module._select_bounded_hash_compact,
            module._ordered_expand,
            module._drop_single_overflow,
            module.run_gate6_compatible_compact_stage_b_tensor,
        ):
            source = inspect.getsource(function)
            for forbidden in (".item(", ".cpu(", ".tolist(", "float(", "argsort(", "topk("):
                self.assertNotIn(forbidden, source, (function.__name__, forbidden))


if __name__ == "__main__":
    unittest.main()
