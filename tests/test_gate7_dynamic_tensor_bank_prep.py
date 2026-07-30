from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import Gate3V1NeuralCandidate, Gate3V1Scorer
from ai_hypothesis.population_compute.gate6_fixed_k_population_scaling import (
    GATE6_STAGE_A_PARENT_SLOTS,
    Gate6EvaluationWorld,
    _advance_parent_batch,
    _bounded_indices,
    _initial_thinning,
    _prune_to_capacity,
    _score_rank,
    _seed_from_parts,
)
from ai_hypothesis.population_compute.gate7_dynamic_tensor_bank_prep import (
    GATE7_DYNAMIC_TENSOR_BANK_PREPARATION_ONLY,
    run_gate6_compatible_dynamic_stage_b_tensor,
)
from ai_hypothesis.population_compute.gate7_tensor_engine_prep import (
    Gate7TensorFrontier,
    build_complete_tensor_frontier,
    path_bits_to_tuple,
    path_tuple_to_bits,
)
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1PublicWorld


def _world(index: int, seed: int, hints: tuple[int, ...]) -> Gate6EvaluationWorld:
    public = Gate3V1PublicWorld(seed=seed, depth=10, noisy_hints=hints)
    public.validate()
    world = Gate6EvaluationWorld(
        world_index=index,
        public=public,
        hidden_path=(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    )
    world.validate()
    return world


def _heap_id(path: tuple[int, ...]) -> int:
    return (1 << len(path)) | path_tuple_to_bits(path)


def _frontier_as_reference_candidates(frontier: Gate7TensorFrontier) -> list[tuple[Gate3V1NeuralCandidate, ...]]:
    rows: list[tuple[Gate3V1NeuralCandidate, ...]] = []
    for world_index in range(frontier.batch_size):
        candidates = []
        for position in range(frontier.population):
            candidates.append(
                Gate3V1NeuralCandidate(
                    path=path_bits_to_tuple(position, frontier.depth),
                    state=frontier.states[world_index, position].clone(),
                    score=float(frontier.scores[world_index, position].item()),
                )
            )
        rows.append(tuple(candidates))
    return rows


def _run_eager_reference(
    model: Gate3V1Scorer,
    worlds: tuple[Gate6EvaluationWorld, ...],
    frontier: Gate7TensorFrontier,
    *,
    population_capacity: int,
    stage_b_slots: int,
    mode: str,
    k: int,
    sampling_group: str,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[int, ...], ...],
    list[tuple[Gate3V1NeuralCandidate, ...]],
]:
    populations = _frontier_as_reference_candidates(frontier)
    populations = [
        _initial_thinning(
            population,
            world_seed=world.public.seed,
            population_size=population_capacity,
        )
        for world, population in zip(worlds, populations, strict=True)
    ]

    selected_by_world: list[list[int]] = [[] for _ in worlds]
    terminals_by_world: list[list[tuple[int, int]]] = [[] for _ in worlds]
    pruned_by_world: list[list[int]] = [[] for _ in worlds]

    for local_slot in range(stage_b_slots):
        absolute_slot = GATE6_STAGE_A_PARENT_SLOTS + local_slot
        selected_rows: list[tuple[Gate3V1NeuralCandidate, ...]] = []
        for world_offset, (world, population) in enumerate(zip(worlds, populations, strict=True)):
            if mode == "global_score":
                parent = _score_rank(
                    population,
                    world_seed=world.public.seed,
                    expansion_index=absolute_slot,
                )[0]
            else:
                ordered = tuple(sorted(population, key=lambda candidate: candidate.path))
                positions = _bounded_indices(
                    count=len(ordered),
                    k=k,
                    world_seed=world.public.seed,
                    slot_index=absolute_slot,
                    group=sampling_group,
                )
                visible = tuple(ordered[position] for position in positions)
                if mode == "bounded_score":
                    parent = _score_rank(
                        visible,
                        world_seed=world.public.seed,
                        expansion_index=absolute_slot,
                    )[0]
                elif mode == "bounded_hash":
                    parent = min(
                        visible,
                        key=lambda candidate: (
                            _seed_from_parts(
                                "gate6-fixed-k-population-scaling-bounded-hash-selection",
                                world.public.seed,
                                absolute_slot,
                                "".join(str(bit) for bit in candidate.path),
                            ),
                            candidate.path,
                        ),
                    )
                else:
                    raise ValueError(mode)

            selected_by_world[world_offset].append(_heap_id(parent.path))
            selected_rows.append((parent,))
            populations[world_offset] = tuple(
                candidate for candidate in population if candidate.path != parent.path
            )

        children_by_world = _advance_parent_batch(
            model,
            worlds,
            tuple(selected_rows),
            device=torch.device("cpu"),
        )
        next_populations = []
        for world_offset, (world, population, parent_tuple, children) in enumerate(
            zip(worlds, populations, selected_rows, children_by_world, strict=True)
        ):
            parent = parent_tuple[0]
            if parent.depth + 1 == world.public.depth:
                terminals_by_world[world_offset].append((_heap_id(children[0].path), _heap_id(children[1].path)))
                updated = population
            else:
                terminals_by_world[world_offset].append((0, 0))
                updated = population + tuple(children)
            retained, pruned = _prune_to_capacity(
                updated,
                world_seed=world.public.seed,
                slot_index=local_slot,
                population_size=population_capacity,
            )
            pruned_by_world[world_offset].append(pruned)
            next_populations.append(retained)
        populations = next_populations

    return (
        tuple(tuple(row) for row in selected_by_world),
        tuple(tuple(row) for row in terminals_by_world),
        tuple(tuple(row) for row in pruned_by_world),
        populations,
    )


class Gate7DynamicTensorBankPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(314159)
        cls.model = Gate3V1Scorer()
        cls.worlds = (
            _world(0, 1111, (0, 1, 1, 0, 1, 0, 0, 1, 1, 0)),
            _world(1, 2222, (1, 0, 0, 1, 0, 1, 1, 0, 0, 1)),
        )
        cls.frontier = build_complete_tensor_frontier(
            cls.model,
            tuple(world.public for world in cls.worlds),
            frontier_depth=8,
            device="cpu",
        )

    def test_preparation_only_boundary(self) -> None:
        self.assertTrue(GATE7_DYNAMIC_TENSOR_BANK_PREPARATION_ONLY)

    def _assert_equivalent(self, *, population: int, slots: int, mode: str, k: int, group: str) -> None:
        expected_selected, expected_terminals, expected_pruned, expected_populations = _run_eager_reference(
            self.model,
            self.worlds,
            self.frontier,
            population_capacity=population,
            stage_b_slots=slots,
            mode=mode,
            k=k,
            sampling_group=group,
        )
        actual = run_gate6_compatible_dynamic_stage_b_tensor(
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

        torch.testing.assert_close(
            actual.selected_heap_ids,
            torch.tensor(expected_selected, dtype=torch.int64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual.terminal_child_heap_ids,
            torch.tensor(expected_terminals, dtype=torch.int64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual.overflow_pruned_count,
            torch.tensor(expected_pruned, dtype=torch.int64),
            rtol=0.0,
            atol=0.0,
        )

        for world_offset, reference_population in enumerate(expected_populations):
            ordered_reference = tuple(sorted(reference_population, key=lambda candidate: candidate.path))
            live_count = len(ordered_reference)
            actual_ids = actual.final_bank.heap_ids[world_offset, :live_count]
            expected_ids = torch.tensor([_heap_id(candidate.path) for candidate in ordered_reference], dtype=torch.int64)
            torch.testing.assert_close(actual_ids, expected_ids, rtol=0.0, atol=0.0)

            reference_states = torch.stack([candidate.state for candidate in ordered_reference])
            reference_scores = torch.tensor([candidate.score for candidate in ordered_reference], dtype=torch.float32)
            torch.testing.assert_close(
                actual.final_bank.states[world_offset, :live_count],
                reference_states,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                actual.final_bank.scores[world_offset, :live_count],
                reference_scores,
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(bool(actual.final_bank.live_mask[world_offset, :live_count].all()))
            self.assertFalse(bool(actual.final_bank.live_mask[world_offset, live_count:].any()))

    def test_dynamic_stage_b_matches_global_reference_at_n128(self) -> None:
        self._assert_equivalent(population=128, slots=24, mode="global_score", k=16, group="k16")

    def test_dynamic_stage_b_matches_k16_score_reference_at_n64(self) -> None:
        self._assert_equivalent(population=64, slots=32, mode="bounded_score", k=16, group="k16")

    def test_dynamic_stage_b_matches_k16_hash_reference_at_n64(self) -> None:
        self._assert_equivalent(population=64, slots=32, mode="bounded_hash", k=16, group="k16")

    def test_dynamic_stage_b_matches_k32_score_reference_at_n128(self) -> None:
        self._assert_equivalent(population=128, slots=24, mode="bounded_score", k=32, group="k32")

    def test_dynamic_stage_b_matches_k64_hash_reference_at_n256(self) -> None:
        self._assert_equivalent(population=256, slots=16, mode="bounded_hash", k=64, group="k64")

    def test_dynamic_hot_path_has_no_cuda_scalar_extraction(self) -> None:
        from ai_hypothesis.population_compute import gate7_dynamic_tensor_bank_prep as module

        for function in (
            module._select_global,
            module._bounded_sample_positions,
            module._select_bounded_score,
            module._select_bounded_hash,
            module._build_selected_child_inputs,
            module.run_gate6_compatible_dynamic_stage_b_tensor,
        ):
            source = inspect.getsource(function)
            for forbidden in (".item(", ".cpu(", ".tolist(", "float("):
                self.assertNotIn(forbidden, source, (function.__name__, forbidden))


if __name__ == "__main__":
    unittest.main()
