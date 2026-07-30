from __future__ import annotations

import inspect
import unittest

import torch

from ai_hypothesis.population_compute.gate3_v1_model import (
    Gate3V1NeuralCandidate,
    Gate3V1Scorer,
    encode_gate3_v1_child_input,
)
from ai_hypothesis.population_compute.gate3_v1_sparse_active_reserve import Gate3V1PublicWorld
from ai_hypothesis.population_compute.gate6_fixed_k_population_scaling import (
    Gate6EvaluationWorld,
    Gate6SchedulerMode,
    _advance_parent_batch,
    _bounded_visible_candidates,
    _score_rank,
)
from ai_hypothesis.population_compute.gate7_tensor_engine_prep import (
    GATE7_TENSOR_ENGINE_PREPARATION_ONLY,
    build_complete_tensor_frontier,
    build_productive_child_inputs,
    path_bits_to_tuple,
    path_tuple_to_bits,
    select_bounded_score_indices,
)


def _public_world(seed: int, hints: tuple[int, ...]) -> Gate3V1PublicWorld:
    world = Gate3V1PublicWorld(seed=seed, depth=10, noisy_hints=hints)
    world.validate()
    return world


def _evaluation_world(index: int, seed: int, hints: tuple[int, ...]) -> Gate6EvaluationWorld:
    world = Gate6EvaluationWorld(
        world_index=index,
        public=_public_world(seed, hints),
        hidden_path=(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    )
    world.validate()
    return world


class Gate7TensorEnginePreparationTests(unittest.TestCase):
    def test_preparation_only_boundary(self) -> None:
        self.assertTrue(GATE7_TENSOR_ENGINE_PREPARATION_ONLY)

    def test_path_integer_round_trip(self) -> None:
        for depth in range(0, 11):
            for value in range(1 << depth):
                path = path_bits_to_tuple(value, depth)
                self.assertEqual(path_tuple_to_bits(path), value)

    def test_vectorized_productive_inputs_match_reference_encoder(self) -> None:
        worlds = (
            _public_world(11, (0, 1, 0, 1, 1, 0, 1, 0, 0, 1)),
            _public_world(22, (1, 0, 1, 0, 0, 1, 0, 1, 1, 0)),
        )
        for child_depth in (1, 5, 10):
            actual = build_productive_child_inputs(
                worlds,
                child_depth=child_depth,
                parent_count=3,
                device="cpu",
            )
            expected = []
            for world in worlds:
                for _parent in range(3):
                    for action in (0, 1):
                        expected.append(
                            encode_gate3_v1_child_input(
                                world=world,
                                child_depth=child_depth,
                                observed_hint=world.noisy_hints[child_depth - 1],
                                branch_action=action,
                                sink=False,
                                device="cpu",
                            )
                        )
            torch.testing.assert_close(actual, torch.stack(expected), rtol=0.0, atol=0.0)

    def test_complete_tensor_frontier_matches_eager_reference(self) -> None:
        torch.manual_seed(12345)
        model = Gate3V1Scorer()
        worlds = (
            _evaluation_world(0, 101, (0, 1, 0, 1, 0, 1, 1, 0, 1, 0)),
            _evaluation_world(1, 202, (1, 0, 1, 0, 1, 0, 0, 1, 0, 1)),
        )

        populations = [
            (
                Gate3V1NeuralCandidate(
                    path=(),
                    state=model.initial_state(1, device="cpu")[0],
                    score=0.0,
                ),
            )
            for _ in worlds
        ]
        for _depth in range(8):
            populations = list(
                _advance_parent_batch(
                    model,
                    worlds,
                    tuple(populations),
                    device=torch.device("cpu"),
                )
            )

        tensor = build_complete_tensor_frontier(
            model,
            tuple(world.public for world in worlds),
            frontier_depth=8,
            device="cpu",
        )
        self.assertEqual(tensor.population, 256)
        self.assertEqual(tensor.depth, 8)

        for world_offset, reference in enumerate(populations):
            self.assertEqual(
                tuple(candidate.path for candidate in reference),
                tuple(path_bits_to_tuple(value, 8) for value in range(256)),
            )
            reference_states = torch.stack([candidate.state for candidate in reference])
            reference_scores = torch.tensor([candidate.score for candidate in reference], dtype=torch.float32)
            torch.testing.assert_close(tensor.states[world_offset], reference_states, rtol=0.0, atol=0.0)
            torch.testing.assert_close(tensor.scores[world_offset], reference_scores, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                tensor.path_bits[world_offset],
                torch.arange(256, dtype=torch.int64),
                rtol=0.0,
                atol=0.0,
            )

    def test_k16_tensor_selection_matches_eager_reference_including_ties(self) -> None:
        depth = 5
        population = 1 << depth
        world_seeds = (123, 456)
        slot = 311
        score_rows = []
        expected_positions = []
        path_rows = tuple(tuple(range(population)) for _ in world_seeds)

        for row_index, world_seed in enumerate(world_seeds):
            scores = [((index * 17 + row_index * 3) % 23) / 100.0 for index in range(population)]
            # Force several quantized ties so deterministic tie-breaking is exercised.
            scores[3] = 0.5014
            scores[9] = 0.5012
            scores[17] = 0.50149
            score_rows.append(scores)
            candidates = tuple(
                Gate3V1NeuralCandidate(
                    path=path_bits_to_tuple(index, depth),
                    state=torch.zeros(64),
                    score=scores[index],
                )
                for index in range(population)
            )
            visible = _bounded_visible_candidates(
                candidates,
                mode=Gate6SchedulerMode.BOUNDED_SCORE_K16,
                world_seed=world_seed,
                slot_index=slot,
            )
            selected = _score_rank(visible, world_seed=world_seed, expansion_index=slot)[0]
            expected_positions.append(path_tuple_to_bits(selected.path))

        actual = select_bounded_score_indices(
            torch.tensor(score_rows, dtype=torch.float32),
            path_bits_by_world=path_rows,
            path_depth=depth,
            world_seeds=world_seeds,
            slot_index=slot,
            k=16,
            sampling_group="k16",
        )
        torch.testing.assert_close(
            actual,
            torch.tensor(expected_positions, dtype=torch.int64),
            rtol=0.0,
            atol=0.0,
        )

    def test_hot_tensor_paths_do_not_extract_cuda_scalars_to_python(self) -> None:
        from ai_hypothesis.population_compute import gate7_tensor_engine_prep as module

        for function in (
            module.build_complete_tensor_frontier,
            module.build_productive_child_inputs,
            module.select_bounded_score_indices,
        ):
            source = inspect.getsource(function)
            self.assertNotIn(".item(", source)
            self.assertNotIn(".cpu(", source)
            self.assertNotIn(".tolist(", source)
            self.assertNotIn("float(", source)

        selection_source = inspect.getsource(module.select_bounded_score_indices)
        self.assertNotIn("sorted(", selection_source)


if __name__ == "__main__":
    unittest.main()
