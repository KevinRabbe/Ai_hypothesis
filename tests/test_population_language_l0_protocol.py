from __future__ import annotations

import unittest

from ai_hypothesis.population_language import l0_protocol as p


class PopulationLanguageL0ProtocolContract(unittest.TestCase):
    def test_locked_dataset_fingerprints(self) -> None:
        self.assertEqual(
            p.dataset_fingerprint("train", 256),
            "65b4fc5e9a9d1e0e9f42544f086555a03f1545b0cfbc0df623126a8ba2150437",
        )
        self.assertEqual(
            p.dataset_fingerprint("validation", 256),
            "1563d2f3f8f247dad8627c677414f18432ee4840fe317be1815c38abb46284cf",
        )
        self.assertEqual(
            p.dataset_fingerprint("test", 256),
            "435c44813776ad034a1e44f6aa805454124e4c471e53f56a346cba6b2ee60485",
        )

    def test_episode_requires_contextual_nonce_binding(self) -> None:
        meanings: dict[str, set[tuple[str, str]]] = {token: set() for token in p.NONCE_TOKENS}
        for ordinal in range(1024):
            episode = p.make_episode("train", ordinal)
            meanings[episode.lhs_nonce].add((episode.lhs_color, episode.lhs_shape))
            meanings[episode.rhs_nonce].add((episode.rhs_color, episode.rhs_shape))
        self.assertTrue(all(len(values) > 1 for values in meanings.values()))

    def test_semantic_splits_do_not_overlap(self) -> None:
        semantics: dict[str, set[tuple[str, str, str, str, str]]] = {}
        for split in ("train", "validation", "test"):
            semantics[split] = {
                (
                    episode.lhs_color,
                    episode.lhs_shape,
                    episode.relation,
                    episode.rhs_color,
                    episode.rhs_shape,
                )
                for episode in (p.make_episode(split, ordinal) for ordinal in range(1024))
            }
        self.assertTrue(semantics["train"].isdisjoint(semantics["validation"]))
        self.assertTrue(semantics["train"].isdisjoint(semantics["test"]))
        self.assertTrue(semantics["validation"].isdisjoint(semantics["test"]))

    def test_answer_span_and_sequence_contract(self) -> None:
        episode = p.make_episode("test", 0)
        self.assertEqual(len(episode.tokens), 28)
        self.assertLessEqual(len(episode.tokens), p.MAX_SEQUENCE_LENGTH)
        self.assertEqual(
            episode.answer_tokens,
            (
                episode.lhs_color,
                episode.lhs_shape,
                episode.relation,
                episode.rhs_color,
                episode.rhs_shape,
            ),
        )
        self.assertEqual(episode.tokens[-1], "<eos>")
        self.assertEqual(len(episode.token_ids), len(episode.tokens))

    def test_parameter_budget_and_worker_independence(self) -> None:
        self.assertEqual(p.transformer_parameter_count(), 18_964_544)
        self.assertEqual(p.population_parameter_count(), 18_964_800)
        self.assertLessEqual(p.relative_parameter_delta(), p.PARAMETER_TOLERANCE_FRACTION)
        self.assertEqual(
            p.population_parameter_count(p.PopulationConfig(training_workers=256)),
            p.population_parameter_count(),
        )

    def test_protocol_validation(self) -> None:
        report = p.validate_protocol()
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "PROTOCOL_ONLY_NO_TRAINING_RESULT")
        self.assertEqual(report["training_workers"], 32)
        self.assertEqual(report["evaluation_workers"], [16, 32, 64, 128, 256])
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
