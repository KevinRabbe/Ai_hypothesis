from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CHARTER = ROOT / "docs" / "project_charter.md"
HYPOTHESIS = ROOT / "docs" / "hypothesis.md"
QUESTIONS = ROOT / "docs" / "research_questions.md"
ROADMAP = ROOT / "docs" / "roadmap.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class PopulationIntelligence300MCharterTests(unittest.TestCase):
    def test_primary_objective_is_consistent(self) -> None:
        required = (
            "smartest population-based language model",
            "300 million learned-parameter budget",
        )
        for path in (README, CHARTER):
            text = read(path)
            for phrase in required:
                self.assertIn(phrase, text, f"{path} lost locked objective phrase: {phrase}")

    def test_repository_entry_point_links_locked_documents(self) -> None:
        text = read(README)
        for relative_path in (
            "docs/project_charter.md",
            "docs/hypothesis.md",
            "docs/research_questions.md",
            "docs/roadmap.md",
        ):
            self.assertIn(relative_path, text)

    def test_roadmap_preserves_gate_order(self) -> None:
        text = read(ROADMAP)
        headings = (
            "## Gate 1 — Population Language L0 reference evidence",
            "## Gate 2 — Bounded Post-Training Learning L0",
            "## Gate 3 — Small-scale intelligence mechanism laboratory",
            "## Gate 4 — Approximately 50M architecture integration",
            "## Gate 5 — Approximately 100M language-and-code model",
            "## Gate 6 — Strongest approximately 300M population model",
            "## Gate 7 — Deterministic interactive world learning",
            "## Gate 8 — Scaling beyond 300M",
            "## Separate later project — Population Edge Runtime",
        )
        positions = [text.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions), "protected roadmap gate order changed")

    def test_edge_runtime_remains_separate(self) -> None:
        for path in (README, CHARTER, HYPOTHESIS, QUESTIONS, ROADMAP):
            text = read(path).lower()
            self.assertIn("edge", text, f"{path} lost the edge-runtime scope boundary")
        charter = read(CHARTER)
        self.assertIn("Population Edge Runtime", charter)
        self.assertIn("must not dictate the present intelligence research", charter)

    def test_material_change_control_is_present(self) -> None:
        charter = read(CHARTER)
        self.assertIn("## 7. Roadmap change control", charter)
        self.assertIn("dedicated charter-version proposal", charter)
        self.assertIn("explicit owner approval before merge", charter)
        self.assertIn("Existing preregistered protocols remain immutable historical records", charter)

    def test_stale_primary_budget_language_is_removed(self) -> None:
        stale_phrases = (
            "long-term reference budget is approximately **1 billion",
            "A long-term reference experiment may use approximately 1 billion",
            "## Step 11 — Scale Toward ~1B Total Parameters",
            "## RQ13 — Scaling toward 1B total parameters",
        )
        for path in (README, CHARTER, HYPOTHESIS, QUESTIONS, ROADMAP):
            text = read(path)
            for phrase in stale_phrases:
                self.assertNotIn(phrase, text, f"{path} retains superseded primary direction")


if __name__ == "__main__":
    unittest.main()
