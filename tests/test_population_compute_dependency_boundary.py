from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path("ai_hypothesis/population_compute")
FORBIDDEN_PREFIXES = (
    "ai_hypothesis.runtime",
    "ai_hypothesis.large_scope",
    "ai_hypothesis.step02",
)


class PopulationComputeDependencyBoundaryTests(unittest.TestCase):
    def test_population_compute_does_not_import_legacy_research_stacks(self) -> None:
        """Keep the current Gate-0/1/2 research line independent of older stacks.

        The runtime, large-scope, and Step-2 packages are historically useful, but the
        current fixed-parameter population-compute experiments must not depend on them
        implicitly.  This is a repository-hygiene contract only; it does not delete or
        deprecate those packages.
        """

        violations: list[str] = []
        paths = sorted(PACKAGE_ROOT.glob("*.py"))
        self.assertTrue(paths, "population_compute package is unexpectedly empty")

        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                else:
                    continue

                for module in modules:
                    if module.startswith(FORBIDDEN_PREFIXES):
                        violations.append(f"{path}:{node.lineno}: {module}")

        self.assertEqual(
            violations,
            [],
            "population_compute unexpectedly depends on older repository stacks:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
