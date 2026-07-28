from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path("ai_hypothesis/population_compute")
FORBIDDEN_MODULE_PREFIXES = (
    "ai_hypothesis.runtime",
    "ai_hypothesis.large_scope",
    "ai_hypothesis.step02",
)
FORBIDDEN_PATH_FRAGMENTS = (
    "ai_hypothesis/runtime",
    "ai_hypothesis/large_scope",
    "ai_hypothesis/step02",
    "tests/test_runtime_",
    "tests/test_incremental_",
    "tests/test_indexed_",
    "tests/test_integration_",
    "tests/test_large_scope_",
    "tests/test_step02_",
)

CANONICAL_TEST_PATHS = (
    Path("tests/test_population_compute_dependency_boundary.py"),
    Path("tests/test_population_compute_contract.py"),
    Path("tests/test_collective_relay.py"),
    Path("tests/test_shared_population_cell.py"),
    Path("tests/test_relay_population_model.py"),
    Path("tests/test_compositional_relay_protocol.py"),
    Path("tests/test_population_state_reset_policy.py"),
    Path("tests/test_relay_experiment.py"),
    Path("tests/test_relay_serial_control.py"),
    Path("tests/test_relay_experiment_v1.py"),
    Path("tests/test_run_relay_scaling_v1.py"),
    Path("tests/test_confirmation_gate_v1.py"),
    Path("tests/test_relay_resource_frontier.py"),
    Path("tests/test_relay_resource_equivalence_diagnostic.py"),
    Path("tests/test_relay_precision_diagnostic.py"),
    Path("tests/test_relay_resource_frontier_v1.py"),
    Path("tests/test_relay_resource_audit.py"),
    Path("tests/test_gate2_persistent_state_capacity.py"),
    Path("tests/test_gate2_persistent_model.py"),
    Path("tests/test_gate2_development.py"),
)

CANONICAL_PYTHON_HELPERS = (
    Path("experiments/population_compute_scaling_v0/diagnose_compositional_relay.py"),
)

CANONICAL_TEXT_SURFACES = (
    Path(".github/workflows/population-compute-gate-ci.yml"),
    Path("scripts/run_gate1_resource_frontier.ps1"),
    Path("scripts/run_gate1_resource_frontier_v1.ps1"),
    Path("scripts/finalize_gate1_resource_frontier_v1_existing.ps1"),
    Path("scripts/run_gate2_development.ps1"),
)


def _absolute_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return imports


class PopulationComputeDependencyBoundaryTests(unittest.TestCase):
    def test_population_compute_python_surfaces_do_not_import_legacy_research_stacks(self) -> None:
        """Keep the Gate-0/1/2 executable Python surface independent of older stacks."""

        package_paths = sorted(PACKAGE_ROOT.glob("*.py"))
        self.assertTrue(package_paths, "population_compute package is unexpectedly empty")

        python_paths = package_paths + list(CANONICAL_TEST_PATHS) + list(CANONICAL_PYTHON_HELPERS)
        missing = [str(path) for path in python_paths if not path.is_file()]
        self.assertEqual(missing, [], "canonical Python boundary paths are missing: " + ", ".join(missing))

        violations: list[str] = []
        for path in python_paths:
            for lineno, module in _absolute_imports(path):
                if module.startswith(FORBIDDEN_MODULE_PREFIXES):
                    violations.append(f"{path}:{lineno}: {module}")

        self.assertEqual(
            violations,
            [],
            "canonical population-compute Python surfaces unexpectedly depend on older stacks:\n"
            + "\n".join(violations),
        )

    def test_population_compute_runners_and_ci_do_not_reference_legacy_paths(self) -> None:
        """Prevent shell/workflow references from surviving after deferred stacks are removed."""

        missing = [str(path) for path in CANONICAL_TEXT_SURFACES if not path.is_file()]
        self.assertEqual(missing, [], "canonical runner/CI paths are missing: " + ", ".join(missing))

        violations: list[str] = []
        for path in CANONICAL_TEXT_SURFACES:
            text = path.read_text(encoding="utf-8")
            normalized = text.replace("\\", "/")
            for fragment in FORBIDDEN_PATH_FRAGMENTS:
                if fragment in normalized:
                    violations.append(f"{path}: {fragment}")
            for module_prefix in FORBIDDEN_MODULE_PREFIXES:
                if module_prefix in text:
                    violations.append(f"{path}: {module_prefix}")

        self.assertEqual(
            violations,
            [],
            "canonical population-compute runner/CI surfaces still reference deferred stacks:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
