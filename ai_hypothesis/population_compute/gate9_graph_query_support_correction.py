"""Data-blind correction to the Gate-9 graph-query support policy.

The original protocol's single query-exclusion flag is narrowed to the isolated
local-induction test. Distributed graph messages may naturally equal one of the
nine support inputs. The graph generator may not reject operators, skip frozen
counters, or alter worlds to prevent such hits; instead exact support-hit rates
are mandatory descriptive evidence. No generator or execution surface opens.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

_PROTOCOL_PATH = pathlib.Path(__file__).with_name(
    "gate9_contextual_operator_induction_protocol.py"
)


def _load_protocol():
    name = "gate9_graph_query_support_correction_protocol_dependency"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen Gate9 protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()

GATE9_QUERY_POLICY_CORRECTION_VERSION = "gate9-graph-query-support-correction-v0"
GATE9_QUERY_POLICY_STATUS = (
    "DATA_BLIND_GATE9_GRAPH_QUERY_SUPPORT_POLICY_CORRECTED_EXECUTION_CLOSED"
)
GATE9_ORIGINAL_PROTOCOL_HEAD = "e5e20e8de6707d35f1a7a9315a5a9a67deacc9a1"
GATE9_OPERATOR_CONTRACT_HEAD = "be6451e1af82b18749bd0313a9c02ca62c4eee5c"

GATE9_LOCAL_QUERY_EXCLUDES_SUPPORT_INPUTS = True
GATE9_GRAPH_QUERY_EXCLUDES_SUPPORT_INPUTS = False
GATE9_GRAPH_SUPPORT_HIT_REPORT_REQUIRED = True
GATE9_GRAPH_OPERATOR_REJECTION_ON_SUPPORT_HIT = False
GATE9_GRAPH_OPERATOR_COUNTER_SKIPPING_ALLOWED = False
GATE9_GRAPH_WORLD_REJECTION_ON_SUPPORT_HIT = False


@dataclass(frozen=True, slots=True)
class Gate9GraphSupportHitEvidence:
    population: int
    depth: int
    worlds: int
    total_path_queries: int
    support_hits: int
    support_hits_by_path_position: tuple[int, ...]

    def validate(self) -> None:
        if (self.population, self.depth) not in protocol.GATE9_VALID_CONDITIONS:
            raise ValueError("Gate9 support-hit evidence is outside the frozen matrix")
        if self.worlds != protocol.GATE9_WORLDS_PER_CONDITION:
            raise ValueError("Gate9 support-hit evidence world count drifted")
        if self.total_path_queries != self.worlds * self.depth:
            raise ValueError("Gate9 support-hit query count drifted")
        if len(self.support_hits_by_path_position) != self.depth:
            raise ValueError("Gate9 support-hit path-position vector drifted")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= self.worlds
            for value in self.support_hits_by_path_position
        ):
            raise ValueError("Gate9 path-position support-hit count is invalid")
        if self.support_hits != sum(self.support_hits_by_path_position):
            raise ValueError("Gate9 support-hit total disagrees with path positions")
        if not 0 <= self.support_hits <= self.total_path_queries:
            raise ValueError("Gate9 support-hit total is invalid")

    @property
    def support_hit_rate(self) -> float:
        self.validate()
        return self.support_hits / self.total_path_queries

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["support_hit_rate"] = self.support_hit_rate
        return payload


def corrected_gate9_query_policy_plan() -> dict[str, Any]:
    return {
        "version": GATE9_QUERY_POLICY_CORRECTION_VERSION,
        "status": GATE9_QUERY_POLICY_STATUS,
        "original_protocol_head": GATE9_ORIGINAL_PROTOCOL_HEAD,
        "operator_contract_head": GATE9_OPERATOR_CONTRACT_HEAD,
        "support_inputs": list(protocol.GATE9_SUPPORT_INPUTS),
        "local_query_excludes_support_inputs": (
            GATE9_LOCAL_QUERY_EXCLUDES_SUPPORT_INPUTS
        ),
        "graph_query_excludes_support_inputs": (
            GATE9_GRAPH_QUERY_EXCLUDES_SUPPORT_INPUTS
        ),
        "graph_support_hit_report_required": (
            GATE9_GRAPH_SUPPORT_HIT_REPORT_REQUIRED
        ),
        "graph_operator_rejection_on_support_hit": (
            GATE9_GRAPH_OPERATOR_REJECTION_ON_SUPPORT_HIT
        ),
        "graph_operator_counter_skipping_allowed": (
            GATE9_GRAPH_OPERATOR_COUNTER_SKIPPING_ALLOWED
        ),
        "graph_world_rejection_on_support_hit": (
            GATE9_GRAPH_WORLD_REJECTION_ON_SUPPORT_HIT
        ),
        "graph_support_hit_evidence_unit": (
            "world_path_position_before_each_relevant_edge_operator"
        ),
        "graph_operator_distribution_conditioned_on_support_hits": False,
        "thresholds_changed": False,
        "condition_matrix_changed": False,
        "operator_ranges_changed": False,
        "operator_generation_admitted": False,
        "graph_world_generation_admitted": False,
        "architecture_admitted": False,
        "training_admitted": False,
        "checkpoint_loading_admitted": False,
        "scientific_test_admitted": False,
        "result_classification_admitted": False,
    }
