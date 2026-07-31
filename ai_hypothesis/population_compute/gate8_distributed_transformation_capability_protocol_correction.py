"""Pre-exposure correction to the frozen Gate-8 capability protocol.

This append-only contract supersedes one inconsistent sentence in the qualified
protocol. The unique relevant root-to-target path contains exactly ``depth``
edges. Therefore relevant path edges are *at most* one eighth of all graph
edges whenever ``8 * depth <= population``; equality holds only when
``population == 8 * depth``.

The module is deliberately standalone so importing it cannot trigger package
initializers, Torch, a generator, an oracle, a model, training, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE8_PROTOCOL_CORRECTION_VERSION = (
    "gate8-distributed-transformation-capability-protocol-correction-v0"
)
GATE8_PROTOCOL_CORRECTION_BASE_HEAD = (
    "e73541115e8ddd122f336463dc1a9ffdbf82df46"
)
GATE8_PROTOCOL_CORRECTION_STATUS = (
    "DATA_FROZEN_GATE8_RELEVANT_EDGE_RULE_CORRECTED_EXECUTION_CLOSED"
)
GATE8_RELEVANT_EDGE_RULE = "unique_path_edges_equal_depth_and_are_at_most_one_eighth"
GATE8_RELEVANT_PATH_EDGE_COUNT = "depth"
GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR = 8
GATE8_RELEVANT_EDGE_FRACTION_BOUND = 1.0 / GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR
GATE8_CORRECTION_POPULATIONS = (32, 64, 128, 256, 512, 1_024)
GATE8_CORRECTION_DEPTHS = (4, 8, 16, 32, 64, 128)
GATE8_CORRECTION_VALID_CONDITIONS = tuple(
    (population, depth)
    for population in GATE8_CORRECTION_POPULATIONS
    for depth in GATE8_CORRECTION_DEPTHS
    if depth * GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR <= population
)


@dataclass(frozen=True, slots=True)
class Gate8RelevantEdgeContract:
    population: int
    depth: int

    def validate(self) -> None:
        if self.population not in GATE8_CORRECTION_POPULATIONS:
            raise ValueError("Gate8 correction population is outside the frozen ladder")
        if self.depth not in GATE8_CORRECTION_DEPTHS:
            raise ValueError("Gate8 correction depth is outside the frozen ladder")
        if (self.population, self.depth) not in GATE8_CORRECTION_VALID_CONDITIONS:
            raise ValueError("Gate8 correction condition is outside the frozen matrix")
        if self.relevant_path_edges != self.depth:
            raise ValueError("Gate8 relevant path must contain exactly depth edges")
        if self.relevant_fraction > GATE8_RELEVANT_EDGE_FRACTION_BOUND:
            raise ValueError("Gate8 relevant path exceeds one eighth of graph edges")

    @property
    def relevant_path_edges(self) -> int:
        return self.depth

    @property
    def total_edges(self) -> int:
        return self.population

    @property
    def distractor_edges(self) -> int:
        return self.population - self.depth

    @property
    def relevant_fraction(self) -> float:
        return self.depth / self.population

    @property
    def bound_is_tight(self) -> bool:
        return self.population == self.depth * GATE8_RELEVANT_EDGE_FRACTION_DENOMINATOR


def gate8_relevant_edge_contracts() -> tuple[Gate8RelevantEdgeContract, ...]:
    rows = tuple(
        Gate8RelevantEdgeContract(population=population, depth=depth)
        for population, depth in GATE8_CORRECTION_VALID_CONDITIONS
    )
    for row in rows:
        row.validate()
    return rows


def gate8_protocol_correction_plan() -> dict[str, object]:
    rows = gate8_relevant_edge_contracts()
    return {
        "version": GATE8_PROTOCOL_CORRECTION_VERSION,
        "base_protocol_head": GATE8_PROTOCOL_CORRECTION_BASE_HEAD,
        "scientific_status": GATE8_PROTOCOL_CORRECTION_STATUS,
        "execution_admitted": False,
        "generator_admitted": False,
        "training_admitted": False,
        "baseline_execution_admitted": False,
        "superseded_text": "relevant edges are exactly one eighth of all graph edges",
        "corrected_rule": GATE8_RELEVANT_EDGE_RULE,
        "relevant_path_edge_count": GATE8_RELEVANT_PATH_EDGE_COUNT,
        "maximum_relevant_fraction": GATE8_RELEVANT_EDGE_FRACTION_BOUND,
        "condition_count": len(rows),
        "tight_bound_conditions": [
            [row.population, row.depth] for row in rows if row.bound_is_tight
        ],
    }
