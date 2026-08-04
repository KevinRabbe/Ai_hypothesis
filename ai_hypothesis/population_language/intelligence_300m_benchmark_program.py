"""Benchmark evidence program for the Population Intelligence 300M roadmap.

The program defines contamination-resistant task families, split roles, exact
oracles, baselines, measurements, and claim limits. It contains no protected
instances, model execution, or result authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import intelligence_300m_mechanism_program as mechanisms

VERSION = "population-intelligence-300m-benchmark-program-v0"
BRANCH = "agent/population-intelligence-300m-benchmark-program-v0"
STATUS = "BENCHMARK_PROGRAM_ONLY_NO_GENERATION_OR_PROTECTED_RESULT_ACCESS"
SOURCE_MECHANISM_HEAD = "9cc8b1b4c1dc4fc95df346ef82b188e276d97fcb"

MINIMUM_INITIALIZATION_SEEDS = 3
CAPABILITY_MODES = mechanisms.CAPABILITY_MODES

SPLIT_ROLES = (
    ("development", "VISIBLE_FOR_IMPLEMENTATION_AND_DEBUGGING"),
    ("calibration", "VISIBLE_ONLY_FOR_PREDECLARED_SELECTION"),
    ("validation", "PROTECTED_UNTIL_ARCHITECTURE_AND_SELECTION_FREEZE"),
    ("test", "PROTECTED_UNTIL_FINAL_EXECUTION_AUTHORIZATION"),
)

PROTECTED_SPLITS = ("validation", "test")
DESIGN_SPLITS = ("development", "calibration")

GLOBAL_BASELINES = (
    "MATCHED_DENSE_TRANSFORMER",
    "MATCHED_RECURRENT_DENSE_MODEL",
    "POPULATION_WITH_TARGET_MECHANISM_ABLATED",
    "EQUAL_COMPUTE_INDEPENDENT_RESAMPLING",
)

GLOBAL_RESULT_FIELDS = (
    "benchmark_id",
    "benchmark_version",
    "split",
    "model_identity",
    "checkpoint_sha256",
    "initialization_seed",
    "capability_mode",
    "total_learned_parameters",
    "active_parameters",
    "trainable_parameters",
    "worker_count",
    "recurrent_rounds",
    "inference_flops",
    "routed_messages",
    "routed_bytes",
    "retrieved_bytes",
    "persisted_bytes",
    "verifier_calls",
    "tool_calls",
    "latency_seconds",
    "peak_ram_bytes",
    "peak_vram_bytes",
    "primary_metric",
    "secondary_metrics",
    "artifact_provenance",
)

DECISIVE_EVIDENCE_RULES = (
    "PROCEDURAL_OR_HELD_OUT_INSTANCES_ARE_PRIMARY",
    "PUBLIC_BENCHMARKS_ARE_SUPPORTIVE_NOT_SOLE_DECISION_EVIDENCE",
    "INSTANCE_GENERATION_AND_SCORING_ARE_VERSIONED",
    "DEVELOPMENT_CALIBRATION_VALIDATION_AND_TEST_SEEDS_ARE_DISJOINT",
    "VALIDATION_AND_TEST_LABELS_ARE_UNAVAILABLE_DURING_DESIGN",
    "NO_THRESHOLD_IS_CHANGED_AFTER_PROTECTED_RESULT_ACCESS",
    "ALL_CAPABILITY_MODES_ARE_REPORTED_SEPARATELY",
    "EXACT_ORACLES_OVERRIDE_LEARNED_JUDGES_WHERE_AVAILABLE",
    "FAILURES_TIMEOUTS_AND_INVALID_OUTPUTS_COUNT_AGAINST_RESULTS",
    "NEGATIVE_AND_NULL_RESULTS_ARE_PUBLISHED",
)

LATER_INTEGRATION_BENCHMARKS = (
    "STRUCTURED_DETERMINISTIC_RPG",
    "SCREEN_AND_EXTRACTED_TEXT_RPG",
    "PIXELS_AND_CONTROLLER_RPG",
    "RANDOMIZED_OR_PROCEDURAL_RPG",
)


@dataclass(frozen=True)
class BenchmarkFamily:
    identifier: str
    title: str
    mechanism_lanes: tuple[str, ...]
    question: str
    exact_oracle: str
    difficulty_axes: tuple[str, ...]
    required_metrics: tuple[str, ...]
    required_baselines: tuple[str, ...]
    required_failure_slices: tuple[str, ...]
    allowed_claim: str
    forbidden_claim: str

    def validate(self) -> "BenchmarkFamily":
        if not self.identifier.startswith("B") or len(self.identifier) != 3:
            raise ValueError("benchmark identifier must use the BNN form")
        if not self.identifier[1:].isdigit():
            raise ValueError("benchmark identifier suffix must be numeric")
        if not self.title or not self.question or not self.exact_oracle:
            raise ValueError("benchmark title, question, and oracle must be nonempty")
        if not self.mechanism_lanes:
            raise ValueError("benchmark must bind at least one mechanism lane")
        for lane in self.mechanism_lanes:
            mechanisms.lane_by_id(lane)
        if len(self.difficulty_axes) < 2:
            raise ValueError("benchmark requires at least two difficulty axes")
        if len(self.required_metrics) < 4:
            raise ValueError("benchmark requires at least four metrics")
        if not self.required_baselines or not self.required_failure_slices:
            raise ValueError("benchmark baselines and failure slices must be nonempty")
        if not self.allowed_claim or not self.forbidden_claim:
            raise ValueError("benchmark claim boundaries must be explicit")
        return self


BENCHMARK_FAMILIES = (
    BenchmarkFamily(
        identifier="B01",
        title="Procedural compositional rule induction",
        mechanism_lanes=("M01", "M02", "M03", "M05", "M06"),
        question=(
            "Can the model infer new latent rules from limited examples and apply "
            "them to deeper unseen compositions while preserving old capability?"
        ),
        exact_oracle="DETERMINISTIC_RULE_EXECUTOR_AVAILABLE_ONLY_TO_SCORING",
        difficulty_axes=(
            "NUMBER_OF_NEW_RULES",
            "COMPOSITION_DEPTH",
            "EXAMPLE_COVERAGE",
            "DISTRACTOR_RULES",
            "SEQUENTIAL_RULE_COUNT",
        ),
        required_metrics=(
            "DIRECT_HOLDOUT_ACCURACY",
            "COMPOSITION_ACCURACY_BY_DEPTH",
            "PAIRED_GAIN_CONFIDENCE_BOUND",
            "RETENTION_DROP",
            "RESTART_DRIFT",
            "ADAPTATION_PARAMETERS_AND_BYTES",
        ),
        required_baselines=(
            "NO_ADAPTATION",
            "FULL_CONTEXT_REPLAY",
            "RETRIEVAL_ONLY",
            "DENSE_ADAPTER_MATCHED_TRAINABLE_PARAMETERS",
        ),
        required_failure_slices=(
            "UNSEEN_OPERATOR",
            "UNSEEN_COMPOSITION",
            "CONFLICTING_EXAMPLES",
            "OUT_OF_BUDGET_ADAPTATION",
        ),
        allowed_claim="THE_MODEL_ACQUIRES_AND_COMPOSES_THE_TESTED_RULE_FAMILY",
        forbidden_claim="THE_RESULT_PROVES_GENERAL_CONTINUAL_LEARNING",
    ),
    BenchmarkFamily(
        identifier="B02",
        title="Algorithmic state tracking",
        mechanism_lanes=("M01", "M03", "M07", "M09"),
        question=(
            "Can recurrent population computation maintain and update exact latent "
            "state over long sequences more reliably than matched controls?"
        ),
        exact_oracle="REFERENCE_STATE_MACHINE",
        difficulty_axes=(
            "SEQUENCE_LENGTH",
            "STATE_DIMENSION",
            "DEPENDENCY_DISTANCE",
            "BRANCHING_FACTOR",
            "IRRELEVANT_EVENT_RATE",
        ),
        required_metrics=(
            "FINAL_STATE_EXACT_ACCURACY",
            "INTERMEDIATE_STATE_ACCURACY",
            "ERROR_ONSET_POSITION",
            "GAIN_PER_INFERENCE_FLOP",
            "ROUTED_BYTES",
            "RECOVERY_AFTER_DISTRACTOR",
        ),
        required_baselines=(
            "MATCHED_DENSE_TRANSFORMER",
            "MATCHED_RECURRENT_DENSE_MODEL",
            "NO_COMMUNICATION_POPULATION",
            "EQUAL_COMPUTE_RESAMPLING",
        ),
        required_failure_slices=(
            "LONG_DEPENDENCY",
            "HIGH_DISTRACTOR_RATE",
            "STATE_COLLISION",
            "ROUND_SATURATION",
        ),
        allowed_claim="THE_ARCHITECTURE_TRACKS_THE_TESTED_STATE_PROCESSES_BETTER",
        forbidden_claim="STATE_TRACKING_ACCURACY_ALONE_PROVES_PLANNING",
    ),
    BenchmarkFamily(
        identifier="B03",
        title="Graph planning and counterfactual search",
        mechanism_lanes=("M01", "M02", "M03", "M07"),
        question=(
            "Can diverse workers search competing plans, preserve minority evidence, "
            "and revise after counterexamples under controlled compute?"
        ),
        exact_oracle="DETERMINISTIC_GRAPH_SEARCH_AND_TRANSITION_SIMULATOR",
        difficulty_axes=(
            "GRAPH_SIZE",
            "PATH_DEPTH",
            "DECEPTIVE_BRANCH_COUNT",
            "CONSTRAINT_COUNT",
            "DYNAMIC_EDGE_CHANGES",
        ),
        required_metrics=(
            "VALID_PLAN_RATE",
            "OPTIMALITY_GAP",
            "COUNTEREXAMPLE_RECOVERY_RATE",
            "MINORITY_RESCUE_RATE",
            "COMPUTE_TO_SOLUTION",
            "HALTING_REGRET",
        ),
        required_baselines=(
            "GREEDY_ONE_PASS",
            "EQUAL_COMPUTE_BEAM_OR_RESAMPLING",
            "FLAT_POPULATION",
            "HIERARCHICAL_POPULATION",
        ),
        required_failure_slices=(
            "DECEPTIVE_LOCAL_OPTIMUM",
            "CONFLICTING_CONSTRAINTS",
            "LATE_INVALIDATION",
            "NO_VALID_PLAN",
        ),
        allowed_claim="THE_POPULATION_IMPROVES_SEARCH_ON_THE_TESTED_GRAPH_FAMILY",
        forbidden_claim="GRAPH_SUCCESS_PROVES_GENERAL_WORLD_MODELING",
    ),
    BenchmarkFamily(
        identifier="B04",
        title="Verified code synthesis",
        mechanism_lanes=("M04", "M10"),
        question=(
            "Can the model generate programs that satisfy hidden executable "
            "specifications, and can verified search be distilled?"
        ),
        exact_oracle="PARSER_COMPILER_STATIC_CHECKS_AND_HIDDEN_TESTS",
        difficulty_axes=(
            "SPECIFICATION_LENGTH",
            "ALGORITHMIC_COMPLEXITY",
            "API_SURFACE",
            "HIDDEN_EDGE_CASE_COUNT",
            "REQUIRED_REVISION_STEPS",
        ),
        required_metrics=(
            "COMPILE_RATE",
            "HIDDEN_TEST_PASS_RATE",
            "EXACT_SUCCESS_RATE",
            "FALSE_VERIFIER_ACCEPT_RATE",
            "ATTEMPTS_TO_SUCCESS",
            "VERIFICATION_COMPUTE",
        ),
        required_baselines=(
            "ONE_PASS_CODE_GENERATION",
            "EQUAL_COMPUTE_RESAMPLING",
            "LEARNED_VERIFIER_ONLY",
            "EXACT_VERIFIER_LOOP",
        ),
        required_failure_slices=(
            "COMPILES_BUT_WRONG",
            "OVERFITS_VISIBLE_TESTS",
            "API_MISUSE",
            "TIME_OR_MEMORY_LIMIT",
        ),
        allowed_claim="EXACT_VERIFICATION_IMPROVES_CODE_SUCCESS_ON_THE_TESTED_TASKS",
        forbidden_claim="COMPILATION_OR_VISIBLE_TESTS_ALONE_PROVE_CORRECTNESS",
    ),
    BenchmarkFamily(
        identifier="B05",
        title="Code diagnosis and repair",
        mechanism_lanes=("M04", "M10"),
        question=(
            "Can the model localize failures and make minimal verified repairs more "
            "efficiently than regeneration?"
        ),
        exact_oracle="FAULT_INJECTION_MANIFEST_AND_HIDDEN_REGRESSION_TESTS",
        difficulty_axes=(
            "FAULT_COUNT",
            "FAULT_DISTANCE_FROM_SYMPTOM",
            "CODEBASE_SIZE",
            "TEST_AMBIGUITY",
            "INTERACTING_FAULTS",
        ),
        required_metrics=(
            "FAULT_LOCALIZATION_ACCURACY",
            "REPAIR_SUCCESS_RATE",
            "REGRESSION_FREE_RATE",
            "PATCH_SIZE",
            "ATTEMPTS_TO_REPAIR",
            "FALSE_DIAGNOSIS_RATE",
        ),
        required_baselines=(
            "FULL_REGENERATION",
            "RANDOM_LOCAL_EDIT",
            "LEARNED_VERIFIER_REPAIR",
            "EXACT_FEEDBACK_REPAIR",
        ),
        required_failure_slices=(
            "MISLEADING_STACK_TRACE",
            "MULTIPLE_ROOT_CAUSES",
            "PLAUSIBLE_INCORRECT_PATCH",
            "NONLOCAL_REGRESSION",
        ),
        allowed_claim="THE_MODEL_DIAGNOSES_AND_REPAIRS_THE_TESTED_FAULT_FAMILIES",
        forbidden_claim="SMALL_PATCH_SIZE_IMPLIES_CORRECT_CAUSAL_UNDERSTANDING",
    ),
    BenchmarkFamily(
        identifier="B06",
        title="Changing organization memory and procedure learning",
        mechanism_lanes=("M05", "M06", "M08"),
        question=(
            "Can the system learn people, roles, procedures, exceptions, and changes "
            "over time without replaying complete history?"
        ),
        exact_oracle="VERSIONED_SYNTHETIC_ORGANIZATION_DATABASE_AND_POLICY_ENGINE",
        difficulty_axes=(
            "ORGANIZATION_SIZE",
            "CHANGE_RATE",
            "PROCEDURE_DEPTH",
            "EXCEPTION_COUNT",
            "TIME_SINCE_OBSERVATION",
        ),
        required_metrics=(
            "CURRENT_FACT_ACCURACY",
            "STALE_FACT_ERROR_RATE",
            "PROCEDURE_SUCCESS",
            "NOVEL_CASE_TRANSFER",
            "RESTART_PERSISTENCE",
            "UNRELATED_RETENTION",
        ),
        required_baselines=(
            "FULL_HISTORY_CONTEXT",
            "RETRIEVAL_ONLY",
            "ADAPTER_ONLY",
            "MEMORY_PLUS_ADAPTER",
        ),
        required_failure_slices=(
            "ROLE_CHANGE",
            "CONFLICTING_UPDATE",
            "EXCEPTION_TO_POLICY",
            "STALE_RETRIEVAL",
        ),
        allowed_claim="THE_SYSTEM_LEARNS_AND_UPDATES_THE_TESTED_ORGANIZATION",
        forbidden_claim="FACT_RECALL_ALONE_PROVES_PROCEDURAL_LEARNING",
    ),
    BenchmarkFamily(
        identifier="B07",
        title="Conditional memory capacity allocation",
        mechanism_lanes=("M05", "M08"),
        question=(
            "At matched total parameters and active compute, what allocation between "
            "dense reasoning and addressable memory maximizes useful capability?"
        ),
        exact_oracle="TASK_SPECIFIC_EXACT_ORACLES_AND_MEMORY_QUERY_MANIFEST",
        difficulty_axes=(
            "MEMORY_PARAMETER_FRACTION",
            "LOOKUP_SPARSITY",
            "KNOWLEDGE_NOVELTY",
            "REASONING_DEPTH",
            "LOOKUP_COLLISION_RATE",
        ),
        required_metrics=(
            "RECALL_ACCURACY",
            "REASONING_ACCURACY",
            "CODE_SUCCESS",
            "ACTIVE_FLOPS",
            "LOOKUP_BANDWIDTH",
            "CAPABILITY_PER_TOTAL_PARAMETER",
        ),
        required_baselines=(
            "ALL_DENSE_PARAMETERS",
            "RETRIEVAL_ONLY_EXTERNAL_MEMORY",
            "MATCHED_TOTAL_PARAMETER_CONDITIONAL_MEMORY",
            "MATCHED_ACTIVE_COMPUTE_DENSE_MODEL",
        ),
        required_failure_slices=(
            "MEMORY_COLLISION",
            "MISSING_ENTRY",
            "MISROUTED_LOOKUP",
            "REASONING_CAPACITY_STARVATION",
        ),
        allowed_claim="THE_MEASURED_ALLOCATION_IMPROVES_THE_TESTED_CAPABILITY_PROFILE",
        forbidden_claim="THE_BEST_SMALL_SCALE_MEMORY_RATIO_IS_FIXED_FOR_300M",
    ),
    BenchmarkFamily(
        identifier="B08",
        title="Adaptive compute challenge set",
        mechanism_lanes=("M01", "M02", "M03", "M09"),
        question=(
            "Can the model identify which instances benefit from more workers, rounds, "
            "search, or verification without spending maximum compute everywhere?"
        ),
        exact_oracle="MIXED_EXACT_ORACLES_WITH_PREDECLARED_INSTANCE_DIFFICULTY_FACTORS",
        difficulty_axes=(
            "MINIMUM_REQUIRED_ROUNDS",
            "MINIMUM_REQUIRED_WORKERS",
            "AMBIGUITY",
            "VERIFICATION_NEED",
            "UNSOLVABLE_INSTANCE_RATE",
        ),
        required_metrics=(
            "QUALITY_COMPUTE_FRONTIER",
            "ORACLE_COMPUTE_REGRET",
            "OVERCOMPUTE_RATE",
            "UNDERCOMPUTE_RATE",
            "UNSOLVABLE_STOP_RATE",
            "DIFFICULTY_CALIBRATION",
        ),
        required_baselines=(
            "FIXED_LOW_COMPUTE",
            "FIXED_MEAN_COMPUTE",
            "FIXED_MAX_COMPUTE",
            "RANDOM_COMPUTE_ALLOCATION",
        ),
        required_failure_slices=(
            "EASY_BUT_UNCERTAIN",
            "HARD_BUT_CONFIDENT",
            "UNSOLVABLE",
            "VERIFIER_MISLEADING",
        ),
        allowed_claim="THE_CONTROLLER_IMPROVES_THE_TESTED_QUALITY_COMPUTE_FRONTIER",
        forbidden_claim="CONTROLLER_CONFIDENCE_IS_A_GENERAL_MEASURE_OF_TASK_DIFFICULTY",
    ),
)


def benchmark_by_id(identifier: str) -> BenchmarkFamily:
    for family in BENCHMARK_FAMILIES:
        if family.identifier == identifier:
            return family
    raise KeyError(identifier)


def validate_program() -> dict[str, Any]:
    mechanism_snapshot = mechanisms.validate_program()
    identifiers = tuple(family.identifier for family in BENCHMARK_FAMILIES)
    if identifiers != tuple(f"B{index:02d}" for index in range(1, 9)):
        raise RuntimeError("benchmark identifiers or order drifted")
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("benchmark identifiers are not unique")
    for family in BENCHMARK_FAMILIES:
        family.validate()

    split_names = tuple(name for name, _ in SPLIT_ROLES)
    checks = {
        "source_mechanism_head_is_pinned": SOURCE_MECHANISM_HEAD
        == "9cc8b1b4c1dc4fc95df346ef82b188e276d97fcb",
        "mechanism_program_is_valid": mechanism_snapshot["valid"] is True,
        "capability_modes_match": CAPABILITY_MODES == mechanisms.CAPABILITY_MODES,
        "split_order_is_exact": split_names
        == ("development", "calibration", "validation", "test"),
        "protected_splits_are_exact": PROTECTED_SPLITS == ("validation", "test"),
        "minimum_seed_count_is_three": MINIMUM_INITIALIZATION_SEEDS == 3,
        "eight_benchmark_families_are_defined": len(BENCHMARK_FAMILIES) == 8,
        "global_result_schema_is_bounded": len(GLOBAL_RESULT_FIELDS) == 25,
        "interactive_worlds_are_later": not any(
            name in mechanisms.BENCHMARK_FAMILIES
            for name in LATER_INTEGRATION_BENCHMARKS
        ),
        "exact_oracles_are_required": all(
            family.exact_oracle for family in BENCHMARK_FAMILIES
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("Population Intelligence 300M benchmark program drifted")
    return {
        "version": VERSION,
        "branch": BRANCH,
        "status": STATUS,
        "source_mechanism_head": SOURCE_MECHANISM_HEAD,
        "minimum_initialization_seeds": MINIMUM_INITIALIZATION_SEEDS,
        "capability_modes": list(CAPABILITY_MODES),
        "split_roles": [list(value) for value in SPLIT_ROLES],
        "protected_splits": list(PROTECTED_SPLITS),
        "design_splits": list(DESIGN_SPLITS),
        "global_baselines": list(GLOBAL_BASELINES),
        "global_result_fields": list(GLOBAL_RESULT_FIELDS),
        "decisive_evidence_rules": list(DECISIVE_EVIDENCE_RULES),
        "later_integration_benchmarks": list(LATER_INTEGRATION_BENCHMARKS),
        "benchmark_families": [
            {
                "identifier": family.identifier,
                "title": family.title,
                "mechanism_lanes": list(family.mechanism_lanes),
                "question": family.question,
                "exact_oracle": family.exact_oracle,
                "difficulty_axes": list(family.difficulty_axes),
                "required_metrics": list(family.required_metrics),
                "required_baselines": list(family.required_baselines),
                "required_failure_slices": list(family.required_failure_slices),
                "allowed_claim": family.allowed_claim,
                "forbidden_claim": family.forbidden_claim,
            }
            for family in BENCHMARK_FAMILIES
        ],
        "checks": checks,
        "valid": True,
    }
