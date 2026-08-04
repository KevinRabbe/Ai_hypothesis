"""Machine-readable evidence program for the Population Intelligence 300M roadmap.

This module freezes the questions, controls, measurements, dependencies, and
promotion rules that must guide the small-scale mechanism laboratory before a
50M, 100M, or 300M architecture is selected. It contains no model execution,
protected data, or authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VERSION = "population-intelligence-300m-mechanism-program-v0"
BRANCH = "agent/population-intelligence-300m-mechanism-program-v0"
STATUS = "MECHANISM_PROGRAM_ONLY_NO_EXPERIMENT_OR_ARCHITECTURE_FREEZE"
SOURCE_ROADMAP_HEAD = "eba20a82710b02092bf5c44b91247c3d277b5694"

PRIMARY_TARGET_PARAMETERS = 300_000_000
MODEL_STAGES = (
    ("diagnostic", 19_000_000),
    ("integration", 50_000_000),
    ("language_and_code", 100_000_000),
    ("primary", PRIMARY_TARGET_PARAMETERS),
)

CAPABILITY_MODES = (
    "ONE_PASS_CLOSED_BOOK",
    "RECURSIVE_POPULATION",
    "FULL_SYSTEM",
)

BENCHMARK_FAMILIES = (
    "PROCEDURAL_COMPOSITION",
    "ALGORITHMIC_STATE_TRACKING",
    "CODE_GENERATION_AND_REPAIR",
    "SEQUENTIAL_RULE_LEARNING",
    "CHANGING_ORGANIZATION",
)

LATER_INTEGRATION_BENCHMARK = "DETERMINISTIC_INTERACTIVE_WORLD"
SEPARATE_LATER_PROJECT = "POPULATION_EDGE_RUNTIME"

UNIVERSAL_CONTROLS = (
    "MATCHED_LEARNED_PARAMETER_BUDGET",
    "TRANSPARENT_ACTIVE_COMPUTE_ACCOUNTING",
    "MATCHED_TRAINING_DATA_AND_CHECKPOINT_SELECTION",
    "AT_LEAST_THREE_PREDECLARED_INITIALIZATION_SEEDS",
    "PROTECTED_EVALUATION_NOT_USED_FOR_DESIGN",
    "NEGATIVE_NULL_AND_FAILURE_RESULTS_PRESERVED",
)

PROMOTION_REQUIREMENTS = (
    "PROSPECTIVE_PREREGISTRATION",
    "PROTOCOL_SPECIFIC_SUCCESS_THRESHOLD_PASSED",
    "MATCHED_DENSE_OR_RECURRENT_CONTROL",
    "MATCHED_COMPUTE_CONTROL",
    "CAUSAL_ABLATION",
    "END_TO_END_COST_ACCOUNTING",
    "REPLICATION_ACROSS_PREDECLARED_SEEDS",
    "KNOWN_FAILURE_REGION_RECORDED",
    "CLAIM_RESTRICTED_TO_OBSERVED_SCALE_AND_TASKS",
)

ARCHITECTURE_FREEZE_REQUIREMENTS = (
    "ONLY_PROMOTED_MECHANISMS_MAY_ENTER_THE_50M_INTEGRATION_CANDIDATE",
    "FAILED_MECHANISMS_REQUIRE_A_NEW_PROSPECTIVE_PROTOCOL_BEFORE_REENTRY",
    "INTERACTIONS_REQUIRE_NEW_ABLATIONS_AT_50M",
    "THE_100M_RESULT_SELECTS_THE_300M_DESIGN",
    "THE_300M_DESIGN_MUST_BE_FROZEN_BEFORE_PROTECTED_FINAL_EVALUATION",
    "NO_EDGE_RUNTIME_CONSTRAINT_MAY_SELECT_THE_INTELLIGENCE_ARCHITECTURE",
)

PARAMETER_ALLOCATION_DOMAINS = (
    "LEXICAL_ENCODER_DECODER",
    "RECURRENT_POPULATION_CORE",
    "ROUTING_AND_COMMUNICATION",
    "VERIFIER_AND_VALUE_SYSTEM",
    "CONDITIONAL_OR_PERSISTENT_MEMORY",
    "POST_TRAINING_PLASTICITY",
)


@dataclass(frozen=True)
class MechanismLane:
    identifier: str
    title: str
    priority: int
    dependencies: tuple[str, ...]
    question: str
    required_controls: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    allowed_claim: str
    forbidden_claim: str

    def validate(self) -> "MechanismLane":
        if not self.identifier.startswith("M") or len(self.identifier) != 3:
            raise ValueError("mechanism identifier must use the MNN form")
        if not self.identifier[1:].isdigit():
            raise ValueError("mechanism identifier suffix must be numeric")
        if self.priority not in (1, 2, 3):
            raise ValueError("mechanism priority lies outside the locked tiers")
        if not self.title or not self.question:
            raise ValueError("mechanism title and question must be nonempty")
        if not self.required_controls or not self.primary_metrics:
            raise ValueError("mechanism controls and metrics must be nonempty")
        if not self.allowed_claim or not self.forbidden_claim:
            raise ValueError("mechanism claim boundaries must be explicit")
        return self


MECHANISM_LANES = (
    MechanismLane(
        identifier="M01",
        title="Recurrent latent depth",
        priority=1,
        dependencies=(),
        question=(
            "Can repeated application of the same learned core improve difficult-task "
            "capability without adding learned parameters?"
        ),
        required_controls=(
            "SAME_CHECKPOINT_AND_WORKER_COUNT",
            "ROUNDS_1_2_4_8_16_32",
            "EQUAL_FLOP_RESAMPLING_OR_WIDER_SHALLOW_CONTROL",
            "NO_HIDDEN_RETRIEVAL_OR_TOOL_USE",
        ),
        primary_metrics=(
            "CAPABILITY_BY_ROUND",
            "GAIN_PER_ADDITIONAL_INFERENCE_FLOP",
            "CALIBRATION_BY_ROUND",
            "SATURATION_OR_INSTABILITY_POINT",
        ),
        allowed_claim="RECURRENCE_HELPS_ON_THE_TESTED_TASKS_AT_THE_TESTED_SCALE",
        forbidden_claim="RECURRENCE_CREATES_UNBOUNDED_INTELLIGENCE_OR_MUST_SCALE",
    ),
    MechanismLane(
        identifier="M02",
        title="Diversity and private deliberation",
        priority=1,
        dependencies=(),
        question=(
            "Can workers develop causally useful independent error modes without "
            "becoming incoherent or collapsing to copies?"
        ),
        required_controls=(
            "UNRESTRICTED_SHARED_WORKERS",
            "EQUAL_COST_INDEPENDENT_RESAMPLING",
            "PRIVATE_BEFORE_PUBLIC_DELIBERATION",
            "DIFFERENTIATED_INFORMATION_ACCESS",
            "ADVERSARIAL_COUNTEREXAMPLE_WORKERS",
        ),
        primary_metrics=(
            "PAIRWISE_STATE_AND_PREDICTION_SIMILARITY",
            "ERROR_OVERLAP",
            "UNIQUE_CORRECTION_RATE",
            "MINORITY_RESCUE_RATE",
            "CAUSAL_WORKER_ABLATION_LOSS",
        ),
        allowed_claim="DIVERSITY_PRODUCES_MEASURABLE_INDEPENDENT_CORRECTION",
        forbidden_claim="MORE_WORKERS_ARE_USEFUL_BECAUSE_WORKER_STATES_DIFFER",
    ),
    MechanismLane(
        identifier="M03",
        title="Adaptive test-time computation",
        priority=1,
        dependencies=("M01", "M02"),
        question=(
            "Can a learned controller allocate workers, rounds, candidates, and "
            "verification only where additional compute is useful?"
        ),
        required_controls=(
            "FIXED_SCHEDULE_MATCHED_MEAN_COMPUTE",
            "FIXED_SCHEDULE_MATCHED_MAXIMUM_COMPUTE",
            "RANDOM_ALLOCATION_CONTROL",
            "CONTROLLER_SIGNAL_ABLATIONS",
        ),
        primary_metrics=(
            "CAPABILITY_COMPUTE_PARETO_FRONTIER",
            "HALTING_REGRET",
            "WASTED_COMPUTE_RATE",
            "DIFFICULTY_ALLOCATION_CORRELATION",
            "FAILURE_TO_ESCALATE_RATE",
        ),
        allowed_claim="ADAPTIVE_COMPUTE_IMPROVES_THE_MEASURED_QUALITY_COST_FRONTIER",
        forbidden_claim="THE_CONTROLLER_IDENTIFIES_TRUE_REASONING_DIFFICULTY_GENERALLY",
    ),
    MechanismLane(
        identifier="M04",
        title="Verifier-guided generation and repair",
        priority=1,
        dependencies=(),
        question=(
            "Does generate-test-diagnose-revise beat one-pass generation and "
            "equal-cost independent resampling when exact evidence is available?"
        ),
        required_controls=(
            "ONE_PASS_GENERATION",
            "EQUAL_COST_INDEPENDENT_RESAMPLING",
            "LEARNED_VERIFIER_ONLY",
            "EXACT_EXECUTION_VERIFIER",
            "VERIFIER_FAILURE_INJECTION",
        ),
        primary_metrics=(
            "EXACT_TASK_SUCCESS",
            "FALSE_ACCEPT_RATE",
            "REPAIR_GAIN_PER_ATTEMPT",
            "DIAGNOSIS_LOCALIZATION_ACCURACY",
            "VERIFICATION_COMPUTE",
        ),
        allowed_claim="EXACT_FEEDBACK_IMPROVES_VERIFIED_SUCCESS_ON_SUPPORTED_TASKS",
        forbidden_claim="A_LEARNED_VERIFIER_IS_AUTHORITATIVE_WHEN_EXACT_EVIDENCE_EXISTS",
    ),
    MechanismLane(
        identifier="M05",
        title="Memory versus learning separation",
        priority=1,
        dependencies=(),
        question=(
            "Which persistent mechanism is responsible for factual recall, procedure "
            "acquisition, transfer, restart survival, and forgetting?"
        ),
        required_controls=(
            "FULL_CONTEXT_REPLAY",
            "RAW_RETRIEVAL_ONLY",
            "COMPRESSED_MEMORY_ONLY",
            "PERSISTENT_ADAPTER_ONLY",
            "MEMORY_PLUS_ADAPTER",
            "NO_PERSISTENCE_CONTROL",
        ),
        primary_metrics=(
            "FACT_RECALL",
            "PROCEDURAL_TRANSFER",
            "UNSEEN_COMPOSITION_GENERALIZATION",
            "FRESH_PROCESS_PERSISTENCE",
            "RETENTION_DROP",
            "PERSISTED_BYTES",
        ),
        allowed_claim="THE_TEST_IDENTIFIES_WHICH_PERSISTENCE_MECHANISM_CAUSES_EACH_GAIN",
        forbidden_claim="RETRIEVAL_ALONE_IS_CONTINUAL_LEARNING",
    ),
    MechanismLane(
        identifier="M06",
        title="Sequential continual learning",
        priority=1,
        dependencies=("M05",),
        question=(
            "Can the frozen-base system acquire several changing skills or procedures "
            "while preserving old capability and rejecting contradictions?"
        ),
        required_controls=(
            "SINGLE_SKILL_BASELINE",
            "MULTIPLE_PREDECLARED_SKILL_ORDERS",
            "NO_REPLAY_CONTROL",
            "BOUNDED_REPLAY_CONTROL",
            "ISOLATED_ADAPTER_CONTROL",
            "CONSOLIDATION_CONTROL",
        ),
        primary_metrics=(
            "ACQUISITION_BY_SKILL",
            "FORWARD_TRANSFER",
            "BACKWARD_TRANSFER",
            "CATASTROPHIC_FORGETTING",
            "CROSS_SKILL_INTERFERENCE",
            "CONTRADICTION_REJECTION",
            "RESTART_PERSISTENCE",
        ),
        allowed_claim="THE_SYSTEM_LEARNS_THE_TESTED_SEQUENCE_WITH_MEASURED_INTERFERENCE",
        forbidden_claim="ONE_SUCCESSFUL_ADAPTATION_PROVES_GENERAL_CONTINUAL_LEARNING",
    ),
    MechanismLane(
        identifier="M07",
        title="Hierarchical communication",
        priority=2,
        dependencies=("M02",),
        question=(
            "Can local groups, summaries, or blackboards preserve decisive evidence "
            "while reducing bandwidth and consensus collapse?"
        ),
        required_controls=(
            "NO_COMMUNICATION",
            "FLAT_ALL_TO_ALL",
            "FLAT_SPARSE_TOP_K",
            "LOCAL_GROUPS_WITH_SUMMARIES",
            "SHARED_BLACKBOARD",
        ),
        primary_metrics=(
            "CAPABILITY",
            "ROUTED_BYTES_AND_MESSAGES",
            "EFFECTIVE_WORKER_UTILIZATION",
            "DECISIVE_EVIDENCE_RETENTION",
            "GROUP_CAUSAL_ABLATION_LOSS",
        ),
        allowed_claim="THE_TOPOLOGY_IMPROVES_THE_MEASURED_CAPABILITY_BANDWIDTH_FRONTIER",
        forbidden_claim="HIERARCHY_IS_BETTER_BECAUSE_IT_USES_FEWER_MESSAGES",
    ),
    MechanismLane(
        identifier="M08",
        title="Conditional memory allocation",
        priority=2,
        dependencies=("M05",),
        question=(
            "Does reallocating learned parameters from dense computation to sparse "
            "addressable memory improve reasoning as well as recall?"
        ),
        required_controls=(
            "DENSE_MATCHED_TOTAL_PARAMETERS",
            "MATCHED_ACTIVE_FLOPS",
            "MATCHED_TRAINING_TOKENS",
            "RETRIEVAL_ONLY_BASELINE",
            "MEMORY_CAPACITY_SWEEP",
        ),
        primary_metrics=(
            "FACTUAL_RECALL",
            "COMPOSITIONAL_REASONING",
            "CODE_AND_ALGORITHM_SUCCESS",
            "ACTIVE_FLOPS",
            "LOOKUP_BANDWIDTH",
            "EFFECTIVE_RECURRENT_DEPTH",
        ),
        allowed_claim="CONDITIONAL_MEMORY_FREES_USEFUL_CAPACITY_ON_THE_TESTED_WORKLOADS",
        forbidden_claim="MORE_STORED_PARAMETERS_ARE_EQUIVALENT_TO_MORE_REASONING_CAPACITY",
    ),
    MechanismLane(
        identifier="M09",
        title="Scale-stable parameterization",
        priority=2,
        dependencies=("M01", "M03", "M07"),
        question=(
            "Can optimizer, activation, routing, recurrence, and message scales tuned "
            "on proxies transfer predictably to larger population models?"
        ),
        required_controls=(
            "INDEPENDENT_RETUNING_BASELINE",
            "PROXY_TO_TARGET_TRANSFER",
            "WIDTH_WORKER_AND_ROUND_SWEEPS",
            "ROUTER_AND_MESSAGE_SCALE_ABLATIONS",
        ),
        primary_metrics=(
            "HYPERPARAMETER_TRANSFER_ERROR",
            "ACTIVATION_SCALE_DRIFT",
            "GRADIENT_SCALE_DRIFT",
            "ROUTER_ENTROPY_DRIFT",
            "TRAINING_STABILITY",
            "TARGET_PERFORMANCE_REGRET",
        ),
        allowed_claim="THE_TESTED_PARAMETERIZATION_TRANSFERS_BETWEEN_THE_MEASURED_SCALES",
        forbidden_claim="A_PROXY_TUNING_RESULT_AUTOMATICALLY_TRANSFERS_TO_300M_OR_LARGER",
    ),
    MechanismLane(
        identifier="M10",
        title="Verified search distillation",
        priority=3,
        dependencies=("M04",),
        question=(
            "Can expensive verified population search be distilled into cheaper direct "
            "behavior without teaching unverifiable reasoning artifacts?"
        ),
        required_controls=(
            "FINAL_ANSWER_ONLY_TRAINING",
            "UNFILTERED_TRAJECTORY_DISTILLATION",
            "VERIFIED_TRAJECTORY_DISTILLATION",
            "STUDENT_GENERATED_TRAJECTORIES",
            "EQUAL_TRAINING_TOKEN_CONTROL",
        ),
        primary_metrics=(
            "DIRECT_STUDENT_SUCCESS",
            "SEARCH_COMPUTE_REDUCTION",
            "VERIFIER_PASS_RATE",
            "OUT_OF_DISTRIBUTION_TRANSFER",
            "TEACHER_DEPENDENCE",
        ),
        allowed_claim="VERIFIED_SEARCH_CAN_BE_COMPRESSED_FOR_THE_TESTED_TASK_FAMILY",
        forbidden_claim="DISTILLATION_PRESERVES_ALL_SEARCH_CAPABILITY_OR_REASONING_FAITHFULNESS",
    ),
)


def lane_by_id(identifier: str) -> MechanismLane:
    for lane in MECHANISM_LANES:
        if lane.identifier == identifier:
            return lane
    raise KeyError(identifier)


def validate_program() -> dict[str, Any]:
    identifiers = tuple(lane.identifier for lane in MECHANISM_LANES)
    if identifiers != tuple(f"M{index:02d}" for index in range(1, 11)):
        raise RuntimeError("mechanism identifiers or order drifted")
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("mechanism identifiers are not unique")

    seen: set[str] = set()
    for lane in MECHANISM_LANES:
        lane.validate()
        if any(dependency not in seen for dependency in lane.dependencies):
            raise RuntimeError("mechanism dependency does not precede the dependent lane")
        seen.add(lane.identifier)

    checks = {
        "source_roadmap_head_is_pinned": SOURCE_ROADMAP_HEAD
        == "eba20a82710b02092bf5c44b91247c3d277b5694",
        "primary_target_is_300m": PRIMARY_TARGET_PARAMETERS == 300_000_000,
        "model_stage_order_is_locked": tuple(name for name, _ in MODEL_STAGES)
        == ("diagnostic", "integration", "language_and_code", "primary"),
        "capability_modes_are_separate": CAPABILITY_MODES
        == ("ONE_PASS_CLOSED_BOOK", "RECURSIVE_POPULATION", "FULL_SYSTEM"),
        "ten_mechanism_lanes_are_defined": len(MECHANISM_LANES) == 10,
        "all_universal_controls_present": len(UNIVERSAL_CONTROLS) == 6,
        "promotion_is_prospective_and_causal": (
            "PROSPECTIVE_PREREGISTRATION" in PROMOTION_REQUIREMENTS
            and "CAUSAL_ABLATION" in PROMOTION_REQUIREMENTS
            and "END_TO_END_COST_ACCOUNTING" in PROMOTION_REQUIREMENTS
        ),
        "interactive_world_is_later": LATER_INTEGRATION_BENCHMARK
        not in BENCHMARK_FAMILIES,
        "edge_runtime_is_separate": SEPARATE_LATER_PROJECT
        == "POPULATION_EDGE_RUNTIME",
        "parameter_allocation_remains_evidence_driven": len(
            PARAMETER_ALLOCATION_DOMAINS
        )
        == 6,
    }
    if not all(checks.values()):
        raise RuntimeError("Population Intelligence 300M mechanism program drifted")
    return {
        "version": VERSION,
        "branch": BRANCH,
        "status": STATUS,
        "source_roadmap_head": SOURCE_ROADMAP_HEAD,
        "primary_target_parameters": PRIMARY_TARGET_PARAMETERS,
        "model_stages": [list(stage) for stage in MODEL_STAGES],
        "capability_modes": list(CAPABILITY_MODES),
        "benchmark_families": list(BENCHMARK_FAMILIES),
        "later_integration_benchmark": LATER_INTEGRATION_BENCHMARK,
        "separate_later_project": SEPARATE_LATER_PROJECT,
        "universal_controls": list(UNIVERSAL_CONTROLS),
        "promotion_requirements": list(PROMOTION_REQUIREMENTS),
        "architecture_freeze_requirements": list(ARCHITECTURE_FREEZE_REQUIREMENTS),
        "parameter_allocation_domains": list(PARAMETER_ALLOCATION_DOMAINS),
        "mechanism_lanes": [
            {
                "identifier": lane.identifier,
                "title": lane.title,
                "priority": lane.priority,
                "dependencies": list(lane.dependencies),
                "question": lane.question,
                "required_controls": list(lane.required_controls),
                "primary_metrics": list(lane.primary_metrics),
                "allowed_claim": lane.allowed_claim,
                "forbidden_claim": lane.forbidden_claim,
            }
            for lane in MECHANISM_LANES
        ],
        "checks": checks,
        "valid": True,
    }
