"""Qualified Post-Training Learning L0 stack manifest and protected handoff state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

VERSION = "population-language-post-training-learning-l0-stack-audit-v0"
BRANCH = "agent/population-language-post-training-learning-l0-stack-audit-v0"
STATUS = "QUALIFIED_STACK_AUDIT_ONLY_NO_EXECUTION_OR_MERGE"
SOURCE_ORCHESTRATOR_HEAD = "52f0e5e1a97a2a78d42b17e6861da4464665c754"

MERGE_POLICY = "REVIEW_BEFORE_MERGE_NO_AUTOMATIC_MERGE"
STACK_STATE = "OPEN_DRAFT_UNMERGED_AT_QUALIFICATION"
FIRST_BASE_SHA = "bfd2111b65f805e6379ad45ecda6f5fe09d2a282"

DEFAULT_PROTECTED_STATE = {
    "active_reference_output_access_allowed": False,
    "calibration_execution_authorized": False,
    "final_world_access_authorized": False,
    "final_execution_authorized": False,
    "automatic_merge_allowed": False,
    "fifty_m_architecture_frozen": False,
    "hundred_m_architecture_frozen": False,
    "three_hundred_m_architecture_frozen": False,
}

MISSING_OPERATIONAL_COMPONENTS = (
    "REAL_AUTHORIZATION_GATED_CALIBRATION_ROW_EXECUTOR",
    "FINAL_EVALUATION_CANDIDATE_LOCK_AND_PLAN",
    "FINAL_RESULT_VERIFIER",
    "POWERSHELL_OPERATOR_RUNBOOK",
)

NEXT_HANDOFF_ORDER = (
    "WAIT_FOR_REFERENCE_TRAINING_TO_COMPLETE",
    "RECEIVE_FINAL_CONSOLE_OUTPUT_WITHOUT_DISCOVERING_OUTPUTS",
    "VERIFY_EXPLICIT_REFERENCE_OUTPUT_WITH_PR_211_BOUNDARY",
    "BUILD_HASH_PINNED_CALIBRATION_PLAN_WITH_PR_212_BOUNDARY",
    "QUALIFY_REAL_CALIBRATION_ROW_EXECUTOR",
    "REQUEST_SEPARATE_EXPLICIT_CALIBRATION_AUTHORIZATION",
    "RUN_ONLY_THE_FROZEN_144_ROW_CALIBRATION",
    "VERIFY_CALIBRATION_RESULT_WITH_PR_215_BOUNDARY",
    "LOCK_SELECTED_CANDIDATE_OR_RECORD_VALID_REJECTION",
    "REQUEST_SEPARATE_EXPLICIT_FINAL_EXECUTION_AUTHORIZATION",
)


@dataclass(frozen=True)
class StackEntry:
    order: int
    pull_request: int
    title: str
    role: str
    branch: str
    base_sha: str
    head_sha: str
    synthetic_merge_sha: str
    qualification: str
    state: str = STACK_STATE
    merge_allowed: bool = False


STACK = (
    StackEntry(
        1,
        203,
        "Preregister Population Language Post-Training Learning L0",
        "FROZEN_SCIENTIFIC_PROTOCOL",
        "agent/population-language-post-training-learning-l0-protocol-v0",
        "bfd2111b65f805e6379ad45ecda6f5fe09d2a282",
        "48e8edb9ff39417bfb5cb44521318efa032a340a",
        "6fcb507fb0574c8db0a1707aaa2a9165b55e7e8f",
        "LOCAL_PROTOCOL_TESTS_AND_REMOTE_SCOPE_QUALIFIED",
    ),
    StackEntry(
        2,
        204,
        "Implement bounded Post-Training Learning L0 adapter",
        "BOUNDED_NEURAL_ADAPTER",
        "agent/population-language-post-training-learning-l0-adapter-v0",
        "48e8edb9ff39417bfb5cb44521318efa032a340a",
        "508a1021f3724a39023d4a4f7c6918d98f379f5c",
        "18a538e1f9ee4c76460604d46cf493a0e16aadee",
        "GITHUB_ACTIONS_30910221919_SUCCESS",
    ),
    StackEntry(
        3,
        207,
        "Lock Post-Training Learning L0 calibration contract",
        "CALIBRATION_GRID_AND_SELECTION_CONTRACT",
        "agent/population-language-post-training-learning-l0-calibration-v0",
        "508a1021f3724a39023d4a4f7c6918d98f379f5c",
        "19aa701c475b19fc5b31409528948f21ad9fbdf4",
        "182146345464b4ce67210cbc5a2e8db5f61ec6fe",
        "GITHUB_ACTIONS_30928864211_SUCCESS",
    ),
    StackEntry(
        4,
        208,
        "Implement Post-Training Learning L0 execution primitives",
        "DETERMINISTIC_EXECUTION_PRIMITIVES",
        "agent/population-language-post-training-learning-l0-execution-primitives-v0",
        "19aa701c475b19fc5b31409528948f21ad9fbdf4",
        "821449afe7381d4becc9c43dc456632b66b8f034",
        "b4d3a1340cac92a31094d6e130dd5551b26c697d",
        "GITHUB_ACTIONS_30929795640_SUCCESS",
    ),
    StackEntry(
        5,
        209,
        "Add strict Post-Training Learning L0 checkpoint contract",
        "STRICT_REFERENCE_CHECKPOINT_LOADER",
        "agent/population-language-post-training-learning-l0-checkpoint-contract-v0",
        "821449afe7381d4becc9c43dc456632b66b8f034",
        "0b43d2cfedcaaf92a9905750ba3cac809645bebd",
        "6c0d9aa9367f995ae0491736e2056a47042a60f5",
        "GITHUB_ACTIONS_30935470745_SUCCESS",
    ),
    StackEntry(
        6,
        210,
        "Add fresh-process Post-Training Learning persistence harness",
        "TRUE_SUBPROCESS_RESTART_BOUNDARY",
        "agent/population-language-post-training-learning-l0-fresh-process-v0",
        "0b43d2cfedcaaf92a9905750ba3cac809645bebd",
        "f0cf83d1be0426fda976f08a379ab040be53ba89",
        "b2423aa0e212060b290ce96cf79380dde7e4c857",
        "GITHUB_ACTIONS_30936129511_SUCCESS",
    ),
    StackEntry(
        7,
        211,
        "Add strict Population Language L0 reference manifest verifier",
        "COMPLETED_REFERENCE_OUTPUT_VERIFIER",
        "agent/population-language-l0-reference-manifest-verifier-v0",
        "f0cf83d1be0426fda976f08a379ab040be53ba89",
        "4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d",
        "1f841356ddd2ff25ef8523ed3437596ca60626ef",
        "GITHUB_ACTIONS_30937048561_SUCCESS",
    ),
    StackEntry(
        8,
        212,
        "Add immutable Post-Training Learning L0 calibration plan",
        "HASH_PINNED_NON_AUTHORIZING_CALIBRATION_PLAN",
        "agent/population-language-post-training-learning-l0-calibration-plan-v0",
        "4bb09762948fd83eb7a7ea2beb5b1f8ecdbd450d",
        "780c19c0c8e63dafe6e7c74bbfe3d579129e53fe",
        "0b23d76c68fadd043e37c1e0d098de10b929f504",
        "GITHUB_ACTIONS_30937548207_SUCCESS",
    ),
    StackEntry(
        9,
        215,
        "Add authorization-gated Post-Training Learning L0 calibration orchestrator",
        "CALIBRATION_WORK_MANIFEST_AND_RESULT_VERIFIER",
        "agent/population-language-post-training-learning-l0-calibration-orchestrator-v0",
        "780c19c0c8e63dafe6e7c74bbfe3d579129e53fe",
        "52f0e5e1a97a2a78d42b17e6861da4464665c754",
        "684acad39f85e26934be301827b038ffb683d519",
        "GITHUB_ACTIONS_30950226040_SUCCESS",
    ),
)

MERGE_ORDER = tuple(entry.pull_request for entry in STACK)


def _is_git_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def stack_manifest() -> dict[str, object]:
    return {
        "version": VERSION,
        "status": STATUS,
        "source_orchestrator_head": SOURCE_ORCHESTRATOR_HEAD,
        "merge_policy": MERGE_POLICY,
        "first_base_sha": FIRST_BASE_SHA,
        "entries": [asdict(entry) for entry in STACK],
        "merge_order": list(MERGE_ORDER),
        "default_protected_state": dict(DEFAULT_PROTECTED_STATE),
        "missing_operational_components": list(MISSING_OPERATIONAL_COMPONENTS),
        "next_handoff_order": list(NEXT_HANDOFF_ORDER),
    }


def stack_manifest_sha256() -> str:
    return hashlib.sha256(_canonical_json_bytes(stack_manifest())).hexdigest()


def validate_entries(entries: Sequence[StackEntry]) -> dict[str, bool]:
    sequence = tuple(entries)
    checks = {
        "entry_count_is_exact": len(sequence) == 9,
        "orders_are_contiguous": tuple(entry.order for entry in sequence)
        == tuple(range(1, 10)),
        "pull_requests_are_exact": tuple(entry.pull_request for entry in sequence)
        == (203, 204, 207, 208, 209, 210, 211, 212, 215),
        "first_base_is_exact": bool(sequence)
        and sequence[0].base_sha == FIRST_BASE_SHA,
        "stack_is_sha_contiguous": all(
            current.base_sha == previous.head_sha
            for previous, current in zip(sequence, sequence[1:])
        ),
        "all_shas_are_valid": all(
            _is_git_sha(value)
            for entry in sequence
            for value in (
                entry.base_sha,
                entry.head_sha,
                entry.synthetic_merge_sha,
            )
        ),
        "head_shas_are_unique": len({entry.head_sha for entry in sequence})
        == len(sequence),
        "branches_are_unique": len({entry.branch for entry in sequence})
        == len(sequence),
        "roles_are_unique": len({entry.role for entry in sequence})
        == len(sequence),
        "all_entries_remain_unmerged": all(entry.state == STACK_STATE for entry in sequence),
        "no_entry_allows_merge": all(entry.merge_allowed is False for entry in sequence),
        "final_head_is_orchestrator_head": bool(sequence)
        and sequence[-1].head_sha == SOURCE_ORCHESTRATOR_HEAD,
    }
    return checks


def validate_stack_audit() -> dict[str, object]:
    entry_checks = validate_entries(STACK)
    protected_checks = {
        "review_before_merge_is_locked": MERGE_POLICY
        == "REVIEW_BEFORE_MERGE_NO_AUTOMATIC_MERGE",
        "automatic_merge_is_disabled": DEFAULT_PROTECTED_STATE[
            "automatic_merge_allowed"
        ]
        is False,
        "active_output_access_is_disabled": DEFAULT_PROTECTED_STATE[
            "active_reference_output_access_allowed"
        ]
        is False,
        "calibration_is_not_authorized": DEFAULT_PROTECTED_STATE[
            "calibration_execution_authorized"
        ]
        is False,
        "final_world_access_is_not_authorized": DEFAULT_PROTECTED_STATE[
            "final_world_access_authorized"
        ]
        is False,
        "final_execution_is_not_authorized": DEFAULT_PROTECTED_STATE[
            "final_execution_authorized"
        ]
        is False,
        "larger_architectures_are_unfrozen": not any(
            DEFAULT_PROTECTED_STATE[key]
            for key in (
                "fifty_m_architecture_frozen",
                "hundred_m_architecture_frozen",
                "three_hundred_m_architecture_frozen",
            )
        ),
        "real_row_executor_is_still_missing": MISSING_OPERATIONAL_COMPONENTS[0]
        == "REAL_AUTHORIZATION_GATED_CALIBRATION_ROW_EXECUTOR",
        "handoff_waits_for_training_completion": NEXT_HANDOFF_ORDER[0]
        == "WAIT_FOR_REFERENCE_TRAINING_TO_COMPLETE",
        "final_authorization_is_last_gate": NEXT_HANDOFF_ORDER[-1]
        == "REQUEST_SEPARATE_EXPLICIT_FINAL_EXECUTION_AUTHORIZATION",
    }
    checks = {**entry_checks, **protected_checks}
    return {
        "status": STATUS,
        "version": VERSION,
        "source_orchestrator_head": SOURCE_ORCHESTRATOR_HEAD,
        "merge_order": list(MERGE_ORDER),
        "manifest_sha256": stack_manifest_sha256(),
        "missing_operational_components": list(MISSING_OPERATIONAL_COMPONENTS),
        "next_handoff_order": list(NEXT_HANDOFF_ORDER),
        "checks": checks,
        "valid": all(checks.values()),
    }
