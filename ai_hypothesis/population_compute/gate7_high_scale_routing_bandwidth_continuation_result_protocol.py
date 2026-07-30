"""Data-blind result interpretation contract for the Gate-7 post-confirmation continuation.

This module is frozen before continuation evidence is inspected. It defines the complete reporting
surface and the family-level next-question decision hierarchy. It contains no artifact loader, world
generator, checkpoint loader, Torch import, execution path, result values or experiment-opening path.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE7_CONTINUATION_RESULT_PROTOCOL_FROZEN = True
GATE7_CONTINUATION_RESULT_RECORD_OPENED = False
GATE7_SECOND_CONTINUATION_OPENED = False
GATE7_CONTINUATION_RESULT_PROTOCOL_VERSION = (
    "gate7-high-scale-routing-bandwidth-continuation-result-protocol-v0"
)
GATE7_CONTINUATION_EXECUTION_HEAD = "19ee6b4e228c56b32a11b11b1c61b35bf640e2c8"
GATE7_CONTINUATION_PROTOCOL_HEAD = "4f05f8b1f9a33aed712edbf28691b927d2e220d3"
GATE7_CONFIRMATION_RESULT_HEAD = "ae8bd8544a03e48f4f397d2ca5ae933d9247e430"
GATE7_CONFIRMATION_RESULT_SHA256 = (
    "725e3749ba5fed7cdcbb6d61df81bcc77a7b69bacfdc82d553efb06f5ff888da"
)
GATE7_CONFIRMATION_AUDIT_SHA256 = (
    "27a46ba0feccf6b3322885334819e0e7a07bb02be930122eb1f063c65d69fb99"
)
GATE7_CONFIRMIRMATION_MANIFEST_SHA256 = (
    "e7c1823dc59a50b58250cab0f7b18b95ca42b831e90182f07295680b6986b263"
)

GATE7_CONTINUATION_POPULATIONS = (16_384, 32_768, 65_536, 131_072)
GATE7_CONTINUATION_K_LADDER = (16, 32, 64, 128, 256, 512)
GATE7_CONTINUATION_CHECKPOINTS = (0, 1, 2)
GATE7_CONTINUATION_WORLD_COUNT = 512
GATE7_CONTINUATION_BOOTSTRAP_SAMPLES = 10_000
GATE7_CONTINUATION_NONINFERIORITY_MARGIN = 0.05

GATE7_TIER_K_REQUIRED = "G7_CONTINUATION_K_REQUIRED"
GATE7_TIER_NO_K_LE_512 = "G7_CONTINUATION_NO_K_LE_512"
GATE7_TIER_REFERENCE_NOT_VIABLE = "G7_CONTINUATION_REFERENCE_NOT_VIABLE"
GATE7_VALID_TIER_OUTCOMES = {
    GATE7_TIER_K_REQUIRED,
    GATE7_TIER_NO_K_LE_512,
    GATE7_TIER_REFERENCE_NOT_VIABLE,
}

GATE7_CAMPAIGN_COMPLETE = "G7_POST_CONFIRMATION_LADDER_COMPLETE"
GATE7_CAMPAIGN_RESOURCE_FRONTIER = "G7_POST_CONFIRMATION_RESOURCE_FRONTIER_REACHED"
GATE7_VALID_CAMPAIGN_OUTCOMES = {
    GATE7_CAMPAIGN_COMPLETE,
    GATE7_CAMPAIGN_RESOURCE_FRONTIER,
}

GATE7_NEXT_RESOURCE_ENGINEERING = "G7_NEXT_QUESTION_RESOURCE_ENGINEERING"
GATE7_NEXT_GLOBAL_SIGNAL_OR_REPRESENTATION = (
    "G7_NEXT_QUESTION_GLOBAL_SIGNAL_OR_REPRESENTATION"
)
GATE7_NEXT_ROUTING_MECHANISM_OR_BUDGET = (
    "G7_NEXT_QUESTION_ROUTING_MECHANISM_OR_BUDGET"
)
GATE7_NEXT_COORDINATION_EFFICIENCY = "G7_NEXT_QUESTION_COORDINATION_EFFICIENCY"

GATE7_REQUIRED_TIER_REPORT_FIELDS = (
    "population",
    "reference_viable",
    "global_reference_by_checkpoint",
    "global_reference_stratified",
    "all_k_by_checkpoint",
    "passing_k",
    "smallest_passing_k",
    "smallest_passing_k_over_n",
    "tier_outcome",
)
GATE7_REQUIRED_PROVENANCE_FIELDS = (
    "execution_head",
    "artifact_valid",
    "audit_errors",
    "result_sha256",
    "audit_sha256",
    "manifest_sha256",
    "training_performed",
    "checkpoint_selection_performed",
    "second_continuation_opened",
)


@dataclass(frozen=True, slots=True)
class Gate7ContinuationTierResultSummary:
    population: int
    reference_viable: bool
    passing_k: tuple[int, ...]
    smallest_passing_k: int | None
    smallest_passing_k_over_n: float | None
    tier_outcome: str

    def validate(self) -> None:
        if self.population not in GATE7_CONTINUATION_POPULATIONS:
            raise ValueError("result tier population is outside the frozen continuation ladder")
        if self.tier_outcome not in GATE7_VALID_TIER_OUTCOMES:
            raise ValueError("unknown continuation result tier outcome")
        expected_passing = tuple(
            k for k in GATE7_CONTINUATION_K_LADDER if k in self.passing_k
        )
        if self.passing_k != expected_passing:
            raise ValueError("passing K values must remain ordered, unique and inside the frozen ladder")
        expected_smallest = self.passing_k[0] if self.passing_k else None
        if self.smallest_passing_k != expected_smallest:
            raise ValueError("smallest passing K is inconsistent with the complete pass set")
        expected_ratio = (
            expected_smallest / self.population if expected_smallest is not None else None
        )
        if self.smallest_passing_k_over_n != expected_ratio:
            raise ValueError("reported K/N ratio is inconsistent")
        if not self.reference_viable:
            if self.tier_outcome != GATE7_TIER_REFERENCE_NOT_VIABLE:
                raise ValueError("a non-viable global reference must retain its own tier outcome")
        elif self.passing_k:
            if self.tier_outcome != GATE7_TIER_K_REQUIRED:
                raise ValueError("a non-empty pass set requires the K-required outcome")
        elif self.tier_outcome != GATE7_TIER_NO_K_LE_512:
            raise ValueError("an empty pass set with viable reference requires no-K<=512")


@dataclass(frozen=True, slots=True)
class Gate7ContinuationCampaignResultSummary:
    campaign_outcome: str
    completed_populations: tuple[int, ...]
    resource_frontier_population: int | None
    tiers: tuple[Gate7ContinuationTierResultSummary, ...]

    def validate(self) -> None:
        if self.campaign_outcome not in GATE7_VALID_CAMPAIGN_OUTCOMES:
            raise ValueError("unknown continuation campaign outcome")
        expected_prefix = GATE7_CONTINUATION_POPULATIONS[: len(self.completed_populations)]
        if self.completed_populations != expected_prefix:
            raise ValueError("completed populations must remain an exact contiguous prefix")
        if tuple(tier.population for tier in self.tiers) != self.completed_populations:
            raise ValueError("tier summaries must exactly match completed populations")
        for tier in self.tiers:
            tier.validate()
        if self.campaign_outcome == GATE7_CAMPAIGN_COMPLETE:
            if self.completed_populations != GATE7_CONTINUATION_POPULATIONS:
                raise ValueError("a completed campaign must report the complete frozen ladder")
            if self.resource_frontier_population is not None:
                raise ValueError("a completed campaign cannot also report a resource frontier")
        else:
            if len(self.completed_populations) >= len(GATE7_CONTINUATION_POPULATIONS):
                raise ValueError("a resource frontier cannot occur after the full ladder completed")
            expected_frontier = GATE7_CONTINUATION_POPULATIONS[len(self.completed_populations)]
            if self.resource_frontier_population != expected_frontier:
                raise ValueError("resource frontier must be the next uncompleted population")


def required_reporting_surface() -> dict[str, object]:
    """Return the frozen result-record surface without accepting or loading evidence."""

    return {
        "protocol_version": GATE7_CONTINUATION_RESULT_PROTOCOL_VERSION,
        "execution_head": GATE7_CONTINUATION_EXECUTION_HEAD,
        "populations": GATE7_CONTINUATION_POPULATIONS,
        "k_ladder": GATE7_CONTINUATION_K_LADDER,
        "checkpoints": GATE7_CONTINUATION_CHECKPOINTS,
        "world_count": GATE7_CONTINUATION_WORLD_COUNT,
        "bootstrap_samples": GATE7_CONTINUATION_BOOTSTRAP_SAMPLES,
        "noninferiority_margin": GATE7_CONTINUATION_NONINFERIORITY_MARGIN,
        "required_provenance_fields": GATE7_REQUIRED_PROVENANCE_FIELDS,
        "required_tier_report_fields": GATE7_REQUIRED_TIER_REPORT_FIELDS,
        "must_report_every_exposed_k": True,
        "must_report_every_checkpoint": True,
        "must_report_complete_passing_k_set": True,
        "allow_asymptotic_scaling_law_fit": False,
        "allow_interpolation_for_unobserved_population": False,
        "allow_pooling_64_world_screen_with_512_world_evidence": False,
        "allow_post_result_rescue_k": False,
        "allow_second_continuation": False,
    }


def choose_next_question_family(
    campaign: Gate7ContinuationCampaignResultSummary,
) -> str:
    """Choose only the next research family, using the frozen pre-result hierarchy.

    This does not choose an architecture, K value, population, sample count or execution protocol.
    """

    campaign.validate()
    if campaign.campaign_outcome == GATE7_CAMPAIGN_RESOURCE_FRONTIER:
        return GATE7_NEXT_RESOURCE_ENGINEERING
    if any(not tier.reference_viable for tier in campaign.tiers):
        return GATE7_NEXT_GLOBAL_SIGNAL_OR_REPRESENTATION
    if any(not tier.passing_k for tier in campaign.tiers):
        return GATE7_NEXT_ROUTING_MECHANISM_OR_BUDGET
    return GATE7_NEXT_COORDINATION_EFFICIENCY


def allowed_cross_population_statements(
    campaign: Gate7ContinuationCampaignResultSummary,
) -> tuple[str, ...]:
    """Return the exact interpretation boundaries allowed after a valid audited result."""

    campaign.validate()
    statements = [
        "Report the observed K_required(N) staircase only for completed viable-reference tiers.",
        "Preserve every passing K set; do not infer monotonicity from the smallest passing value.",
        "Report K/N as a descriptive ratio, not as an asymptotic exponent.",
        "Keep the earlier 64-world screen separate from 512-world confirmation/continuation evidence.",
        "Treat no-K<=512 as a tested-budget result, not proof that bounded routing fails.",
        "Treat reference-not-viable as a global-signal or representation result, not a routing result.",
        "Treat a resource frontier as engineering evidence, not scientific non-viability.",
    ]
    if campaign.campaign_outcome == GATE7_CAMPAIGN_RESOURCE_FRONTIER:
        statements.append(
            "Do not infer outcomes for the resource-frontier population or any larger unexposed tier."
        )
    else:
        statements.append("The fixed N16384..N131072 ladder completed without a resource truncation.")
    return tuple(statements)
