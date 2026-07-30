from __future__ import annotations

import pytest

from ai_hypothesis.population_compute.gate7_high_scale_routing_bandwidth_continuation_result_protocol import (
    GATE7_CAMPAIGN_COMPLETE,
    GATE7_CAMPAIGN_RESOURCE_FRONTIER,
    GATE7_CONTINUATION_EXECUTION_HEAD,
    GATE7_CONTINUATION_K_LADDER,
    GATE7_CONTINUATION_POPULATIONS,
    GATE7_NEXT_COORDINATION_EFFICIENCY,
    GATE7_NEXT_GLOBAL_SIGNAL_OR_REPRESENTATION,
    GATE7_NEXT_RESOURCE_ENGINEERING,
    GATE7_NEXT_ROUTING_MECHANISM_OR_BUDGET,
    GATE7_TIER_K_REQUIRED,
    GATE7_TIER_NO_K_LE_512,
    GATE7_TIER_REFERENCE_NOT_VIABLE,
    Gate7ContinuationCampaignResultSummary,
    Gate7ContinuationTierResultSummary,
    allowed_cross_population_statements,
    choose_next_question_family,
    required_reporting_surface,
)


def tier(
    population: int,
    *,
    reference_viable: bool = True,
    passing_k: tuple[int, ...] = (256, 512),
) -> Gate7ContinuationTierResultSummary:
    if not reference_viable:
        outcome = GATE7_TIER_REFERENCE_NOT_VIABLE
        passing_k = ()
    elif passing_k:
        outcome = GATE7_TIER_K_REQUIRED
    else:
        outcome = GATE7_TIER_NO_K_LE_512
    smallest = passing_k[0] if passing_k else None
    return Gate7ContinuationTierResultSummary(
        population=population,
        reference_viable=reference_viable,
        passing_k=passing_k,
        smallest_passing_k=smallest,
        smallest_passing_k_over_n=(smallest / population if smallest is not None else None),
        tier_outcome=outcome,
    )


def complete_campaign(
    tiers: tuple[Gate7ContinuationTierResultSummary, ...],
) -> Gate7ContinuationCampaignResultSummary:
    return Gate7ContinuationCampaignResultSummary(
        campaign_outcome=GATE7_CAMPAIGN_COMPLETE,
        completed_populations=GATE7_CONTINUATION_POPULATIONS,
        resource_frontier_population=None,
        tiers=tiers,
    )


def test_reporting_surface_is_exactly_data_blind_and_nonadaptive() -> None:
    surface = required_reporting_surface()
    assert surface["execution_head"] == GATE7_CONTINUATION_EXECUTION_HEAD
    assert surface["populations"] == GATE7_CONTINUATION_POPULATIONS
    assert surface["k_ladder"] == GATE7_CONTINUATION_K_LADDER
    assert surface["world_count"] == 512
    assert surface["bootstrap_samples"] == 10_000
    assert surface["must_report_every_exposed_k"] is True
    assert surface["must_report_every_checkpoint"] is True
    assert surface["must_report_complete_passing_k_set"] is True
    assert surface["allow_asymptotic_scaling_law_fit"] is False
    assert surface["allow_interpolation_for_unobserved_population"] is False
    assert surface["allow_pooling_64_world_screen_with_512_world_evidence"] is False
    assert surface["allow_post_result_rescue_k"] is False
    assert surface["allow_second_continuation"] is False


def test_complete_all_passing_ladder_selects_coordination_efficiency() -> None:
    campaign = complete_campaign(
        tuple(tier(population) for population in GATE7_CONTINUATION_POPULATIONS)
    )
    assert choose_next_question_family(campaign) == GATE7_NEXT_COORDINATION_EFFICIENCY


def test_any_viable_no_k_tier_selects_routing_mechanism_or_budget() -> None:
    tiers = tuple(
        tier(population, passing_k=() if population == 65_536 else (256, 512))
        for population in GATE7_CONTINUATION_POPULATIONS
    )
    assert choose_next_question_family(complete_campaign(tiers)) == (
        GATE7_NEXT_ROUTING_MECHANISM_OR_BUDGET
    )


def test_reference_failure_outranks_no_k_tier() -> None:
    tiers = (
        tier(16_384, passing_k=()),
        tier(32_768, reference_viable=False),
        tier(65_536),
        tier(131_072),
    )
    assert choose_next_question_family(complete_campaign(tiers)) == (
        GATE7_NEXT_GLOBAL_SIGNAL_OR_REPRESENTATION
    )


def test_resource_frontier_outranks_completed_scientific_tiers() -> None:
    campaign = Gate7ContinuationCampaignResultSummary(
        campaign_outcome=GATE7_CAMPAIGN_RESOURCE_FRONTIER,
        completed_populations=(16_384, 32_768),
        resource_frontier_population=65_536,
        tiers=(tier(16_384), tier(32_768, passing_k=())),
    )
    assert choose_next_question_family(campaign) == GATE7_NEXT_RESOURCE_ENGINEERING
    assert any("unexposed" in statement for statement in allowed_cross_population_statements(campaign))


def test_passing_set_must_remain_ordered_unique_and_complete() -> None:
    with pytest.raises(ValueError, match="ordered, unique"):
        tier_result = Gate7ContinuationTierResultSummary(
            population=16_384,
            reference_viable=True,
            passing_k=(512, 256),
            smallest_passing_k=512,
            smallest_passing_k_over_n=512 / 16_384,
            tier_outcome=GATE7_TIER_K_REQUIRED,
        )
        tier_result.validate()


def test_smallest_k_and_ratio_must_match_complete_pass_set() -> None:
    with pytest.raises(ValueError, match="smallest passing K"):
        Gate7ContinuationTierResultSummary(
            population=16_384,
            reference_viable=True,
            passing_k=(256, 512),
            smallest_passing_k=512,
            smallest_passing_k_over_n=512 / 16_384,
            tier_outcome=GATE7_TIER_K_REQUIRED,
        ).validate()
    with pytest.raises(ValueError, match="K/N ratio"):
        Gate7ContinuationTierResultSummary(
            population=16_384,
            reference_viable=True,
            passing_k=(256, 512),
            smallest_passing_k=256,
            smallest_passing_k_over_n=0.5,
            tier_outcome=GATE7_TIER_K_REQUIRED,
        ).validate()


def test_complete_campaign_requires_all_four_ordered_tiers() -> None:
    with pytest.raises(ValueError, match="complete frozen ladder"):
        Gate7ContinuationCampaignResultSummary(
            campaign_outcome=GATE7_CAMPAIGN_COMPLETE,
            completed_populations=(16_384,),
            resource_frontier_population=None,
            tiers=(tier(16_384),),
        ).validate()


def test_resource_frontier_must_be_next_uncompleted_population() -> None:
    with pytest.raises(ValueError, match="next uncompleted population"):
        Gate7ContinuationCampaignResultSummary(
            campaign_outcome=GATE7_CAMPAIGN_RESOURCE_FRONTIER,
            completed_populations=(16_384,),
            resource_frontier_population=65_536,
            tiers=(tier(16_384),),
        ).validate()


def test_interpretation_contract_forbids_scaling_claims() -> None:
    campaign = complete_campaign(
        tuple(tier(population) for population in GATE7_CONTINUATION_POPULATIONS)
    )
    statements = allowed_cross_population_statements(campaign)
    assert any("not as an asymptotic exponent" in statement for statement in statements)
    assert any("64-world screen separate" in statement for statement in statements)
    assert any("not proof that bounded routing fails" in statement for statement in statements)
