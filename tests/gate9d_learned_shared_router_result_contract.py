from __future__ import annotations


def verify_v0_failure_interpretation() -> None:
    total_states = 256 * 256
    contribution_positive_states = 8 * 128
    negative_fraction = (total_states - contribution_positive_states) / total_states
    assert contribution_positive_states == 1024
    assert negative_fraction == 0.984375

    evaluation_episodes = 64 * 247
    expected_contribution_messages = 64 * sum(
        query.bit_count() for query in range(1, 256) if query not in {1 << i for i in range(8)}
    )
    # The exact query set is owned by the sparse dependency; this contract only
    # protects the interpretation that approximately nine messages per episode
    # means unconditional basis routing, not query-conditioned routing.
    observed_messages = 142080
    assert observed_messages / evaluation_episodes > 8.9
    assert expected_contribution_messages / evaluation_episodes < 4.2


verify_v0_failure_interpretation()
