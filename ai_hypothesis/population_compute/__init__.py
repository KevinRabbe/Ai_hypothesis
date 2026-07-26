"""Fixed-parameter population-compute research contracts."""

from .collective_relay import (
    COLLECTIVE_RELAY_VERSION,
    RELAY_DIFFICULTIES,
    RelayDifficulty,
    RelayRecord,
    RelayWorld,
    generate_relay_dataset,
    generate_relay_world,
    resolve_relay,
)
from .contract import (
    DEVELOPMENT_POPULATION_SIZES,
    CommunicationMode,
    CurveAssessment,
    GateCriteria,
    PopulationCondition,
    PopulationRunMetrics,
    assess_scaling_curve,
    validate_fixed_parameter_identity,
)

__all__ = [
    "COLLECTIVE_RELAY_VERSION",
    "DEVELOPMENT_POPULATION_SIZES",
    "RELAY_DIFFICULTIES",
    "CommunicationMode",
    "CurveAssessment",
    "GateCriteria",
    "PopulationCondition",
    "PopulationRunMetrics",
    "RelayDifficulty",
    "RelayRecord",
    "RelayWorld",
    "assess_scaling_curve",
    "generate_relay_dataset",
    "generate_relay_world",
    "resolve_relay",
    "validate_fixed_parameter_identity",
]
