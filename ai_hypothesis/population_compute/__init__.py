"""Fixed-parameter population-compute research contracts."""

from .collective_relay import (
    COLLECTIVE_RELAY_VERSION,
    RELAY_DIFFICULTIES,
    RELAY_WORLD_SIZE,
    RelayDifficulty,
    RelayRecord,
    RelayWorld,
    generate_relay_dataset,
    generate_relay_world,
    information_complete_at,
    relay_population_points,
    relay_scope_thresholds,
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
from .model import (
    PopulationForwardOutput,
    PopulationTelemetry,
    SharedPopulationCell,
    SharedPopulationConfig,
)
from .relay_serial_control import (
    RelayScheduleOutput,
    RelayScheduleTelemetry,
    normalized_parallel_forward,
    normalized_serial_forward,
)

__all__ = [
    "COLLECTIVE_RELAY_VERSION",
    "DEVELOPMENT_POPULATION_SIZES",
    "RELAY_DIFFICULTIES",
    "RELAY_WORLD_SIZE",
    "CommunicationMode",
    "CurveAssessment",
    "GateCriteria",
    "PopulationCondition",
    "PopulationForwardOutput",
    "PopulationRunMetrics",
    "PopulationTelemetry",
    "RelayDifficulty",
    "RelayRecord",
    "RelayScheduleOutput",
    "RelayScheduleTelemetry",
    "RelayWorld",
    "SharedPopulationCell",
    "SharedPopulationConfig",
    "assess_scaling_curve",
    "generate_relay_dataset",
    "generate_relay_world",
    "information_complete_at",
    "normalized_parallel_forward",
    "normalized_serial_forward",
    "relay_population_points",
    "relay_scope_thresholds",
    "resolve_relay",
    "validate_fixed_parameter_identity",
]
