# wheeled_biped/validation/__init__.py
"""Balance-core validation infrastructure."""

from wheeled_biped.validation.telemetry_schema_checker import (
    TelemetrySchemaChecker,
    MissingFieldError,
)
from wheeled_biped.validation.structural_invariant_checker import (
    StructuralInvariantChecker,
    ArchitectureRegressionError,
)
from wheeled_biped.validation.failure_classifier import (
    FailureClassifier,
    FailureMode,
    ClassificationResult,
    ThresholdCrossing,
)

__all__ = [
    "TelemetrySchemaChecker",
    "MissingFieldError",
    "StructuralInvariantChecker",
    "ArchitectureRegressionError",
    "FailureClassifier",
    "FailureMode",
    "ClassificationResult",
    "ThresholdCrossing",
]
