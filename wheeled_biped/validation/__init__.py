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
from wheeled_biped.validation.classification_report import (
    ClassificationReportGenerator,
)
from wheeled_biped.validation.fix_cycle_reporter import (
    FixCycleReporter,
    FixCycleRecord,
)
from wheeled_biped.validation.balance_core_validator import (
    BalanceCoreValidator,
    ValidationResult,
)
from wheeled_biped.validation.study_aggregator import (
    StudyAggregator,
    StudyCaseResult,
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
    "ClassificationReportGenerator",
    "FixCycleReporter",
    "FixCycleRecord",
    "BalanceCoreValidator",
    "ValidationResult",
    "StudyAggregator",
    "StudyCaseResult",
]
