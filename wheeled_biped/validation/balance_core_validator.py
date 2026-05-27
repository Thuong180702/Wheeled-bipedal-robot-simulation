# wheeled_biped/validation/balance_core_validator.py
"""Main validation orchestrator for balance-core controller with duration ladder."""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import subprocess
import pandas as pd

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
    ClassificationResult,
)
from wheeled_biped.validation.classification_report import (
    ClassificationReportGenerator,
)


@dataclass
class ValidationResult:
    """Result of validating a single duration."""
    passed: bool
    duration_steps: int
    schema_valid: bool
    structural_invariants_valid: bool
    duration_completed: bool
    failure_classification: Optional[ClassificationResult]
    failure_report: Optional[str]
    error_message: str


class BalanceCoreValidator:
    """Main validation orchestrator with progressive duration gating.

    Orchestrates:
    1. Schema checking (TelemetrySchemaChecker)
    2. Structural invariant checking (StructuralInvariantChecker)
    3. Duration completion checking
    4. Failure classification (FailureClassifier)
    5. Report generation (ClassificationReportGenerator)
    6. Progressive duration ladder (100→200→500→1000)
    """

    DURATION_LADDER = [100, 200, 500, 1000]

    def __init__(self):
        """Initialize validator with all checkers."""
        self.schema_checker = TelemetrySchemaChecker()
        self.invariant_checker = StructuralInvariantChecker()
        self.failure_classifier = FailureClassifier()
        self.report_generator = ClassificationReportGenerator()

    def validate_duration(
        self,
        telemetry_path: str,
        expected_steps: int,
    ) -> ValidationResult:
        """Validate telemetry for a single duration.

        Args:
            telemetry_path: Path to telemetry CSV file
            expected_steps: Expected number of simulation steps

        Returns:
            ValidationResult with pass/fail status and diagnostics
        """
        # Load telemetry
        try:
            df = pd.read_csv(telemetry_path)
        except Exception as e:
            return ValidationResult(
                passed=False,
                duration_steps=0,
                schema_valid=False,
                structural_invariants_valid=False,
                duration_completed=False,
                failure_classification=None,
                failure_report=None,
                error_message=f"Failed to load telemetry: {e}",
            )

        actual_steps = len(df)

        # Check schema
        try:
            self.schema_checker.validate(df)
            schema_valid = True
            schema_error = ""
        except MissingFieldError as e:
            return ValidationResult(
                passed=False,
                duration_steps=actual_steps,
                schema_valid=False,
                structural_invariants_valid=False,
                duration_completed=False,
                failure_classification=None,
                failure_report=None,
                error_message=str(e),
            )

        # Check structural invariants
        try:
            self.invariant_checker.check_all(df)
            structural_invariants_valid = True
            structural_error = ""
        except ArchitectureRegressionError as e:
            return ValidationResult(
                passed=False,
                duration_steps=actual_steps,
                schema_valid=True,
                structural_invariants_valid=False,
                duration_completed=False,
                failure_classification=None,
                failure_report=None,
                error_message=str(e),
            )

        # Check duration completion
        duration_completed = actual_steps >= expected_steps

        # Check for failures (threshold crossings) even if duration completed
        classification = None
        report = None
        has_failure = False

        try:
            classification = self.failure_classifier.classify(df)
            report = self.report_generator.to_markdown(classification)
            has_failure = True
        except ValueError as e:
            # No threshold crossings found - this is expected for successful runs
            if "No threshold crossings found" in str(e):
                has_failure = False
            else:
                # Unexpected error
                return ValidationResult(
                    passed=False,
                    duration_steps=actual_steps,
                    schema_valid=True,
                    structural_invariants_valid=True,
                    duration_completed=duration_completed,
                    failure_classification=None,
                    failure_report=None,
                    error_message=f"Failure classification error: {e}",
                )
        except Exception as e:
            # Unexpected error during classification
            return ValidationResult(
                passed=False,
                duration_steps=actual_steps,
                schema_valid=True,
                structural_invariants_valid=True,
                duration_completed=duration_completed,
                failure_classification=None,
                failure_report=None,
                error_message=f"Unexpected error during classification: {e}",
            )

        # Determine overall pass/fail
        if has_failure:
            # Failure detected (threshold crossing)
            error_msg = f"Failure detected: {classification.primary_failure_mode.value}"
            if not duration_completed:
                error_msg += f" (duration incomplete: {actual_steps}/{expected_steps} steps)"

            return ValidationResult(
                passed=False,
                duration_steps=actual_steps,
                schema_valid=True,
                structural_invariants_valid=True,
                duration_completed=duration_completed,
                failure_classification=classification,
                failure_report=report,
                error_message=error_msg,
            )
        elif not duration_completed:
            # Duration incomplete but no threshold crossings detected
            return ValidationResult(
                passed=False,
                duration_steps=actual_steps,
                schema_valid=True,
                structural_invariants_valid=True,
                duration_completed=False,
                failure_classification=None,
                failure_report=None,
                error_message=f"Duration incomplete: {actual_steps}/{expected_steps} steps (no threshold crossings detected)",
            )
        else:
            # All checks passed
            return ValidationResult(
                passed=True,
                duration_steps=actual_steps,
                schema_valid=True,
                structural_invariants_valid=True,
                duration_completed=True,
                failure_classification=None,
                failure_report=None,
                error_message="",
            )

    def run_simulation(self, steps: int, output_dir: str) -> Path:
        """Run simulation for specified number of steps.

        Args:
            steps: Number of simulation steps
            output_dir: Output directory for telemetry

        Returns:
            Path to generated telemetry CSV file

        Raises:
            RuntimeError: If simulation fails
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        telemetry_path = output_path / f"telemetry_{steps}.csv"

        # Run simulate_hierarchical_controller.py
        cmd = [
            "python",
            "scripts/simulate_hierarchical_controller.py",
            "--steps", str(steps),
            "--output", str(telemetry_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Simulation failed for {steps} steps:\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )

        if not telemetry_path.exists():
            raise RuntimeError(
                f"Simulation completed but telemetry file not found: {telemetry_path}"
            )

        return telemetry_path

    def validate_ladder(
        self,
        output_dir: str,
        start_duration: Optional[int] = None,
    ) -> List[ValidationResult]:
        """Run progressive duration ladder validation.

        Runs simulations at increasing durations (100→200→500→1000) and validates
        each. Stops at first failure.

        Args:
            output_dir: Output directory for telemetry and reports
            start_duration: Optional starting duration (default: first in ladder)

        Returns:
            List of ValidationResult for each duration tested
        """
        results = []

        # Determine starting index
        if start_duration is None:
            start_idx = 0
        else:
            try:
                start_idx = self.DURATION_LADDER.index(start_duration)
            except ValueError:
                raise ValueError(
                    f"Invalid start_duration {start_duration}. "
                    f"Must be one of {self.DURATION_LADDER}"
                )

        # Run ladder
        for duration in self.DURATION_LADDER[start_idx:]:
            print(f"\n{'='*60}")
            print(f"Validating duration: {duration} steps")
            print(f"{'='*60}")

            # Run simulation
            try:
                telemetry_path = self.run_simulation(duration, output_dir)
            except RuntimeError as e:
                # Simulation failed - create failure result
                result = ValidationResult(
                    passed=False,
                    duration_steps=0,
                    schema_valid=False,
                    structural_invariants_valid=False,
                    duration_completed=False,
                    failure_classification=None,
                    failure_report=None,
                    error_message=str(e),
                )
                results.append(result)
                print(f"❌ Simulation failed: {e}")
                break

            # Validate telemetry
            result = self.validate_duration(str(telemetry_path), duration)
            results.append(result)

            if result.passed:
                print(f"[PASS] Duration {duration} PASSED")
            else:
                print(f"[FAIL] Duration {duration} FAILED")
                print(f"Error: {result.error_message}")

                # Write failure report if available
                if result.failure_report:
                    report_path = Path(output_dir) / f"failure_report_{duration}.md"
                    report_path.write_text(result.failure_report)
                    print(f"Failure report written to: {report_path}")

                # Stop at first failure
                print("\nStopping ladder at first failure")
                break

        return results
