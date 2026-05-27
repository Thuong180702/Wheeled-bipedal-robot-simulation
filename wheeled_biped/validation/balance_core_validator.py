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
    """Result of a single duration validation."""
    passed: bool
    duration_steps: int
    actual_steps: int
    structural_invariants_passed: bool
    failure_mode: Optional['FailureMode']
    classification_result: Optional[ClassificationResult]
    telemetry_path: Path
    report_path: Optional[Path]


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
                duration_steps=expected_steps,
                actual_steps=0,
                structural_invariants_passed=False,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
            )

        actual_steps = len(df)

        # Check schema
        try:
            self.schema_checker.validate(df)
        except MissingFieldError as e:
            return ValidationResult(
                passed=False,
                duration_steps=expected_steps,
                actual_steps=actual_steps,
                structural_invariants_passed=False,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
            )

        # Check structural invariants
        try:
            self.invariant_checker.check_all(df)
        except ArchitectureRegressionError as e:
            return ValidationResult(
                passed=False,
                duration_steps=expected_steps,
                actual_steps=actual_steps,
                structural_invariants_passed=False,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
            )

        # Check duration completion
        duration_completed = actual_steps >= expected_steps

        # Check for failures (threshold crossings) even if duration completed
        classification = None
        report_path = None
        has_failure = False

        try:
            classification = self.failure_classifier.classify(df)
            has_failure = True

            # Generate report and save it
            report = self.report_generator.to_markdown(classification)
            report_file = Path(telemetry_path).parent / f"failure_report_{expected_steps}.md"
            report_file.write_text(report)
            report_path = report_file
        except ValueError as e:
            # No threshold crossings found - this is expected for successful runs
            if "No threshold crossings found" in str(e):
                has_failure = False
            else:
                # Unexpected error
                return ValidationResult(
                    passed=False,
                    duration_steps=expected_steps,
                    actual_steps=actual_steps,
                    structural_invariants_passed=True,
                    failure_mode=None,
                    classification_result=None,
                    telemetry_path=Path(telemetry_path),
                    report_path=None,
                )
        except Exception as e:
            # Unexpected error during classification
            return ValidationResult(
                passed=False,
                duration_steps=expected_steps,
                actual_steps=actual_steps,
                structural_invariants_passed=True,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
            )

        # Determine overall pass/fail
        if has_failure:
            # Failure detected (threshold crossing)
            return ValidationResult(
                passed=False,
                duration_steps=expected_steps,
                actual_steps=actual_steps,
                structural_invariants_passed=True,
                failure_mode=classification.primary_failure_mode,
                classification_result=classification,
                telemetry_path=Path(telemetry_path),
                report_path=report_path,
            )
        elif not duration_completed:
            # Duration incomplete but no threshold crossings detected
            return ValidationResult(
                passed=False,
                duration_steps=expected_steps,
                actual_steps=actual_steps,
                structural_invariants_passed=True,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
            )
        else:
            # All checks passed
            return ValidationResult(
                passed=True,
                duration_steps=expected_steps,
                actual_steps=actual_steps,
                structural_invariants_passed=True,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
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
        import shutil
        import glob

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run simulate_hierarchical_controller.py
        # Note: script outputs to outputs/hierarchical_controller_sim/telemetry_{timestamp}.csv
        cmd = [
            "python",
            "scripts/simulate_hierarchical_controller.py",
            "--controller-mode", "balance-core",
            "--steps", str(steps),
        ]

        # Get list of existing telemetry files before simulation
        sim_output_dir = Path("outputs/hierarchical_controller_sim")
        existing_files = set(sim_output_dir.glob("telemetry_*.csv")) if sim_output_dir.exists() else set()

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

        # Find the newly created telemetry file
        new_files = set(sim_output_dir.glob("telemetry_*.csv")) - existing_files
        if not new_files:
            raise RuntimeError(
                f"Simulation completed but no new telemetry file found in {sim_output_dir}"
            )

        source_telemetry = max(new_files, key=lambda p: p.stat().st_mtime)

        # Copy to desired output directory with predictable name
        dest_telemetry = output_path / f"telemetry_{steps}.csv"
        shutil.copy2(source_telemetry, dest_telemetry)

        return dest_telemetry

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
                    duration_steps=duration,
                    actual_steps=0,
                    structural_invariants_passed=False,
                    failure_mode=None,
                    classification_result=None,
                    telemetry_path=Path(output_dir) / f"telemetry_{duration}.csv",
                    report_path=None,
                )
                results.append(result)
                print(f"❌ Simulation failed: {e}")
                break

            # Validate telemetry
            result = self.validate_duration(str(telemetry_path), duration)
            results.append(result)

            if result.passed:
                print(f"✓ [PASS] Duration {duration} PASSED")
            else:
                print(f"✗ [FAIL] Duration {duration} FAILED")
                if result.failure_mode:
                    print(f"  Failure mode: {result.failure_mode.value}")
                    print(f"  Actual steps: {result.actual_steps}/{result.duration_steps}")

                # Report already written by validate_duration()
                if result.report_path:
                    print(f"  Failure report: {result.report_path}")

                # Stop at first failure
                print("\nStopping ladder at first failure")
                break

        return results
