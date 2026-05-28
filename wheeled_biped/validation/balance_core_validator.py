# wheeled_biped/validation/balance_core_validator.py
"""Main validation orchestrator for balance-core controller with duration ladder."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional, List, Sequence
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
    termination_reason: Optional[str] = None
    failure_window_path: Optional[Path] = None
    summary_sidecar_path: Optional[Path] = None
    classification_source: str = "main_telemetry"
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    requested_steps: Optional[int] = None
    survived_steps: Optional[int] = None
    terminated: Optional[bool] = None
    final_sim_time_s: Optional[float] = None
    primary_failure_mode: Optional[str] = None
    secondary_failure_modes: list[str] = field(default_factory=list)


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
        failure_window_path: Optional[Path] = None,
        summary_sidecar_path: Optional[Path] = None,
    ) -> ValidationResult:
        """Validate telemetry for a single duration.

        Args:
            telemetry_path: Path to telemetry CSV file
            expected_steps: Expected number of simulation steps
            failure_window_path: Optional path to full-rate failure-window telemetry
            summary_sidecar_path: Optional path to whole-run summary sidecar JSON

        Returns:
            ValidationResult with pass/fail status and diagnostics
        """
        telemetry_path_obj = Path(telemetry_path)
        summary_metrics: dict[str, Any] = {}
        termination_reason = None
        classification_source = "main_telemetry"
        requested_steps: Optional[int] = expected_steps
        survived_steps: Optional[int] = None
        terminated: Optional[bool] = None
        final_sim_time_s: Optional[float] = None

        def build_result(
            *,
            passed: bool,
            actual_steps_value: int,
            structural_invariants_passed: bool,
            failure_mode=None,
            classification_result=None,
            report_path_value: Optional[Path] = None,
            classification_source_value: Optional[str] = None,
        ) -> ValidationResult:
            primary_failure_mode = None
            secondary_failure_modes: list[str] = []
            if classification_result is not None:
                primary_failure_mode = classification_result.primary_failure_mode.value
                secondary_failure_modes = [
                    crossing.failure_mode.value
                    for crossing in classification_result.secondary_threshold_crossings
                ]
            return ValidationResult(
                passed=passed,
                duration_steps=expected_steps,
                actual_steps=actual_steps_value,
                structural_invariants_passed=structural_invariants_passed,
                failure_mode=failure_mode,
                classification_result=classification_result,
                telemetry_path=telemetry_path_obj,
                report_path=report_path_value,
                termination_reason=termination_reason,
                failure_window_path=failure_window_path,
                summary_sidecar_path=summary_sidecar_path,
                classification_source=(classification_source_value or classification_source),
                summary_metrics=summary_metrics,
                requested_steps=requested_steps,
                survived_steps=survived_steps,
                terminated=terminated,
                final_sim_time_s=final_sim_time_s,
                primary_failure_mode=primary_failure_mode,
                secondary_failure_modes=secondary_failure_modes,
            )

        # Load telemetry
        try:
            df = pd.read_csv(telemetry_path_obj)
        except Exception:
            return build_result(
                passed=False,
                actual_steps_value=0,
                structural_invariants_passed=False,
            )

        actual_steps = len(df)
        if "termination_reason" in df.columns and not df["termination_reason"].empty:
            last_reason = df["termination_reason"].iloc[-1]
            if pd.notna(last_reason):
                last_reason_str = str(last_reason).strip()
                if last_reason_str:
                    termination_reason = last_reason_str

        if summary_sidecar_path is not None and summary_sidecar_path.exists():
            try:
                summary_metrics = json.loads(summary_sidecar_path.read_text(encoding="utf-8"))
                requested_steps = int(summary_metrics.get("requested_steps", expected_steps))
                actual_steps = int(
                    summary_metrics.get(
                        "actual_steps",
                        summary_metrics.get("survived_steps", actual_steps),
                    )
                )
                survived_steps = int(summary_metrics.get("survived_steps", actual_steps))
                terminated = bool(summary_metrics.get("terminated", actual_steps < expected_steps))
                final_sim_time_s = float(summary_metrics.get("final_sim_time_s", 0.0))
                termination_reason = summary_metrics.get("termination_reason", termination_reason)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                summary_metrics = {}
                requested_steps = expected_steps
                survived_steps = actual_steps
                terminated = actual_steps < expected_steps
                final_sim_time_s = None
        else:
            survived_steps = actual_steps
            terminated = actual_steps < expected_steps
        # Check schema
        try:
            self.schema_checker.validate(df)
        except MissingFieldError:
            return build_result(
                passed=False,
                actual_steps_value=actual_steps,
                structural_invariants_passed=False,
            )

        # Check structural invariants
        try:
            self.invariant_checker.check_all(df)
        except ArchitectureRegressionError:
            return build_result(
                passed=False,
                actual_steps_value=actual_steps,
                structural_invariants_passed=False,
            )

        # Check duration completion
        duration_completed = actual_steps >= expected_steps

        # Check for failures (threshold crossings) even if duration completed
        classification = None
        report_path = None
        has_failure = False
        classification_df = df

        if failure_window_path is not None and failure_window_path.exists():
            try:
                classification_df = pd.read_csv(failure_window_path)
                classification_source = "failure_window"
            except Exception:
                classification_df = df
                classification_source = "main_telemetry"

        try:
            classification = self.failure_classifier.classify(classification_df)
            has_failure = True

            # Generate report and save it
            report = self.report_generator.to_markdown(classification)
            report_file = telemetry_path_obj.parent / f"failure_report_{expected_steps}.md"
            report_file.write_text(report)
            report_path = report_file
        except ValueError as e:
            # No threshold crossings found - this is expected for successful runs
            if "No threshold crossings found" in str(e):
                has_failure = False
            else:
                # Unexpected error
                return build_result(
                    passed=False,
                    actual_steps_value=actual_steps,
                    structural_invariants_passed=True,
                )
        except Exception:
            # Unexpected error during classification
            return build_result(
                passed=False,
                actual_steps_value=actual_steps,
                structural_invariants_passed=True,
            )

        # Determine overall pass/fail
        if has_failure:
            # Failure detected (threshold crossing)
            return build_result(
                passed=False,
                actual_steps_value=actual_steps,
                structural_invariants_passed=True,
                failure_mode=classification.primary_failure_mode,
                classification_result=classification,
                report_path_value=report_path,
                classification_source_value=classification_source,
            )
        elif not duration_completed:
            # Duration incomplete but no threshold crossings detected
            return build_result(
                passed=False,
                actual_steps_value=actual_steps,
                structural_invariants_passed=True,
            )
        else:
            # All checks passed
            return build_result(
                passed=True,
                actual_steps_value=actual_steps,
                structural_invariants_passed=True,
            )

    def run_simulation(
        self,
        steps: int,
        output_dir: str,
        sim_args: Optional[Sequence[str]] = None,
        long_run_options: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Run simulation for specified number of steps.

        Args:
            steps: Number of simulation steps
            output_dir: Output directory for telemetry
            sim_args: Optional extra simulator CLI arguments
            long_run_options: Optional logging-only simulator options

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
        if sim_args:
            cmd.extend(sim_args)
        if long_run_options:
            if long_run_options.get("telemetry_decimation") is not None:
                cmd.extend(["--telemetry-decimation", str(long_run_options["telemetry_decimation"])])
            if long_run_options.get("failure_window_steps") is not None:
                cmd.extend(["--failure-window-steps", str(long_run_options["failure_window_steps"])])
            if long_run_options.get("write_run_summary_sidecar"):
                cmd.append("--write-run-summary-sidecar")

        # Get list of existing telemetry and artifact files before simulation
        sim_output_dir = Path("outputs/hierarchical_controller_sim")
        existing_files = set(sim_output_dir.glob("telemetry_*.csv")) if sim_output_dir.exists() else set()
        existing_failure_windows = set(sim_output_dir.glob("failure_window_*.csv")) if sim_output_dir.exists() else set()
        existing_sidecars = set(sim_output_dir.glob("telemetry_*.summary.json")) if sim_output_dir.exists() else set()

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

        new_failure_windows = set(sim_output_dir.glob("failure_window_*.csv")) - existing_failure_windows
        for source_failure_window in new_failure_windows:
            dest_failure_window = output_path / f"failure_window_{steps}.csv"
            shutil.copy2(source_failure_window, dest_failure_window)

        new_sidecars = set(sim_output_dir.glob("telemetry_*.summary.json")) - existing_sidecars
        for source_sidecar in new_sidecars:
            dest_sidecar = output_path / f"telemetry_{steps}.summary.json"
            shutil.copy2(source_sidecar, dest_sidecar)

        return dest_telemetry

    def validate_ladder(
        self,
        output_dir: str,
        start_duration: Optional[int] = None,
        durations: Optional[Sequence[int]] = None,
        stop_on_first_failure: bool = True,
        sim_args: Optional[Sequence[str]] = None,
        long_run_options: Optional[dict[str, Any]] = None,
    ) -> List[ValidationResult]:
        """Run progressive duration ladder validation.

        Runs simulations at increasing durations and validates each.

        Args:
            output_dir: Output directory for telemetry and reports
            start_duration: Optional starting duration (skip lower values)
            durations: Custom list of step counts. Defaults to DURATION_LADDER.
            stop_on_first_failure: If True (default), stop at first failure.
                If False, run all durations for full diagnostics.
            sim_args: Optional extra simulator CLI arguments.
            long_run_options: Optional logging-only simulator options.

        Returns:
            List of ValidationResult for each duration tested
        """
        results = []

        ladder = list(durations) if durations is not None else list(self.DURATION_LADDER)

        # Determine starting index
        if start_duration is not None:
            try:
                start_idx = ladder.index(start_duration)
            except ValueError:
                raise ValueError(
                    f"Invalid start_duration {start_duration}. "
                    f"Must be one of {ladder}"
                )
        else:
            start_idx = 0

        # Run ladder
        for duration in ladder[start_idx:]:
            print(f"\n{'='*60}")
            print(f"Validating duration: {duration} steps")
            print(f"{'='*60}")

            # Run simulation
            try:
                telemetry_path = self.run_simulation(
                    duration,
                    output_dir,
                    sim_args=sim_args,
                    long_run_options=long_run_options,
                )
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
                    requested_steps=duration,
                    survived_steps=0,
                    terminated=True,
                    final_sim_time_s=0.0,
                )
                results.append(result)
                print(f"❌ Simulation failed: {e}")
                break

            failure_window_path = Path(output_dir) / f"failure_window_{duration}.csv"
            if not failure_window_path.exists():
                failure_window_path = None

            summary_sidecar_path = Path(output_dir) / f"telemetry_{duration}.summary.json"
            if not summary_sidecar_path.exists():
                summary_sidecar_path = None

            # Validate telemetry
            result = self.validate_duration(
                str(telemetry_path),
                duration,
                failure_window_path=failure_window_path,
                summary_sidecar_path=summary_sidecar_path,
            )
            results.append(result)

            if result.passed:
                print(f"[PASS] Duration {duration} PASSED")
            else:
                print(f"[FAIL] Duration {duration} FAILED")
                if result.failure_mode:
                    print(f"  Failure mode: {result.failure_mode.value}")
                    print(f"  Actual steps: {result.actual_steps}/{result.duration_steps}")

                # Report already written by validate_duration()
                if result.report_path:
                    print(f"  Failure report: {result.report_path}")

                if stop_on_first_failure:
                    print("\nStopping ladder at first failure")
                    break

        return results
