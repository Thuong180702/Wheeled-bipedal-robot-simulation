# tests/test_balance_core_validation_workflow.py
"""Tests for balance-core validation workflow with duration ladder."""

import argparse
import json
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from wheeled_biped.validation.balance_core_validator import (
    BalanceCoreValidator,
    ValidationResult,
)
from wheeled_biped.validation.study_aggregator import StudyAggregator


class TestBalanceCoreValidationWorkflow:
    """Test balance-core validation workflow."""

    def _csv_vector(self, values) -> str:
        return ",".join(str(v) for v in values)

    def _create_valid_telemetry(self, steps: int) -> pd.DataFrame:
        """Create valid telemetry dataframe with all required fields.

        Args:
            steps: Number of simulation steps

        Returns:
            Valid telemetry dataframe
        """
        # Create base data
        data = {
            # Metadata
            "control_mode": ["balance-core"] * steps,
            "controller_mode": ["balance-core"] * steps,
            "step": list(range(steps)),
            "time": [i * 0.002 for i in range(steps)],  # 500 Hz

            # State fields
            "pitch_x_rad": [0.01] * steps,
            "roll_y_rad": [0.005] * steps,
            "yaw_z_rad": [0.0] * steps,
            "pitch_rate_rad_s": [0.0] * steps,
            "roll_rate_rad_s": [0.0] * steps,
            "yaw_rate_rad_s": [0.0] * steps,
            "com_x_m": [0.0] * steps,
            "com_y_m": [0.0] * steps,
            "com_z_m": [0.45] * steps,

            # Posture fields (10-element CSV strings)
            "joint_positions": [self._csv_vector([0.0] * 10)] * steps,
            "joint_velocities": [self._csv_vector([0.0] * 10)] * steps,

            # Contact fields
            "contact_supervisor_state": ["DOUBLE_CONTACT"] * steps,
            "contact_duration_s": [i * 0.002 for i in range(steps)],

            # Torque fields (10-element CSV strings)
            "tau_shape_posture_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "tau_support_feedforward_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "tau_sagittal_wheel_balance_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "tau_lateral_roll_balance_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "tau_total_raw_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "tau_total_clipped_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "tau_final_per_joint": [self._csv_vector([0.0] * 10)] * steps,
            "active_torque_owner_per_joint": [self._csv_vector(["shape_posture"] * 10)] * steps,
            "ownership_violation_count": [0] * steps,

            # Actuator fields
            "actuator_ctrl_per_joint": [self._csv_vector([0.0] * 10)] * steps,

            # Safety fields (10-element boolean CSV strings)
            "torque_saturation_mask_per_joint": [self._csv_vector([False] * 10)] * steps,
            "torque_rate_saturation_mask_per_joint": [self._csv_vector([False] * 10)] * steps,

            # Hidden torque fields
            "hidden_torque_norm": [0.0] * steps,
        }

        return pd.DataFrame(data)

    def test_100_step_validation_pass(self):
        """Test that valid 100-step telemetry passes validation."""
        validator = BalanceCoreValidator()

        # Create valid telemetry
        df = self._create_valid_telemetry(100)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            telemetry_path = f.name
            df.to_csv(f, index=False)

        try:
            # Validate
            result = validator.validate_duration(telemetry_path, expected_steps=100)

            # Check result
            assert result.passed is True
            assert result.duration_steps == 100
            assert result.actual_steps == 100
            assert result.structural_invariants_passed is True
            assert result.failure_mode is None
            assert result.classification_result is None
            assert result.report_path is None
        finally:
            os.unlink(telemetry_path)

    def test_duration_ladder_stops_at_first_failure(self):
        """Test that duration ladder stops at first failure."""
        validator = BalanceCoreValidator()

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            # Mock run_simulation to create telemetry files
            # For this test, we'll create files that fail at 200 steps

            def mock_run_simulation(steps: int, output_dir_path: str, sim_args=None):
                """Mock simulation that creates telemetry."""
                telemetry_path = Path(output_dir_path) / f"telemetry_{steps}.csv"

                if steps == 100:
                    # 100 steps succeeds
                    df = self._create_valid_telemetry(100)
                elif steps == 200:
                    # 200 steps fails - pitch divergence at step 150
                    df = self._create_valid_telemetry(200)
                    # Make pitch diverge at step 150
                    df.loc[150:, "pitch_x_rad"] = 0.35  # Exceeds 0.30 threshold
                else:
                    # Should not reach here
                    raise AssertionError(f"Should not simulate {steps} steps")

                df.to_csv(telemetry_path, index=False)
                return telemetry_path

            # Replace run_simulation with mock
            original_run_simulation = validator.run_simulation
            validator.run_simulation = mock_run_simulation

            try:
                # Run ladder starting from 100
                results = validator.validate_ladder(output_dir, start_duration=100)

                # Check results
                assert len(results) == 2  # Should stop after 200 fails

                # First result (100 steps) should pass
                assert results[0].passed is True
                assert results[0].duration_steps == 100

                # Second result (200 steps) should fail
                assert results[1].passed is False
                assert results[1].duration_steps == 200
                assert results[1].classification_result is not None
                assert results[1].report_path is not None
            finally:
                validator.run_simulation = original_run_simulation

    def test_validation_detects_incomplete_duration(self):
        """Test that validation detects when simulation ends early."""
        validator = BalanceCoreValidator()

        # Create telemetry with only 50 steps when expecting 100
        # Include a failure (pitch divergence) to get classification
        df = self._create_valid_telemetry(50)
        # Make pitch diverge at step 40
        df.loc[40:, "pitch_x_rad"] = 0.35  # Exceeds 0.30 threshold

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            telemetry_path = f.name
            df.to_csv(f, index=False)

        try:
            # Validate expecting 100 steps
            result = validator.validate_duration(telemetry_path, expected_steps=100)

            # Check result
            assert result.passed is False
            assert result.duration_steps == 100  # Expected steps
            assert result.actual_steps == 50  # Actual steps achieved
            assert result.classification_result is not None  # Pitch divergence detected
        finally:
            os.unlink(telemetry_path)

    def test_validation_detects_schema_errors(self):
        """Test that validation detects schema errors."""
        validator = BalanceCoreValidator()

        # Create telemetry missing required field
        df = self._create_valid_telemetry(100)
        df = df.drop(columns=["pitch_x_rad"])

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            telemetry_path = f.name
            df.to_csv(f, index=False)

        try:
            # Validate
            result = validator.validate_duration(telemetry_path, expected_steps=100)

            # Check result
            assert result.passed is False
            assert result.structural_invariants_passed is False  # Schema check failed before structural checks
        finally:
            os.unlink(telemetry_path)

    def test_validation_detects_structural_invariant_violations(self):
        """Test that validation detects structural invariant violations."""
        validator = BalanceCoreValidator()

        # Create telemetry with ownership violations
        df = self._create_valid_telemetry(100)
        df.loc[50:, "ownership_violation_count"] = 5

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            telemetry_path = f.name
            df.to_csv(f, index=False)

        try:
            # Validate
            result = validator.validate_duration(telemetry_path, expected_steps=100)

            # Check result
            assert result.passed is False
            assert result.structural_invariants_passed is False
        finally:
            os.unlink(telemetry_path)

    def test_arbitrary_duration_ladder_with_custom_durations(self):
        """Test that validate_ladder accepts custom durations."""
        validator = BalanceCoreValidator()

        with tempfile.TemporaryDirectory() as output_dir:
            call_log = []

            def mock_run_simulation(steps, output_dir_path, sim_args=None):
                call_log.append(steps)
                telemetry_path = Path(output_dir_path) / f"telemetry_{steps}.csv"
                df = self._create_valid_telemetry(steps)
                df.to_csv(telemetry_path, index=False)
                return telemetry_path

            original = validator.run_simulation
            validator.run_simulation = mock_run_simulation
            try:
                results = validator.validate_ladder(
                    output_dir,
                    durations=[500, 1000, 2000],
                )
                assert len(results) == 3
                assert results[0].duration_steps == 500
                assert results[1].duration_steps == 1000
                assert results[2].duration_steps == 2000
                assert all(r.passed for r in results)
                assert call_log == [500, 1000, 2000]
            finally:
                validator.run_simulation = original

    def test_default_stops_at_first_failure(self):
        """Test that default validate_ladder stops at first failure."""
        validator = BalanceCoreValidator()

        with tempfile.TemporaryDirectory() as output_dir:

            def mock_run_simulation(steps, output_dir_path, sim_args=None):
                telemetry_path = Path(output_dir_path) / f"telemetry_{steps}.csv"
                df = self._create_valid_telemetry(steps)
                if steps == 1000:
                    df.loc[500:, "pitch_x_rad"] = 0.35
                df.to_csv(telemetry_path, index=False)
                return telemetry_path

            original = validator.run_simulation
            validator.run_simulation = mock_run_simulation
            try:
                results = validator.validate_ladder(
                    output_dir,
                    durations=[500, 1000, 2000],
                    stop_on_first_failure=True,
                )
                assert len(results) == 2  # stops after 1000 fails
                assert results[0].passed is True
                assert results[1].passed is False
            finally:
                validator.run_simulation = original

    def test_continue_all_runs_every_duration(self):
        """Test that stop_on_first_failure=False runs all durations."""
        validator = BalanceCoreValidator()

        with tempfile.TemporaryDirectory() as output_dir:

            def mock_run_simulation(steps, output_dir_path, sim_args=None):
                telemetry_path = Path(output_dir_path) / f"telemetry_{steps}.csv"
                df = self._create_valid_telemetry(steps)
                if steps == 1000:
                    df.loc[500:, "pitch_x_rad"] = 0.35
                df.to_csv(telemetry_path, index=False)
                return telemetry_path

            original = validator.run_simulation
            validator.run_simulation = mock_run_simulation
            try:
                results = validator.validate_ladder(
                    output_dir,
                    durations=[500, 1000, 2000],
                    stop_on_first_failure=False,
                )
                assert len(results) == 3  # all run
                assert results[0].passed is True
                assert results[1].passed is False
                assert results[2].passed is True
            finally:
                validator.run_simulation = original

    def test_sim_args_forwarded_to_run_simulation(self):
        """Test that sim_args are forwarded through run_simulation."""
        validator = BalanceCoreValidator()
        captured_args = []

        original_run = validator.run_simulation

        def mock_run(steps, output_dir_path, sim_args=None):
            captured_args.append(list(sim_args or []))
            telemetry_path = Path(output_dir_path) / f"telemetry_{steps}.csv"
            df = self._create_valid_telemetry(steps)
            df.to_csv(telemetry_path, index=False)
            return telemetry_path

        validator.run_simulation = mock_run
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                validator.validate_ladder(
                    output_dir,
                    durations=[100],
                    sim_args=["--initial-root-z-perturbation", "0.02"],
                )
                assert captured_args[0] == ["--initial-root-z-perturbation", "0.02"]
            finally:
                validator.run_simulation = original_run

    def test_study_aggregator_classifies_invalid_initial_setup_before_controller_failure(self):
        aggregator = StudyAggregator()
        df = self._create_valid_telemetry(100)
        df["left_wheel_contact"] = [False] * 100
        df["right_wheel_contact"] = [False] * 100
        df["min_wheel_contact_dist_m"] = [0.01] * 100
        df.loc[20:, "pitch_x_rad"] = 0.35

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            telemetry_path = f.name
            df.to_csv(f, index=False)

        try:
            result = aggregator.evaluate_case_from_telemetry(
                case_id="invalid_setup_case",
                height_test_type="root_z_perturbation",
                duration_steps=100,
                telemetry_path=telemetry_path,
                sim_args=["--initial-root-z-perturbation", "0.02"],
            )
        finally:
            os.unlink(telemetry_path)

        assert result.setup_valid is False
        assert result.setup_failure_reason == "floating_start"
        assert result.failure_mode == "invalid_initial_setup"
        assert result.responsible_component is None
        assert result.passed is False

    def test_study_aggregator_preserves_height_test_type_and_classification(self):
        aggregator = StudyAggregator()
        df = self._create_valid_telemetry(100)
        df["left_wheel_contact"] = [True] * 100
        df["right_wheel_contact"] = [True] * 100
        df["min_wheel_contact_dist_m"] = [-5e-4] * 100
        df["nominal_equilibrium_com_z_m"] = [0.45] * 100
        df["initial_com_z_m_after_perturbation"] = [0.47] * 100
        df.loc[50:, "pitch_x_rad"] = 0.35

        with tempfile.TemporaryDirectory() as output_dir:
            telemetry_path = Path(output_dir) / "telemetry.csv"
            df.to_csv(telemetry_path, index=False)
            result = aggregator.evaluate_case_from_telemetry(
                case_id="root_z_plus_020mm_100",
                height_test_type="root_z_perturbation",
                duration_steps=100,
                telemetry_path=telemetry_path,
                sim_args=["--initial-root-z-perturbation", "0.02"],
            )

        assert result.setup_valid is True
        assert result.height_test_type == "root_z_perturbation"
        assert result.failure_mode == "F2.1"
        assert result.responsible_component == "SagittalWheelBalanceController"
        assert result.initial_com_z_m == pytest.approx(0.47)
        assert result.equilibrium_com_z_m == pytest.approx(0.45)

    def test_study_aggregator_writes_json_and_markdown_summaries(self):
        aggregator = StudyAggregator()
        results = [
            aggregator._to_study_case_result(
                case_id="longevity_1000",
                height_test_type="longevity",
                validation_result=ValidationResult(
                    passed=True,
                    duration_steps=1000,
                    actual_steps=1000,
                    structural_invariants_passed=True,
                    failure_mode=None,
                    classification_result=None,
                    telemetry_path=Path("telemetry_1000.csv"),
                    report_path=None,
                ),
                sim_args=[],
                summary_metrics={"requested_steps": 1000, "survival_steps": 1000},
                setup_verdict={
                    "initial_contact_state": "DOUBLE_CONTACT",
                    "min_wheel_contact_dist_m": -5e-4,
                    "equilibrium_com_z_m": 0.45,
                    "initial_com_z_m": 0.45,
                },
            )
        ]

        with tempfile.TemporaryDirectory() as output_dir:
            json_path = Path(output_dir) / "summary.json"
            markdown_path = Path(output_dir) / "summary.md"
            aggregator.write_summary_files(
                results,
                json_path=json_path,
                markdown_path=markdown_path,
                conclusion="long_duration_survival_passed_up_to_1000_steps",
            )

            payload = json.loads(json_path.read_text())
            markdown = markdown_path.read_text()



def test_main_writes_known_study_summaries():
    from scripts import validate_balance_core

    with tempfile.TemporaryDirectory() as output_dir:
        with patch.object(validate_balance_core.sys, "argv", [
            "validate_balance_core.py",
            "--write-known-study-summaries",
            "--output-dir", output_dir,
        ]):
            exit_code = validate_balance_core.main()

        assert exit_code == 0

        summary_dir = Path(output_dir)
        long_json = json.loads((summary_dir / "long_duration_summary.json").read_text())
        root_json = json.loads((summary_dir / "root_z_perturbation_summary.json").read_text())
        true_height_md = (summary_dir / "true_height_feasibility_summary.md").read_text()

    assert long_json["conclusion"] == "long_duration_survival_passed_up_to_10000_steps"
    assert long_json["max_confirmed_passing_duration_steps"] == 10000
    assert root_json["conclusion"].startswith("root_z_perturbation_robustness_narrow")
    assert root_json["case_count"] == 9
    assert "true_height_variant_test_blocked" in true_height_md
    from scripts.validate_balance_core import _parse_durations

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_durations(",,,")



def test_parse_durations_parses_comma_separated_ints():
    from scripts.validate_balance_core import _parse_durations

    assert _parse_durations("1000, 2000,5000") == [1000, 2000, 5000]



def test_main_forwards_initial_root_z_perturbation_flag():
    from scripts import validate_balance_core

    class FakeValidator:
        def __init__(self):
            self.run_calls = []

        def run_simulation(self, steps, output_dir, sim_args=None):
            self.run_calls.append((steps, output_dir, list(sim_args or [])))
            return Path(output_dir) / f"telemetry_{steps}.csv"

        def validate_duration(self, telemetry_path, expected_steps):
            return ValidationResult(
                passed=True,
                duration_steps=expected_steps,
                actual_steps=expected_steps,
                structural_invariants_passed=True,
                failure_mode=None,
                classification_result=None,
                telemetry_path=Path(telemetry_path),
                report_path=None,
            )

    fake_validator = FakeValidator()

    with patch.object(validate_balance_core, "BalanceCoreValidator", return_value=fake_validator):
        with patch.object(validate_balance_core.sys, "argv", [
            "validate_balance_core.py",
            "--single-duration", "1000",
            "--initial-root-z-perturbation", "0.02",
        ]):
            exit_code = validate_balance_core.main()

    assert exit_code == 0
    assert fake_validator.run_calls == [
        (1000, str(Path("outputs/balance_core_validation")), ["--initial-root-z-perturbation", "0.02"])
    ]



def test_main_forwards_initial_root_z_perturbation_in_ladder_mode():
    from scripts import validate_balance_core

    class FakeValidator:
        def __init__(self):
            self.ladder_calls = []

        def validate_ladder(
            self,
            output_dir,
            start_duration=None,
            durations=None,
            stop_on_first_failure=True,
            sim_args=None,
        ):
            self.ladder_calls.append({
                "output_dir": output_dir,
                "start_duration": start_duration,
                "durations": durations,
                "stop_on_first_failure": stop_on_first_failure,
                "sim_args": list(sim_args or []),
            })
            return [
                ValidationResult(
                    passed=True,
                    duration_steps=1000,
                    actual_steps=1000,
                    structural_invariants_passed=True,
                    failure_mode=None,
                    classification_result=None,
                    telemetry_path=Path(output_dir) / "telemetry_1000.csv",
                    report_path=None,
                )
            ]

    fake_validator = FakeValidator()

    with patch.object(validate_balance_core, "BalanceCoreValidator", return_value=fake_validator):
        with patch.object(validate_balance_core.sys, "argv", [
            "validate_balance_core.py",
            "--durations", "1000,2000",
            "--continue-all",
            "--initial-root-z-perturbation", "0.01",
        ]):
            exit_code = validate_balance_core.main()

    assert exit_code == 0
    assert fake_validator.ladder_calls == [{
        "output_dir": str(Path("outputs/balance_core_validation")),
        "start_duration": None,
        "durations": [1000, 2000],
        "stop_on_first_failure": False,
        "sim_args": ["--initial-root-z-perturbation", "0.01"],
    }]
