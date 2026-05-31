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
from wheeled_biped.validation.step_e_audit_helpers import (
    classify_steady_state_balance_torque_bias,
)


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

            def mock_run_simulation(steps: int, output_dir_path: str, sim_args=None, long_run_options=None):
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

            def mock_run_simulation(steps, output_dir_path, sim_args=None, long_run_options=None):
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

            def mock_run_simulation(steps, output_dir_path, sim_args=None, long_run_options=None):
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

            def mock_run_simulation(steps, output_dir_path, sim_args=None, long_run_options=None):
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

        def mock_run(steps, output_dir_path, sim_args=None, long_run_options=None):
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

    def test_long_run_options_forwarded_to_run_simulation(self):
        """Test that long-run logging options are forwarded through validate_ladder."""
        validator = BalanceCoreValidator()
        captured = {}

        original_run = validator.run_simulation

        def mock_run(steps, output_dir_path, sim_args=None, long_run_options=None):
            captured["steps"] = steps
            captured["long_run_options"] = dict(long_run_options or {})
            telemetry_path = Path(output_dir_path) / f"telemetry_{steps}.csv"
            df = self._create_valid_telemetry(steps)
            df.to_csv(telemetry_path, index=False)
            return telemetry_path

        validator.run_simulation = mock_run
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                validator.validate_ladder(
                    output_dir,
                    durations=[10000],
                    long_run_options={"telemetry_decimation": 20, "failure_window_steps": 400},
                )
                assert captured["steps"] == 10000
                assert captured["long_run_options"]["telemetry_decimation"] == 20
                assert captured["long_run_options"]["failure_window_steps"] == 400
            finally:
                validator.run_simulation = original_run

    def test_validate_duration_prefers_failure_window_and_sidecar_actual_steps(self):
        """Test that failure-window telemetry drives classification and sidecar drives actual steps."""
        validator = BalanceCoreValidator()

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            decimated = self._create_valid_telemetry(50)
            decimated_path = output_path / "telemetry_1000.csv"
            decimated.to_csv(decimated_path, index=False)

            failure_window = self._create_valid_telemetry(200)
            failure_window.loc[150:, "pitch_x_rad"] = 0.35
            failure_window_path = output_path / "failure_window_1000.csv"
            failure_window.to_csv(failure_window_path, index=False)

            summary_sidecar_path = output_path / "telemetry_1000.summary.json"
            summary_sidecar_path.write_text(json.dumps({
                "requested_steps": 1000,
                "survived_steps": 1000,
                "actual_steps": 1000,
                "terminated": False,
                "written_telemetry_rows": 50,
                "termination_reason": "completed",
                "final_sim_time_s": 2.0,
                "wheel_velocity_trend": 0.15,
                "metric_integrity": {"source": "full_rate_online", "limitations": []},
            }), encoding="utf-8")

            result = validator.validate_duration(
                str(decimated_path),
                expected_steps=1000,
                failure_window_path=failure_window_path,
                summary_sidecar_path=summary_sidecar_path,
            )

            assert result.passed is False
            assert result.classification_result is not None
            assert result.classification_source == "failure_window"
            assert result.actual_steps == 1000
            assert result.requested_steps == 1000
            assert result.survived_steps == 1000
            assert result.terminated is False
            assert result.final_sim_time_s == pytest.approx(2.0)
            assert result.failure_window_path == failure_window_path
            assert result.summary_sidecar_path == summary_sidecar_path
            assert result.termination_reason == "completed"
            assert result.summary_metrics["metric_integrity"]["source"] == "full_rate_online"
            assert result.summary_metrics["written_telemetry_rows"] == 50
            assert result.summary_metrics["wheel_velocity_trend"] == pytest.approx(0.15)

    def test_run_simulation_copies_failure_window_and_sidecar_to_expected_artifact_paths(self):
        validator = BalanceCoreValidator()

        with tempfile.TemporaryDirectory() as output_dir:
            sim_output_dir = Path("outputs/hierarchical_controller_sim")
            sim_output_dir.mkdir(parents=True, exist_ok=True)

            telemetry_source = sim_output_dir / "telemetry_123456.csv"
            failure_window_source = sim_output_dir / "failure_window_123456.csv"
            sidecar_source = sim_output_dir / "telemetry_123456.summary.json"

            telemetry_source.write_text("time\n0.0\n", encoding="utf-8")
            failure_window_source.write_text("time\n0.0\n", encoding="utf-8")
            sidecar_source.write_text(json.dumps({"actual_steps": 1000}), encoding="utf-8")

            try:
                telemetry_source.unlink(missing_ok=True)
                failure_window_source.unlink(missing_ok=True)
                sidecar_source.unlink(missing_ok=True)

                def fake_run(*args, **kwargs):
                    telemetry_source.write_text("time\n0.0\n", encoding="utf-8")
                    failure_window_source.write_text("time\n0.0\n", encoding="utf-8")
                    sidecar_source.write_text(json.dumps({"actual_steps": 1000}), encoding="utf-8")
                    return None

                with patch("wheeled_biped.validation.balance_core_validator.subprocess.run", side_effect=fake_run):
                    dest_telemetry = validator.run_simulation(
                        1000,
                        output_dir,
                        long_run_options={
                            "telemetry_decimation": 20,
                            "failure_window_steps": 500,
                            "write_run_summary_sidecar": True,
                        },
                    )

                assert dest_telemetry == Path(output_dir) / "telemetry_1000.csv"
                assert dest_telemetry.exists()
                assert (Path(output_dir) / "failure_window_1000.csv").exists()
                assert (Path(output_dir) / "telemetry_1000.summary.json").exists()
            finally:
                telemetry_source.unlink(missing_ok=True)
                failure_window_source.unlink(missing_ok=True)
                sidecar_source.unlink(missing_ok=True)


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
                    summary_metrics={"requested_steps": 1000, "survival_steps": 1000},
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

            assert payload["max_confirmed_passing_duration_steps"] == 1000
            assert payload["long_duration_survival_passed_up_to_100000_steps"] is False
            assert payload["first_failing_duration_steps"] is None
            assert "Max confirmed passing duration: 1000 steps" in markdown
            assert "Passed 100000 steps: no" in markdown



def test_step_e_steady_state_balance_torque_bias_classification():
    df = pd.DataFrame({
        "support_position_velocity_m_s": [0.0, 1e-6, -1e-6, 0.0],
        "support_position_error_m": [0.0527, 0.0528, 0.0526, 0.0527],
        "tau_balance_before_position": [1.0534, 1.0533, 1.0535, 1.0534],
        "tau_position_clipped": [-1.0534, -1.0533, -1.0535, -1.0534],
        "hidden_torque_norm": [0.0, 0.0, 0.0, 0.0],
        "ownership_violation_count": [0, 0, 0, 0],
        "wheel_torque_saturation_left": [False, False, False, False],
        "wheel_torque_saturation_right": [False, False, False, False],
    })

    result = classify_steady_state_balance_torque_bias(df)

    assert result.primary_classification == "steady_state_balance_torque_bias"
    assert abs(result.support_position_velocity_mean) <= 1e-3
    assert result.support_position_error_mean > 0.0
    assert result.tau_balance_before_position_mean > 0.0
    assert result.tau_position_clipped_mean < 0.0
    assert abs(result.net_balance_mean) <= 1e-3
    assert result.physical_motor_limit is False
    assert result.sign_error is False
    assert result.continuous_position_drift is False
    assert result.WBC_active is False
    assert result.E0_logic_active is False


def test_main_routes_step_a_orchestration_to_summary_dir_from_default_output_dir():
    from scripts import validate_balance_core
def test_main_routes_step_a_orchestration_to_custom_output_dir():
    from scripts import validate_balance_core

    captured_output_dirs = []

    def fake_write_known_study_summaries(output_dir):
        captured_output_dirs.append(Path(output_dir))

    with tempfile.TemporaryDirectory() as output_dir:
        with patch.object(validate_balance_core, "_write_known_study_summaries", side_effect=fake_write_known_study_summaries):
            with patch.object(validate_balance_core.sys, "argv", [
                "validate_balance_core.py",
                "--step-a-orchestration",
                "--output-dir", output_dir,
            ]):
                exit_code = validate_balance_core.main()

    assert exit_code == 0
    assert captured_output_dirs == [Path(output_dir)]



def test_parse_durations_parses_comma_separated_ints():
    from scripts.validate_balance_core import _parse_durations

    assert _parse_durations("1000, 2000,5000") == [1000, 2000, 5000]



def test_main_forwards_initial_root_z_perturbation_flag():
    from scripts import validate_balance_core

    class FakeValidator:
        def __init__(self):
            self.run_calls = []

        def run_simulation(self, steps, output_dir, sim_args=None, long_run_options=None):
            self.run_calls.append((steps, output_dir, list(sim_args or []), dict(long_run_options or {})))
            return Path(output_dir) / f"telemetry_{steps}.csv"

        def validate_duration(self, telemetry_path, expected_steps, failure_window_path=None, summary_sidecar_path=None):
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
        (1000, str(Path("outputs/balance_core_validation")), ["--initial-root-z-perturbation", "0.02"], {})
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
            long_run_options=None,
        ):
            self.ladder_calls.append({
                "output_dir": output_dir,
                "start_duration": start_duration,
                "durations": durations,
                "stop_on_first_failure": stop_on_first_failure,
                "sim_args": list(sim_args or []),
                "long_run_options": dict(long_run_options or {}),
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
                    requested_steps=1000,
                    survived_steps=1000,
                    terminated=False,
                    final_sim_time_s=2.0,
                    summary_metrics={
                        "pitch_x": {"min": 0.0, "max": 0.1, "rms": 0.02},
                        "roll_y": {"min": 0.0, "max": 0.1, "rms": 0.02},
                        "com_z": {"min": 0.4, "max": 0.45, "drift": 0.0},
                        "wheel_vel_mean": {"min": -0.2, "max": 0.3, "rms": 0.1},
                        "wheel_velocity_trend": 0.02,
                        "ownership_violation_count_max": 0,
                        "hidden_torque_norm_max": 0.0,
                        "tau_wbc_norm_max": 0.0,
                        "contact_state_summary": {"counts": {"DOUBLE_CONTACT": 1000}, "most_common_state": "DOUBLE_CONTACT"},
                        "torque_saturation": {"fraction_max": 0.0, "fraction_mean": 0.0},
                        "torque_rate_saturation": {"fraction_max": 0.0, "fraction_mean": 0.0},
                        "metric_integrity": {"source": "full_rate_online", "limitations": []},
                    },
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
        "long_run_options": {},
    }]


def test_main_forwards_long_run_options_to_ladder_mode(tmp_path):
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
            long_run_options=None,
        ):
            self.ladder_calls.append({
                "output_dir": output_dir,
                "start_duration": start_duration,
                "durations": durations,
                "stop_on_first_failure": stop_on_first_failure,
                "sim_args": list(sim_args or []),
                "long_run_options": dict(long_run_options or {}),
            })
            return [
                ValidationResult(
                    passed=True,
                    duration_steps=10000,
                    actual_steps=10000,
                    structural_invariants_passed=True,
                    failure_mode=None,
                    classification_result=None,
                    telemetry_path=Path(output_dir) / "telemetry_10000.csv",
                    report_path=None,
                    requested_steps=10000,
                    survived_steps=10000,
                    terminated=False,
                    final_sim_time_s=20.0,
                    summary_metrics={
                        "pitch_x": {"min": 0.0, "max": 0.1, "rms": 0.02},
                        "roll_y": {"min": 0.0, "max": 0.1, "rms": 0.02},
                        "com_z": {"min": 0.4, "max": 0.45, "drift": 0.0},
                        "wheel_vel_mean": {"min": -0.2, "max": 0.3, "rms": 0.1},
                        "wheel_velocity_trend": 0.05,
                        "ownership_violation_count_max": 0,
                        "hidden_torque_norm_max": 0.0,
                        "tau_wbc_norm_max": 0.0,
                        "contact_state_summary": {"counts": {"DOUBLE_CONTACT": 10000}, "most_common_state": "DOUBLE_CONTACT"},
                        "torque_saturation": {"fraction_max": 0.0, "fraction_mean": 0.0},
                        "torque_rate_saturation": {"fraction_max": 0.0, "fraction_mean": 0.0},
                        "metric_integrity": {"source": "full_rate_online", "limitations": []},
                    },
                )
            ]

    fake_validator = FakeValidator()

    with patch.object(validate_balance_core, "BalanceCoreValidator", return_value=fake_validator):
        with patch.object(validate_balance_core.sys, "argv", [
            "validate_balance_core.py",
            "--durations", "10000",
            "--telemetry-decimation", "20",
            "--failure-window-steps", "500",
            "--write-run-summary-sidecar",
            "--output-dir", str(tmp_path),
        ]):
            exit_code = validate_balance_core.main()

    assert exit_code == 0
    assert fake_validator.ladder_calls == [{
        "output_dir": str(tmp_path),
        "start_duration": None,
        "durations": [10000],
        "stop_on_first_failure": True,
        "sim_args": [],
        "long_run_options": {
            "telemetry_decimation": 20,
            "failure_window_steps": 500,
            "write_run_summary_sidecar": True,
        },
    }]


def test_build_extended_longevity_summary_reports_first_failure_and_100k_status(tmp_path):
    from scripts import validate_balance_core

    passing = ValidationResult(
        passed=True,
        duration_steps=10000,
        actual_steps=10000,
        structural_invariants_passed=True,
        failure_mode=None,
        classification_result=None,
        telemetry_path=tmp_path / "telemetry_10000.csv",
        report_path=None,
        requested_steps=10000,
        survived_steps=10000,
        terminated=False,
        final_sim_time_s=20.0,
        summary_metrics={
            "pitch_x": {"min": 0.0, "max": 0.1, "rms": 0.02},
            "roll_y": {"min": 0.0, "max": 0.1, "rms": 0.02},
            "com_z": {"min": 0.4, "max": 0.45, "drift": 0.0},
            "wheel_vel_mean": {"min": -0.2, "max": 0.3, "rms": 0.1},
            "wheel_velocity_trend": 0.05,
            "ownership_violation_count_max": 0,
            "hidden_torque_norm_max": 0.0,
            "tau_wbc_norm_max": 0.0,
            "contact_state_summary": {"counts": {"DOUBLE_CONTACT": 10000}, "most_common_state": "DOUBLE_CONTACT"},
            "torque_saturation": {"fraction_max": 0.0, "fraction_mean": 0.0},
            "torque_rate_saturation": {"fraction_max": 0.0, "fraction_mean": 0.0},
            "metric_integrity": {"source": "full_rate_online", "limitations": []},
        },
    )
    failing = ValidationResult(
        passed=False,
        duration_steps=20000,
        actual_steps=15321,
        structural_invariants_passed=True,
        failure_mode=None,
        classification_result=None,
        telemetry_path=tmp_path / "telemetry_20000.csv",
        report_path=tmp_path / "failure_report_20000.md",
        requested_steps=20000,
        survived_steps=15321,
        terminated=True,
        termination_reason="fell",
        final_sim_time_s=30.642,
        primary_failure_mode="F2.1",
        secondary_failure_modes=["F1.2"],
        summary_sidecar_path=tmp_path / "telemetry_20000.summary.json",
        failure_window_path=tmp_path / "failure_window_20000.csv",
        summary_metrics={
            "pitch_x": {"min": 0.0, "max": 0.35, "rms": 0.08},
            "roll_y": {"min": 0.0, "max": 0.1, "rms": 0.02},
            "com_z": {"min": 0.35, "max": 0.45, "drift": -0.08},
            "wheel_vel_mean": {"min": -0.2, "max": 0.7, "rms": 0.2},
            "wheel_velocity_trend": 0.2,
            "ownership_violation_count_max": 0,
            "hidden_torque_norm_max": 0.0,
            "tau_wbc_norm_max": 0.0,
            "contact_state_summary": {"counts": {"DOUBLE_CONTACT": 12000, "NO_CONTACT": 3321}, "most_common_state": "DOUBLE_CONTACT"},
            "torque_saturation": {"fraction_max": 0.1, "fraction_mean": 0.01},
            "torque_rate_saturation": {"fraction_max": 0.2, "fraction_mean": 0.03},
            "metric_integrity": {"source": "full_rate_online", "limitations": []},
            "written_telemetry_rows": 767,
        },
    )

    summary = validate_balance_core._write_extended_longevity_summary([passing, failing], tmp_path)

    assert summary["maximum_confirmed_survival_steps"] == 10000
    assert summary["passed_100000_steps"] is False
    assert summary["first_failing_duration"] == 20000
    assert summary["primary_failure_mode"] == "F2.1"
    assert summary["conclusion"] == "long_duration_survival_confirmed_up_to_10000_steps"
    assert (tmp_path / "extended_longevity_summary.json").exists()
    assert (tmp_path / "extended_longevity_summary.md").exists()


def test_resolve_sysid_output_dir_uses_position_aware_namespace():
    from scripts.collect_sagittal_balance_sysid_data import resolve_sysid_output_dir

    path = resolve_sysid_output_dir(Path("outputs"))
    assert path == Path("outputs/sagittal_position_aware_balance/sysid")


def test_build_sysid_run_metadata_marks_closed_loop_collection():
    from scripts.collect_sagittal_balance_sysid_data import build_sysid_run_metadata

    metadata = build_sysid_run_metadata(
        scenario="nominal",
        duration_steps=5000,
        controller_mode="balance-core",
    )

    assert metadata["collection_mode"] == "closed_loop"
    assert metadata["controller_mode"] == "balance-core"
    assert metadata["duration_steps"] == 5000
