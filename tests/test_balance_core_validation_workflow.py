# tests/test_balance_core_validation_workflow.py
"""Tests for balance-core validation workflow with duration ladder."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from wheeled_biped.validation.balance_core_validator import (
    BalanceCoreValidator,
    ValidationResult,
)


class TestBalanceCoreValidationWorkflow:
    """Test balance-core validation workflow."""

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

            # Posture fields (10-element lists as strings)
            "joint_positions": [str([0.0] * 10)] * steps,
            "joint_velocities": [str([0.0] * 10)] * steps,

            # Contact fields
            "contact_supervisor_state": ["DOUBLE_CONTACT"] * steps,
            "contact_duration_s": [i * 0.002 for i in range(steps)],

            # Torque fields (10-element lists as strings)
            "tau_shape_posture_per_joint": [str([0.0] * 10)] * steps,
            "tau_support_feedforward_per_joint": [str([0.0] * 10)] * steps,
            "tau_sagittal_wheel_balance_per_joint": [str([0.0] * 10)] * steps,
            "tau_lateral_roll_balance_per_joint": [str([0.0] * 10)] * steps,
            "tau_total_raw_per_joint": [str([0.0] * 10)] * steps,
            "tau_total_clipped_per_joint": [str([0.0] * 10)] * steps,
            "tau_final_per_joint": [str([0.0] * 10)] * steps,
            "active_torque_owner_per_joint": [str(["shape_posture"] * 10)] * steps,
            "ownership_violation_count": [0] * steps,

            # Actuator fields
            "actuator_ctrl_per_joint": [str([0.0] * 10)] * steps,

            # Safety fields (10-element boolean lists as strings)
            "torque_saturation_mask_per_joint": [str([False] * 10)] * steps,
            "torque_rate_saturation_mask_per_joint": [str([False] * 10)] * steps,

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
            assert result.schema_valid is True
            assert result.structural_invariants_valid is True
            assert result.duration_completed is True
            assert result.failure_classification is None
            assert result.failure_report is None
        finally:
            os.unlink(telemetry_path)

    def test_duration_ladder_stops_at_first_failure(self):
        """Test that duration ladder stops at first failure."""
        validator = BalanceCoreValidator()

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            # Mock run_simulation to create telemetry files
            # For this test, we'll create files that fail at 200 steps

            def mock_run_simulation(steps: int, output_dir_path: str):
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
                assert results[1].failure_classification is not None
                assert results[1].failure_report is not None
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
            assert result.duration_steps == 50
            assert result.duration_completed is False
            assert result.failure_classification is not None
            assert "Duration incomplete" in result.error_message or "Failure detected" in result.error_message
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
            assert result.schema_valid is False
            assert "pitch_x_rad" in result.error_message
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
            assert result.structural_invariants_valid is False
            assert "ownership" in result.error_message.lower()
        finally:
            os.unlink(telemetry_path)
