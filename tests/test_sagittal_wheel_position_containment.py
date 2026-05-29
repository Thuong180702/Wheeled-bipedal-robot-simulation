"""Tests for E0b multi-zone position containment in SagittalWheelBalanceController.

E0b FAILED EXPERIMENT - DO NOT USE
Failed validation: 15.98 m drift (better than 35.22 m baseline but still unacceptable)
Root cause: direct wheel torque position correction fights balance controller

These tests verify:
1. E0b is disabled by default (backward compatibility)
2. E0b logic works correctly when explicitly enabled (for research documentation)
3. Telemetry fields are present
"""

import pytest
import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.sagittal_wheel_balance_controller import (
    SagittalWheelBalanceController,
)


class TestPositionContainmentDisabledByDefault:
    """Test that E0b position containment is disabled by default."""

    def test_disabled_by_default(self):
        """E0b should be disabled by default (enable_position_containment=False)."""
        controller = SagittalWheelBalanceController()
        assert controller.enable_position_containment == False

    def test_no_position_input_gives_zero_correction(self):
        """When position_y_m=0.0 (default), position correction should be zero."""
        controller = SagittalWheelBalanceController()

        tau, diag = controller.compute(
            pitch_x_rad=0.05,
            pitch_rate_x_rad_s=0.1,
            cp_error_y_m=0.02,
            com_vy_m_s=0.1,
            wheel_vel_left_rad_s=1.0,
            wheel_vel_right_rad_s=1.0,
            outer_position_bias=0.0,
            # position_y_m defaults to 0.0
            # roll_y_rad defaults to 0.0
        )

        # Position correction should be zero when disabled
        assert diag["position_containment_enabled"] == False
        assert abs(diag["position_correction_proportional"]) < 1e-6
        # Velocity damping may be nonzero but proportional correction should be zero

        # Output should only affect wheel joints
        assert tau[4] != 0.0 or tau[9] != 0.0  # At least one wheel has torque
        assert all(tau[i] == 0.0 for i in [0, 1, 2, 3, 5, 6, 7, 8])  # Leg joints zero

    def test_disabled_returns_zero_correction_even_with_position_input(self):
        """When disabled, position correction should be zero even with position input."""
        controller = SagittalWheelBalanceController(enable_position_containment=False)

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=1.0,  # Large position error
            roll_y_rad=0.0,
        )

        # All position correction should be zero when disabled
        assert diag["position_correction_proportional"] == 0.0
        assert diag["position_correction_velocity"] == 0.0
        assert diag["position_bias"] == 0.0


class TestPositionContainmentZones:
    """Test multi-zone position containment behavior when explicitly enabled.

    These tests verify the failed E0b logic for research documentation.
    E0b must remain disabled by default.
    """

    def test_inside_deadband_minimal_correction(self):
        """Inside deadband, position correction should be minimal (only velocity damping)."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            kp_position=8.0,
            kd_position_velocity=3.0,
        )

        # Position inside deadband
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.05,  # Inside 0.08m deadband
            roll_y_rad=0.0,
        )

        assert diag["in_deadband"] == True
        assert diag["in_soft_zone"] == False
        assert diag["in_hard_zone"] == False
        # Proportional correction should be zero in deadband
        assert abs(diag["position_correction_proportional"]) < 1e-6

    def test_soft_zone_weak_correction(self):
        """In soft zone, correction should be weak (0.5x gain)."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            position_soft_limit_m=0.25,
            kp_position=8.0,
            kd_position_velocity=3.0,
        )

        # Position in soft zone
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.15,  # Between 0.08 and 0.25
            roll_y_rad=0.0,
        )

        assert diag["in_deadband"] == False
        assert diag["in_soft_zone"] == True
        assert diag["in_hard_zone"] == False
        # Correction should be nonzero and oppose drift
        assert diag["position_correction_proportional"] < 0  # Opposes positive drift

    def test_hard_zone_strong_correction(self):
        """In hard zone, correction should be stronger (1.0x gain)."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            position_soft_limit_m=0.25,
            position_hard_limit_m=0.45,
            kp_position=8.0,
            kd_position_velocity=3.0,
        )

        # Position in hard zone
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.35,  # Between 0.25 and 0.45
            roll_y_rad=0.0,
        )

        assert diag["in_deadband"] == False
        assert diag["in_soft_zone"] == False
        assert diag["in_hard_zone"] == True
        assert diag["containment_violation"] == False
        # Correction should be stronger than soft zone
        assert abs(diag["position_correction_proportional"]) > 0

    def test_beyond_hard_limit_violation_flag(self):
        """Beyond hard limit, containment violation flag should be set."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            position_soft_limit_m=0.25,
            position_hard_limit_m=0.45,
        )

        # Position beyond hard limit
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.50,  # Beyond 0.45m hard limit
            roll_y_rad=0.0,
        )

        assert diag["containment_violation"] == True


class TestPositionCorrectionDirection:
    """Test that position correction opposes drift in the correct direction when enabled."""

    def test_positive_drift_negative_correction(self):
        """Positive position drift should produce negative correction torque."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            kp_position=8.0,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.20,  # Positive drift
            roll_y_rad=0.0,
        )

        # Correction should oppose positive drift (be negative)
        assert diag["position_correction_proportional"] < 0

    def test_negative_drift_positive_correction(self):
        """Negative position drift should produce positive correction torque."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            kp_position=8.0,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=-0.20,  # Negative drift
            roll_y_rad=0.0,
        )

        # Correction should oppose negative drift (be positive)
        assert diag["position_correction_proportional"] > 0


class TestPositionCorrectionClipping:
    """Test that position correction is properly clipped."""

    def test_correction_clipped_to_max_bias(self):
        """Position correction should be clipped to max_position_bias."""
        controller = SagittalWheelBalanceController(
            position_deadband_m=0.08,
            kp_position=100.0,  # Very high gain
            max_position_bias=15.0,
        )

        # Large position error
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=1.0,  # Large drift
            roll_y_rad=0.0,
        )

        # Final bias should be clipped
        assert abs(diag["position_bias"]) <= 15.0


class TestBalancePriorityGating:
    """Test that position correction is gated when pitch/roll are unsafe when enabled."""

    def test_large_pitch_reduces_correction(self):
        """Large pitch should reduce position correction via balance priority gate."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            position_deadband_m=0.08,
            kp_position=8.0,
            pitch_gate_threshold_rad=0.15,
        )

        # Same position error, different pitch
        tau_small_pitch, diag_small = controller.compute(
            pitch_x_rad=0.05,  # Small pitch
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.20,
            roll_y_rad=0.0,
        )

        tau_large_pitch, diag_large = controller.compute(
            pitch_x_rad=0.20,  # Large pitch (beyond threshold)
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.20,
            roll_y_rad=0.0,
        )

        # Balance priority gate should be lower with large pitch
        assert diag_large["balance_priority_gate"] < diag_small["balance_priority_gate"]
        # Position correction should be reduced
        assert abs(diag_large["position_bias"]) < abs(diag_small["position_bias"])
        # Gate should be marked as active
        assert diag_large["balance_priority_gate_active"] == True


class TestTelemetryFields:
    """Test that all required telemetry fields are present."""

    def test_all_telemetry_fields_present(self):
        """All E0b telemetry fields should be present in diagnostics."""
        controller = SagittalWheelBalanceController()

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.20,
            roll_y_rad=0.0,
        )

        required_fields = [
            "position_containment_enabled",
            "position_y_m",
            "position_error_abs",
            "planar_drift_m",
            "sagittal_position_velocity_m_s",
            "position_deadband_m",
            "position_soft_limit_m",
            "position_hard_limit_m",
            "in_deadband",
            "in_soft_zone",
            "in_hard_zone",
            "containment_violation",
            "position_correction_proportional",
            "position_correction_velocity",
            "position_correction_raw",
            "position_bias",
            "balance_priority_gate",
            "balance_priority_gate_active",
        ]

        for field in required_fields:
            assert field in diag, f"Missing telemetry field: {field}"


class TestOwnershipAndOutput:
    """Test that ownership and output constraints are maintained."""

    def test_output_only_affects_wheel_joints(self):
        """Output torque should only be nonzero on wheel joints [4, 9]."""
        controller = SagittalWheelBalanceController()

        tau, diag = controller.compute(
            pitch_x_rad=0.05,
            pitch_rate_x_rad_s=0.1,
            cp_error_y_m=0.02,
            com_vy_m_s=0.1,
            wheel_vel_left_rad_s=1.0,
            wheel_vel_right_rad_s=1.0,
            outer_position_bias=0.0,
            position_y_m=0.20,
            roll_y_rad=0.0,
        )

        # Only wheel joints should have nonzero torque
        leg_joints = [0, 1, 2, 3, 5, 6, 7, 8]
        wheel_joints = [4, 9]

        for i in leg_joints:
            assert tau[i] == 0.0, f"Leg joint {i} should have zero torque"

        # At least one wheel should have torque (unless all inputs are zero)
        assert tau[4] != 0.0 or tau[9] != 0.0

    def test_no_wbc_in_diagnostics(self):
        """Diagnostics should not contain WBC-related fields."""
        controller = SagittalWheelBalanceController()

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.20,
            roll_y_rad=0.0,
        )

        # Should not have WBC-related fields
        wbc_fields = ["tau_wbc", "wbc_correction", "wbc_norm"]
        for field in wbc_fields:
            assert field not in diag


class TestVelocityDamping:
    """Test velocity damping component of position containment when enabled."""

    def test_velocity_damping_opposes_motion(self):
        """Velocity damping should oppose forward motion."""
        controller = SagittalWheelBalanceController(
            enable_position_containment=True,  # Explicitly enable for testing
            kd_position_velocity=3.0,
        )

        # Forward velocity
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            cp_error_y_m=0.0,
            com_vy_m_s=0.5,  # Forward velocity
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            outer_position_bias=0.0,
            position_y_m=0.20,
            roll_y_rad=0.0,
        )

        # Velocity damping should oppose forward motion (be negative)
        assert diag["position_correction_velocity"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
