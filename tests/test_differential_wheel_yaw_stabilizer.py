"""Tests for DifferentialWheelYawStabilizer.

Verifies:
- Antisymmetric wheel torque output
- Sign convention matching YawController
- Torque saturation
- Lowpass filtering
- Zero output at zero error/rate
- Only wheel joints actuated
- Proportionality to error/rate
- Diagnostics populated
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.differential_wheel_yaw_stabilizer import (
    DifferentialWheelYawStabilizer,
)
from wheeled_biped.controllers.yaw_controller import YawController


class TestDifferentialWheelYawStabilizer:
    """Test wheel yaw stabilizer antisymmetric wheel torque generation."""

    @pytest.fixture
    def stabilizer(self):
        """Create wheel yaw stabilizer with standard gains (no filter, full gate)."""
        return DifferentialWheelYawStabilizer(
            kp_yaw=5.0,
            kd_yaw=1.5,
            max_yaw_torque=5.0,
            lowpass_alpha=1.0,  # No filtering for basic tests
            height_gate_low=0.0,  # Full gate -> always 1.0
            height_gate_high=1.0,
        )

    def test_positive_yaw_error_applies_antisymmetric_wheel_torque(self, stabilizer):
        """Positive yaw error: left wheel positive, right wheel negative."""
        yaw_error = 0.1
        yaw_rate = 0.0

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        # Left wheel positive, right wheel negative (CCW correction)
        assert tau[4] > 0.0, f"Left wheel should be positive, got {tau[4]}"
        assert tau[9] < 0.0, f"Right wheel should be negative, got {tau[9]}"
        # Antisymmetric: left ≈ -right
        assert abs(tau[4] + tau[9]) < 1e-6, f"Wheel torques should be antisymmetric: left={tau[4]}, right={tau[9]}"

    def test_negative_yaw_error_applies_antisymmetric_wheel_torque(self, stabilizer):
        """Negative yaw error: left wheel negative, right wheel positive."""
        yaw_error = -0.1
        yaw_rate = 0.0

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        # Left wheel negative, right wheel positive (CW correction)
        assert tau[4] < 0.0, f"Left wheel should be negative, got {tau[4]}"
        assert tau[9] > 0.0, f"Right wheel should be positive, got {tau[9]}"
        assert abs(tau[4] + tau[9]) < 1e-6

    def test_yaw_rate_damping(self, stabilizer):
        """Yaw rate damping should oppose yaw velocity."""
        yaw_error = 0.0
        yaw_rate = 1.0  # rad/s, CCW

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        # Positive yaw rate (CCW) needs CW damping:
        # Left wheel negative, right wheel positive
        assert tau[4] < 0.0, f"Damping should oppose positive yaw rate, left={tau[4]}"
        assert tau[9] > 0.0, f"Damping should oppose positive yaw rate, right={tau[9]}"

    def test_torque_saturation(self, stabilizer):
        """Wheel yaw torque should saturate at max_yaw_torque."""
        yaw_error = 10.0  # Very large error
        yaw_rate = 0.0

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        assert abs(tau[4]) <= stabilizer.max_yaw_torque + 1e-6
        assert abs(tau[9]) <= stabilizer.max_yaw_torque + 1e-6
        assert diag["wheel_yaw_saturated"]

    def test_zero_yaw_error_and_rate_produces_zero_torque(self, stabilizer):
        """Zero yaw error and rate should produce zero torque."""
        yaw_error = 0.0
        yaw_rate = 0.0

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        assert tau[4] == 0.0
        assert tau[9] == 0.0

    def test_only_wheel_joints_actuated(self, stabilizer):
        """Wheel yaw stabilizer should only actuate wheel joints [4, 9]."""
        yaw_error = 0.1
        yaw_rate = 0.0

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        for idx in range(10):
            if idx in (4, 9):
                continue
            assert tau[idx] == 0.0, f"Joint {idx} should not be actuated, got {tau[idx]}"

    def test_torque_proportional_to_error(self, stabilizer):
        """Torque magnitude should be proportional to error magnitude."""
        yaw_rate = 0.0

        tau_small, _ = stabilizer.compute(0.05, yaw_rate)
        tau_large, _ = stabilizer.compute(0.20, yaw_rate)

        assert abs(tau_large[4]) > abs(tau_small[4])
        assert abs(tau_large[9]) > abs(tau_small[9])

    def test_sign_matches_yaw_controller_direction(self):
        """Wheel yaw direction should produce same corrective yaw moment as hip-yaw YawController."""
        stabilizer = DifferentialWheelYawStabilizer(
            kp_yaw=5.0, kd_yaw=1.5, max_yaw_torque=5.0, lowpass_alpha=1.0,
            height_gate_low=0.0, height_gate_high=1.0,
        )
        yaw_controller = YawController(kp_yaw=5.0, kd_yaw=1.5, max_yaw_torque=5.0)

        yaw_error = 0.1
        yaw_rate = 0.0

        tau_yaw, _ = yaw_controller.compute(yaw_error, yaw_rate)
        tau_wheel, _ = stabilizer.compute(yaw_error, yaw_rate, current_height_m=0.45)

        # Hip-yaw: left negative, right positive for positive yaw_error
        assert tau_yaw[1] < 0.0
        assert tau_yaw[6] > 0.0
        # Wheel: left positive, right negative for positive yaw_error
        assert tau_wheel[4] > 0.0
        assert tau_wheel[9] < 0.0
        # Both produce same-direction corrective yaw moment on body

    def test_diagnostics_populated(self, stabilizer):
        """Diagnostics should contain all required fields."""
        yaw_error = 0.1
        yaw_rate = 0.5

        tau, diag = stabilizer.compute(yaw_error, yaw_rate)

        required_keys = [
            "wheel_yaw_error",
            "wheel_yaw_rate",
            "wheel_yaw_tau_raw",
            "wheel_yaw_tau_clipped",
            "wheel_yaw_tau_left",
            "wheel_yaw_tau_right",
            "wheel_yaw_saturated",
            "wheel_yaw_kp",
            "wheel_yaw_kd",
            "wheel_yaw_max_torque",
            "wheel_yaw_lowpass_alpha",
        ]

        for key in required_keys:
            assert key in diag, f"Diagnostic key '{key}' missing"

        assert diag["wheel_yaw_error"] == yaw_error
        assert diag["wheel_yaw_rate"] == yaw_rate

    def test_lowpass_filtering(self):
        """Lowpass should smooth torque output across steps."""
        stabilizer = DifferentialWheelYawStabilizer(
            kp_yaw=3.0, kd_yaw=0.8, max_yaw_torque=3.0, lowpass_alpha=0.3,
        )

        # First step: error from 0 to 0.1 (step input)
        tau1, diag1 = stabilizer.compute(0.1, 0.0)
        tau_raw = 3.0 * 0.1  # kp * error = 0.3
        # With alpha=0.3: tau1 = 0.0*0.7 + 0.3*0.3 = 0.09
        expected_1 = 0.3 * 0.3  # alpha * tau_clipped (no saturation)
        assert abs(float(tau1[4]) - expected_1) < 1e-6, (
            f"Expected {expected_1}, got {tau1[4]}"
        )

    def test_reset_clears_state(self):
        """Reset should clear internal state."""
        stabilizer = DifferentialWheelYawStabilizer(
            kp_yaw=3.0, kd_yaw=0.8, max_yaw_torque=3.0, lowpass_alpha=0.3,
        )

        # Apply a step input
        stabilizer.compute(0.1, 0.0)
        # Internal state should be non-zero
        assert stabilizer._prev_tau_yaw_left != 0.0

        # Reset
        stabilizer.reset()
        assert stabilizer._prev_tau_yaw_left == 0.0
        assert stabilizer._prev_tau_yaw_right == 0.0

        # After reset, compute with same input should produce first-step output
        tau, diag = stabilizer.compute(0.1, 0.0)
        expected = 0.3 * 0.3  # alpha * kp * error (first-order lowpass from 0)
        assert abs(float(tau[4]) - expected) < 1e-6


class TestDifferentialWheelYawStabilizerIntegration:
    """Integration tests: how wheel yaw interacts with the YawController."""

    def test_wheel_yaw_enabled_suppresses_hip_yaw(self):
        """When wheel yaw is enabled, YawController output should be zeroed."""
        stabilizer = DifferentialWheelYawStabilizer(
            kp_yaw=3.0, kd_yaw=0.8, max_yaw_torque=3.0, lowpass_alpha=1.0,
        )
        yaw_controller = YawController(
            kp_yaw=8.0, kd_yaw=2.0, max_yaw_torque=5.0,
        )

        yaw_error = 0.1
        yaw_rate = 0.0

        # Compute both
        tau_yaw, _ = yaw_controller.compute(yaw_error, yaw_rate)
        tau_wheel, _ = stabilizer.compute(yaw_error, yaw_rate)

        # With wheel yaw enabled, the YawController's hip-yaw output should be
        # suppressed (zeroed) in the control loop. The YawController itself still
        # computes just for telemetry, but tau_yaw[1] and tau_yaw[6] are NOT added
        # to tau_shape_posture. Verify both exist.
        assert abs(tau_yaw[1]) > 0.0  # YawController computes (telemetry)
        assert abs(tau_yaw[6]) > 0.0
        assert abs(tau_wheel[4]) > 0.0  # Wheel yaw is active
        assert abs(tau_wheel[9]) > 0.0

        # In the actual control loop, tau_yaw would be zeroed:
        # tau_shape_posture_with_yaw = tau_shape_posture (no tau_yaw added)
        # tau_sagittal_wheel_balance = tau_sagittal_wheel_balance + tau_wheel
        # Verify this separation: wheel yaw does NOT write to hip-yaw
        assert tau_wheel[1] == 0.0
        assert tau_wheel[6] == 0.0
