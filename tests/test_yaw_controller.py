"""Tests for yaw controller functionality."""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.yaw_controller import YawController


class TestYawController:
    """Test yaw controller antisymmetric torque generation."""

    @pytest.fixture
    def controller(self):
        """Create yaw controller with standard gains."""
        return YawController(
            kp_yaw=5.0,
            kd_yaw=1.0,
            max_yaw_torque=3.0,
        )

    def test_positive_yaw_error_applies_antisymmetric_torque(self, controller):
        """Positive yaw error (robot yawed right of reference) should apply corrective torque."""
        yaw_error = 0.1  # rad, robot needs to yaw left (counter-clockwise)
        yaw_rate = 0.0

        tau, diag = controller.compute(yaw_error, yaw_rate)

        # Antisymmetric torque: left and right have opposite signs
        # Positive yaw error needs positive yaw moment (CCW viewed from above)
        # This requires: left negative, right positive
        assert tau[1] < 0.0, f"Left hip-yaw should be negative for positive yaw error, got {tau[1]}"
        assert tau[6] > 0.0, f"Right hip-yaw should be positive for positive yaw error, got {tau[6]}"
        assert abs(tau[1] + tau[6]) < 1e-6, f"Torques should be antisymmetric: left={tau[1]}, right={tau[6]}"

    def test_negative_yaw_error_applies_antisymmetric_torque(self, controller):
        """Negative yaw error (robot yawed left of reference) should apply corrective torque."""
        yaw_error = -0.1  # rad, robot needs to yaw right (clockwise)
        yaw_rate = 0.0

        tau, diag = controller.compute(yaw_error, yaw_rate)

        # Negative yaw error needs negative yaw moment (CW viewed from above)
        # This requires: left positive, right negative
        assert tau[1] > 0.0, f"Left hip-yaw should be positive for negative yaw error, got {tau[1]}"
        assert tau[6] < 0.0, f"Right hip-yaw should be negative for negative yaw error, got {tau[6]}"
        assert abs(tau[1] + tau[6]) < 1e-6, f"Torques should be antisymmetric"

    def test_yaw_rate_damping(self, controller):
        """Yaw rate damping should oppose yaw velocity."""
        yaw_error = 0.0
        yaw_rate = 1.0  # rad/s, spinning CCW

        tau, diag = controller.compute(yaw_error, yaw_rate)

        # Positive yaw rate (CCW) needs negative yaw moment (CW) to slow down
        # This requires: left positive, right negative
        assert tau[1] > 0.0, "Damping should oppose positive yaw rate"
        assert tau[6] < 0.0, "Damping should oppose positive yaw rate"

    def test_torque_saturation(self, controller):
        """Yaw torque should saturate at max_yaw_torque."""
        yaw_error = 10.0  # Very large error
        yaw_rate = 0.0

        tau, diag = controller.compute(yaw_error, yaw_rate)

        # Should saturate at max_yaw_torque
        assert abs(tau[1]) <= controller.max_yaw_torque + 1e-6, "Left torque should saturate"
        assert abs(tau[6]) <= controller.max_yaw_torque + 1e-6, "Right torque should saturate"
        assert diag["yaw_saturated"], "Should report saturation"

    def test_zero_yaw_error_and_rate_produces_zero_torque(self, controller):
        """Zero yaw error and rate should produce zero torque."""
        yaw_error = 0.0
        yaw_rate = 0.0

        tau, diag = controller.compute(yaw_error, yaw_rate)

        assert tau[1] == 0.0, "Zero error/rate should produce zero torque"
        assert tau[6] == 0.0, "Zero error/rate should produce zero torque"

    def test_only_hip_yaw_joints_actuated(self, controller):
        """Yaw controller should only actuate hip-yaw joints [1, 6]."""
        yaw_error = 0.1
        yaw_rate = 0.0

        tau, diag = controller.compute(yaw_error, yaw_rate)

        # All other joints should be zero
        for idx in [0, 2, 3, 4, 5, 7, 8, 9]:
            assert tau[idx] == 0.0, f"Joint {idx} should not be actuated by yaw controller"

    def test_torque_proportional_to_error(self, controller):
        """Torque magnitude should be proportional to error magnitude."""
        yaw_rate = 0.0

        tau_small, _ = controller.compute(0.05, yaw_rate)
        tau_large, _ = controller.compute(0.20, yaw_rate)

        # Larger error should produce larger torque
        assert abs(tau_large[1]) > abs(tau_small[1]), "Larger error should produce larger torque"
        assert abs(tau_large[6]) > abs(tau_small[6]), "Larger error should produce larger torque"

    def test_diagnostics_populated(self, controller):
        """Diagnostics should contain all required fields."""
        yaw_error = 0.1
        yaw_rate = 0.5

        tau, diag = controller.compute(yaw_error, yaw_rate)

        required_keys = [
            "yaw_error",
            "yaw_rate",
            "tau_yaw_antisym_raw",
            "tau_yaw_antisym",
            "tau_yaw_left",
            "tau_yaw_right",
            "yaw_saturated",
            "kp_yaw",
            "kd_yaw",
            "max_yaw_torque",
        ]

        for key in required_keys:
            assert key in diag, f"Diagnostic key '{key}' missing"

        # Verify values
        assert diag["yaw_error"] == yaw_error
        assert diag["yaw_rate"] == yaw_rate
        assert diag["tau_yaw_left"] == float(tau[1])
        assert diag["tau_yaw_right"] == float(tau[6])
