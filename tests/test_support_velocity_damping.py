"""Tests for support position velocity damping in SagittalVelocityDampedBalanceController.

Step E position regulator fix: add explicit support-center velocity damping to prevent
transient position excursions during nominal standing.
"""

import pytest
import jax.numpy as jnp

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
)


class TestSupportVelocityDamping:
    """Test support position velocity damping term."""

    def test_support_velocity_computation_forward_drift(self):
        """Test support velocity is positive when position error increases (forward drift)."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            dt=0.01,
        )

        # First step: position error = 0.0m
        tau1, diag1 = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )
        assert diag1["support_position_velocity_m_s"] == 0.0

        # Second step: position error = 0.01m (forward drift)
        tau2, diag2 = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )
        # velocity = (0.01 - 0.0) / 0.01 = 1.0 m/s
        assert diag2["support_position_velocity_m_s"] == pytest.approx(1.0, abs=1e-6)

    def test_support_velocity_computation_backward_drift(self):
        """Test support velocity is negative when position error decreases (backward drift)."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            dt=0.01,
        )

        # First step: position error = 0.01m
        tau1, diag1 = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )

        # Second step: position error = 0.0m (backward drift)
        tau2, diag2 = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )
        # velocity = (0.0 - 0.01) / 0.01 = -1.0 m/s
        assert diag2["support_position_velocity_m_s"] == pytest.approx(-1.0, abs=1e-6)

    def test_tau_support_velocity_opposes_forward_drift(self):
        """Test tau_support_velocity is negative (opposes) when support drifts forward."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            dt=0.01,
        )

        # Initialize with zero
        controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        # Forward drift: position error increases
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )

        # velocity = +1.0 m/s (forward)
        # tau_support_velocity = -k_support_velocity * velocity = -10.0 * 1.0 = -10.0 Nm
        assert diag["support_position_velocity_m_s"] == pytest.approx(1.0, abs=1e-6)
        assert diag["tau_support_velocity"] == pytest.approx(-10.0, abs=1e-6)

    def test_tau_support_velocity_opposes_backward_drift(self):
        """Test tau_support_velocity is positive (opposes) when support drifts backward."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            dt=0.01,
        )

        # Initialize with forward position
        controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )

        # Backward drift: position error decreases
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        # velocity = -1.0 m/s (backward)
        # tau_support_velocity = -k_support_velocity * velocity = -10.0 * (-1.0) = +10.0 Nm
        assert diag["support_position_velocity_m_s"] == pytest.approx(-1.0, abs=1e-6)
        assert diag["tau_support_velocity"] == pytest.approx(10.0, abs=1e-6)

    def test_tau_support_velocity_zero_when_disabled(self):
        """Test tau_support_velocity is zero when k_support_velocity = 0.0."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=0.0,  # disabled
            dt=0.01,
        )

        # Initialize
        controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        # Forward drift
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )

        # velocity is computed but tau_support_velocity should be zero
        assert diag["support_position_velocity_m_s"] == pytest.approx(1.0, abs=1e-6)
        assert diag["tau_support_velocity"] == pytest.approx(0.0, abs=1e-6)

    def test_tau_support_velocity_included_in_total_torque(self):
        """Test tau_support_velocity is included in total wheel torque."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            k_position=0.0,  # disable other terms for isolation
            k_velocity=0.0,
            kp_pitch=0.0,
            kd_pitch=0.0,
            kp_cp=0.0,
            kd_com_vy=0.0,
            k_wheel_velocity=0.0,
            dt=0.01,
        )

        # Initialize
        controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        # Forward drift
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )

        # tau_support_velocity = -10.0 Nm
        # With all other terms disabled, total torque should equal tau_support_velocity
        assert diag["tau_support_velocity"] == pytest.approx(-10.0, abs=1e-6)
        assert diag["tau_left"] == pytest.approx(-10.0, abs=1e-6)
        assert diag["tau_right"] == pytest.approx(-10.0, abs=1e-6)
        assert float(tau[4]) == pytest.approx(-10.0, abs=1e-6)  # left wheel
        assert float(tau[9]) == pytest.approx(-10.0, abs=1e-6)  # right wheel

    def test_support_velocity_damping_with_position_control(self):
        """Test support velocity damping works alongside position control."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            k_position=20.0,
            max_position_tau=3.0,
            k_velocity=0.0,
            kp_pitch=0.0,
            kd_pitch=0.0,
            kp_cp=0.0,
            kd_com_vy=0.0,
            k_wheel_velocity=0.0,
            dt=0.01,
        )

        # Initialize
        controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        # Forward drift: position error = 0.01m
        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
        )

        # tau_position = -20.0 * 0.01 = -0.2 Nm
        # tau_support_velocity = -10.0 * 1.0 = -10.0 Nm
        # total = -0.2 + (-10.0) = -10.2 Nm
        assert diag["tau_position"] == pytest.approx(-0.2, abs=1e-6)
        assert diag["tau_support_velocity"] == pytest.approx(-10.0, abs=1e-6)
        assert diag["tau_left"] == pytest.approx(-10.2, abs=1e-6)
        assert diag["tau_right"] == pytest.approx(-10.2, abs=1e-6)

    def test_diagnostics_include_k_support_velocity(self):
        """Test diagnostics include k_support_velocity gain."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=15.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        assert "k_support_velocity" in diag
        assert diag["k_support_velocity"] == 15.0

    def test_no_wbc_no_legacy_sources(self):
        """Test controller does not use WBC or legacy torque sources."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.1,
            pitch_rate_x_rad_s=0.05,
            sagittal_velocity_m_s=0.02,
            wheel_vel_left_rad_s=1.0,
            wheel_vel_right_rad_s=1.0,
            sagittal_position_error_m=0.05,
        )

        # Controller should only output wheel torques
        # Leg joints should be zero
        assert float(tau[0]) == 0.0  # l_hip_roll
        assert float(tau[1]) == 0.0  # l_hip_yaw
        assert float(tau[2]) == 0.0  # l_hip_pitch
        assert float(tau[3]) == 0.0  # l_knee
        assert float(tau[5]) == 0.0  # r_hip_roll
        assert float(tau[6]) == 0.0  # r_hip_yaw
        assert float(tau[7]) == 0.0  # r_hip_pitch
        assert float(tau[8]) == 0.0  # r_knee

        # Wheel joints should be nonzero
        assert float(tau[4]) != 0.0  # l_wheel
        assert float(tau[9]) != 0.0  # r_wheel

    def test_kp_cp_remains_disabled(self):
        """Test kp_cp remains disabled (0.0) as required by Step E."""
        controller = SagittalVelocityDampedBalanceController(
            k_support_velocity=10.0,
            kp_cp=0.0,  # must remain disabled
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.05,
        )

        # tau_cp should be zero
        assert diag["tau_cp"] == 0.0
        assert controller.kp_cp == 0.0
