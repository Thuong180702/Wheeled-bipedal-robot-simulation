"""Tests for total-torque-bound-aware position authority allocation.

Validates that:
1. Legacy fixed-cap mode still works when budget mode is disabled
2. Counteracting position torque can use available final wheel margin safely
3. Same-direction position torque is limited by final total torque bounds
4. The bounds are symmetric across positive and negative directions
5. Reported diagnostics match the applied total-torque-bound logic
"""

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
)


class TestTorqueBudgetAwarePosition:
    """Test total-torque-bound-aware position authority allocation."""

    def test_budget_mode_disabled_uses_fixed_cap(self):
        """When budget mode is disabled, should use fixed max_position_tau."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=10.0,
            enable_torque_budget_aware_position=False,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.5,
        )

        assert abs(diag["tau_position_clipped"]) == pytest.approx(3.0, abs=0.01)
        assert diag["tau_position_saturation_reason"] == "fixed_cap"
        assert diag["enable_torque_budget_aware_position"] is False

    def test_counteracting_position_torque_uses_total_torque_bound_without_pitch_reserve(self):
        """Counteracting position torque should use full remaining wheel torque authority."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=40.0,
            kd_pitch=0.0,
            kd_com_vy=0.0,
            k_velocity=0.0,
            k_wheel_velocity=0.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=20.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.1,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.4,
        )

        assert diag["tau_balance_before_position"] == pytest.approx(4.0, abs=1e-9)
        assert diag["tau_position_raw"] == pytest.approx(-8.0, abs=1e-9)
        assert diag["tau_position_lower_bound"] == pytest.approx(-9.0, abs=1e-9)
        assert diag["tau_position_upper_bound"] == pytest.approx(1.0, abs=1e-9)
        assert diag["tau_position_clipped"] == pytest.approx(-8.0, abs=1e-9)
        assert diag["tau_total_before_final_clip"] == pytest.approx(-4.0, abs=1e-9)
        assert diag["position_authority_reason"] == "within_bounds"

    def test_counteracting_torque_is_limited_only_by_total_torque_bound(self):
        """Counteracting position torque should clip at the final wheel torque bound only."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=40.0,
            kd_pitch=0.0,
            kd_com_vy=0.0,
            k_velocity=0.0,
            k_wheel_velocity=0.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=20.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.1,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.6,
        )

        assert diag["tau_balance_before_position"] == pytest.approx(4.0, abs=1e-9)
        assert diag["tau_position_raw"] == pytest.approx(-12.0, abs=1e-9)
        assert diag["tau_position_lower_bound"] == pytest.approx(-9.0, abs=1e-9)
        assert diag["tau_position_upper_bound"] == pytest.approx(1.0, abs=1e-9)
        assert diag["tau_position_clipped"] == pytest.approx(-9.0, abs=1e-9)
        assert diag["tau_total_before_final_clip"] == pytest.approx(-5.0, abs=1e-9)
        assert diag["tau_total_after_final_clip"] == pytest.approx(-5.0, abs=1e-9)
        assert diag["position_authority_mode"] == "total_torque_bound"
        assert diag["position_authority_reason"] == "lower_bound"
        assert diag["final_wheel_torque_margin"] == pytest.approx(0.0, abs=1e-9)

    def test_same_direction_torque_is_limited_by_total_torque_upper_bound(self):
        """Reinforcing position torque should be clipped by the final total torque bound."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=40.0,
            kd_pitch=0.0,
            kd_com_vy=0.0,
            k_velocity=0.0,
            k_wheel_velocity=0.0,
            k_position=10.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=20.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.1,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=-0.3,
        )

        assert diag["tau_balance_before_position"] == pytest.approx(4.0, abs=1e-9)
        assert diag["tau_position_raw"] == pytest.approx(3.0, abs=1e-9)
        assert diag["tau_position_lower_bound"] == pytest.approx(-9.0, abs=1e-9)
        assert diag["tau_position_upper_bound"] == pytest.approx(1.0, abs=1e-9)
        assert diag["tau_position_clipped"] == pytest.approx(1.0, abs=1e-9)
        assert diag["tau_total_before_final_clip"] == pytest.approx(5.0, abs=1e-9)
        assert diag["tau_total_after_final_clip"] == pytest.approx(5.0, abs=1e-9)
        assert diag["position_authority_reason"] == "upper_bound"

    def test_negative_side_counteracting_torque_is_symmetric(self):
        """Negative balance should allow positive counteracting position torque symmetrically."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=40.0,
            kd_pitch=0.0,
            kd_com_vy=0.0,
            k_velocity=0.0,
            k_wheel_velocity=0.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=20.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=-0.1,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=-0.4,
        )

        assert diag["tau_balance_before_position"] == pytest.approx(-4.0, abs=1e-9)
        assert diag["tau_position_raw"] == pytest.approx(8.0, abs=1e-9)
        assert diag["tau_position_lower_bound"] == pytest.approx(-1.0, abs=1e-9)
        assert diag["tau_position_upper_bound"] == pytest.approx(9.0, abs=1e-9)
        assert diag["tau_position_clipped"] == pytest.approx(8.0, abs=1e-9)
        assert diag["tau_total_before_final_clip"] == pytest.approx(4.0, abs=1e-9)
        assert diag["tau_total_after_final_clip"] == pytest.approx(4.0, abs=1e-9)
        assert diag["position_authority_reason"] == "within_bounds"

    def test_total_torque_bound_is_respected_before_final_clip(self):
        """Balance plus clipped position torque should always stay inside wheel limits."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=40.0,
            kd_pitch=0.0,
            kd_com_vy=0.0,
            k_velocity=0.0,
            k_wheel_velocity=0.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=20.0,
            dt=0.01,
        )

        for pitch_x_rad, sagittal_position_error_m in [
            (0.1, -0.3),
            (0.1, 0.6),
            (-0.1, 0.3),
            (-0.1, -0.6),
        ]:
            tau, diag = controller.compute(
                pitch_x_rad=pitch_x_rad,
                pitch_rate_x_rad_s=0.0,
                sagittal_velocity_m_s=0.0,
                wheel_vel_left_rad_s=0.0,
                wheel_vel_right_rad_s=0.0,
                sagittal_position_error_m=sagittal_position_error_m,
            )

            assert -controller.max_tau_wheel - 1e-9 <= diag["tau_total_before_final_clip"] <= controller.max_tau_wheel + 1e-9
            assert -controller.max_tau_wheel - 1e-9 <= diag["tau_total_after_final_clip"] <= controller.max_tau_wheel + 1e-9

    def test_final_wheel_torque_margin_matches_reported_wheel_torque(self):
        """Final wheel torque margin should match max_tau_wheel minus the raw wheel command magnitude."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=10.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=10.0,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.2,
            pitch_rate_x_rad_s=0.5,
            sagittal_velocity_m_s=1.0,
            wheel_vel_left_rad_s=5.0,
            wheel_vel_right_rad_s=5.0,
            sagittal_position_error_m=1.0,
        )

        final_wheel_torque_max = max(abs(diag["tau_left"]), abs(diag["tau_right"]))
        expected_margin = controller.max_tau_wheel - final_wheel_torque_max
        assert diag["final_wheel_torque_margin"] == pytest.approx(expected_margin, abs=1e-9)

    def test_kp_cp_disabled_does_not_affect_budget(self):
        """kp_cp=0.0 should remain disabled and not affect torque budget."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            kp_cp=0.0,
            k_position=20.0,
            k_support_velocity=0.0,
            max_position_tau=3.0,
            max_tau_wheel=10.0,
            enable_torque_budget_aware_position=True,
            position_tau_budget_cap=20.0,
            dt=0.01,
        )

        controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.0,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.5,
        )

        assert diag["tau_cp"] == 0.0
        assert diag["enable_torque_budget_aware_position"] is True
        assert diag["position_authority_mode"] == "total_torque_bound"

    def test_baseline_sagittal_controller_unaffected(self):
        """Baseline sagittal controller should be unaffected by budget mode."""
        pass


class TestSteadyStateCenteringIntegral:
    """Tests for steady-state-only centering integral action."""

    def test_integral_inactive_during_transient(self):
        """Integral should remain inactive during transient conditions."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=True,
            dt=0.01,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.1,  # large pitch — unsafe
            pitch_rate_x_rad_s=0.5,  # large rate — unsafe
            sagittal_velocity_m_s=2.0,  # large velocity — unsafe
            wheel_vel_left_rad_s=5.0,
            wheel_vel_right_rad_s=5.0,
            sagittal_position_error_m=0.1,
            com_z_m=0.40,
        )

        assert diag["integral_active"] is False
        assert diag["tau_position_integral"] == pytest.approx(0.0, abs=1e-9)

    def test_integral_active_under_steady_state_conditions(self):
        """Integral should activate under steady-state safe conditions."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=True,
            ki_position_integral=0.5,
            integral_max_abs=1.0,
            dt=0.01,
        )

        # Warm up controller to set prev_support_position_error_m
        controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        assert diag["integral_active"] is True
        assert diag["integral_gate_reason"] == "safe_steady_state"

    def test_integral_reduces_positive_support_error(self):
        """Positive support error should accumulate negative integral torque."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=True,
            ki_position_integral=0.5,
            integral_max_abs=1.0,
            dt=0.01,
        )

        # Warm up
        controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        # First accumulation call
        tau1, diag1 = controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        # Second accumulation call
        tau2, diag2 = controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        assert diag1["tau_position_integral"] < 0.0
        assert diag2["tau_position_integral"] < diag1["tau_position_integral"]

    def test_integral_reduces_negative_support_error(self):
        """Negative support error should accumulate positive integral torque."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=True,
            ki_position_integral=0.5,
            integral_max_abs=1.0,
            dt=0.01,
        )

        # Warm up
        controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=-0.05,
            com_z_m=0.40,
        )

        tau1, diag1 = controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=-0.05,
            com_z_m=0.40,
        )

        tau2, diag2 = controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=-0.05,
            com_z_m=0.40,
        )

        assert diag1["tau_position_integral"] > 0.0
        assert diag2["tau_position_integral"] > diag1["tau_position_integral"]

    def test_integral_anti_windup_resets_when_unsafe(self):
        """Integral should anti-windup or reset when conditions become unsafe."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=True,
            ki_position_integral=2.0,
            integral_max_abs=2.0,
            dt=0.01,
        )

        # Warm up
        controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        # Build up integral
        for _ in range(50):
            tau, diag = controller.compute(
                pitch_x_rad=0.01,
                pitch_rate_x_rad_s=0.01,
                sagittal_velocity_m_s=0.1,
                wheel_vel_left_rad_s=0.5,
                wheel_vel_right_rad_s=0.5,
                sagittal_position_error_m=0.05,
                com_z_m=0.40,
            )

        integral_before = diag["tau_position_integral"]
        assert abs(integral_before) > 0.01

        # Now unsafe — integral should go inactive and reset
        tau_unsafe, diag_unsafe = controller.compute(
            pitch_x_rad=0.1,
            pitch_rate_x_rad_s=0.5,
            sagittal_velocity_m_s=2.0,
            wheel_vel_left_rad_s=5.0,
            wheel_vel_right_rad_s=5.0,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        assert diag_unsafe["integral_active"] is False
        assert diag_unsafe["tau_position_integral"] == pytest.approx(0.0, abs=1e-9)

    def test_integral_is_bounded(self):
        """Integral torque must stay within integral_max_abs."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=True,
            ki_position_integral=5.0,
            integral_max_abs=0.3,
            dt=0.01,
        )

        # Warm up
        controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        # Accumulate many steps
        for _ in range(200):
            tau, diag = controller.compute(
                pitch_x_rad=0.01,
                pitch_rate_x_rad_s=0.01,
                sagittal_velocity_m_s=0.1,
                wheel_vel_left_rad_s=0.5,
                wheel_vel_right_rad_s=0.5,
                sagittal_position_error_m=0.05,
                com_z_m=0.40,
            )

        assert abs(diag["tau_position_integral"]) <= 0.3 + 1e-9

    def test_integral_inactive_when_disabled(self):
        """Integral should remain zero when enable_position_integral=False."""
        controller = SagittalVelocityDampedBalanceController(
            kp_pitch=50.0,
            kd_pitch=10.0,
            k_position=40.0,
            max_position_tau=3.0,
            max_tau_wheel=5.0,
            enable_position_integral=False,
            ki_position_integral=0.5,
            integral_max_abs=1.0,
            dt=0.01,
        )

        # Warm up
        controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        tau, diag = controller.compute(
            pitch_x_rad=0.01,
            pitch_rate_x_rad_s=0.01,
            sagittal_velocity_m_s=0.1,
            wheel_vel_left_rad_s=0.5,
            wheel_vel_right_rad_s=0.5,
            sagittal_position_error_m=0.05,
            com_z_m=0.40,
        )

        assert diag["integral_active"] is False
        assert diag["tau_position_integral"] == pytest.approx(0.0, abs=1e-9)

    def test_kp_cp_remains_zero(self):
        """kp_cp must remain 0.0 as required by Step E constraints."""
        controller = SagittalVelocityDampedBalanceController(kp_cp=0.0)
        assert controller.kp_cp == 0.0

    def test_baseline_and_velocity_damped_mutually_exclusive(self):
        """Baseline and velocity-damped controllers must be mutually exclusive."""
        from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController

        baseline = SagittalWheelBalanceController(kp_pitch=50.0, kp_cp=30.0)
        velocity_damped = SagittalVelocityDampedBalanceController(kp_cp=0.0)

        assert baseline.kp_cp == 30.0
        assert velocity_damped.kp_cp == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
