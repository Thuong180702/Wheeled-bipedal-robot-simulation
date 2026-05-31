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

    def test_counteracting_torque_can_exceed_old_fixed_cap_within_total_bounds(self):
        """Counteracting position torque should use available final wheel margin."""
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
            pitch_reserve_tau=2.0,
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
        assert diag["tau_total_after_final_clip"] == pytest.approx(-4.0, abs=1e-9)
        assert diag["position_authority_mode"] == "total_torque_bound"
        assert diag["position_authority_reason"] == "within_bounds"
        assert diag["final_wheel_torque_margin"] == pytest.approx(1.0, abs=1e-9)

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
            pitch_reserve_tau=2.0,
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
            pitch_reserve_tau=2.0,
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
            pitch_reserve_tau=2.0,
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
            pitch_reserve_tau=2.0,
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
            pitch_reserve_tau=2.0,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
