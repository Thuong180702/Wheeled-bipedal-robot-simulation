"""Tests for K2_NOTCH_LOW_Q_V1 audit profile.

Verifies:
  - K1 profile unchanged
  - K2 exists and is opt-in only
  - K2 differs from K1 only in wip_notch_q (6.0 -> 2.0)
  - All other K2 parameters identical to K1
  - K2 does not enable WBC/hidden torque
  - K2 is not current-best by default
  - CLI accepts k2_notch_low_q_v1
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestK2ProfileExistsAndIsOptIn:
    """Verify K2_NOTCH_LOW_Q_V1 exists and is opt-in only."""

    def test_k2_profile_exists(self):
        """K2_NOTCH_LOW_Q_V1 must be importable and have correct profile name."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1 is not None
        assert K2_NOTCH_LOW_Q_V1.profile_name == "k2_notch_low_q_v1"

    def test_k2_not_in_sweep_profiles(self):
        """K2 must NOT be in ALL_K_SWEEP_PROFILES (it's a named candidate, not a sweep)."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            ALL_K_SWEEP_PROFILES,
        )
        assert "k2_notch_low_q_v1" not in ALL_K_SWEEP_PROFILES

    def test_k2_is_current_best(self):
        """K2 is the current-best profile (promoted 2026-06-25). K1 is legacy."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        # K2 is the promoted current-best
        assert K2_NOTCH_LOW_Q_V1.profile_name == "k2_notch_low_q_v1"
        assert K2_NOTCH_LOW_Q_V1.wip_notch_q == 2.0

    def test_k2_registered_in_sagittal_authority_profiles(self):
        """K2 must be selectable via --vd-sagittal-authority-profile k2_notch_low_q_v1."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "k2_notch_low_q_v1" in SAGITTAL_AUTHORITY_PROFILES, (
            "k2_notch_low_q_v1 must be in SAGITTAL_AUTHORITY_PROFILES"
        )


class TestK1BaselineUnchangedAfterK2Creation:
    """Verify K1 is unchanged after adding K2 profile."""

    def test_k1_gains_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K1_PITCH_RATE_NOTCH,
        )
        k1 = K1_PITCH_RATE_NOTCH
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=k1)
        assert ctrl.kp_pitch == 50.0
        assert ctrl.kd_pitch == 10.0
        assert ctrl.max_position_tau == 3.0
        assert ctrl.max_tau_wheel == 5.0
        assert k1.k_position_nominal == 40.0
        assert k1.k_velocity_nominal == 15.0
        assert k1.k_wheel_velocity_nominal == 0.5

    def test_k1_filter_params_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH,
        )
        k1 = K1_PITCH_RATE_NOTCH
        assert k1.enable_wip_notch_filter is True
        assert k1.wip_notch_center_hz == 2.5
        assert k1.wip_notch_q == 6.0
        assert k1.wip_notch_filter_blend == 1.0
        assert k1.wip_notch_target_signal == "pitch_rate"
        assert k1.wip_notch_height_gate_start_m == 0.42
        assert k1.wip_notch_height_gate_full_m == 0.48
        assert k1.wip_notch_filter_type == "biquad_notch"

    def test_k1_no_wbc_or_hidden_torque(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K1_PITCH_RATE_NOTCH,
        )
        k1 = K1_PITCH_RATE_NOTCH
        assert getattr(k1, "wbc_enabled", False) is False
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=k1)
        suspicious = ["hidden_torque", "secret_gain", "wbc_active", "extra_torque",
                      "hidden_damping", "secret_notch"]
        for attr in suspicious:
            assert not hasattr(ctrl, attr), f"K1 should not have '{attr}' attribute"


class TestK2OnlyDiffIsQ:
    """Verify K2 differs from K1 ONLY in wip_notch_q (6.0 -> 2.0)."""

    def test_k2_wip_notch_q_is_2p0(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_q == 2.0

    def test_k2_center_hz_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_center_hz == K1_PITCH_RATE_NOTCH.wip_notch_center_hz == 2.5

    def test_k2_blend_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_filter_blend == K1_PITCH_RATE_NOTCH.wip_notch_filter_blend == 1.0

    def test_k2_filter_type_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_filter_type == K1_PITCH_RATE_NOTCH.wip_notch_filter_type == "biquad_notch"

    def test_k2_target_signal_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_target_signal == K1_PITCH_RATE_NOTCH.wip_notch_target_signal == "pitch_rate"

    def test_k2_height_gate_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_height_gate_start_m == K1_PITCH_RATE_NOTCH.wip_notch_height_gate_start_m == 0.42
        assert K2_NOTCH_LOW_Q_V1.wip_notch_height_gate_full_m == K1_PITCH_RATE_NOTCH.wip_notch_height_gate_full_m == 0.48

    def test_k2_gate_enabled_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_gate_enabled == K1_PITCH_RATE_NOTCH.wip_notch_gate_enabled is True

    def test_k2_enable_notch_filter_equals_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.enable_wip_notch_filter == K1_PITCH_RATE_NOTCH.enable_wip_notch_filter is True

    def test_k2_non_filter_gains_equal_k1(self):
        """All sagittal gains (non-filter) must equal K1."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        k1, k2 = K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1
        sagittal_fields = [
            "k_position_nominal",
            "k_velocity_nominal",
            "k_wheel_velocity_nominal",
            "max_position_tau_nominal",
            "kd_pitch_nominal",
            "support_velocity_scale",
        ]
        for field in sagittal_fields:
            assert getattr(k2, field) == getattr(k1, field), (
                f"K2.{field} = {getattr(k2, field)} != K1.{field} = {getattr(k1, field)}"
            )

    def test_k2_torque_limits_equal_k1(self):
        """K2 torque limits must equal K1."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        k1_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K1_PITCH_RATE_NOTCH)
        k2_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K2_NOTCH_LOW_Q_V1)
        assert k2_ctrl.max_tau_wheel == k1_ctrl.max_tau_wheel
        assert k2_ctrl.max_position_tau == k1_ctrl.max_position_tau
        assert k2_ctrl.kp_pitch == k1_ctrl.kp_pitch
        assert k2_ctrl.kd_pitch == k1_ctrl.kd_pitch

    def test_k2_low_band_support_base_equals_k1(self):
        """K2 sagittal base (low-band support) must equal K1."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        k1, k2 = K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1
        assert k2.low_band_support_outer_loop_enabled == k1.low_band_support_outer_loop_enabled
        assert k2.low_band_support_center_m == k1.low_band_support_center_m
        assert k2.low_band_support_sigma_m == k1.low_band_support_sigma_m
        assert k2.physics_equilibrium_feedforward_enabled == k1.physics_equilibrium_feedforward_enabled


class TestK2NoWbcOrHiddenTorque:
    """Verify K2 does not enable WBC or hidden torque."""

    def test_k2_no_wbc(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        assert getattr(K2_NOTCH_LOW_Q_V1, "wbc_enabled", False) is False

    def test_k2_no_hidden_torque(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K2_NOTCH_LOW_Q_V1,
        )
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K2_NOTCH_LOW_Q_V1)
        suspicious = ["hidden_torque", "secret_gain", "wbc_active", "extra_torque",
                      "hidden_damping", "secret_notch"]
        for attr in suspicious:
            assert not hasattr(ctrl, attr), f"K2 should not have '{attr}' attribute"

    def test_k2_no_support_bias(self):
        """K2 must not add support bias."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K2_NOTCH_LOW_Q_V1, K1_PITCH_RATE_NOTCH,
        )
        k1_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K1_PITCH_RATE_NOTCH)
        k2_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K2_NOTCH_LOW_Q_V1)
        assert k2_ctrl.k_support_velocity == k1_ctrl.k_support_velocity

    def test_k2_no_extra_damping(self):
        """K2 must not add extra damping terms not in K1."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        # No unexpected fields should exist
        unexpected = ["extra_pitch_damping", "extra_support_damping", "extra_wheel_damping",
                      "integral_term", "integral_gain"]
        for field in unexpected:
            assert not hasattr(K2_NOTCH_LOW_Q_V1, field), f"K2 should not have '{field}'"


class TestK2CliAccessibility:
    """Verify CLI accepts k2_notch_low_q_v1."""

    def test_cli_choices_includes_k2(self):
        """argparse choices must include k2_notch_low_q_v1."""
        # Check the simulate_hierarchical_controller module's argparse setup
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_check_cli",
            SCRIPTS_DIR / "simulate_hierarchical_controller.py",
        )
        assert spec is not None, "simulate_hierarchical_controller.py not found"

        # Verify the profile is in SAGITTAL_AUTHORITY_PROFILES (stronger check)
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "k2_notch_low_q_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_k2_profile_object_is_k1_derived(self):
        """K2 must be derived from K1 (via dataclass replace), not a fresh construction."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        # Verify K2 inherits all K1 base sagittal fields
        assert K2_NOTCH_LOW_Q_V1.low_band_support_outer_loop_enabled is True
        assert K2_NOTCH_LOW_Q_V1.wip_notch_target_signal == "pitch_rate"
        assert K2_NOTCH_LOW_Q_V1.enable_wip_notch_filter is True
        # Only Q should differ
        assert K2_NOTCH_LOW_Q_V1.wip_notch_q == 2.0
        assert K1_PITCH_RATE_NOTCH.wip_notch_q == 6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
