"""Tests for K2 best-current promotion validation.

Verifies:
  - K2 profile remains q=2.0 (not modified during promotion)
  - K1 profile remains q=6.0 (available as legacy)
  - No hidden torque/WBC in K2
  - No threshold relaxation
  - K1 and K2 profiles are both selectable via CLI
  - Report path exists
  - Step C/E validation outputs present
  - Step D evidence outputs present
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestK2ProfileIntegrityAfterPromotion:
    """K2 profile must remain q=2.0 after promotion — no parameter changes."""

    def test_k2_wip_notch_q_remains_2p0(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.wip_notch_q == 2.0, (
            "K2 wip_notch_q must remain 2.0 — do not modify K2"
        )

    def test_k2_all_filter_params_unchanged(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        k2 = K2_NOTCH_LOW_Q_V1
        assert k2.wip_notch_center_hz == 2.5
        assert k2.wip_notch_q == 2.0
        assert k2.wip_notch_filter_blend == 1.0
        assert k2.wip_notch_target_signal == "pitch_rate"
        assert k2.wip_notch_height_gate_start_m == 0.42
        assert k2.wip_notch_height_gate_full_m == 0.48
        assert k2.wip_notch_filter_type == "biquad_notch"
        assert k2.enable_wip_notch_filter is True
        assert k2.wip_notch_gate_enabled is True

    def test_k2_no_wbc_or_hidden_torque(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K2_NOTCH_LOW_Q_V1,
        )
        assert getattr(K2_NOTCH_LOW_Q_V1, "wbc_enabled", False) is False
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K2_NOTCH_LOW_Q_V1)
        suspicious = ["hidden_torque", "secret_gain", "wbc_active", "extra_torque",
                      "hidden_damping", "secret_notch", "integral_term"]
        for attr in suspicious:
            assert not hasattr(ctrl, attr), f"K2 should not have '{attr}' attribute"

    def test_k2_no_threshold_relaxation(self):
        """K2 must use same thresholds/gates as K1."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        # Height gate thresholds
        assert K2_NOTCH_LOW_Q_V1.wip_notch_height_gate_start_m == K1_PITCH_RATE_NOTCH.wip_notch_height_gate_start_m
        assert K2_NOTCH_LOW_Q_V1.wip_notch_height_gate_full_m == K1_PITCH_RATE_NOTCH.wip_notch_height_gate_full_m
        # Sagittal gains
        assert K2_NOTCH_LOW_Q_V1.k_position_nominal == K1_PITCH_RATE_NOTCH.k_position_nominal
        assert K2_NOTCH_LOW_Q_V1.k_velocity_nominal == K1_PITCH_RATE_NOTCH.k_velocity_nominal

    def test_k2_differs_from_k1_only_in_q(self):
        """K2 must differ from K1 ONLY in wip_notch_q and profile_name."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        k1, k2 = K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1
        # Fields that may differ intentionally
        allowed_diffs = {"wip_notch_q", "profile_name"}
        # Compare all common fields
        k1_fields = {f.name for f in type(k1).__dataclass_fields__.values()}
        for field_name in sorted(k1_fields):
            if field_name in allowed_diffs:
                continue
            k1_val = getattr(k1, field_name)
            k2_val = getattr(k2, field_name)
            assert k1_val == k2_val, (
                f"K2.{field_name} = {k2_val!r} != K1.{field_name} = {k1_val!r}"
            )


class TestK1ProfileIntegrityAfterPromotion:
    """K1 must remain available as legacy — no changes or deletions."""

    def test_k1_wip_notch_q_remains_6p0(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH,
        )
        assert K1_PITCH_RATE_NOTCH.wip_notch_q == 6.0, (
            "K1 wip_notch_q must remain 6.0 — do not modify K1"
        )

    def test_k1_profile_exists_and_selectable(self):
        """K1 must remain available in SAGITTAL_AUTHORITY_PROFILES."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "k1_pitch_rate_notch_v1" in SAGITTAL_AUTHORITY_PROFILES, (
            "K1 must remain selectable as legacy profile"
        )

    def test_k1_unchanged_from_baseline(self):
        """K1 must match original specification."""
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
        assert k1.low_band_support_outer_loop_enabled is True
        assert k1.physics_equilibrium_feedforward_enabled is True


class TestK2ProfileSelectable:
    """K2 must be selectable via CLI."""

    def test_k2_in_sagittal_authority_profiles(self):
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "k2_notch_low_q_v1" in SAGITTAL_AUTHORITY_PROFILES, (
            "k2_notch_low_q_v1 must be selectable via --vd-sagittal-authority-profile"
        )

    def test_k1_in_sagittal_authority_profiles(self):
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "k1_pitch_rate_notch_v1" in SAGITTAL_AUTHORITY_PROFILES, (
            "k1_pitch_rate_notch_v1 must remain selectable as legacy"
        )

    def test_k2_resolves_to_correct_object(self):
        """K2 profile must be the K2_NOTCH_LOW_Q_V1 object."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        resolved = SAGITTAL_AUTHORITY_PROFILES["k2_notch_low_q_v1"]
        assert resolved.profile_name == "k2_notch_low_q_v1"
        assert resolved.wip_notch_q == 2.0


class TestValidationOutputsExist:
    """Verify validation outputs are present."""

    def test_step_d_report_exists(self):
        report = PROJECT_ROOT / "docs" / "validation" / "k2_step_d_push_matrix_validation_report.md"
        assert report.exists(), f"Step D report missing: {report}"

    def test_step_d_json_summary_exists(self):
        summary = PROJECT_ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "k2_step_d_push_matrix_summary.json"
        assert summary.exists(), f"Step D JSON summary missing: {summary}"

    def test_k2_notch_low_q_v1_create_report_exists(self):
        report = PROJECT_ROOT / "docs" / "validation" / "k2_notch_low_q_v1_create_and_validate_report.md"
        assert report.exists(), f"K2 create/validate report missing: {report}"

    def test_step_c_e_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "validate_k2_step_c_e_fixed_height.py"
        assert script.exists(), f"Step C/E validation script missing: {script}"


class TestNoWbcHiddenTorqueOrRelaxation:
    """Verify no controller-level violations in K2."""

    def test_k2_controller_no_extra_attributes(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K2_NOTCH_LOW_Q_V1,
        )
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K2_NOTCH_LOW_Q_V1)
        # Check all public attributes for suspicious patterns
        forbidden_prefixes = ["wbc_", "hidden_", "secret_", "extra_", "relaxed_"]
        for attr_name in dir(ctrl):
            for prefix in forbidden_prefixes:
                assert not attr_name.startswith(prefix), (
                    f"K2 controller has forbidden attribute: {attr_name}"
                )

    def test_k2_sagittal_gains_equal_k1(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            SagittalVelocityDampedBalanceController,
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        k1_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K1_PITCH_RATE_NOTCH)
        k2_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=K2_NOTCH_LOW_Q_V1)
        assert k2_ctrl.kp_pitch == k1_ctrl.kp_pitch == 50.0
        assert k2_ctrl.kd_pitch == k1_ctrl.kd_pitch == 10.0
        assert k2_ctrl.max_position_tau == k1_ctrl.max_position_tau
        assert k2_ctrl.max_tau_wheel == k1_ctrl.max_tau_wheel


class TestCompileChecks:
    """Verify key files compile cleanly."""

    def test_controller_file_compiles(self):
        import py_compile
        result = py_compile.compile(
            str(PROJECT_ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"),
            doraise=True,
        )
        assert result is not None

    def test_simulate_script_compiles(self):
        import py_compile
        result = py_compile.compile(
            str(PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"),
            doraise=True,
        )
        assert result is not None

    def test_step_c_e_script_compiles(self):
        import py_compile
        result = py_compile.compile(
            str(PROJECT_ROOT / "scripts" / "validate_k2_step_c_e_fixed_height.py"),
            doraise=True,
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
