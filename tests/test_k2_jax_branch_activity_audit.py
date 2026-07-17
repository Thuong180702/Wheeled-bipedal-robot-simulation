"""Stage 4: K2 branch activity audit.

Verifies that:
- DISABLED_INACTIVE branches never execute under K2
- ENABLED_ACTIVE branches are confirmed active
- No UNEXPECTED_ACTIVE branches
"""

import pytest


class TestK2BranchActivityAudit:
    """Verify K2 profile enables only the expected strategy branches."""

    def test_disabled_strategies_inactive(self):
        """All strategies disabled by K2 profile remain disabled."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        s = K2_NOTCH_LOW_Q_V1

        # These MUST be False (disabled) for K2
        disabled_must_be_false = [
            ("enable_unified_sagittal_state_feedback", s.enable_unified_sagittal_state_feedback),
            ("enable_active_pitch_crossing", s.enable_active_pitch_crossing),
            ("enable_phase_aware_recenter", s.enable_phase_aware_recenter),
            ("enable_hysteresis_recenter", s.enable_hysteresis_recenter),
            ("enable_bias_cancel", s.enable_bias_cancel),
            ("enable_coordinated_sagittal_feedback", s.enable_coordinated_sagittal_feedback),
            ("enable_early_zero_crossing_recenter", s.enable_early_zero_crossing_recenter),
            ("enable_lp_priority_allocator", s.enable_lp_priority_allocator),
            ("enable_lr_replacement_feedback", s.enable_lr_replacement_feedback),
            ("enable_pitch_aware_position_scaling", s.enable_pitch_aware_position_scaling),
            ("enable_position_integral", s.enable_position_integral),
            ("enable_zero_crossing_recenter", s.enable_zero_crossing_recenter),
            ("enable_body_yaw_wheel_stabilization", s.enable_body_yaw_wheel_stabilization),
        ]
        for name, value in disabled_must_be_false:
            assert not value, f"DISABLED_INACTIVE strategy '{name}' is unexpectedly True"

    def test_enabled_strategies_active(self):
        """All strategies enabled by K2 profile are confirmed active."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        s = K2_NOTCH_LOW_Q_V1

        enabled_must_be_true = [
            ("enable_wip_notch_filter", s.enable_wip_notch_filter),
            ("outer_loop_enabled", s.outer_loop_enabled),
            ("calibrated_outer_loop_enabled", s.calibrated_outer_loop_enabled),
            ("physics_equilibrium_feedforward_enabled", s.physics_equilibrium_feedforward_enabled),
            ("low_band_support_outer_loop_enabled", s.low_band_support_outer_loop_enabled),
        ]
        for name, value in enabled_must_be_true:
            assert value, f"ENABLED_ACTIVE strategy '{name}' is unexpectedly False"

    def test_k2_notch_params_correct(self):
        """K2 notch params match specification."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        s = K2_NOTCH_LOW_Q_V1
        assert s.wip_notch_q == 2.0
        assert s.wip_notch_center_hz == 2.5
        assert s.wip_notch_filter_blend == 1.0
        assert s.wip_notch_height_gate_start_m == 0.42
        assert s.wip_notch_height_gate_full_m == 0.48

    def test_no_wbc_enabled(self):
        """WBC is not used in K2."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        s = K2_NOTCH_LOW_Q_V1
        # No WBC flags should be present
        for attr in dir(s):
            if "wbc" in attr.lower():
                val = getattr(s, attr)
                if isinstance(val, bool):
                    assert not val, f"WBC flag '{attr}' is True"

    def test_no_hidden_torque_flags(self):
        """No hidden torque injection flags are active."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        s = K2_NOTCH_LOW_Q_V1
        hidden_patterns = ["hidden_torque", "secret", "extra_torque", "bias_torque"]
        for attr in dir(s):
            for pat in hidden_patterns:
                assert pat not in attr.lower(), f"Hidden torque flag found: {attr}"

    def test_branch_audit_classification(self):
        """Produce branch classification audit."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        s = K2_NOTCH_LOW_Q_V1

        classifications = []

        # DISABLED_INACTIVE branches (verified False on K2 profile)
        disabled = [
            "enable_unified_sagittal_state_feedback",
            "enable_active_pitch_crossing",
            "enable_phase_aware_recenter",
            "enable_hysteresis_recenter",
            "enable_bias_cancel",
            "enable_coordinated_sagittal_feedback",
            "enable_early_zero_crossing_recenter",
            "enable_lp_priority_allocator",
            "enable_lr_replacement_feedback",
            "enable_pitch_aware_position_scaling",
            "enable_position_integral",
            "enable_zero_crossing_recenter",
            "enable_body_yaw_wheel_stabilization",
        ]
        for name in disabled:
            val = getattr(s, name, None)
            status = "DISABLED_INACTIVE" if val is False else "UNEXPECTED_ACTIVE"
            classifications.append({"branch": name, "classification": status, "value": val})

        # ENABLED_ACTIVE branches
        enabled = [
            "enable_wip_notch_filter",
            "outer_loop_enabled",
            "calibrated_outer_loop_enabled",
            "physics_equilibrium_feedforward_enabled",
            "low_band_support_outer_loop_enabled",
            "adaptive_bias_trim_enabled",
        ]
        for name in enabled:
            val = getattr(s, name, None)
            status = "ENABLED_ACTIVE" if val is True else "UNEXPECTED_INACTIVE"
            classifications.append({"branch": name, "classification": status, "value": val})

        # Verify no UNEXPECTED
        for c in classifications:
            assert "UNEXPECTED" not in c["classification"], (
                f"Branch '{c['branch']}' classified as {c['classification']} (value={c['value']})"
            )

        # All disabled must be DISABLED_INACTIVE
        for c in classifications:
            if "DISABLED" in c["classification"]:
                assert c["value"] is False, f"{c['branch']} should be False"
            if "ENABLED_ACTIVE" in c["classification"]:
                assert c["value"] is True, f"{c['branch']} should be True"

        assert len(classifications) == len(disabled) + len(enabled)
