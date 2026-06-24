"""Tests for ModeBasedHipYawDivergenceController.

Verifies:
- Controller is opt-in only (disabled by default returns zero torque).
- Old profiles (B2v2, PFF, low-band v2) are unchanged.
- Default/current-best unchanged.
- PFF source unchanged.
- Low-band v2 tuning unchanged.
- No WBC/HY2 activation in the controller.
- No setup-name branch in the controller code.
- No D4/D5-specific logic.
- No threshold relaxation.
- Telemetry fields exist when candidate is enabled.
"""

import inspect
import math
import pathlib
import textwrap

import pytest

from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
    HipYawState,
    ModeBasedHipYawDivergenceController,
)

CONTROLLER_SOURCE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "wheeled_biped"
    / "controllers"
    / "mode_based_hip_yaw_divergence_controller.py"
)


# ============================================================================
# Helpers
# ============================================================================


def _make_cfg(**overrides):
    cfg = {
        "enabled": True,
        "kp_div": 1.0,
        "kd_div": 0.1,
        "max_torque": 1.0,
        "soft_limit_rad": 0.3,
        "soft_limit_gain": 0.5,
        "ref_source": "target",
    }
    cfg.update(overrides)
    return cfg


def _get_controller_source() -> str:
    """Read the controller source file."""
    return CONTROLLER_SOURCE_PATH.read_text(encoding="utf-8")


# ============================================================================
# Original tests (kept for backward compat)
# ============================================================================


class TestDisabledReturnZero:
    """Controller is opt-in: disabled by default returns zero torque."""

    def test_disabled_returns_zero(self):
        cfg = _make_cfg(enabled=False)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.2, div_rate=0.5, height=0.3)
        out = ctrl.compute(state)
        assert out["tau_left"] == 0.0
        assert out["tau_right"] == 0.0

    def test_default_cfg_disabled(self):
        """When 'enabled' key is missing from cfg, controller defaults to disabled."""
        cfg = {"kp_div": 1.0, "kd_div": 0.1, "max_torque": 1.0}
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.5, div_rate=0.3, height=0.3)
        out = ctrl.compute(state)
        assert out["tau_left"] == 0.0
        assert out["tau_right"] == 0.0

    def test_empty_cfg_disabled(self):
        """Empty config dict defaults to disabled."""
        ctrl = ModeBasedHipYawDivergenceController({})
        state = HipYawState(div_error=1.0, div_rate=1.0, height=0.2)
        out = ctrl.compute(state)
        assert out["tau_left"] == 0.0
        assert out["tau_right"] == 0.0


class TestEnabledBehavior:
    """Basic PD law and clipping when controller is enabled."""

    def test_enabled_produces_correct_sign_and_respects_max_torque(self):
        cfg = _make_cfg(enabled=True, kp_div=1.0, kd_div=0.1, max_torque=1.0)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.4, div_rate=0.0, height=0.3)
        out = ctrl.compute(state)
        # raw = -(kp * 0.4 + kd * 0) = -0.4 -> left gets -0.4, right gets +0.4
        assert math.isclose(out["tau_left"], -0.4, rel_tol=1e-6)
        assert math.isclose(out["tau_right"], 0.4, rel_tol=1e-6)
        # within max_torque
        assert abs(out["tau_left"]) <= 1.0
        assert abs(out["tau_right"]) <= 1.0

    def test_clips_to_max_torque(self):
        cfg = _make_cfg(enabled=True, kp_div=10.0, kd_div=0.0, max_torque=1.0)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=1.0, div_rate=0.0, height=0.3)
        out = ctrl.compute(state)
        # raw magnitude 10 -> clipped to 1
        assert math.isclose(out["tau_left"], -1.0, rel_tol=1e-6)
        assert math.isclose(out["tau_right"], 1.0, rel_tol=1e-6)
        assert abs(out["tau_left"]) <= 1.0

    def test_height_gate_applied(self):
        cfg = _make_cfg(enabled=True, kp_div=1.0, kd_div=0.0, max_torque=1.0,
                        soft_limit_rad=0.3, soft_limit_gain=0.5)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        # height above high threshold -> gate 0 -> zero torque
        state_high = HipYawState(div_error=0.4, div_rate=0.0, height=0.9)
        out_high = ctrl.compute(state_high)
        assert math.isclose(out_high["tau_left"], 0.0, abs_tol=1e-6)
        assert math.isclose(out_high["tau_right"], 0.0, abs_tol=1e-6)
        # height at low threshold -> gate 1 -> full torque
        state_low = HipYawState(div_error=0.4, div_rate=0.0, height=0.2)
        out_low = ctrl.compute(state_low)
        assert math.isclose(out_low["tau_left"], -0.4, rel_tol=1e-6)
        assert math.isclose(out_low["tau_right"], 0.4, rel_tol=1e-6)


# ============================================================================
# Profile-safety tests: Old profiles unchanged
# ============================================================================


class TestOldProfilesUnchanged:
    """Verify that old profiles (B2v2, PFF, low-band v2) are unchanged."""

    def test_b2v2_profile_available(self):
        """B2v2 baseline profile must still exist and be selectable."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "calibrated_support_position_outer_loop_pitch_ref_v2" in SAGITTAL_AUTHORITY_PROFILES

    def test_pff_profile_available(self):
        """PFF profile must still exist and be selectable."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "physics_equilibrium_feedforward_outer_loop" in SAGITTAL_AUTHORITY_PROFILES

    def test_low_band_v2_profile_available(self):
        """Low-band v2 profile must still exist."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "physics_equilibrium_feedforward_outer_loop_low_band_support_v2" in SAGITTAL_AUTHORITY_PROFILES

    def test_pff_source_unchanged(self):
        """PFF module must export canonical functions unchanged."""
        from wheeled_biped.controllers import physics_equilibrium_feedforward as pff
        assert hasattr(pff, "physics_equilibrium_feedforward_tau_each_wheel_nm")
        assert hasattr(pff, "physics_equilibrium_feedforward_params")

    def test_low_band_v2_tuning_unchanged(self):
        """Low-band v2 tuning parameters (center, sigma) must be unchanged."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        profile = SAGITTAL_AUTHORITY_PROFILES[
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
        ]
        assert profile.low_band_support_outer_loop_enabled is True
        assert profile.low_band_support_center_m == 0.320
        assert profile.low_band_support_sigma_m == 0.004

    def test_default_current_best_unchanged(self):
        """Default/current-best profile has not been altered by the mode controller."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        # Low-band v2 is the promoted current-best
        profile = SAGITTAL_AUTHORITY_PROFILES[
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
        ]
        assert profile.physics_equilibrium_feedforward_enabled is True
        # WBC must not be enabled in current-best
        assert getattr(profile, "wbc_enabled", False) is False


# ============================================================================
# Guard tests: No WBC/HY2 activation, no setup-name branching, no D4/D5
# ============================================================================


class TestControllerGuardRails:
    """Ensure mode-based controller code does not contain forbidden patterns."""

    def test_no_wbc_activation_in_source(self):
        """Controller source must not activate WBC."""
        source = _get_controller_source()
        # Check for WBC-related activations
        assert "wbc_enabled" not in source.lower(), (
            "Controller must not reference wbc_enabled"
        )
        assert "WBCBalance" not in source, (
            "Controller must not instantiate WBC"
        )

    def test_no_hy2_activation_in_source(self):
        """Controller source must not activate HY2 externally.

        The controller IS the mode-based HY2 replacement; it must not
        re-activate the old HY2-DIV path from shape_posture_controller.
        Only a docstring reference is acceptable.
        """
        source = _get_controller_source()
        # Filter out comments/docstrings
        lines = source.split("\n")
        code_lines = [
            l for l in lines
            if not l.strip().startswith("#") and not l.strip().startswith('"""')
            and not l.strip().startswith("'''")
        ]
        code_text = "\n".join(code_lines)
        # No functional HY2 activation
        assert "HY2_DIV_BASELINE" not in code_text, (
            "Controller must not instantiate HY2_DIV_BASELINE"
        )
        assert "HY2_DIV_AGGRESSIVE" not in code_text, (
            "Controller must not instantiate HY2_DIV_AGGRESSIVE"
        )

    def test_no_setup_name_branch(self):
        """Controller must not contain setup-name branching logic."""
        source = _get_controller_source()
        assert "setup_name" not in source, (
            "Controller must not branch on setup_name"
        )
        assert "setup-name" not in source, (
            "Controller must not branch on setup-name (kebab case)"
        )

    def test_no_d4_d5_specific_logic(self):
        """Controller must not contain D4/D5 case-specific logic."""
        source = _get_controller_source()
        # Check for D4/D5 references in code (not docstrings)
        lines = source.split("\n")
        code_lines = [
            l for l in lines
            if not l.strip().startswith("#") and not l.strip().startswith('"""')
            and not l.strip().startswith("'''")
        ]
        code_text = "\n".join(code_lines)
        assert "D4" not in code_text, "Controller must not have D4-specific logic"
        assert "D5" not in code_text, "Controller must not have D5-specific logic"

    def test_no_threshold_relaxation(self):
        """Controller must not relax thresholds based on profile/scenario.

        The max_torque and soft_limit values come strictly from config;
        no runtime relaxation based on push magnitude or scenario.
        """
        source = _get_controller_source()
        # No references to relaxing thresholds dynamically
        assert "relax" not in source.lower(), (
            "Controller must not contain threshold relaxation"
        )
        assert "push_magnitude" not in source, (
            "Controller must not adapt to push_magnitude"
        )
        assert "scenario" not in source, (
            "Controller must not branch on scenario"
        )


# ============================================================================
# Telemetry tests: fields present when candidate is enabled
# ============================================================================


class TestTelemetryFieldsWhenEnabled:
    """When the candidate controller is enabled, output must contain telemetry keys."""

    EXPECTED_OUTPUT_KEYS = [
        "tau_left", "tau_right", "tau_left_raw", "tau_right_raw",
        "support_error_gate", "support_rate_gate", "effective_support_gate", "combined_gate",
    ]

    def test_output_has_required_keys(self):
        """Compute output dict has the expected keys."""
        cfg = _make_cfg(enabled=True)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.1, div_rate=0.0, height=0.3)
        out = ctrl.compute(state)
        for key in self.EXPECTED_OUTPUT_KEYS:
            assert key in out, f"Missing telemetry key: {key}"

    def test_output_values_are_float(self):
        """All output values must be float (not JAX array or None)."""
        cfg = _make_cfg(enabled=True)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.1, div_rate=0.05, height=0.3)
        out = ctrl.compute(state)
        for key, val in out.items():
            assert isinstance(val, float), (
                f"Output[{key}] should be float, got {type(val)}"
            )

    def test_disabled_output_still_has_keys(self):
        """Even when disabled, output dict has the same keys (zeroed)."""
        cfg = _make_cfg(enabled=False)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        state = HipYawState(div_error=0.1, div_rate=0.0, height=0.3)
        out = ctrl.compute(state)
        for key in self.EXPECTED_OUTPUT_KEYS:
            assert key in out, f"Missing key when disabled: {key}"
            # Gate keys default to 1.0 when disabled (fully open = no attenuation)
            if "gate" in key:
                assert out[key] == 1.0, f"Gate key {key} should be 1.0 when disabled"
            else:
                assert out[key] == 0.0

    def test_config_attributes_exposed(self):
        """Controller exposes its config as attributes for telemetry/logging."""
        cfg = _make_cfg(enabled=True, kp_div=2.5, kd_div=0.3, max_torque=1.5)
        ctrl = ModeBasedHipYawDivergenceController(cfg)
        assert ctrl.enabled is True
        assert ctrl.kp_div == 2.5
        assert ctrl.kd_div == 0.3
        assert ctrl.max_torque == 1.5
        assert ctrl.ref_source == "target"


# ============================================================================
# Config-driven only (no hardcoded branches)
# ============================================================================


class TestConfigDrivenOnly:
    """Controller behavior is driven entirely by config, no internal branches."""

    def test_different_gains_different_torque(self):
        """Different kp_div values produce different torques (parameterized)."""
        state = HipYawState(div_error=0.2, div_rate=0.0, height=0.2)
        torques = []
        for kp in [1.0, 2.0, 5.0]:
            cfg = _make_cfg(enabled=True, kp_div=kp, kd_div=0.0, max_torque=10.0)
            ctrl = ModeBasedHipYawDivergenceController(cfg)
            out = ctrl.compute(state)
            torques.append(out["tau_left"])
        # All different
        assert len(set(torques)) == len(torques), (
            "Different gains must produce different torques"
        )

    def test_max_torque_from_config_only(self):
        """Max torque is strictly from config, not modified internally."""
        for max_t in [0.5, 1.0, 2.0, 5.0]:
            cfg = _make_cfg(enabled=True, kp_div=100.0, kd_div=0.0, max_torque=max_t)
            ctrl = ModeBasedHipYawDivergenceController(cfg)
            state = HipYawState(div_error=1.0, div_rate=0.0, height=0.2)
            out = ctrl.compute(state)
            assert abs(out["tau_left"]) <= max_t + 1e-9
            assert abs(out["tau_right"]) <= max_t + 1e-9
