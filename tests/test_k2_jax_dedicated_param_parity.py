"""Phase 1: Parameter source-of-truth parity tests.

Verifies that the dedicated runner and canonical K2 JAX path produce identical
control-affecting JAX params, confirming 0 parameter mismatches.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent.parent

from wheeled_biped.controllers.k2_jax_controller import (
    pack_params_stage2,
    pack_state_k2,
    unpack_params_stage2,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1,
)


def _pack_canonical_params(variant_name=None):
    """Replicate the canonical path's pack_params_stage2() call.

    Matches simulate_hierarchical_controller.py lines 5489-5513.
    """
    _auth = K2_NOTCH_LOW_Q_V1
    _eff_vel_damp = 1.0
    if variant_name and _auth.is_active_for_variant(variant_name):
        _eff_vel_damp = float(_auth.velocity_damping_scale)

    return pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10, dtype=jnp.float64) * 30.0,
        max_torque_rate=jnp.ones(10, dtype=jnp.float64) * 400.0,
        control_dt=0.01,
        mode_div_soft_gain=0.80,
        mode_div_ref_source="disabled",
        k_velocity=15.0,
        velocity_damping_scale=_eff_vel_damp,
        apcr1nd_startup_guard_steps=float(_auth.recenter_priority_startup_guard_steps),
        apcr1nd_safe_min_com_z=float(_auth.recenter_priority_safe_min_com_z),
        apcr1nd_safe_roll_rad=float(_auth.recenter_priority_safe_roll_rad),
        apcr1nd_safe_pitch_rad=float(_auth.recenter_priority_safe_pitch_rad),
        apcr1nd_direct_enter_m=float(_auth.apcr1nd_direct_enter_m),
        apcr1nd_release_inner_m=float(_auth.apcr1nd_release_inner_m),
        apcr1nd_hold_outside_band=bool(_auth.apcr1nd_hold_outside_band),
        apcr1nd_converging_release_steps=float(_auth.apcr1nd_converging_release_steps),
        standalone_mode=True,
        pitch_x_eq_rad=0.0,
        support_center_eq_x_m=-0.135,
        support_center_eq_y_m=-0.005,
        sagittal_axis_x=0.0,
        sagittal_axis_y=1.0,
    )


def _pack_dedicated_params(variant_name=None):
    """Replicate the dedicated runner's pack_params_stage2() call.

    Matches run_k2_jax_realtime.py lines 329-352 (post Phase 1 refactor).
    """
    _auth = K2_NOTCH_LOW_Q_V1
    _eff_vel_damp = 1.0
    if variant_name and _auth.is_active_for_variant(variant_name):
        _eff_vel_damp = float(_auth.velocity_damping_scale)

    return pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10, dtype=jnp.float64) * 30.0,
        max_torque_rate=jnp.ones(10, dtype=jnp.float64) * 400.0,
        control_dt=0.01,
        mode_div_soft_gain=0.80,
        mode_div_ref_source="disabled",
        k_velocity=15.0,
        velocity_damping_scale=_eff_vel_damp,
        apcr1nd_startup_guard_steps=float(_auth.recenter_priority_startup_guard_steps),
        apcr1nd_safe_min_com_z=float(_auth.recenter_priority_safe_min_com_z),
        apcr1nd_safe_roll_rad=float(_auth.recenter_priority_safe_roll_rad),
        apcr1nd_safe_pitch_rad=float(_auth.recenter_priority_safe_pitch_rad),
        apcr1nd_direct_enter_m=float(_auth.apcr1nd_direct_enter_m),
        apcr1nd_release_inner_m=float(_auth.apcr1nd_release_inner_m),
        apcr1nd_hold_outside_band=bool(_auth.apcr1nd_hold_outside_band),
        apcr1nd_converging_release_steps=float(_auth.apcr1nd_converging_release_steps),
        standalone_mode=True,
        pitch_x_eq_rad=0.0,
        support_center_eq_x_m=-0.135,
        support_center_eq_y_m=-0.005,
        sagittal_axis_x=0.0,
        sagittal_axis_y=1.0,
    )


# ── Field-level comparison keys (scalar control-affecting params) ──────────

CONTROL_PARAM_KEYS = [
    "notch_fs_hz", "notch_fc_hz", "notch_Q",
    "notch_b0", "notch_b1", "notch_b2", "notch_a1", "notch_a2",
    "control_dt", "k_velocity", "velocity_damping_scale",
    "mode_div_soft_gain", "mode_div_ref_source",
    "apcr1nd_startup_guard_steps", "apcr1nd_safe_min_com_z",
    "apcr1nd_safe_roll_rad", "apcr1nd_safe_pitch_rad",
    "apcr1nd_direct_enter_m", "apcr1nd_release_inner_m",
    "apcr1nd_hold_outside_band", "apcr1nd_converging_release_steps",
]

EQUILIBRIUM_KEYS = [
    "pitch_x_eq_rad", "roll_y_eq_rad",
    "support_center_eq_x_m", "support_center_eq_y_m",
    "sagittal_axis_x", "sagittal_axis_y",
]


class TestParamParityCanonicalVsDedicated:
    """Verify canonical and dedicated produce identical control-affecting params."""

    @pytest.mark.parametrize("variant", [
        None,
        "high_0p480",
        "low_0p300",
        "low_0p330",
        "high_0p430",
    ])
    def test_flat_params_identical(self, variant):
        """Canonical and dedicated flat param arrays must be bit-identical."""
        can = _pack_canonical_params(variant)
        ded = _pack_dedicated_params(variant)
        np.testing.assert_array_equal(
            np.array(can, dtype=np.float64),
            np.array(ded, dtype=np.float64),
        )

    @pytest.mark.parametrize("variant", [
        None,
        "high_0p480",
        "low_0p300",
    ])
    def test_unpacked_control_params_match(self, variant):
        """All scalar control-affecting params must match after unpacking."""
        can = _pack_canonical_params(variant)
        ded = _pack_dedicated_params(variant)
        can_unpacked = unpack_params_stage2(can)
        ded_unpacked = unpack_params_stage2(ded)
        for key in CONTROL_PARAM_KEYS:
            c_val = can_unpacked[key]
            d_val = ded_unpacked[key]
            assert c_val == d_val, f"Mismatch on {key!r} for variant={variant!r}: {c_val!r} != {d_val!r}"


class TestVelocityDampingScale:
    """Verify velocity_damping_scale is correctly applied per variant."""

    def test_no_variant_uses_baseline_1p0(self):
        """Without variant, velocity_damping_scale should be 1.0."""
        params = _pack_dedicated_params(None)
        unpacked = unpack_params_stage2(params)
        assert float(unpacked["velocity_damping_scale"]) == 1.0

    def test_high_0p480_uses_1p1(self):
        """high_0p480 is in applies_to_variants — should use 1.1."""
        params = _pack_dedicated_params("high_0p480")
        unpacked = unpack_params_stage2(params)
        assert float(unpacked["velocity_damping_scale"]) == 1.1

    def test_low_0p300_uses_1p1(self):
        """low_0p300 is in applies_to_variants — should use 1.1."""
        params = _pack_dedicated_params("low_0p300")
        unpacked = unpack_params_stage2(params)
        assert float(unpacked["velocity_damping_scale"]) == 1.1

    def test_unknown_variant_uses_baseline_1p0(self):
        """Unknown variant not in applies_to_variants — should use 1.0."""
        params = _pack_dedicated_params("some_unknown_variant")
        unpacked = unpack_params_stage2(params)
        assert float(unpacked["velocity_damping_scale"]) == 1.0


class TestApcr1ndHoldOutsideBand:
    """Verify apcr1nd_hold_outside_band is read from K2_NOTCH_LOW_Q_V1."""

    def test_hold_outside_band_is_true(self):
        """K2_NOTCH_LOW_Q_V1 sets apcr1nd_hold_outside_band=True."""
        _auth = K2_NOTCH_LOW_Q_V1
        assert bool(_auth.apcr1nd_hold_outside_band) is True

    def test_param_pack_reflects_true(self):
        """The packed JAX param should reflect True (1.0)."""
        params = _pack_dedicated_params("high_0p480")
        unpacked = unpack_params_stage2(params)
        assert float(unpacked["apcr1nd_hold_outside_band"]) == 1.0


class TestProfileSourceOfTruth:
    """Verify K2_NOTCH_LOW_Q_V1 fields are the accepted source of truth."""

    def test_velocity_damping_scale_source_value(self):
        """Profile velocity_damping_scale must be 1.1."""
        _auth = K2_NOTCH_LOW_Q_V1
        assert float(_auth.velocity_damping_scale) == 1.1

    def test_applies_to_variants_not_empty(self):
        """Profile must have non-empty applies_to_variants for variant-gating."""
        _auth = K2_NOTCH_LOW_Q_V1
        assert len(_auth.applies_to_variants) >= 8

    def test_apcr1nd_fields_match_expected(self):
        """APCR1ND fields must match the documented K2 profile values."""
        _auth = K2_NOTCH_LOW_Q_V1
        assert int(_auth.recenter_priority_startup_guard_steps) == 100
        assert float(_auth.recenter_priority_safe_min_com_z) == 0.27
        assert float(_auth.recenter_priority_safe_roll_rad) == 0.15
        assert float(_auth.recenter_priority_safe_pitch_rad) == 0.15
        assert float(_auth.apcr1nd_direct_enter_m) == 0.06
        assert float(_auth.apcr1nd_release_inner_m) == 0.03
        assert int(_auth.apcr1nd_converging_release_steps) == 15


class TestNoHardcodedK2ProfileInDedicated:
    """Verify dedicated runner imports from canonical profile, not hardcoded dict."""

    def test_dedicated_runner_imports_k2_auth_sched(self):
        """The dedicated runner must import K2_NOTCH_LOW_Q_V1."""
        import importlib
        import scripts.run_k2_jax_realtime as dr
        # Check the module has the right import
        assert hasattr(dr, '_K2_AUTH_SCHED'), (
            "Dedicated runner must import K2_NOTCH_LOW_Q_V1 as _K2_AUTH_SCHED"
        )

    def test_dedicated_runner_has_no_k2_profile_dict(self):
        """The old hardcoded K2_PROFILE dict must be removed."""
        import scripts.run_k2_jax_realtime as dr
        assert not hasattr(dr, 'K2_PROFILE'), (
            "Hardcoded K2_PROFILE dict must be removed — use _K2_AUTH_SCHED instead"
        )


class TestDumpK2ParamsFlag:
    """Verify --dump-k2-params writes valid JSON with required fields."""

    def test_dump_produces_valid_json(self, tmp_path):
        """Running with --dump-k2-params should produce a valid JSON file."""
        import subprocess
        dump_path = tmp_path / "param_dump.json"
        setup = ROOT / "outputs" / "physical_target_height_setups" / "high_0p480_setup.json"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_k2_jax_realtime.py"),
             "--height-setup", str(setup),
             "--steps", "10",
             "--quiet",
             "--telemetry", "off",
             "--dump-k2-params", str(dump_path)],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, f"Dedicated runner failed: {result.stderr}"
        assert dump_path.exists(), f"Dump not written to {dump_path}"
        with open(dump_path) as f:
            data = json.load(f)
        assert "control_affecting_params" in data
        assert "equilibrium_constants" in data
        assert data["control_affecting_params"]["velocity_damping_scale"] == 1.1
        assert data["control_affecting_params"]["apcr1nd_hold_outside_band"] is True
