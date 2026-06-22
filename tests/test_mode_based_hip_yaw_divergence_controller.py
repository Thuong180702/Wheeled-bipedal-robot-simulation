"""Tests for ModeBasedHipYawDivergenceController."""

import math

import pytest

from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
    HipYawState,
    ModeBasedHipYawDivergenceController,
)


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


def test_disabled_returns_zero():
    cfg = _make_cfg(enabled=False)
    ctrl = ModeBasedHipYawDivergenceController(cfg)
    state = HipYawState(div_error=0.2, div_rate=0.5, height=0.3)
    out = ctrl.compute(state)
    assert out["tau_left"] == 0.0
    assert out["tau_right"] == 0.0


def test_enabled_produces_correct_sign_and_respects_max_torque():
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


def test_clips_to_max_torque():
    cfg = _make_cfg(enabled=True, kp_div=10.0, kd_div=0.0, max_torque=1.0)
    ctrl = ModeBasedHipYawDivergenceController(cfg)
    state = HipYawState(div_error=1.0, div_rate=0.0, height=0.3)
    out = ctrl.compute(state)
    # raw magnitude 10 -> clipped to 1
    assert math.isclose(out["tau_left"], -1.0, rel_tol=1e-6)
    assert math.isclose(out["tau_right"], 1.0, rel_tol=1e-6)
    assert abs(out["tau_left"]) <= 1.0


def test_height_gate_applied():
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