"""
Tests for wheeled_biped.eval.standing_quality.compute_standing_signals.

All tests are pure-numpy -- no JAX, no MuJoCo, no GPU required.
Safe for CI (included in the standard test suite).

Test strategy
-------------
- Build synthetic telemetry dicts with known values.
- Verify flags are triggered only when thresholds are exceeded.
- Verify flag messages mention the expected exploit pattern.
- Verify all expected output keys are present.
- Verify edge cases (empty input, NaN handling).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.eval.standing_quality import THRESHOLDS, compute_standing_signals  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: build a synthetic "good standing" telemetry dict
# ---------------------------------------------------------------------------

_CTRL_KEYS = [
    "l_hip_roll_ctrl",
    "l_hip_yaw_ctrl",
    "l_hip_pitch_ctrl",
    "l_knee_ctrl",
    "l_wheel_ctrl",
    "r_hip_roll_ctrl",
    "r_hip_yaw_ctrl",
    "r_hip_pitch_ctrl",
    "r_knee_ctrl",
    "r_wheel_ctrl",
]


def _make_tele(
    T: int = 400,  # noqa: N803
    torso_z: float = 0.69,
    l_wheel_vel: float = 0.0,
    r_wheel_vel: float = 0.0,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    body_wx: float = 0.0,
    body_wy: float = 0.0,
    l_hip_pitch_pos: float = 0.3,
    r_hip_pitch_pos: float = 0.3,
    l_knee_pos: float = 0.5,
    r_knee_pos: float = 0.5,
    xy_drift_total: float = 0.0,
    ctrl_amplitude: float = 0.0,  # half-amplitude of alternating ctrl oscillation
) -> dict[str, np.ndarray]:
    """Return a minimal telemetry dict with constant/simple signals.

    All fields present; ``xy_drift_total`` spreads linearly over the episode;
    ``ctrl_amplitude`` makes all ctrl channels alternate ± amplitude per step.
    """
    t = np.arange(T, dtype=float)
    tele: dict[str, np.ndarray] = {
        "time_s": t * 0.02,
        "torso_x": np.linspace(0.0, xy_drift_total, T),
        "torso_y": np.zeros(T),
        "torso_z": np.full(T, torso_z),
        "roll_rad": np.full(T, roll_rad),
        "pitch_rad": np.full(T, pitch_rad),
        "yaw_rad": np.zeros(T),
        "body_wx": np.full(T, body_wx),
        "body_wy": np.full(T, body_wy),
        "body_vx": np.zeros(T),
        "body_vy": np.zeros(T),
        "body_vz": np.zeros(T),
        "l_wheel_vel": np.full(T, l_wheel_vel),
        "r_wheel_vel": np.full(T, r_wheel_vel),
        "l_hip_pitch_pos": np.full(T, l_hip_pitch_pos),
        "r_hip_pitch_pos": np.full(T, r_hip_pitch_pos),
        "l_knee_pos": np.full(T, l_knee_pos),
        "r_knee_pos": np.full(T, r_knee_pos),
    }
    # ctrl channels: constant baseline + optional alternating oscillation
    baseline = [0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1.0, 1.5, 0.0]
    sign = np.where((t % 2).astype(bool), 1.0, -1.0)
    for i, key in enumerate(_CTRL_KEYS):
        tele[key] = np.full(T, baseline[i]) + ctrl_amplitude * sign
    return tele


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeStandingSignals:
    # ── Expected output structure ────────────────────────────────────────────

    def test_output_has_all_expected_keys(self):
        """All documented signal keys are present in the output."""
        result = compute_standing_signals(_make_tele())
        required = [
            "wheel_spin_mean_rads",
            "height_mean_m",
            "height_std_m",
            "height_tracking_rmse_m",
            "xy_drift_max_m",
            "xy_drift_final_m",
            "roll_mean_abs_deg",
            "pitch_mean_abs_deg",
            "ctrl_jitter_mean_nm",
            "leg_asymmetry_hip_pitch_rad",
            "leg_asymmetry_knee_rad",
            "leg_asymmetry_mean_rad",
            "ang_vel_rms_rads",
            "flags",
            "num_suspicious",
        ]
        for key in required:
            assert key in result, f"Missing output key: {key}"

    def test_flags_is_list(self):
        result = compute_standing_signals(_make_tele())
        assert isinstance(result["flags"], list)

    def test_num_suspicious_equals_len_flags(self):
        result = compute_standing_signals(_make_tele())
        assert result["num_suspicious"] == len(result["flags"])

    # ── Ideal standing: no flags ─────────────────────────────────────────────

    def test_good_standing_no_flags(self):
        """Ideal standing produces no flags."""
        result = compute_standing_signals(_make_tele(), height_cmd=0.69)
        assert result["num_suspicious"] == 0, (
            f"Expected 0 flags for ideal standing, got: {result['flags']}"
        )

    # ── Edge case: empty input ───────────────────────────────────────────────

    def test_empty_telemetry_returns_gracefully(self):
        """Empty dict returns a result with a descriptive flag, no exception."""
        result = compute_standing_signals({})
        assert "flags" in result
        assert result["num_suspicious"] >= 1
        assert any("no telemetry" in f.lower() for f in result["flags"])

    # ── height_cmd handling ──────────────────────────────────────────────────

    def test_height_tracking_rmse_computed_when_cmd_given(self):
        """RMSE is finite and numerically correct when height_cmd is provided."""
        tele = _make_tele(torso_z=0.65)
        result = compute_standing_signals(tele, height_cmd=0.69)
        assert math.isfinite(result["height_tracking_rmse_m"])
        assert abs(result["height_tracking_rmse_m"] - 0.04) < 1e-4

    def test_height_tracking_rmse_nan_when_no_cmd(self):
        """height_tracking_rmse_m is NaN when height_cmd is not provided."""
        result = compute_standing_signals(_make_tele(), height_cmd=None)
        assert math.isnan(result["height_tracking_rmse_m"])

    # ── Wheel spin ───────────────────────────────────────────────────────────

    def test_high_wheel_spin_flagged(self):
        """Wheel spin well above threshold triggers a flag."""
        tele = _make_tele(l_wheel_vel=10.0, r_wheel_vel=10.0)
        result = compute_standing_signals(tele)
        assert result["wheel_spin_mean_rads"] > THRESHOLDS["wheel_spin_mean_rads"]
        assert result["num_suspicious"] >= 1
        assert any("wheel" in f.lower() for f in result["flags"])

    def test_wheel_spin_exactly_at_threshold_not_flagged(self):
        """Signal exactly equal to the threshold is NOT flagged (strict >)."""
        thresh = THRESHOLDS["wheel_spin_mean_rads"]
        # mean((|thresh| + |thresh|) / 2) == thresh  → not > thresh
        tele = _make_tele(l_wheel_vel=thresh, r_wheel_vel=thresh)
        result = compute_standing_signals(tele)
        assert not any("wheel" in f.lower() for f in result["flags"])

    def test_low_wheel_spin_not_flagged(self):
        """Small wheel spin is normal and should not be flagged."""
        tele = _make_tele(l_wheel_vel=0.5, r_wheel_vel=0.3)
        result = compute_standing_signals(tele)
        assert not any("wheel" in f.lower() for f in result["flags"])

    # ── Height oscillation ───────────────────────────────────────────────────

    def test_height_oscillation_flagged(self):
        """Height std above threshold triggers a flag."""
        T = 400  # noqa: N806
        tele = _make_tele()
        tele["torso_z"] = 0.69 + 0.10 * np.sin(np.linspace(0, 4 * np.pi, T))
        result = compute_standing_signals(tele)
        assert result["height_std_m"] > THRESHOLDS["height_std_m"]
        assert any(
            "height" in f.lower() or "oscillat" in f.lower() or "bounce" in f.lower()
            for f in result["flags"]
        )

    def test_stable_height_not_flagged(self):
        """Constant height produces no height flag."""
        tele = _make_tele(torso_z=0.69)
        result = compute_standing_signals(tele)
        assert not any("height" in f.lower() and "oscillat" in f.lower() for f in result["flags"])

    # ── XY drift ─────────────────────────────────────────────────────────────

    def test_large_xy_drift_flagged(self):
        """Drift beyond threshold triggers a flag."""
        tele = _make_tele(xy_drift_total=0.50)  # 50 cm
        result = compute_standing_signals(tele)
        assert result["xy_drift_max_m"] > THRESHOLDS["xy_drift_max_m"]
        assert any("drift" in f.lower() for f in result["flags"])

    def test_small_xy_drift_not_flagged(self):
        """Small drift (5 cm) is within tolerance."""
        tele = _make_tele(xy_drift_total=0.05)
        result = compute_standing_signals(tele)
        assert not any("drift" in f.lower() for f in result["flags"])

    # ── Roll / pitch lean ────────────────────────────────────────────────────

    def test_chronic_roll_flagged(self):
        """Roll above threshold triggers a flag."""
        tele = _make_tele(roll_rad=np.radians(10.0))
        result = compute_standing_signals(tele)
        assert result["roll_mean_abs_deg"] > THRESHOLDS["roll_mean_abs_deg"]
        assert any("roll" in f.lower() for f in result["flags"])

    def test_chronic_pitch_flagged(self):
        """Pitch above threshold triggers a flag."""
        tele = _make_tele(pitch_rad=np.radians(10.0))
        result = compute_standing_signals(tele)
        assert result["pitch_mean_abs_deg"] > THRESHOLDS["pitch_mean_abs_deg"]
        assert any("pitch" in f.lower() for f in result["flags"])

    def test_small_roll_not_flagged(self):
        """Roll under 1 degree is not flagged."""
        tele = _make_tele(roll_rad=np.radians(0.5))
        result = compute_standing_signals(tele)
        assert not any("roll" in f.lower() for f in result["flags"])

    # ── Control jitter ───────────────────────────────────────────────────────

    def test_ctrl_jitter_flagged(self):
        """High step-to-step ctrl change triggers a flag."""
        # All 10 channels alternating ±2 Nm: mean |diff| = 4 / 10 channels = 4 > 0.5
        # Actually mean over all channels and steps:
        # Each channel diff = |2 - (-2)| = 4 per step,  mean(4 across all 10 channels) = 4
        tele = _make_tele(ctrl_amplitude=2.0)
        result = compute_standing_signals(tele)
        assert result["ctrl_jitter_mean_nm"] > THRESHOLDS["ctrl_jitter_mean_nm"]
        assert any("jitter" in f.lower() or "chatter" in f.lower() for f in result["flags"])

    def test_smooth_ctrl_not_flagged(self):
        """Constant ctrl (zero amplitude) produces no jitter flag."""
        tele = _make_tele(ctrl_amplitude=0.0)
        result = compute_standing_signals(tele)
        assert result["ctrl_jitter_mean_nm"] == pytest.approx(0.0, abs=1e-6)
        assert not any("jitter" in f.lower() for f in result["flags"])

    # ── Leg asymmetry ─────────────────────────────────────────────────────────

    def test_leg_asymmetry_flagged(self):
        """Large hip_pitch difference triggers a flag."""
        tele = _make_tele(l_hip_pitch_pos=0.3, r_hip_pitch_pos=0.8)  # 0.5 rad diff
        result = compute_standing_signals(tele)
        assert result["leg_asymmetry_mean_rad"] > THRESHOLDS["leg_asymmetry_mean_rad"]
        assert any("asymmet" in f.lower() for f in result["flags"])

    def test_symmetric_legs_not_flagged(self):
        """Symmetric legs produce no asymmetry flag."""
        tele = _make_tele(l_hip_pitch_pos=0.3, r_hip_pitch_pos=0.3)
        result = compute_standing_signals(tele)
        assert not any("asymmet" in f.lower() for f in result["flags"])

    # ── Torso angular velocity ────────────────────────────────────────────────

    def test_torso_wobble_flagged(self):
        """High angular velocity RMS triggers a flag."""
        tele = _make_tele(body_wx=1.5, body_wy=1.5)  # 1.5 rad/s constant wobble
        result = compute_standing_signals(tele)
        assert result["ang_vel_rms_rads"] > THRESHOLDS["ang_vel_rms_rads"]
        assert any("wobble" in f.lower() or "oscillat" in f.lower() for f in result["flags"])

    def test_low_ang_vel_not_flagged(self):
        """Negligible angular velocity is not flagged."""
        tele = _make_tele(body_wx=0.05, body_wy=0.05)
        result = compute_standing_signals(tele)
        assert not any("wobble" in f.lower() for f in result["flags"])

    # ── Multiple flags counted correctly ─────────────────────────────────────

    def test_multiple_flags_counted(self):
        """num_suspicious counts all triggered flags correctly."""
        tele = _make_tele(
            l_wheel_vel=10.0,  # triggers wheel_spin
            xy_drift_total=0.50,  # triggers xy_drift
            roll_rad=np.radians(10.0),  # triggers roll
        )
        result = compute_standing_signals(tele)
        assert result["num_suspicious"] == len(result["flags"])
        assert result["num_suspicious"] >= 3
