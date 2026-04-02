"""
Tests for LQRBalanceController (pure Python, no MuJoCo/JAX needed).

_build_height_ik() is mocked at the module level so no robot XML is required.
Full integration tests (real IK scan accuracy, physics rollout) are NOT covered
here and require a running MuJoCo installation — they live in tests/test_env.py.

What IS tested:
  - Controller construction (default + custom gains)
  - reset() clears episode state
  - compute_action() output shape and range
  - Obs-size assertion (38-dim → ValueError; 41-dim → ok)
  - Sign conventions (forward lean → positive wheel; yaw error → correct diff)
  - kd_yaw damping contribution
  - Lateral lean → antisymmetric hip-roll
  - Different height commands produce different hip/knee targets
  - gains_info() completeness
  - _compute_lqr_gains() structural properties (K shape, sign)
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock IK scan result used in all tests that instantiate the controller.
# Polynomial coefficients (highest-degree first) for a plausible hip/knee map.
# ---------------------------------------------------------------------------
_MOCK_HIP_POLY = np.array([0.0, -2.0, 1.5])
_MOCK_KNEE_POLY = np.array([0.0, -4.0, 3.0])
_MOCK_H_MIN = 0.38
_MOCK_H_MAX = 0.72

_IK_PATCH = patch(
    "wheeled_biped.controllers.lqr_balance._build_height_ik",
    return_value=(_MOCK_HIP_POLY, _MOCK_KNEE_POLY, _MOCK_H_MIN, _MOCK_H_MAX),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller(**kwargs):
    """Build an LQRBalanceController with a mocked IK scan."""
    from wheeled_biped.controllers.lqr_balance import LQRBalanceController

    with _IK_PATCH:
        return LQRBalanceController(model_path="fake_model.xml", **kwargs)


def _make_obs(
    grav_x: float = 0.0,
    grav_y: float = 0.0,
    grav_z: float = -1.0,
    lin_vel_x: float = 0.0,
    lin_vel_y: float = 0.0,
    lin_vel_z: float = 0.0,
    ang_vel_x: float = 0.0,
    ang_vel_y: float = 0.0,
    ang_vel_z: float = 0.0,
    joint_pos: np.ndarray | None = None,
    joint_vel: np.ndarray | None = None,
    prev_action: np.ndarray | None = None,
    height_cmd_norm: float = 0.5,
    yaw_error: float = 0.0,
) -> np.ndarray:
    """Build a synthetic 41-dim BalanceEnv observation."""
    if joint_pos is None:
        joint_pos = np.zeros(10)
    if joint_vel is None:
        joint_vel = np.zeros(10)
    if prev_action is None:
        prev_action = np.zeros(10)
    return np.array(
        [
            grav_x,  # [0]
            grav_y,  # [1]
            grav_z,  # [2]
            lin_vel_x,  # [3]
            lin_vel_y,  # [4]
            lin_vel_z,  # [5]
            ang_vel_x,  # [6]
            ang_vel_y,  # [7]
            ang_vel_z,  # [8]
            *joint_pos,  # [9:19]
            *joint_vel,  # [19:29]
            *prev_action,  # [29:39]
            height_cmd_norm,  # [39]
            yaw_error,  # [40]
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# _compute_lqr_gains
# ---------------------------------------------------------------------------


class TestComputeLqrGains:
    def test_returns_4_gains(self):
        from wheeled_biped.controllers.lqr_balance import _compute_lqr_gains

        k = _compute_lqr_gains()
        assert k.shape == (4,)

    def test_all_gains_negative(self):
        """All K elements should be negative for this TWIP sign convention.

        u = -(K · x) → positive wheel command for forward lean when K[0] < 0.
        """
        from wheeled_biped.controllers.lqr_balance import _compute_lqr_gains

        k = _compute_lqr_gains()
        assert np.all(k < 0), f"Expected all negative K, got {k}"

    def test_custom_q_r_changes_gains(self):
        from wheeled_biped.controllers.lqr_balance import _compute_lqr_gains

        k_default = _compute_lqr_gains()
        k_custom = _compute_lqr_gains(q_diag=(20.0, 5.0, 3.0, 0.3), r_val=0.5)
        assert not np.allclose(k_default, k_custom)


# ---------------------------------------------------------------------------
# Controller construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        ctrl = _make_controller()
        info = ctrl.gains_info()
        assert info["K_pitch"] > 0
        assert info["K_pitch_rate"] > 0
        assert info["kp_roll"] == pytest.approx(0.4)
        assert info["kd_roll"] == pytest.approx(0.08)
        assert info["kp_yaw"] == pytest.approx(2.5)
        assert info["kd_yaw"] == pytest.approx(0.2)

    def test_custom_gains(self):
        ctrl = _make_controller(kp_roll=0.8, kd_roll=0.1, kp_yaw=3.0, kd_yaw=0.3)
        info = ctrl.gains_info()
        assert info["kp_roll"] == pytest.approx(0.8)
        assert info["kd_roll"] == pytest.approx(0.1)
        assert info["kp_yaw"] == pytest.approx(3.0)
        assert info["kd_yaw"] == pytest.approx(0.3)

    def test_custom_lqr_params(self):
        ctrl = _make_controller(lqr_q=(20.0, 4.0, 5.0, 0.5), lqr_r=0.5)
        info = ctrl.gains_info()
        # Gains should be different from defaults
        ctrl_default = _make_controller()
        info_default = ctrl_default.gains_info()
        assert info["K_pitch"] != pytest.approx(info_default["K_pitch"])

    def test_wheel_vel_limit_from_config(self):
        config = {"low_level_pid": {"wheel_vel_limit": 15.0}}
        ctrl = _make_controller(config=config)
        assert ctrl.gains_info()["wheel_vel_limit_rads"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_drift(self):
        ctrl = _make_controller()
        obs = _make_obs(lin_vel_y=-0.5)  # forward velocity → accumulates drift
        ctrl.reset(height_cmd_m=0.65)
        # Step several times to accumulate drift
        for _ in range(50):
            ctrl.compute_action(obs)
        # Reset should zero out drift
        ctrl.reset(height_cmd_m=0.65)
        assert ctrl._fwd_pos_drift == pytest.approx(0.0)

    def test_reset_sets_height_cmd(self):
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.50)
        assert ctrl._height_cmd_m == pytest.approx(0.50)

    def test_reset_clamps_height_to_range(self):
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.99)  # above MAX_HEIGHT_CMD=0.70
        assert ctrl._height_cmd_m <= 0.70
        ctrl.reset(height_cmd_m=0.10)  # below MIN_HEIGHT_CMD=0.40
        assert ctrl._height_cmd_m >= 0.40

    def test_reset_default_height(self):
        ctrl = _make_controller()
        ctrl.reset()  # default = 0.65
        assert 0.40 <= ctrl._height_cmd_m <= 0.70


# ---------------------------------------------------------------------------
# compute_action() — shape and range
# ---------------------------------------------------------------------------


class TestComputeActionShapeRange:
    def test_output_shape(self):
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.65)
        action = ctrl.compute_action(_make_obs())
        assert action.shape == (10,)

    def test_output_dtype(self):
        ctrl = _make_controller()
        ctrl.reset()
        action = ctrl.compute_action(_make_obs())
        assert action.dtype == np.float32

    def test_output_clipped_to_unit_range(self):
        ctrl = _make_controller()
        ctrl.reset()
        # Use extreme inputs that could push out of range
        obs = _make_obs(grav_y=-0.99, ang_vel_x=5.0, lin_vel_y=-2.0)
        action = ctrl.compute_action(obs)
        assert np.all(action >= -1.0), f"action below -1: {action}"
        assert np.all(action <= 1.0), f"action above +1: {action}"

    def test_upright_at_rest_action_near_zero(self):
        """Upright, zero velocity, zero yaw error → wheel commands near zero."""
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.65)
        obs = _make_obs(grav_z=-1.0)  # upright, no lean
        action = ctrl.compute_action(obs)
        # Wheel channels are indices 4 (l_wheel) and 9 (r_wheel)
        assert abs(action[4]) < 0.1, f"left wheel not near zero at upright: {action[4]}"
        assert abs(action[9]) < 0.1, f"right wheel not near zero at upright: {action[9]}"


# ---------------------------------------------------------------------------
# Obs-size assertion
# ---------------------------------------------------------------------------


class TestObsSizeAssertion:
    def test_41_dim_obs_accepted(self):
        ctrl = _make_controller()
        ctrl.reset()
        obs = _make_obs()
        assert obs.shape == (41,)
        action = ctrl.compute_action(obs)  # should not raise
        assert action.shape == (10,)

    def test_38_dim_obs_raises_value_error(self):
        """38-dim obs (lin_vel_mode='disabled') must raise a clear ValueError."""
        ctrl = _make_controller()
        ctrl.reset()
        bad_obs = np.zeros(38, dtype=np.float32)
        with pytest.raises(ValueError, match="41-dim"):
            ctrl.compute_action(bad_obs)

    def test_wrong_dim_raises_with_helpful_message(self):
        ctrl = _make_controller()
        ctrl.reset()
        bad_obs = np.zeros(20, dtype=np.float32)
        with pytest.raises(ValueError, match="lin_vel_mode"):
            ctrl.compute_action(bad_obs)


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


class TestSignConventions:
    def test_forward_lean_gives_positive_avg_wheel(self):
        """Forward lean (grav_y < 0) must drive both wheels forward (positive avg)."""
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.65)
        # grav_y = -0.2 ≈ 11.5° forward lean
        obs = _make_obs(grav_y=-0.2, grav_z=-math.sqrt(1 - 0.04))
        action = ctrl.compute_action(obs)
        avg_wheel = (float(action[4]) + float(action[9])) / 2.0
        assert avg_wheel > 0.0, (
            f"Expected positive avg wheel for forward lean, got {avg_wheel:.4f}. "
            f"l_wheel={action[4]:.4f}, r_wheel={action[9]:.4f}"
        )

    def test_backward_lean_gives_negative_avg_wheel(self):
        """Backward lean (grav_y > 0) must drive both wheels backward (negative avg)."""
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.65)
        obs = _make_obs(grav_y=0.2, grav_z=-math.sqrt(1 - 0.04))
        action = ctrl.compute_action(obs)
        avg_wheel = (float(action[4]) + float(action[9])) / 2.0
        assert avg_wheel < 0.0, (
            f"Expected negative avg wheel for backward lean, got {avg_wheel:.4f}"
        )

    def test_ccw_yaw_error_left_wheel_faster(self):
        """CCW yaw drift (yaw_error > 0) → CW correction: left wheel faster than right.

        Sign derivation (robot faces -Y, left side at +X):
          CW from above = left side moves forward → omega_l > omega_r
        """
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.65)
        # Upright, no lean — only yaw correction active
        obs = _make_obs(grav_z=-1.0, yaw_error=0.5)
        action = ctrl.compute_action(obs)
        l_wheel = float(action[4])
        r_wheel = float(action[9])
        assert l_wheel > r_wheel, (
            f"Expected l_wheel > r_wheel for CCW drift, got l={l_wheel:.4f}, r={r_wheel:.4f}"
        )

    def test_cw_yaw_error_right_wheel_faster(self):
        """CW yaw drift (yaw_error < 0) → CCW correction: right wheel faster than left."""
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.65)
        obs = _make_obs(grav_z=-1.0, yaw_error=-0.5)
        action = ctrl.compute_action(obs)
        l_wheel = float(action[4])
        r_wheel = float(action[9])
        assert r_wheel > l_wheel, (
            f"Expected r_wheel > l_wheel for CW drift, got l={l_wheel:.4f}, r={r_wheel:.4f}"
        )

    def test_ccw_yaw_rate_damps_correctly(self):
        """CCW spin (yaw_rate > 0) → kd_yaw damps it: left wheel faster than right."""
        ctrl = _make_controller(kp_yaw=0.0, kd_yaw=1.0)  # only rate term active
        ctrl.reset(height_cmd_m=0.65)
        obs = _make_obs(grav_z=-1.0, ang_vel_z=1.0)  # CCW spin
        action = ctrl.compute_action(obs)
        l_wheel = float(action[4])
        r_wheel = float(action[9])
        assert l_wheel > r_wheel, (
            f"Expected l_wheel > r_wheel for CCW spin damping, got l={l_wheel:.4f}, r={r_wheel:.4f}"
        )

    def test_left_lean_increases_l_hip_roll(self):
        """Left lean (grav_x > 0) → l_hip_roll and r_hip_roll antisymmetric."""
        ctrl = _make_controller()
        ctrl.reset()
        obs_left = _make_obs(grav_x=0.3)
        obs_right = _make_obs(grav_x=-0.3)
        action_left = ctrl.compute_action(obs_left)
        action_right = ctrl.compute_action(obs_right)
        # l_hip_roll is index 0, r_hip_roll is index 5
        # left lean → positive roll correction
        assert action_left[0] > action_right[0], "l_hip_roll should be higher for left lean"
        assert action_left[5] < action_right[5], (
            "r_hip_roll should be lower for left lean (antisymmetric)"
        )

    def test_zero_yaw_gains_means_no_differential(self):
        """kp_yaw=0 and kd_yaw=0 → no differential wheel correction."""
        ctrl = _make_controller(kp_yaw=0.0, kd_yaw=0.0)
        ctrl.reset(height_cmd_m=0.65)
        obs = _make_obs(grav_z=-1.0, yaw_error=0.8, ang_vel_z=2.0)
        action = ctrl.compute_action(obs)
        l_wheel = float(action[4])
        r_wheel = float(action[9])
        assert abs(l_wheel - r_wheel) < 1e-6, (
            f"Expected l_wheel == r_wheel with zero yaw gains, "
            f"got diff={abs(l_wheel - r_wheel):.6f}"
        )


# ---------------------------------------------------------------------------
# Height regulation
# ---------------------------------------------------------------------------


class TestHeightRegulation:
    def test_different_heights_different_hip_knee(self):
        """Different height commands should produce different hip/knee targets."""
        ctrl = _make_controller()
        ctrl.reset(height_cmd_m=0.40)
        obs_low = _make_obs(height_cmd_norm=0.0)  # normalised 0 = min height
        action_low = ctrl.compute_action(obs_low)

        ctrl.reset(height_cmd_m=0.70)
        obs_high = _make_obs(height_cmd_norm=1.0)  # normalised 1 = max height
        action_high = ctrl.compute_action(obs_high)

        # hip_pitch is index 2 (l_hip_pitch) and 7 (r_hip_pitch)
        assert action_low[2] != pytest.approx(action_high[2], abs=0.05), (
            "l_hip_pitch should differ between min and max height"
        )

    def test_hip_yaw_neutral(self):
        """Hip yaw joints should always target neutral (midpoint = 0 rad)."""
        ctrl = _make_controller()
        ctrl.reset()
        obs = _make_obs()
        action = ctrl.compute_action(obs)
        # l_hip_yaw index 1, r_hip_yaw index 6
        # Neutral target for joint range [-0.4, 0.4] is 0 rad → normalised = 0.0
        assert action[1] == pytest.approx(0.0, abs=0.01), (
            f"l_hip_yaw should be neutral, got {action[1]}"
        )
        assert action[6] == pytest.approx(0.0, abs=0.01), (
            f"r_hip_yaw should be neutral, got {action[6]}"
        )


# ---------------------------------------------------------------------------
# kd_yaw contribution
# ---------------------------------------------------------------------------


class TestKdYawContribution:
    def test_kd_yaw_zero_vs_nonzero_changes_output(self):
        """When yaw_rate is nonzero, kd_yaw > 0 changes the wheel differential."""
        ctrl_no_kd = _make_controller(kp_yaw=2.5, kd_yaw=0.0)
        ctrl_with_kd = _make_controller(kp_yaw=2.5, kd_yaw=0.5)
        ctrl_no_kd.reset(height_cmd_m=0.65)
        ctrl_with_kd.reset(height_cmd_m=0.65)

        obs = _make_obs(grav_z=-1.0, yaw_error=0.0, ang_vel_z=1.0)  # only rate

        action_no_kd = ctrl_no_kd.compute_action(obs)
        action_with_kd = ctrl_with_kd.compute_action(obs)

        diff_no_kd = float(action_no_kd[4]) - float(action_no_kd[9])
        diff_with_kd = float(action_with_kd[4]) - float(action_with_kd[9])

        # With kp_yaw=0 for the error part, kd_yaw should create a differential
        assert abs(diff_with_kd) > abs(diff_no_kd) + 0.001, (
            f"kd_yaw=0.5 should produce larger differential than kd_yaw=0. "
            f"no_kd_diff={diff_no_kd:.4f}, with_kd_diff={diff_with_kd:.4f}"
        )

    def test_kd_yaw_included_in_gains_info(self):
        """gains_info() must include kd_yaw for paper reporting."""
        ctrl = _make_controller(kd_yaw=0.35)
        info = ctrl.gains_info()
        assert "kd_yaw" in info, "kd_yaw missing from gains_info()"
        assert info["kd_yaw"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# gains_info() completeness
# ---------------------------------------------------------------------------


class TestGainsInfo:
    _REQUIRED_KEYS = [
        "lqr_gains_K",
        "K_pitch",
        "K_pitch_rate",
        "K_fwd_vel",
        "K_fwd_pos",
        "kp_roll",
        "kd_roll",
        "kp_yaw",
        "kd_yaw",
        "wheel_vel_limit_rads",
        "l_com_m",
        "r_wheel_m",
        "ik_h_min_m",
        "ik_h_max_m",
    ]

    def test_all_keys_present(self):
        ctrl = _make_controller()
        info = ctrl.gains_info()
        for k in self._REQUIRED_KEYS:
            assert k in info, f"gains_info() missing key: {k}"

    def test_lqr_gains_k_is_list_of_4(self):
        ctrl = _make_controller()
        info = ctrl.gains_info()
        assert isinstance(info["lqr_gains_K"], list)
        assert len(info["lqr_gains_K"]) == 4

    def test_physical_gains_all_positive(self):
        """Absolute-value gains (K_pitch etc.) reported in gains_info should be > 0."""
        ctrl = _make_controller()
        info = ctrl.gains_info()
        for key in ("K_pitch", "K_pitch_rate", "K_fwd_vel", "K_fwd_pos"):
            assert info[key] > 0, f"gains_info()['{key}'] should be positive, got {info[key]}"

    def test_ik_range_from_mock(self):
        ctrl = _make_controller()
        info = ctrl.gains_info()
        assert info["ik_h_min_m"] == pytest.approx(_MOCK_H_MIN)
        assert info["ik_h_max_m"] == pytest.approx(_MOCK_H_MAX)
