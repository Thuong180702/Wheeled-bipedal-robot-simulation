"""
Tests for sim helper functions.

Covers:
  apply_push_disturbance  (sim.push_disturbance)
  pid_control             (sim.low_level_control)
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def fake_mjx_data(mj_model):
    """Minimal MJX data in standing pose."""
    from mujoco import mjx
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


# ---------------------------------------------------------------------------
# apply_push_disturbance
# ---------------------------------------------------------------------------

class TestApplyPushDisturbance:
    """Tests for wheeled_biped.sim.push_disturbance.apply_push_disturbance."""

    def test_returns_correct_types(self, mj_model, fake_mjx_data):
        """Returns (mjx.Data, jax.Array) with no exceptions."""
        from wheeled_biped.sim.push_disturbance import apply_push_disturbance

        rng = jax.random.PRNGKey(0)
        step_count = jnp.int32(0)
        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        mjx_data_out, new_rng = apply_push_disturbance(
            fake_mjx_data,
            rng,
            body_id=torso_id,
            step_count=step_count,
            push_interval=200,
            push_duration=5,
            push_magnitude=20.0,
            push_enabled=True,
        )
        assert new_rng.shape == rng.shape
        assert mjx_data_out.xfrc_applied.shape == fake_mjx_data.xfrc_applied.shape

    def test_disabled_clears_force(self, mj_model, fake_mjx_data):
        """push_enabled=False always produces zero xfrc_applied."""
        from wheeled_biped.sim.push_disturbance import apply_push_disturbance

        rng = jax.random.PRNGKey(1)
        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        mjx_data_out, _ = apply_push_disturbance(
            fake_mjx_data,
            rng,
            body_id=torso_id,
            step_count=jnp.int32(0),
            push_enabled=False,
        )
        assert np.allclose(np.array(mjx_data_out.xfrc_applied), 0.0), (
            "push_enabled=False should clear all external forces"
        )

    def test_outside_window_clears_force(self, mj_model, fake_mjx_data):
        """When step_count is outside the push window, force is cleared."""
        from wheeled_biped.sim.push_disturbance import apply_push_disturbance

        rng = jax.random.PRNGKey(2)
        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # step_count=50 with push_interval=200, push_duration=5 → not in window
        mjx_data_out, _ = apply_push_disturbance(
            fake_mjx_data,
            rng,
            body_id=torso_id,
            step_count=jnp.int32(50),
            push_interval=200,
            push_duration=5,
            push_enabled=True,
        )
        assert np.allclose(np.array(mjx_data_out.xfrc_applied), 0.0), (
            "Outside push window, xfrc_applied should be zero"
        )

    def test_inside_window_applies_nonzero_force(self, mj_model, fake_mjx_data):
        """When step_count is inside the push window, xfrc_applied is non-zero."""
        from wheeled_biped.sim.push_disturbance import apply_push_disturbance

        rng = jax.random.PRNGKey(3)
        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # step_count=0 → inside window [0, push_duration)
        mjx_data_out, _ = apply_push_disturbance(
            fake_mjx_data,
            rng,
            body_id=torso_id,
            step_count=jnp.int32(0),
            push_interval=200,
            push_duration=5,
            push_magnitude=20.0,
            push_enabled=True,
        )
        xfrc = np.array(mjx_data_out.xfrc_applied)
        assert not np.allclose(xfrc, 0.0), (
            "Inside push window with push_enabled=True, xfrc_applied should be non-zero"
        )

    def test_rng_changes_between_calls(self, mj_model, fake_mjx_data):
        """Returned rng differs from input rng (key was split)."""
        from wheeled_biped.sim.push_disturbance import apply_push_disturbance

        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        rng = jax.random.PRNGKey(42)
        _, new_rng = apply_push_disturbance(
            fake_mjx_data, rng, body_id=torso_id, step_count=jnp.int32(0)
        )
        assert not np.array_equal(np.array(rng), np.array(new_rng))


# ---------------------------------------------------------------------------
# pid_control
# ---------------------------------------------------------------------------

# Minimal PID config for 10-joint robot
_NUM_JOINTS = 10
_KP = jnp.ones(_NUM_JOINTS) * 50.0
_KI = jnp.ones(_NUM_JOINTS) * 0.5
_KD = jnp.ones(_NUM_JOINTS) * 3.0
_JOINT_MINS = jnp.full(_NUM_JOINTS, -1.0)
_JOINT_MAXS = jnp.full(_NUM_JOINTS, 1.0)
# Wheels at indices 4 and 9
_WHEEL_MASK = jnp.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=jnp.float32)
_CTRL_MIN = jnp.full(_NUM_JOINTS, -5.0)
_CTRL_MAX = jnp.full(_NUM_JOINTS, 5.0)


class TestPidControl:
    """Tests for wheeled_biped.sim.low_level_control.pid_control."""

    def _call(self, fake_mjx_data, target=None, integral=None):
        from wheeled_biped.sim.low_level_control import pid_control
        t = target if target is not None else jnp.zeros(_NUM_JOINTS)
        i = integral if integral is not None else jnp.zeros(_NUM_JOINTS)
        return pid_control(
            fake_mjx_data, t, i,
            kp=_KP, ki=_KI, kd=_KD,
            joint_mins=_JOINT_MINS, joint_maxs=_JOINT_MAXS,
            wheel_mask=_WHEEL_MASK,
            wheel_vel_limit=20.0,
            i_limit=0.3,
            ctrl_min=_CTRL_MIN, ctrl_max=_CTRL_MAX,
            control_dt=0.02,
        )

    def test_output_shapes(self, fake_mjx_data):
        """ctrl and new_integral both have shape (num_joints,)."""
        ctrl, new_integral = self._call(fake_mjx_data)
        assert ctrl.shape == (_NUM_JOINTS,)
        assert new_integral.shape == (_NUM_JOINTS,)

    def test_no_nan_in_output(self, fake_mjx_data):
        """No NaN in ctrl or new_integral for a valid standing pose."""
        ctrl, new_integral = self._call(fake_mjx_data)
        assert not np.any(np.isnan(np.array(ctrl))), "NaN in ctrl"
        assert not np.any(np.isnan(np.array(new_integral))), "NaN in integral"

    def test_ctrl_clipped_to_range(self, fake_mjx_data):
        """ctrl is always within [ctrl_min, ctrl_max]."""
        ctrl, _ = self._call(fake_mjx_data, target=jnp.ones(_NUM_JOINTS))
        ctrl_np = np.array(ctrl)
        assert np.all(ctrl_np >= -5.0 - 1e-6)
        assert np.all(ctrl_np <= 5.0 + 1e-6)

    def test_integral_clamped_by_i_limit(self, fake_mjx_data):
        """Integral is anti-windup clamped by i_limit=0.3."""
        # Start with a huge integral to trigger the clamp
        big_integral = jnp.full(_NUM_JOINTS, 100.0)
        _, new_integral = self._call(fake_mjx_data, integral=big_integral)
        new_i_np = np.array(new_integral)
        assert np.all(np.abs(new_i_np) <= 0.3 + 1e-6), (
            f"Integral not clamped: max={np.max(np.abs(new_i_np))}"
        )

    def test_integral_updates_each_call(self, fake_mjx_data):
        """Integral changes between calls (not frozen at zero)."""
        zero_integral = jnp.zeros(_NUM_JOINTS)
        # Non-zero target should produce nonzero error → integral grows
        target = jnp.ones(_NUM_JOINTS)  # all joints at max target
        _, new_integral = self._call(fake_mjx_data, target=target, integral=zero_integral)
        # At least some integral entries should be non-zero
        assert not np.allclose(np.array(new_integral), 0.0), (
            "Integral should update when error is nonzero"
        )
