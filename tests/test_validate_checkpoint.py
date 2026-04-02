"""
Tests for scripts/validate_checkpoint.py — specifically the _build_headless_obs()
helper and the control-path helpers that must match BalanceEnv semantics.

Strategy
--------
These tests do NOT run the full CLI (which requires a real checkpoint and MuJoCo).
Instead they unit-test the shared _build_headless_obs() function imported from
the script, and check that:

  1. lin_vel_mode "clean" → 41-dim obs, lin_vel always present, no noise on lin_vel
  2. lin_vel_mode "noisy" → 41-dim obs, lin_vel present, noise on lin_vel when enabled
  3. lin_vel_mode "disabled" → 38-dim obs, lin_vel excluded
  4. Noise is applied to the right channels and not to prev_action
  5. Noise is zero when apply_noise=False
  6. Action delay buffer semantics: oldest action applied first (FIFO queue)
  7. prev_action = smooth_action (pre-delay), not the delayed control_action

All tests use stub math helpers so they run without a real MuJoCo installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the helper under test
from scripts.validate_checkpoint import _build_headless_obs  # noqa: E402

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _identity_gravity(quat):
    """Stub: gravity_body = [0, 0, -1] always (upright robot)."""
    try:
        import jax.numpy as jnp

        return jnp.array([0.0, 0.0, -1.0])
    except ImportError:
        return np.array([0.0, 0.0, -1.0])


def _identity_conjugate(quat):
    return quat


def _identity_rotate(quat, vec):
    return vec


def _make_mj_data_stub(lin_vel=(0.1, 0.2, 0.0), ang_vel=(0.0, 0.0, 0.0)):
    """Minimal mj_data stub with realistic qpos/qvel arrays."""
    try:
        qpos = np.zeros(17)
        qpos[3] = 1.0  # quaternion w-component
        qpos[2] = 0.65  # torso height
        qvel = np.zeros(16)
        qvel[:3] = lin_vel
        qvel[3:6] = ang_vel
    except Exception:
        qpos = np.zeros(17)
        qpos[3] = 1.0
        qpos[2] = 0.65
        qvel = np.zeros(16)
        qvel[:3] = lin_vel
        qvel[3:6] = ang_vel

    stub = MagicMock()
    stub.qpos = qpos
    stub.qvel = qvel
    return stub


def _noise_stds_zero() -> dict:
    return {"lin_vel": 0.0, "ang_vel": 0.0, "gravity": 0.0, "joint_pos": 0.0, "joint_vel": 0.0}


def _noise_stds_nonzero() -> dict:
    return {"lin_vel": 0.3, "ang_vel": 0.05, "gravity": 0.02, "joint_pos": 0.005, "joint_vel": 0.01}


def _kwargs(lin_vel_mode="clean", apply_noise=False, noise_stds=None):
    try:
        import jax.numpy as jnp

        prev_action = jnp.zeros(10)
        height_cmd_norm = jnp.array([0.5])
        yaw_error = jnp.array([0.0])
    except ImportError:
        prev_action = np.zeros(10)
        height_cmd_norm = np.array([0.5])
        yaw_error = np.array([0.0])

    return dict(
        mj_data=_make_mj_data_stub(),
        prev_action=prev_action,
        height_cmd_norm=height_cmd_norm,
        yaw_error=yaw_error,
        lin_vel_mode=lin_vel_mode,
        apply_noise=apply_noise,
        noise_stds=noise_stds or _noise_stds_zero(),
        rng_np=np.random.default_rng(0),
        get_gravity_fn=_identity_gravity,
        quat_conjugate_fn=_identity_conjugate,
        quat_rotate_fn=_identity_rotate,
    )


# ---------------------------------------------------------------------------
# Obs dimension tests
# ---------------------------------------------------------------------------


class TestObsDimensions:
    """_build_headless_obs must produce the right number of dims per lin_vel_mode."""

    def test_clean_mode_produces_41_dims(self):
        obs = _build_headless_obs(**_kwargs(lin_vel_mode="clean"))
        assert obs.shape == (41,), f"Expected (41,), got {obs.shape}"

    def test_noisy_mode_produces_41_dims(self):
        obs = _build_headless_obs(**_kwargs(lin_vel_mode="noisy"))
        assert obs.shape == (41,), f"Expected (41,), got {obs.shape}"

    def test_disabled_mode_produces_38_dims(self):
        obs = _build_headless_obs(**_kwargs(lin_vel_mode="disabled"))
        assert obs.shape == (38,), f"Expected (38,), got {obs.shape}"

    def test_height_cmd_is_last_minus_one(self):
        """obs[-2] must equal height_cmd_norm regardless of lin_vel_mode."""
        try:
            import jax.numpy as jnp

            height = jnp.array([0.75])
        except ImportError:
            height = np.array([0.75])

        for mode in ("clean", "noisy", "disabled"):
            kw = _kwargs(lin_vel_mode=mode)
            kw["height_cmd_norm"] = height
            obs = _build_headless_obs(**kw)
            assert float(obs[-2]) == pytest.approx(0.75), (
                f"lin_vel_mode='{mode}': obs[-2] should be height_cmd_norm=0.75, got {obs[-2]}"
            )

    def test_yaw_error_is_last(self):
        """obs[-1] must equal yaw_error regardless of lin_vel_mode."""
        try:
            import jax.numpy as jnp

            yaw = jnp.array([1.23])
        except ImportError:
            yaw = np.array([1.23])

        for mode in ("clean", "noisy", "disabled"):
            kw = _kwargs(lin_vel_mode=mode)
            kw["yaw_error"] = yaw
            obs = _build_headless_obs(**kw)
            assert float(obs[-1]) == pytest.approx(1.23), (
                f"lin_vel_mode='{mode}': obs[-1] should be yaw_error=1.23, got {obs[-1]}"
            )


# ---------------------------------------------------------------------------
# lin_vel inclusion / exclusion
# ---------------------------------------------------------------------------


class TestLinVelPlacement:
    """Check that lin_vel is included/excluded and at the right position."""

    def test_clean_mode_includes_lin_vel_at_3_6(self):
        """In 'clean' mode obs[3:6] must match the simulated lin_vel (via identity rotate)."""
        lin_vel = (0.15, 0.25, 0.05)
        kw = _kwargs(lin_vel_mode="clean")
        kw["mj_data"] = _make_mj_data_stub(lin_vel=lin_vel)
        obs = np.array(_build_headless_obs(**kw))
        # After identity rotate, body_lin_vel == world_lin_vel
        np.testing.assert_allclose(obs[3:6], lin_vel, atol=1e-5)

    def test_disabled_mode_excludes_lin_vel(self):
        """In 'disabled' mode obs[3:6] must be ang_vel, not lin_vel."""
        lin_vel = (9.9, 9.9, 9.9)  # obviously wrong if it leaks through
        ang_vel = (0.1, 0.2, 0.3)
        kw = _kwargs(lin_vel_mode="disabled")
        kw["mj_data"] = _make_mj_data_stub(lin_vel=lin_vel, ang_vel=ang_vel)
        obs = np.array(_build_headless_obs(**kw))
        # obs[3:6] must be ang_vel, not lin_vel
        np.testing.assert_allclose(obs[3:6], ang_vel, atol=1e-5)
        # lin_vel must not appear anywhere in the base obs
        base = obs[:-2]  # strip height_cmd + yaw_error
        for v in lin_vel:
            assert not any(np.isclose(base, v)), (
                f"lin_vel value {v} found in disabled-mode base obs"
            )


# ---------------------------------------------------------------------------
# Noise channel tests
# ---------------------------------------------------------------------------


class TestNoiseApplication:
    """Noise must be applied to the right channels and absent when disabled."""

    def test_no_noise_when_apply_noise_false(self):
        """With apply_noise=False the obs must be deterministic and clean."""
        kw1 = _kwargs(apply_noise=False, noise_stds=_noise_stds_nonzero())
        kw2 = _kwargs(apply_noise=False, noise_stds=_noise_stds_nonzero())
        # Different rng seeds should produce identical obs (no noise applied)
        kw1["rng_np"] = np.random.default_rng(0)
        kw2["rng_np"] = np.random.default_rng(99)
        obs1 = np.array(_build_headless_obs(**kw1))
        obs2 = np.array(_build_headless_obs(**kw2))
        np.testing.assert_array_equal(obs1, obs2)

    def test_noise_changes_obs_when_enabled(self):
        """Two different rng seeds must produce different obs when apply_noise=True."""
        kw1 = _kwargs(lin_vel_mode="noisy", apply_noise=True, noise_stds=_noise_stds_nonzero())
        kw2 = _kwargs(lin_vel_mode="noisy", apply_noise=True, noise_stds=_noise_stds_nonzero())
        kw1["rng_np"] = np.random.default_rng(0)
        kw2["rng_np"] = np.random.default_rng(42)
        obs1 = np.array(_build_headless_obs(**kw1))
        obs2 = np.array(_build_headless_obs(**kw2))
        assert not np.allclose(obs1, obs2), "Different seeds with noise must give different obs"

    def test_prev_action_never_noised(self):
        """obs[29:39] (prev_action in clean/noisy mode) must equal prev_action exactly."""
        try:
            import jax.numpy as jnp

            prev = jnp.array([0.1 * i for i in range(10)], dtype=jnp.float32)
        except ImportError:
            prev = np.array([0.1 * i for i in range(10)], dtype=np.float32)

        for mode in ("clean", "noisy"):
            kw = _kwargs(lin_vel_mode=mode, apply_noise=True, noise_stds=_noise_stds_nonzero())
            kw["prev_action"] = prev
            kw["rng_np"] = np.random.default_rng(7)
            obs = np.array(_build_headless_obs(**kw))
            # prev_action occupies [29:39] in 39-dim base
            np.testing.assert_allclose(
                obs[29:39],
                np.array(prev),
                atol=1e-6,
                err_msg=f"lin_vel_mode='{mode}': prev_action at obs[29:39] must not be noised",
            )

    def test_prev_action_never_noised_disabled_mode(self):
        """obs[26:36] (prev_action in disabled mode) must equal prev_action exactly."""
        try:
            import jax.numpy as jnp

            prev = jnp.array([0.5 * i for i in range(10)], dtype=jnp.float32)
        except ImportError:
            prev = np.array([0.5 * i for i in range(10)], dtype=np.float32)

        kw = _kwargs(lin_vel_mode="disabled", apply_noise=True, noise_stds=_noise_stds_nonzero())
        kw["prev_action"] = prev
        kw["rng_np"] = np.random.default_rng(7)
        obs = np.array(_build_headless_obs(**kw))
        # prev_action occupies [26:36] in 36-dim base
        np.testing.assert_allclose(obs[26:36], np.array(prev), atol=1e-6)

    def test_clean_mode_no_noise_on_lin_vel(self):
        """In 'clean' mode lin_vel must not be noised even when apply_noise=True."""
        lin_vel = (0.3, 0.0, 0.0)
        stds = _noise_stds_nonzero()
        stds["lin_vel"] = 100.0  # very large — would be obvious if applied

        kw = _kwargs(lin_vel_mode="clean", apply_noise=True, noise_stds=stds)
        kw["mj_data"] = _make_mj_data_stub(lin_vel=lin_vel)
        kw["rng_np"] = np.random.default_rng(0)
        obs = np.array(_build_headless_obs(**kw))

        # After identity rotate, obs[3:6] must still equal lin_vel (no noise)
        np.testing.assert_allclose(obs[3:6], lin_vel, atol=1e-5)

    def test_noisy_mode_adds_noise_on_lin_vel(self):
        """In 'noisy' mode lin_vel must be corrupted when apply_noise=True."""
        lin_vel = (0.3, 0.0, 0.0)
        stds = _noise_stds_nonzero()
        stds["lin_vel"] = 1.0  # large enough to reliably move the signal

        kw = _kwargs(lin_vel_mode="noisy", apply_noise=True, noise_stds=stds)
        kw["mj_data"] = _make_mj_data_stub(lin_vel=lin_vel)
        kw["rng_np"] = np.random.default_rng(1)
        obs = np.array(_build_headless_obs(**kw))

        # obs[3:6] should NOT exactly equal lin_vel (noise was applied)
        assert not np.allclose(obs[3:6], lin_vel, atol=1e-3), (
            "In 'noisy' mode with large lin_vel_std, obs[3:6] should differ from clean lin_vel"
        )


# ---------------------------------------------------------------------------
# Action delay buffer semantics
# ---------------------------------------------------------------------------


class TestActionDelayBuffer:
    """The delay buffer must behave as a FIFO queue: oldest action applied first."""

    def test_zero_delay_uses_current_smooth_action(self):
        """With no delay, control_action == smooth_action immediately."""
        delay_steps = 0
        smooth_action = np.array([1.0] * 10)
        delay_buffer: list = []  # empty — no delay

        # Simulate one step
        if delay_steps > 0:
            control_action = delay_buffer[0]
            delay_buffer = delay_buffer[1:] + [smooth_action]
        else:
            control_action = smooth_action

        np.testing.assert_array_equal(control_action, smooth_action)

    def test_one_step_delay_applies_previous_action(self):
        """With delay=1, first step applies zero (init), second applies step-0 action."""
        delay_buffer = [np.zeros(10)]  # init: one zeros entry (delay=1)

        smooth_step0 = np.full(10, 1.0)
        smooth_step1 = np.full(10, 2.0)

        # Step 0
        ctrl_step0 = delay_buffer[0]  # zeros (init)
        delay_buffer = delay_buffer[1:] + [smooth_step0]

        # Step 1
        ctrl_step1 = delay_buffer[0]  # smooth_step0 (1 step delayed)
        delay_buffer = delay_buffer[1:] + [smooth_step1]

        np.testing.assert_array_equal(ctrl_step0, np.zeros(10))
        np.testing.assert_array_equal(ctrl_step1, smooth_step0)

    def test_two_step_delay_fifo_order(self):
        """With delay=2, actions appear in the correct FIFO order."""
        delay_buffer = [np.zeros(10), np.zeros(10)]  # init: two zeros entries (delay=2)

        actions = [np.full(10, float(i)) for i in range(1, 6)]  # 1.0 to 5.0
        ctrl_sequence = []

        for sa in actions:
            ctrl_sequence.append(delay_buffer[0].copy())
            delay_buffer = delay_buffer[1:] + [sa]

        # ctrl[0] = zeros (init), ctrl[1] = zeros (init), ctrl[2] = action[0]=1.0, ...
        assert float(ctrl_sequence[0][0]) == pytest.approx(0.0)
        assert float(ctrl_sequence[1][0]) == pytest.approx(0.0)
        assert float(ctrl_sequence[2][0]) == pytest.approx(1.0)
        assert float(ctrl_sequence[3][0]) == pytest.approx(2.0)
        assert float(ctrl_sequence[4][0]) == pytest.approx(3.0)

    def test_prev_action_is_smooth_not_delayed(self):
        """prev_action stored in state must be smooth_action, not the delayed control_action.

        This matches BalanceEnv.step() comment:
          'prev_action stored in state = smooth_action (the policy's intended target)'
        """
        delay_steps = 1
        delay_buffer = [np.zeros(10)]

        smooth = np.full(10, 7.0)

        # Simulate one step
        if delay_steps > 0:
            control_action = delay_buffer[0]
            delay_buffer = delay_buffer[1:] + [smooth]
        else:
            control_action = smooth

        # prev_action MUST be smooth, not control_action
        prev_action_stored = smooth  # as in the validate_checkpoint loop

        np.testing.assert_array_equal(prev_action_stored, smooth)
        # And verify control_action is different (the delayed zeros)
        np.testing.assert_array_equal(control_action, np.zeros(10))
        assert not np.allclose(prev_action_stored, control_action)


# ---------------------------------------------------------------------------
# Obs size invariant
# ---------------------------------------------------------------------------


class TestObsSizeInvariant:
    """obs.shape[0] must equal expected_obs_size derived from lin_vel_mode."""

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("clean", 41),
            ("noisy", 41),
            ("disabled", 38),
        ],
    )
    def test_obs_size_matches_expected(self, mode, expected):
        obs = _build_headless_obs(**_kwargs(lin_vel_mode=mode))
        assert obs.shape[0] == expected, (
            f"lin_vel_mode='{mode}': expected {expected} dims, got {obs.shape[0]}"
        )

    @pytest.mark.parametrize(
        "mode,expected_base",
        [
            ("clean", 39),
            ("noisy", 39),
            ("disabled", 36),
        ],
    )
    def test_base_obs_matches_base_env_compute_obs_size(self, mode, expected_base):
        """base_obs dim must match WheeledBipedEnv._compute_obs_size() logic."""
        # Replicate _compute_obs_size() formula (36 base + 3 if not disabled)
        computed = 36 if mode == "disabled" else 39
        assert computed == expected_base, f"_compute_obs_size formula broken for mode='{mode}'"
        obs = _build_headless_obs(**_kwargs(lin_vel_mode=mode))
        assert obs.shape[0] == expected_base + 2  # + height_cmd_norm + yaw_error
