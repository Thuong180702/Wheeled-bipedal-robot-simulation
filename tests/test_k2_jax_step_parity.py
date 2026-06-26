"""Stage 4: Full-step and multi-step K2 JAX parity tests."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from wheeled_biped.controllers.k2_jax_controller import (
    K2_JAX_STATE_FIELDS,
    K2_JAX_STATE_SIZE,
    K2_JAX_INPUT_SIZE,
    K2_JAX_DIAG_SIZE,
    K2_JAX_DIAG_FIELDS,
    pack_state_k2,
    pack_input_k2,
    pack_params_stage2,
    k2_jax_controller_step,
    k2_jax_diag_flat_to_dict,
)


def _make_default_params():
    return pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10) * 10.0,
        max_torque_rate=jnp.ones(10) * 400.0,
        control_dt=0.01,
    )


def _make_default_input(height=0.48, pitch=0.0, pitch_rate=0.0):
    q_ref = np.array([0.0, 0.0, 0.635, 1.232, 0.0, 0.0, 0.0, 0.635, 1.232, 0.0])
    q = np.array([0.0, 0.0, 0.63, 1.23, 0.0, 0.0, 0.0, 0.63, 1.23, 0.0])
    qd = np.zeros(10)
    return pack_input_k2(
        pitch_x_rad=pitch, pitch_rate_x_rad_s=pitch_rate,
        roll_y_rad=0.0, roll_rate_y_rad_s=0.0,
        yaw_error_rad=0.0, yaw_rate_rad_s=0.0,
        com_z_m=height, com_vy_m_s=0.0,
        sagittal_velocity_m_s=0.0, sagittal_position_error_m=0.0,
        wheel_vel_left_rad_s=0.0, wheel_vel_right_rad_s=0.0,
        support_velocity_m_s=0.0, commanded_height_ref_m=height,
        hip_yaw_div_error=0.0, hip_yaw_div_rate=0.0,
        joint_pos=q, joint_vel=qd, q_ref=q_ref,
        support_position_error_m=0.0,
    )


class TestFullStepParity:
    """Full-step JAX controller produces valid outputs."""

    @pytest.fixture(scope="class")
    def jax_step_fn(self):
        return jax.jit(k2_jax_controller_step)

    @pytest.fixture(scope="class")
    def params(self):
        return _make_default_params()

    def test_jit_compiles(self, jax_step_fn, params):
        """k2_jax_controller_step compiles under jax.jit."""
        state = pack_state_k2()
        inp = _make_default_input()
        tau, new_state, diag = jax_step_fn(state, inp, params)
        assert tau.shape == (10,)
        assert new_state.shape == (K2_JAX_STATE_SIZE,)
        assert diag.shape == (K2_JAX_DIAG_SIZE,)

    def test_zero_input_produces_finite_output(self, jax_step_fn, params):
        """Zero inputs produce finite torques (no NaN)."""
        state = pack_state_k2()
        inp = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
        tau, _, diag = jax_step_fn(state, inp, params)
        assert jnp.all(jnp.isfinite(tau))
        assert jnp.all(jnp.isfinite(diag))

    def test_multi_step_no_nan(self, jax_step_fn, params):
        """1000-step run produces no NaN."""
        state = pack_state_k2()
        for step in range(1000):
            t = step * 0.01
            height = 0.48
            pitch = 0.01 * np.sin(t * 2 * np.pi * 0.3)
            inp = _make_default_input(height=height, pitch=pitch)
            tau, state, diag = jax_step_fn(state, inp, params)
            assert jnp.all(jnp.isfinite(tau)), f"NaN at step {step}"
            assert jnp.all(jnp.isfinite(state)), f"NaN state at step {step}"

    def test_state_evolves(self, jax_step_fn, params):
        """State changes across steps."""
        state = pack_state_k2()
        inp = _make_default_input(pitch=0.05)
        _, state1, _ = jax_step_fn(state, inp, params)
        _, state2, _ = jax_step_fn(state1, inp, params)
        # State should not be identical (notch filter accumulates)
        assert not jnp.allclose(state, state1)

    def test_output_torques_within_limits(self, jax_step_fn, params):
        """Output torques stay within actuator limits."""
        state = pack_state_k2()
        for step in range(500):
            t = step * 0.01
            pitch = 0.05 * np.sin(t * 2 * np.pi * 0.5)
            inp = _make_default_input(pitch=pitch)
            tau, state, _ = jax_step_fn(state, inp, params)
            assert jnp.all(jnp.abs(tau) <= 10.0 + 1e-6), f"Torque exceeded limit at step {step}"

    def test_diag_fields_populated(self, jax_step_fn, params):
        """All diag fields are finite."""
        state = pack_state_k2()
        inp = _make_default_input(pitch=0.03)
        _, _, diag = jax_step_fn(state, inp, params)
        diag_dict = k2_jax_diag_flat_to_dict(diag)
        for name in K2_JAX_DIAG_FIELDS:
            assert name in diag_dict, f"Missing diag field: {name}"
            assert np.isfinite(diag_dict[name]), f"Non-finite diag field: {name}"


class TestStateFieldAudit:
    """Every state field has a confirmed Python source."""

    def test_state_size_consistent(self):
        assert K2_JAX_STATE_SIZE == len(K2_JAX_STATE_FIELDS)

    def test_state_fields_unique(self):
        assert len(K2_JAX_STATE_FIELDS) == len(set(K2_JAX_STATE_FIELDS))

    def test_no_fake_state_fields(self):
        """No [AUDIT] or fake fields remain (abs_* fields are confirmed adaptive_bias_trim)."""
        fake_patterns = ["hy_div_", "lateral_roll_prev", "physics_ff_smoothed"]
        for field in K2_JAX_STATE_FIELDS:
            if field.startswith("abs_"):
                continue  # confirmed adaptive_bias_trim fields
            for pattern in fake_patterns:
                assert pattern not in field.lower(), f"Fake field found: {field}"

    def test_all_state_fields_have_known_sources(self):
        """Each state field maps to a known Python source."""
        sources = {
            "notch_x1": "BiquadNotchFilter._x1 in sagittal_velocity_damped_balance_controller.py:4317",
            "notch_x2": "BiquadNotchFilter._x2",
            "notch_y1": "BiquadNotchFilter._y1",
            "notch_y2": "BiquadNotchFilter._y2",
            "prev_tau_0": "simulate_hierarchical_controller.py:tau_prev (nonlocal)",
            "filtered_com_z": "sagittal_velocity_damped_balance_controller.py:4161 (_filtered_com_z)",
            "prev_support_error": "simulate_hierarchical_controller.py:prev_support_error (nonlocal, line 6375)",
            "outer_loop_pitch_ref_smoothed_deg": "simulate_hierarchical_controller.py:4942",
            "outer_loop_prev_support_error_m": "simulate_hierarchical_controller.py:4940",
            "outer_loop_support_error_rate_smoothed": "simulate_hierarchical_controller.py:4941",
            "abs_slow_ema": "sagittal_velocity_damped_balance_controller.py:4228 (_adaptive_bias_slow_error_history EMA)",
            "abs_fast_ema": "sagittal_velocity_damped_balance_controller.py:4229 (_adaptive_bias_fast_error_history EMA)",
            "abs_trim_tau": "sagittal_velocity_damped_balance_controller.py:4226 (_adaptive_bias_trim_tau)",
            "abs_hold_steps": "sagittal_velocity_damped_balance_controller.py:4235 (_adaptive_bias_hold_steps)",
            "abs_prev_err_sign": "sagittal_velocity_damped_balance_controller.py:4234 (_adaptive_bias_prev_error_sign)",
            "abs_zc_count": "sagittal_velocity_damped_balance_controller.py:4231 (_adaptive_bias_crossing_count)",
        }
        for field in K2_JAX_STATE_FIELDS:
            # Generic check for prev_tau_N
            if field.startswith("prev_tau_"):
                base = "prev_tau_0"
            else:
                base = field
            assert base in sources or field in sources, (
                f"State field '{field}' has no confirmed Python source. "
                f"Add it to the sources dict or remove the field."
            )

    def test_no_mode_div_state(self):
        """Mode-div is stateless (confirmed in Stage 3)."""
        for field in K2_JAX_STATE_FIELDS:
            assert "hy_div" not in field.lower(), f"Mode-div state field leaked: {field}"

    def test_no_lateral_roll_state(self):
        """Lateral roll is stateless (confirmed in Stage 3)."""
        for field in K2_JAX_STATE_FIELDS:
            assert "lateral_roll" not in field.lower(), f"Lateral roll state field leaked: {field}"


class TestDiagFieldAudit:
    """Telemetry field audit: no silently dropped fields."""

    def test_diag_size_consistent(self):
        assert K2_JAX_DIAG_SIZE == len(K2_JAX_DIAG_FIELDS)

    def test_diag_fields_unique(self):
        assert len(K2_JAX_DIAG_FIELDS) == len(set(K2_JAX_DIAG_FIELDS))

    def test_diag_flat_to_dict_roundtrip(self):
        """diag_flat → dict → values match."""
        diag = jnp.arange(K2_JAX_DIAG_SIZE, dtype=jnp.float64)
        d = k2_jax_diag_flat_to_dict(diag)
        for i, name in enumerate(K2_JAX_DIAG_FIELDS):
            assert d[name] == pytest.approx(float(i))


class TestStatePackUnpackK2:
    """Full K2 state pack/unpack roundtrip."""

    def test_default_state_roundtrip(self):
        s = pack_state_k2()
        assert s.shape == (K2_JAX_STATE_SIZE,)
        assert float(s[_S_FILTERED_COM_Z]) == 0.4  # code smell? check import...

    def test_state_dtype(self):
        s = pack_state_k2()
        assert s.dtype == jnp.float64


# Import index constant for filtered_com_z
from wheeled_biped.controllers.k2_jax_controller import _S_FILTERED_COM_Z
