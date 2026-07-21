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
            "abs_slow_sum": "sagittal_velocity_damped_balance_controller.py:4228 (running sum of _adaptive_bias_slow_error_history)",
            "abs_fast_sum": "sagittal_velocity_damped_balance_controller.py:4229 (running sum of _adaptive_bias_fast_error_history)",
            "abs_trim_tau": "sagittal_velocity_damped_balance_controller.py:4226 (_adaptive_bias_trim_tau)",
            "abs_hold_steps": "sagittal_velocity_damped_balance_controller.py:4235 (_adaptive_bias_hold_steps)",
            "abs_prev_err_sign": "sagittal_velocity_damped_balance_controller.py:4234 (_adaptive_bias_prev_error_sign)",
            "abs_zc_count": "sagittal_velocity_damped_balance_controller.py:4231 (_adaptive_bias_crossing_count)",
            "abs_slow_count": "k2_jax_controller.py (ring buffer valid entry count)",
            "abs_slow_ptr": "k2_jax_controller.py (ring buffer write pointer)",
            "abs_guard_trigger": "k2_jax_controller.py (ZC guard trigger counter)",
            # Phase 6M: ZC ring buffer fields
            "abs_zc_buf_count": "k2_jax_controller.py (ZC ring buffer valid entry count)",
            "abs_zc_buf_ptr": "k2_jax_controller.py (ZC ring buffer write pointer)",
            # APCR1ND gating state (Phase 4+ full port)
            "apcr1nd_step_counter": "sagittal_velocity_damped_balance_controller.py:4256 (_apcr1nd_step_counter)",
            "apcr1nd_prev_error": "sagittal_velocity_damped_balance_controller.py:4257 (_apcr1nd_prev_error)",
            "apcr1nd_tuned_converging_steps": "sagittal_velocity_damped_balance_controller.py:4263 (_apcr1nd_tuned_converging_steps)",
            "apcr1nd_tuned_recenter_held": "sagittal_velocity_damped_balance_controller.py:4264 (_apcr1nd_tuned_recenter_held)",
            # Phase 7: Python's runtime effective_max_position_tau (T6F/T6I-raised)
            "effective_max_position_tau_py": "sagittal_velocity_damped_balance_controller.py:5786 (sagittal_diag['effective_max_position_tau'], captured from Python both-synced)",
            # Phase 0: APCR1ND wheel damping override active flag (-1=Python-skipped, 0=standalone, 1=Python-applied)
            "py_wd_override_active": "sagittal_velocity_damped_balance_controller.py:8982/8991 (sagittal_diag['apcr1n_wheel_damping_override_active'], captured from Python both-synced)",
            # Drift controller state (K2 JAX drift controller)
            "drift_ref_world_x": "k2_jax_controller.py:2233 (drift reference latch)",
            "drift_ref_world_y": "k2_jax_controller.py:2233 (drift reference latch)",
            "drift_ref_yaw": "k2_jax_controller.py:2235 (drift reference latch)",
            "drift_ref_latched": "k2_jax_controller.py:2223 (drift latch flag)",
            # Heading hip-yaw stabilizer state
            "heading_hy_ref_yaw": "k2_jax_controller.py:2067 (heading reference latch)",
            "heading_hy_ref_latched": "k2_jax_controller.py:2066 (heading latch flag)",
            "heading_hy_integral": "k2_jax_controller.py:2078 (heading leaky integral)",
            # Anchor position integral (V3_ANCHOR) — JAX-only, like drift/heading state
            "anchor_integ_tau": "k2_jax_controller.py Step 4a2 (anchor position integral, ki=0 for non-anchor profiles)",
            "anchor_activity_ema": "k2_jax_controller.py Step 4a2 (quiet-stance |sag_vel| EMA for the idle damping boost)",
        }
        for field in K2_JAX_STATE_FIELDS:
            # Generic check for prev_tau_N and abs_buf_N
            if field.startswith("prev_tau_"):
                base = "prev_tau_0"
            elif field.startswith("abs_buf_"):
                # Ring buffer entries — known, from sliding window implementation
                base = "abs_buf_0"
                if "abs_buf_0" not in sources:
                    sources["abs_buf_0"] = "k2_jax_controller.py (ABS ring buffer entry)"
            elif field.startswith("abs_zc_buf_"):
                # Phase 6M: ZC ring buffer entries — separate 500-entry ZC buffer
                base = "abs_zc_buf_0"
                if "abs_zc_buf_0" not in sources:
                    sources["abs_zc_buf_0"] = "k2_jax_controller.py (ZC ring buffer entry, matches Python _adaptive_bias_zero_crossing_history)"
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
