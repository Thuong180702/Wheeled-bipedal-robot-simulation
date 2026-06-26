"""Stage 2: JAX notch filter + torque composer component parity tests.

Verifies that JAX pure-function implementations produce identical results
to the Python reference implementations (BiquadNotchFilter and
BalanceCoreTorqueComposer).

All tolerances are strict: <= 1e-10 for notch and composer outputs.
"""

import math
import random

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from wheeled_biped.controllers.signal_filters import (
    BiquadNotchFilter,
    smoothstep_gate,
    smoothstep_gate_jax,
    biquad_notch_coefficients,
    biquad_notch_update,
)
from wheeled_biped.controllers.k2_jax_controller import (
    K2_JAX_STATE_FIELDS_STAGE2,
    K2_JAX_STATE_SIZE_STAGE2,
    K2_JAX_PARAMS_FIELDS_STAGE2,
    K2_JAX_PARAMS_SIZE_STAGE2,
    pack_state_stage2,
    unpack_state_stage2,
    pack_params_stage2,
    unpack_params_stage2,
    k2_jax_notch_step,
    k2_jax_torque_composer_step,
    python_biquad_notch_update,
    python_torque_composer,
    _IDX_NOTCH_X1,
    _IDX_NOTCH_X2,
    _IDX_NOTCH_Y1,
    _IDX_NOTCH_Y2,
    _IDX_PREV_TAU_START,
)

# ===========================================================================
# Notch coefficient parity
# ===========================================================================


class TestNotchCoefficientParity:
    """Biquad notch coefficients match between pure function and class."""

    def test_k2_coefficients_match(self):
        """K2 notch (fs=100, fc=2.5, Q=2.0) coefficients match exactly."""
        fs, fc, Q = 100.0, 2.5, 2.0

        # Pure function
        b0_fn, b1_fn, b2_fn, a1_fn, a2_fn = biquad_notch_coefficients(fs, fc, Q)

        # Class
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        b0_cls, b1_cls, b2_cls, a1_cls, a2_cls = nf.coefficients()

        assert b0_fn == pytest.approx(b0_cls, abs=1e-15)
        assert b1_fn == pytest.approx(b1_cls, abs=1e-15)
        assert b2_fn == pytest.approx(b2_cls, abs=1e-15)
        assert a1_fn == pytest.approx(a1_cls, abs=1e-15)
        assert a2_fn == pytest.approx(a2_cls, abs=1e-15)

    def test_k1_coefficients_match(self):
        """K1 notch (fs=100, fc=2.5, Q=6.0) coefficients match exactly."""
        fs, fc, Q = 100.0, 2.5, 6.0

        b0_fn, b1_fn, b2_fn, a1_fn, a2_fn = biquad_notch_coefficients(fs, fc, Q)
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        b0_cls, b1_cls, b2_cls, a1_cls, a2_cls = nf.coefficients()

        assert b0_fn == pytest.approx(b0_cls, abs=1e-15)
        assert b1_fn == pytest.approx(b1_cls, abs=1e-15)
        assert b2_fn == pytest.approx(b2_cls, abs=1e-15)
        assert a1_fn == pytest.approx(a1_cls, abs=1e-15)
        assert a2_fn == pytest.approx(a2_cls, abs=1e-15)

    @pytest.mark.parametrize("fc_hz", [1.0, 2.0, 2.5, 3.0, 5.0, 10.0])
    @pytest.mark.parametrize("Q", [1.0, 2.0, 4.0, 6.0, 8.0])
    def test_coefficients_match_wide_range(self, fc_hz, Q):
        """Coefficients match across a range of fc and Q values."""
        fs = 100.0
        b0_fn, b1_fn, b2_fn, a1_fn, a2_fn = biquad_notch_coefficients(fs, fc_hz, Q)
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc_hz, Q=Q)
        b0_cls, b1_cls, b2_cls, a1_cls, a2_cls = nf.coefficients()

        assert b0_fn == pytest.approx(b0_cls, abs=1e-15)
        assert b1_fn == pytest.approx(b1_cls, abs=1e-15)
        assert b2_fn == pytest.approx(b2_cls, abs=1e-15)
        assert a1_fn == pytest.approx(a1_cls, abs=1e-15)
        assert a2_fn == pytest.approx(a2_cls, abs=1e-15)


# ===========================================================================
# Notch update parity
# ===========================================================================


class TestNotchUpdateParity:
    """Biquad notch update matches between pure function and class."""

    @staticmethod
    def _class_update(x, nf, state):
        """Apply class-based update given explicit state."""
        nf._x1, nf._x2, nf._y1, nf._y2 = state
        y = nf.update(x)
        return y, (nf._x1, nf._x2, nf._y1, nf._y2)

    @staticmethod
    def _fn_update(x, state, coeffs):
        """Apply pure-function update.

        Returns:
            (y, (x1_new, x2_new, y1_new, y2_new))
        """
        x1, x2, y1, y2 = state
        b0, b1, b2, a1, a2 = coeffs
        y, x1n, x2n, y1n, y2n = biquad_notch_update(x, x1, x2, y1, y2, b0, b1, b2, a1, a2)
        return y, (x1n, x2n, y1n, y2n)

    def test_zero_state_zero_input(self):
        """Zero state + zero input produces zero output."""
        fs, fc, Q = 100.0, 2.5, 2.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        coeffs = biquad_notch_coefficients(fs, fc, Q)

        y_cls, state_cls = self._class_update(0.0, nf, (0.0, 0.0, 0.0, 0.0))
        y_fn, state_fn = self._fn_update(0.0, (0.0, 0.0, 0.0, 0.0), coeffs)

        assert y_fn == pytest.approx(y_cls, abs=1e-15)
        for i in range(4):
            assert state_fn[i] == pytest.approx(state_cls[i], abs=1e-15)

    def test_10k_random_inputs(self):
        """10,000 random (x, state) pairs: max output diff <= 1e-10."""
        fs, fc, Q = 100.0, 2.5, 2.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        coeffs = biquad_notch_coefficients(fs, fc, Q)

        rng = random.Random(42)
        max_y_diff = 0.0
        max_state_diff = 0.0

        for _ in range(10_000):
            x = rng.uniform(-10.0, 10.0)
            state = tuple(rng.uniform(-5.0, 5.0) for _ in range(4))

            y_cls, state_cls = self._class_update(x, nf, state)
            y_fn, state_fn = self._fn_update(x, state, coeffs)

            max_y_diff = max(max_y_diff, abs(y_fn - y_cls))
            for i in range(4):
                max_state_diff = max(max_state_diff, abs(state_fn[i] - state_cls[i]))

        assert max_y_diff <= 1e-10, f"Max output diff: {max_y_diff:.2e} > 1e-10"
        assert max_state_diff <= 1e-10, f"Max state diff: {max_state_diff:.2e} > 1e-10"

    def test_impulse_response_matches(self):
        """Impulse response matches between class and function over 100 steps."""
        fs, fc, Q = 100.0, 2.5, 2.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        coeffs = biquad_notch_coefficients(fs, fc, Q)

        # Zero initial state
        fn_state = (0.0, 0.0, 0.0, 0.0)
        nf.reset()

        # Impulse at step 0
        for step in range(100):
            x = 1.0 if step == 0 else 0.0

            y_cls, cls_state = self._class_update(x, nf, (nf._x1, nf._x2, nf._y1, nf._y2))
            y_fn, fn_state = self._fn_update(x, fn_state, coeffs)

            assert y_fn == pytest.approx(y_cls, abs=1e-10), (
                f"Step {step}: output diff {abs(y_fn - y_cls):.2e}"
            )
            for i in range(4):
                assert fn_state[i] == pytest.approx(cls_state[i], abs=1e-10), (
                    f"Step {step}, state[{i}]: diff {abs(fn_state[i] - cls_state[i]):.2e}"
                )


# ===========================================================================
# Notch 1000-step stream parity
# ===========================================================================


class TestNotchStreamParity:
    """Long-stream notch filter parity: BiquadNotchFilter vs pure function."""

    def test_1000_step_stream_final_state(self):
        """1000-step sinusoidal input: final state diff <= 1e-10."""
        fs, fc, Q = 100.0, 2.5, 2.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        coeffs = biquad_notch_coefficients(fs, fc, Q)

        nf.reset()
        fn_state = (0.0, 0.0, 0.0, 0.0)

        dt = 1.0 / fs
        max_y_diff = 0.0

        for step in range(1000):
            t = step * dt
            # Mix of low-freq (0.5 Hz) and WIP-mode (2.5 Hz)
            x = 0.5 * math.sin(2 * math.pi * 0.5 * t) + 0.3 * math.sin(2 * math.pi * 2.5 * t)

            # Class update
            cls_state_before = (nf._x1, nf._x2, nf._y1, nf._y2)
            y_cls = nf.update(x)

            # Function update
            y_fn, x1n, x2n, y1n, y2n = biquad_notch_update(x, *fn_state, *coeffs)
            fn_state = (x1n, x2n, y1n, y2n)

            max_y_diff = max(max_y_diff, abs(y_fn - y_cls))

        assert max_y_diff <= 1e-10, f"Max output diff over 1000 steps: {max_y_diff:.2e}"

        # Final state parity
        cls_final = (nf._x1, nf._x2, nf._y1, nf._y2)
        for i in range(4):
            assert fn_state[i] == pytest.approx(cls_final[i], abs=1e-10), (
                f"Final state[{i}]: diff {abs(fn_state[i] - cls_final[i]):.2e}"
            )

    def test_1000_step_k2_jax_notch_step_parity(self):
        """k2_jax_notch_step matches BiquadNotchFilter over 1000 steps."""
        fs, fc, Q = 100.0, 2.5, 2.0
        nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc, Q=Q)
        nf.reset()

        params = pack_params_stage2(fs_hz=fs, fc_hz=fc, Q=Q,
                                     torque_limit=jnp.ones(10) * 10.0,
                                     max_torque_rate=jnp.ones(10) * 400.0,
                                     control_dt=0.01)
        state_jax = pack_state_stage2()

        dt = 1.0 / fs
        max_y_diff = 0.0

        for step in range(1000):
            t = step * dt
            x = 0.5 * math.sin(2 * math.pi * 0.5 * t) + 0.3 * math.sin(2 * math.pi * 2.5 * t)

            # Class
            y_cls = nf.update(x)

            # JAX k2_jax_notch_step
            y_jax, state_jax = k2_jax_notch_step(
                jnp.array(x, dtype=jnp.float64), state_jax, params
            )

            max_y_diff = max(max_y_diff, abs(float(y_jax) - y_cls))

        assert max_y_diff <= 1e-10, f"Max output diff: {max_y_diff:.2e}"

        # Final state parity
        cls_final = (nf._x1, nf._x2, nf._y1, nf._y2)
        state_unpacked = unpack_state_stage2(state_jax)
        assert state_unpacked["notch_x1"] == pytest.approx(cls_final[0], abs=1e-10)
        assert state_unpacked["notch_x2"] == pytest.approx(cls_final[1], abs=1e-10)
        assert state_unpacked["notch_y1"] == pytest.approx(cls_final[2], abs=1e-10)
        assert state_unpacked["notch_y2"] == pytest.approx(cls_final[3], abs=1e-10)


# ===========================================================================
# Smoothstep gate parity
# ===========================================================================


class TestSmoothstepGateParity:
    """smoothstep_gate_jax matches smoothstep_gate."""

    def test_boundary_values(self):
        """Test at and around gate boundaries."""
        test_points = [
            (0.40, 0.42, 0.48),  # below start
            (0.42, 0.42, 0.48),  # at start
            (0.45, 0.42, 0.48),  # middle
            (0.48, 0.42, 0.48),  # at end
            (0.50, 0.42, 0.48),  # above end
        ]
        for value, start, end in test_points:
            py_result = smoothstep_gate(value, start, end)
            jax_result = float(smoothstep_gate_jax(jnp.array(value, dtype=jnp.float64), start, end))
            assert jax_result == pytest.approx(py_result, abs=1e-15), (
                f"value={value}, start={start}, end={end}: "
                f"py={py_result}, jax={jax_result}"
            )

    def test_random_values(self):
        """Random values match within 1e-15."""
        rng = random.Random(123)
        for _ in range(1000):
            value = rng.uniform(0.30, 0.55)
            start = rng.uniform(0.40, 0.44)
            end = rng.uniform(0.46, 0.50)
            if end <= start:
                continue

            py_result = smoothstep_gate(value, start, end)
            jax_result = float(smoothstep_gate_jax(jnp.array(value, dtype=jnp.float64), start, end))
            assert jax_result == pytest.approx(py_result, abs=1e-15)

    def test_inverted_range(self):
        """end <= start returns step function (1.0 if value >= end)."""
        # With value=0.5, start=1.0, end=0.0: end <= start True, value >= end True → 1.0
        assert smoothstep_gate(0.5, 1.0, 0.0) == 1.0
        # With value=0.5, start=0.0, end=0.0: end <= start True, value >= end True → 1.0
        assert smoothstep_gate(0.5, 0.0, 0.0) == 1.0
        # With value=-1.0, start=1.0, end=0.0: end <= start True, value >= end False → 0.0
        assert smoothstep_gate(-1.0, 1.0, 0.0) == 0.0

        # JAX
        assert float(smoothstep_gate_jax(jnp.array(0.5), 1.0, 0.0)) == 1.0
        assert float(smoothstep_gate_jax(jnp.array(0.5), 0.0, 0.0)) == 1.0
        assert float(smoothstep_gate_jax(jnp.array(-1.0), 1.0, 0.0)) == 0.0


# ===========================================================================
# Torque composer parity
# ===========================================================================


class TestTorqueComposerParity:
    """JAX torque composer matches BalanceCoreTorqueComposer.compose()."""

    @staticmethod
    def _run_comparison(tau_sum, tau_prev, torque_limit, max_torque_rate, control_dt):
        """Run both Python and JAX composers and return diffs."""
        # Python reference
        py_tau_final, py_tau_clipped, py_sat, py_rate_sat = python_torque_composer(
            np.asarray(tau_sum, dtype=np.float64),
            np.asarray(tau_prev, dtype=np.float64),
            np.asarray(torque_limit, dtype=np.float64),
            np.asarray(max_torque_rate, dtype=np.float64),
            float(control_dt),
        )

        # JAX via params
        params = pack_params_stage2(
            torque_limit=jnp.asarray(torque_limit, dtype=jnp.float64),
            max_torque_rate=jnp.asarray(max_torque_rate, dtype=jnp.float64),
            control_dt=float(control_dt),
        )
        jax_tau_final, jax_tau_clipped, jax_sat, jax_rate_sat = k2_jax_torque_composer_step(
            jnp.asarray(tau_sum, dtype=jnp.float64),
            jnp.asarray(tau_prev, dtype=jnp.float64),
            params,
        )

        return {
            "tau_final_diff": np.max(np.abs(np.asarray(jax_tau_final) - py_tau_final)),
            "tau_clipped_diff": np.max(np.abs(np.asarray(jax_tau_clipped) - py_tau_clipped)),
            "sat_mismatch": int(np.sum(np.asarray(jax_sat) != py_sat)),
            "rate_sat_mismatch": int(np.sum(np.asarray(jax_rate_sat) != py_rate_sat)),
        }

    def test_zero_torques(self):
        """Zero inputs produce zero outputs."""
        result = self._run_comparison(
            tau_sum=np.zeros(10),
            tau_prev=np.zeros(10),
            torque_limit=np.ones(10) * 10.0,
            max_torque_rate=np.ones(10) * 400.0,
            control_dt=0.01,
        )
        assert result["tau_final_diff"] <= 1e-15
        assert result["tau_clipped_diff"] <= 1e-15
        assert result["sat_mismatch"] == 0
        assert result["rate_sat_mismatch"] == 0

    def test_within_limits(self):
        """Torques within limits pass through unchanged."""
        tau_sum = np.array([1.0, -0.5, 2.0, -1.5, 0.3, -1.0, 0.5, -2.0, 1.5, -0.3])
        tau_prev = np.zeros(10)
        result = self._run_comparison(
            tau_sum=tau_sum,
            tau_prev=tau_prev,
            torque_limit=np.ones(10) * 10.0,
            max_torque_rate=np.ones(10) * 400.0,
            control_dt=0.01,
        )
        assert result["tau_final_diff"] <= 1e-10
        assert result["tau_clipped_diff"] <= 1e-15
        assert result["sat_mismatch"] == 0
        assert result["rate_sat_mismatch"] == 0

    def test_clipping_saturation(self):
        """Torques exceeding limits are clipped."""
        tau_sum = np.array([15.0, -15.0, 8.0, -8.0, 5.0, -5.0, 3.0, -3.0, 2.0, -2.0])
        tau_prev = np.zeros(10)
        result = self._run_comparison(
            tau_sum=tau_sum,
            tau_prev=tau_prev,
            torque_limit=np.ones(10) * 10.0,
            max_torque_rate=np.ones(10) * 400.0,
            control_dt=0.01,
        )
        assert result["tau_final_diff"] <= 1e-10
        assert result["sat_mismatch"] == 0  # same joints flagged

    def test_rate_limiting(self):
        """Large step change triggers rate limiting."""
        tau_sum = np.ones(10) * 5.0
        tau_prev = np.ones(10) * (-5.0)  # 10 Nm jump
        result = self._run_comparison(
            tau_sum=tau_sum,
            tau_prev=tau_prev,
            torque_limit=np.ones(10) * 20.0,
            max_torque_rate=np.ones(10) * 200.0,  # 200 Nm/s → 2 Nm/step max
            control_dt=0.01,
        )
        assert result["tau_final_diff"] <= 1e-10
        assert result["rate_sat_mismatch"] == 0

    def test_10k_random_inputs(self):
        """10,000 random torque vectors: max per-joint diff <= 1e-10."""
        rng = random.Random(99)
        max_tau_diff = 0.0

        for _ in range(10_000):
            tau_sum = np.array([rng.uniform(-15.0, 15.0) for _ in range(10)])
            tau_prev = np.array([rng.uniform(-10.0, 10.0) for _ in range(10)])
            torque_limit = np.ones(10) * rng.uniform(5.0, 20.0)
            max_rate = np.ones(10) * rng.uniform(100.0, 800.0)

            result = self._run_comparison(tau_sum, tau_prev, torque_limit, max_rate, 0.01)
            max_tau_diff = max(max_tau_diff, result["tau_final_diff"])
            assert result["sat_mismatch"] == 0, "Saturation mask mismatch"
            assert result["rate_sat_mismatch"] == 0, "Rate saturation mask mismatch"

        assert max_tau_diff <= 1e-10, f"Max tau diff: {max_tau_diff:.2e} > 1e-10"


# ===========================================================================
# State pack/unpack roundtrip
# ===========================================================================


class TestStatePackUnpackStage2:
    """Packing and unpacking preserves all state values."""

    def test_zero_initial_state(self):
        """Default (zero) state roundtrips correctly."""
        state = pack_state_stage2()
        unpacked = unpack_state_stage2(state)

        assert unpacked["notch_x1"] == 0.0
        assert unpacked["notch_x2"] == 0.0
        assert unpacked["notch_y1"] == 0.0
        assert unpacked["notch_y2"] == 0.0
        assert np.all(unpacked["prev_tau"] == 0.0)

    def test_nonzero_state_roundtrip(self):
        """Nonzero state values survive pack→unpack."""
        notch_state = (0.1, -0.2, 0.3, -0.4)
        prev_tau = np.array([1.0, -0.5, 2.0, -1.5, 0.3, -1.0, 0.5, -2.0, 1.5, -0.3])

        state = pack_state_stage2(
            notch_x1=notch_state[0],
            notch_x2=notch_state[1],
            notch_y1=notch_state[2],
            notch_y2=notch_state[3],
            prev_tau=prev_tau,
        )
        unpacked = unpack_state_stage2(state)

        assert unpacked["notch_x1"] == pytest.approx(notch_state[0])
        assert unpacked["notch_x2"] == pytest.approx(notch_state[1])
        assert unpacked["notch_y1"] == pytest.approx(notch_state[2])
        assert unpacked["notch_y2"] == pytest.approx(notch_state[3])
        np.testing.assert_array_almost_equal(unpacked["prev_tau"], prev_tau)

    def test_state_size_consistent(self):
        """State size matches field count."""
        assert K2_JAX_STATE_SIZE_STAGE2 == len(K2_JAX_STATE_FIELDS_STAGE2)
        state = pack_state_stage2()
        assert state.shape == (K2_JAX_STATE_SIZE_STAGE2,)

    def test_state_fields_unique(self):
        """No duplicate field names."""
        assert len(K2_JAX_STATE_FIELDS_STAGE2) == len(set(K2_JAX_STATE_FIELDS_STAGE2))


class TestParamsPackUnpackStage2:
    """Params packing and unpacking preserves all values."""

    def test_k2_default_params(self):
        """K2 default params roundtrip correctly."""
        params = pack_params_stage2(
            fs_hz=100.0, fc_hz=2.5, Q=2.0,
            torque_limit=jnp.ones(10) * 10.0,
            max_torque_rate=jnp.ones(10) * 400.0,
            control_dt=0.01,
        )
        unpacked = unpack_params_stage2(params)

        assert unpacked["notch_fs_hz"] == 100.0
        assert unpacked["notch_fc_hz"] == 2.5
        assert unpacked["notch_Q"] == 2.0
        assert unpacked["control_dt"] == 0.01
        np.testing.assert_array_almost_equal(unpacked["torque_limit"], np.ones(10) * 10.0)
        np.testing.assert_array_almost_equal(unpacked["max_torque_rate"], np.ones(10) * 400.0)

    def test_coefficients_match_biquad_notch_filter(self):
        """Packed notch coefficients match BiquadNotchFilter for K2."""
        nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=2.0)
        params = pack_params_stage2(fs_hz=100.0, fc_hz=2.5, Q=2.0)
        unpacked = unpack_params_stage2(params)

        cls_b0, cls_b1, cls_b2, cls_a1, cls_a2 = nf.coefficients()
        assert unpacked["notch_b0"] == pytest.approx(cls_b0, abs=1e-15)
        assert unpacked["notch_b1"] == pytest.approx(cls_b1, abs=1e-15)
        assert unpacked["notch_b2"] == pytest.approx(cls_b2, abs=1e-15)
        assert unpacked["notch_a1"] == pytest.approx(cls_a1, abs=1e-15)
        assert unpacked["notch_a2"] == pytest.approx(cls_a2, abs=1e-15)

    def test_params_size_consistent(self):
        """Params size matches field count."""
        assert K2_JAX_PARAMS_SIZE_STAGE2 == len(K2_JAX_PARAMS_FIELDS_STAGE2)
        params = pack_params_stage2()
        assert params.shape == (K2_JAX_PARAMS_SIZE_STAGE2,)

    def test_params_fields_unique(self):
        """No duplicate param field names."""
        assert len(K2_JAX_PARAMS_FIELDS_STAGE2) == len(set(K2_JAX_PARAMS_FIELDS_STAGE2))


# ===========================================================================
# Index constant sanity checks
# ===========================================================================


class TestIndexConstants:
    """Index constants are consistent with field layout."""

    def test_prev_tau_indices(self):
        """prev_tau fields are at indices 4-13 in state."""
        for i in range(10):
            field_name = f"prev_tau_{i}"
            idx = K2_JAX_STATE_FIELDS_STAGE2.index(field_name)
            assert idx == _IDX_PREV_TAU_START + i

    def test_notch_state_indices(self):
        """Notch state fields are at indices 0-3."""
        assert K2_JAX_STATE_FIELDS_STAGE2.index("notch_x1") == _IDX_NOTCH_X1
        assert K2_JAX_STATE_FIELDS_STAGE2.index("notch_x2") == _IDX_NOTCH_X2
        assert K2_JAX_STATE_FIELDS_STAGE2.index("notch_y1") == _IDX_NOTCH_Y1
        assert K2_JAX_STATE_FIELDS_STAGE2.index("notch_y2") == _IDX_NOTCH_Y2
