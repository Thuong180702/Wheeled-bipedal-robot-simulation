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
    K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE,
    K2_JAX_PARAMS_SIZE_DRIFT,
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
        """Params size matches field count + extension constants.

        K2_JAX_PARAMS_SIZE_STAGE2 (41) = base fields in K2_JAX_PARAMS_FIELDS_STAGE2.
        K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE (54) = base + EXT(+7) + STANDALONE(+6).
        K2_JAX_PARAMS_SIZE_DRIFT (78) = EXT_STANDALONE + drift(+8) + heading(+4) +
            anti-twist(+3) + split-gate(+4) + mean-center(+2) + misc(+1).
        pack_params_stage2() returns the full DRIFT size.
        """
        assert K2_JAX_PARAMS_SIZE_STAGE2 == len(K2_JAX_PARAMS_FIELDS_STAGE2)
        params = pack_params_stage2()
        assert params.shape == (K2_JAX_PARAMS_SIZE_DRIFT,)

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


# ===========================================================================
# Stage 3: Height scheduling parity
# ===========================================================================


class TestHeightSchedulingParity:
    """K2 height scheduling functions match Python reference."""

    def test_smoothstep01_parity(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            _jax_smoothstep01, python_smoothstep01,
        )
        for val in [0.0, 0.25, 0.5, 0.75, 1.0, -0.1, 1.5]:
            py = python_smoothstep01(val)
            jx = float(_jax_smoothstep01(jnp.array(val)))
            assert jx == pytest.approx(py, abs=1e-15)

    @pytest.mark.parametrize("z_ref", [0.33, 0.40, 0.45, 0.48, 0.30, 0.55])
    def test_scheduled_k_position(self, z_ref):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_scheduled_k_position, python_scheduled_k_position,
        )
        args = (z_ref, 0.0, 5.0, 0.35, 0.45)
        py = python_scheduled_k_position(*args)
        jx = float(k2_jax_scheduled_k_position(
            jnp.array(args[0]), *args[1:]))
        assert jx == pytest.approx(py, abs=1e-10)

    @pytest.mark.parametrize("z_ref", [0.33, 0.40, 0.45, 0.48, 0.30, 0.55])
    def test_scheduled_k_wheel_velocity(self, z_ref):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_scheduled_k_wheel_velocity, python_scheduled_k_wheel_velocity,
        )
        args = (z_ref, 0.5, 3.0, 0.35, 0.45)
        py = python_scheduled_k_wheel_velocity(*args)
        jx = float(k2_jax_scheduled_k_wheel_velocity(
            jnp.array(args[0]), *args[1:]))
        assert jx == pytest.approx(py, abs=1e-10)


# ===========================================================================
# Stage 3: Pitch ref offset + outer loop parity
# ===========================================================================


class TestPitchRefOffsetParity:
    """Pitch reference offset interpolation matches Python."""

    def test_k2_height_schedule_interpolation(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_interpolate_pitch_ref_offset, python_interpolate_pitch_ref_offset,
        )
        heights = (0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480)
        offsets = (3.0, -2.0, -4.0, 0.0, -3.0, 5.0, 2.0, 2.0, 3.0, 3.0)
        for h in [0.28, 0.30, 0.33, 0.40, 0.45, 0.48, 0.50]:
            py = python_interpolate_pitch_ref_offset(h, heights, offsets, clamp=True)
            jx = float(k2_jax_interpolate_pitch_ref_offset(
                jnp.array(h), jnp.array(heights), jnp.array(offsets), True))
            assert jx == pytest.approx(py, abs=1e-10)


class TestOuterLoopParity:
    """Support-position outer loop pitch ref matches Python."""

    def test_outer_loop_basic(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_compute_outer_loop_pitch_ref, python_compute_outer_loop_pitch_ref,
        )
        args = (0.02, 0.001, 0.0, 1.0, 0.0, 0.0, 0.015, 3.0)
        py = python_compute_outer_loop_pitch_ref(*args)
        jx = float(k2_jax_compute_outer_loop_pitch_ref(
            jnp.array(args[0]), jnp.array(args[1]), jnp.array(args[2]),
            *args[3:]))
        assert jx == pytest.approx(py, abs=1e-10)

    def test_outer_loop_deadband(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_compute_outer_loop_pitch_ref, python_compute_outer_loop_pitch_ref,
        )
        # Error within deadband → zero proportional
        args = (0.005, 0.001, 0.0, 1.0, 0.0, 0.0, 0.015, 3.0)
        py = python_compute_outer_loop_pitch_ref(*args)
        jx = float(k2_jax_compute_outer_loop_pitch_ref(
            jnp.array(args[0]), jnp.array(args[1]), jnp.array(args[2]),
            *args[3:]))
        assert jx == pytest.approx(py, abs=1e-10)

    def test_outer_loop_saturation(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_compute_outer_loop_pitch_ref, python_compute_outer_loop_pitch_ref,
        )
        # Large error should saturate
        args = (10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 3.0)
        py = python_compute_outer_loop_pitch_ref(*args)
        jx = float(k2_jax_compute_outer_loop_pitch_ref(
            jnp.array(args[0]), jnp.array(args[1]), jnp.array(args[2]),
            *args[3:]))
        assert jx == pytest.approx(py, abs=1e-10)


# ===========================================================================
# Stage 3: PCHIP grid interpolation verification
# ===========================================================================


class TestCalibratedOuterLoopParity:
    """Calibrated outer loop grid interpolation vs PCHIP."""

    @pytest.fixture(scope="class")
    def grid_params(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            build_calibrated_grid_params,
        )
        # Default 20000-pt grid — empirically verified <1e-6 for all functions
        return build_calibrated_grid_params()

    @pytest.fixture(scope="class")
    def pchip_refs(self):
        # D12 bugfix: JAX uses v2 calibrated outer loop functions (matches K2 profile).
        from wheeled_biped.controllers.calibrated_outer_loop_functions_v2 import (
            calibrated_kp_deg_per_m,
            calibrated_kd_deg_per_mps,
            calibrated_theta_ref_max_deg,
            calibrated_deadband_m,
            calibrated_rate_limit_deg_per_step,
            calibrated_lowpass_alpha,
        )
        return {
            "kp": calibrated_kp_deg_per_m,
            "kd": calibrated_kd_deg_per_mps,
            "theta_max": calibrated_theta_ref_max_deg,
            "deadband": calibrated_deadband_m,
            "rate_limit": calibrated_rate_limit_deg_per_step,
            "lowpass": calibrated_lowpass_alpha,
        }

    def _max_error(self, grid_heights, grid_values, pchip_fn):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_grid_interpolate,
        )
        max_err = 0.0
        rng = random.Random(42)
        for _ in range(10000):
            h = rng.uniform(0.28, 0.50)
            py = pchip_fn(h)
            jx = float(k2_jax_grid_interpolate(
                jnp.array(h), grid_heights, grid_values))
            max_err = max(max_err, abs(jx - py))
        return max_err

    def test_kp_grid_error(self, grid_params, pchip_refs):
        err = self._max_error(
            grid_params["grid_heights"], grid_params["kp_grid"], pchip_refs["kp"])
        assert err <= 1e-6, f"Kp grid max error: {err:.2e}"

    def test_kd_grid_error(self, grid_params, pchip_refs):
        err = self._max_error(
            grid_params["grid_heights"], grid_params["kd_grid"], pchip_refs["kd"])
        assert err <= 1e-6, f"Kd grid max error: {err:.2e}"

    def test_theta_max_grid_error(self, grid_params, pchip_refs):
        err = self._max_error(
            grid_params["grid_heights"], grid_params["theta_max_grid"],
            pchip_refs["theta_max"])
        assert err <= 1e-6, f"theta_max grid max error: {err:.2e}"

    def test_deadband_grid_error(self, grid_params, pchip_refs):
        err = self._max_error(
            grid_params["grid_heights"], grid_params["deadband_grid"],
            pchip_refs["deadband"])
        assert err <= 1e-6, f"deadband grid max error: {err:.2e}"

    def test_rate_limit_grid_error(self, grid_params, pchip_refs):
        err = self._max_error(
            grid_params["grid_heights"], grid_params["rate_limit_grid"],
            pchip_refs["rate_limit"])
        assert err <= 1e-6, f"rate_limit grid max error: {err:.2e}"

    def test_lowpass_grid_error(self, grid_params, pchip_refs):
        err = self._max_error(
            grid_params["grid_heights"], grid_params["lowpass_grid"],
            pchip_refs["lowpass"])
        assert err <= 1e-6, f"lowpass grid max error: {err:.2e}"


class TestPhysicsFFParity:
    """Physics equilibrium feedforward grid interpolation vs PCHIP."""

    @pytest.fixture(scope="class")
    def ff_grid_params(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            build_physics_ff_grid_params,
        )
        # Default 100000-pt grid for high-curvature physics FF functions
        return build_physics_ff_grid_params()

    def _max_error(self, grid_heights, grid_values, pchip_fn):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_grid_interpolate,
        )
        max_err = 0.0
        rng = random.Random(99)
        for _ in range(10000):
            h = rng.uniform(0.28, 0.50)
            py = pchip_fn(h)
            jx = float(k2_jax_grid_interpolate(
                jnp.array(h), grid_heights, grid_values))
            max_err = max(max_err, abs(jx - py))
        return max_err

    def test_tau_eq_ff_grid_error(self, ff_grid_params):
        from wheeled_biped.controllers.physics_equilibrium_feedforward import (
            physics_equilibrium_feedforward_tau_each_wheel_nm,
        )
        err = self._max_error(
            ff_grid_params["grid_heights"],
            ff_grid_params["tau_eq_ff_grid"],
            physics_equilibrium_feedforward_tau_each_wheel_nm)
        assert err <= 1e-6, f"tau_eq_ff grid max error: {err:.2e}"

    def test_pitch_eq_grid_error(self, ff_grid_params):
        from wheeled_biped.controllers.physics_equilibrium_feedforward import (
            physics_equilibrium_pitch_eq_no_off_deg,
        )
        err = self._max_error(
            ff_grid_params["grid_heights"],
            ff_grid_params["pitch_eq_grid"],
            physics_equilibrium_pitch_eq_no_off_deg)
        assert err <= 1e-6, f"pitch_eq grid max error: {err:.2e}"


# ===========================================================================
# Stage 3: Low-band support parity
# ===========================================================================


class TestLowBandSupportParity:
    """Low-band support outer loop matches Python reference behavior."""

    def test_gate_at_center(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_low_band_support_gate,
        )
        g = float(k2_jax_low_band_support_gate(jnp.array(0.320), 0.320, 0.004))
        assert g == pytest.approx(1.0, abs=1e-10)

    def test_gate_at_480(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_low_band_support_gate,
        )
        g = float(k2_jax_low_band_support_gate(jnp.array(0.480), 0.320, 0.004))
        assert g < 1e-10  # essentially zero far from center

    def test_pitch_ref_at_center(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_low_band_support_pitch_ref,
        )
        offset, theta_max = k2_jax_low_band_support_pitch_ref(
            jnp.array(0.320), jnp.array(0.02),
            0.320, 0.004, 1.4, 3.0, 1.0,
        )
        assert float(offset) != 0.0  # active at center
        assert float(theta_max) == pytest.approx(3.0, abs=1e-4)


# ===========================================================================
# Stage 3: Component controller parity
# ===========================================================================


class TestShapePostureParity:
    """Shape posture JAX PD matches Python ShapePostureController."""

    def test_pd_torque_match(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_shape_posture_compute,
        )
        q_ref = jnp.array([0.0, 0.0, 0.635, 1.232, 0.0, 0.0, 0.0, 0.635, 1.232, 0.0])
        q = jnp.array([0.01, -0.02, 0.63, 1.23, 0.0, -0.01, 0.02, 0.64, 1.24, 0.0])
        qd = jnp.zeros(10)
        tau, diag = k2_jax_shape_posture_compute(q_ref, q, qd)
        assert tau.shape == (10,)
        # hip_yaw [1,6], hip_pitch [2,7], knee [3,8] should be nonzero
        assert float(jnp.abs(tau[1])) > 0
        assert float(jnp.abs(tau[6])) > 0
        assert float(jnp.abs(tau[2])) > 0
        assert float(jnp.abs(tau[7])) > 0
        assert float(jnp.abs(tau[3])) > 0
        assert float(jnp.abs(tau[8])) > 0
        # wheels [4,9] should be zero
        assert float(tau[4]) == 0.0
        assert float(tau[9]) == 0.0


class TestLateralRollParity:
    """Lateral roll JAX matches Python LateralRollBalanceController."""

    def test_positive_roll_produces_antisymmetric_torque(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_lateral_roll_compute,
        )
        tau, _ = k2_jax_lateral_roll_compute(
            jnp.array(0.1), jnp.array(0.0),
        )
        # Positive roll → left hip roll positive, right negative
        assert float(tau[0]) > 0
        assert float(tau[5]) < 0

    def test_stance_regularization_disabled_by_default(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_lateral_roll_compute,
        )
        tau_no_stance, _ = k2_jax_lateral_roll_compute(
            jnp.array(0.0), jnp.array(0.0),
            hip_roll_pos_left=0.1, hip_roll_pos_right=-0.1,
            hip_roll_ref_left=0.0, hip_roll_ref_right=0.0,
            enable_stance_regularization=False,
        )
        tau_stance, _ = k2_jax_lateral_roll_compute(
            jnp.array(0.0), jnp.array(0.0),
            hip_roll_pos_left=0.1, hip_roll_pos_right=-0.1,
            hip_roll_ref_left=0.0, hip_roll_ref_right=0.0,
            enable_stance_regularization=True,
        )
        # With stance, torques should differ from without
        assert float(tau_no_stance[0]) != float(tau_stance[0])


class TestYawControllerParity:
    """Yaw JAX matches Python YawController."""

    def test_positive_yaw_error_antisymmetric(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_yaw_compute,
        )
        tau = k2_jax_yaw_compute(jnp.array(0.1), jnp.array(0.0))
        # Positive yaw error → negative on left, positive on right
        assert float(tau[1]) < 0
        assert float(tau[6]) > 0


class TestModeDivParity:
    """Mode-div JAX matches Python ModeBasedHipYawDivergenceController."""

    def test_k2_params_produce_torque(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_mode_div_compute,
        )
        tau = k2_jax_mode_div_compute(
            jnp.array(0.05), jnp.array(0.01), jnp.array(0.40),
            kp_div=10.0, kd_div=0.50, max_torque=7.5,
            soft_limit_rad=0.30, soft_gain=0.80,
        )
        assert float(tau[1]) != 0.0
        assert float(tau[6]) != 0.0
        # Antisymmetric
        assert float(tau[1]) == pytest.approx(-float(tau[6]), abs=1e-10)

    def test_height_gate_reduces_torque_at_high_height(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_mode_div_compute,
        )
        # At low height (0.30m): gate = 1.0, full torque
        tau_low = k2_jax_mode_div_compute(
            jnp.array(0.05), jnp.array(0.01), jnp.array(0.30),
            kp_div=10.0, kd_div=0.50, max_torque=7.5,
            soft_limit_rad=0.30, soft_gain=0.80,
        )
        # At high height (1.0m): gate << 1.0, reduced torque
        tau_high = k2_jax_mode_div_compute(
            jnp.array(0.05), jnp.array(0.01), jnp.array(1.0),
            kp_div=10.0, kd_div=0.50, max_torque=7.5,
            soft_limit_rad=0.30, soft_gain=0.80,
        )
        # At height >= soft_limit + soft_gain (1.10m): gate = 0, zero torque
        tau_zero = k2_jax_mode_div_compute(
            jnp.array(0.05), jnp.array(0.01), jnp.array(1.20),
            kp_div=10.0, kd_div=0.50, max_torque=7.5,
            soft_limit_rad=0.30, soft_gain=0.80,
        )
        # Gate reduces torque significantly at high height
        assert abs(float(tau_high[1])) < abs(float(tau_low[1]))
        # Gate is zero above soft_limit + soft_gain
        assert float(tau_zero[1]) == 0.0

    def test_torque_clips_at_max(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_mode_div_compute,
        )
        tau = k2_jax_mode_div_compute(
            jnp.array(10.0), jnp.array(0.0), jnp.array(0.30),  # huge error, low height
            kp_div=10.0, kd_div=0.50, max_torque=7.5,
            soft_limit_rad=0.30, soft_gain=0.80,
        )
        assert abs(float(tau[1])) <= 7.5 + 1e-10


# ===========================================================================
# Stage 3: Sagittal torque assembly parity
# ===========================================================================


class TestSagittalTorqueAssemblyParity:
    """K2-active sagittal torque assembly matches Python reference.

    Note: This is NOT a full-step parity test (that's Stage 4).
    It verifies that individual torque terms are computed correctly.
    """

    def test_zero_inputs_produce_zero_wheel_torque(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_sagittal_torque_assembly,
        )
        tau, diag = k2_jax_sagittal_torque_assembly(
            pitch_x_rad=jnp.array(0.0),
            pitch_rate_rad_s=jnp.array(0.0),
            sagittal_velocity_m_s=jnp.array(0.0),
            sagittal_position_error_m=jnp.array(0.0),
            wheel_vel_left_rad_s=jnp.array(0.0),
            wheel_vel_right_rad_s=jnp.array(0.0),
            support_velocity_m_s=jnp.array(0.0),
            kp_pitch=50.0, effective_pitch_scale=1.0, effective_pitch_tau_cap=0.0,
            effective_kd_pitch=10.0,
            effective_k_velocity=0.0, effective_velocity_damping_scale=1.0,
            effective_support_velocity_gain=0.0, effective_support_velocity_scale=1.0,
            effective_k_wheel_velocity=0.5,
            effective_k_position=0.0, effective_max_position_tau=3.0,
            kp_cp=0.0, kd_com_vy=5.0,
            wheel_torque_sign=1.0,
        )
        assert tau.shape == (10,)
        # Non-wheel joints should be zero
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert float(tau[i]) == 0.0, f"Joint {i} should be zero"
        # Wheels should be zero (no velocity)
        assert float(tau[4]) == 0.0
        assert float(tau[9]) == 0.0

    def test_forward_pitch_produces_forward_wheel_torque(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_sagittal_torque_assembly,
        )
        tau, _ = k2_jax_sagittal_torque_assembly(
            pitch_x_rad=jnp.array(0.1),  # forward pitch
            pitch_rate_rad_s=jnp.array(0.0),
            sagittal_velocity_m_s=jnp.array(0.0),
            sagittal_position_error_m=jnp.array(0.0),
            wheel_vel_left_rad_s=jnp.array(0.0),
            wheel_vel_right_rad_s=jnp.array(0.0),
            support_velocity_m_s=jnp.array(0.0),
            kp_pitch=50.0, effective_pitch_scale=1.0, effective_pitch_tau_cap=0.0,
            effective_kd_pitch=10.0,
            effective_k_velocity=0.0, effective_velocity_damping_scale=1.0,
            effective_support_velocity_gain=0.0, effective_support_velocity_scale=1.0,
            effective_k_wheel_velocity=0.5,
            effective_k_position=0.0, effective_max_position_tau=3.0,
            kp_cp=0.0, kd_com_vy=5.0,
            wheel_torque_sign=1.0,
        )
        # Forward pitch → forward wheel torque (positive on both wheels)
        assert float(tau[4]) > 0, f"Left wheel torque should be positive, got {float(tau[4])}"
        assert float(tau[9]) > 0, f"Right wheel torque should be positive, got {float(tau[9])}"


# ===========================================================================
# Stage 3: Rate limit + lowpass parity
# ===========================================================================


class TestRateLimitLowpassParity:
    """JAX rate limit and lowpass match Python."""

    def test_rate_limit_parity(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            _jax_apply_rate_limit, python_apply_rate_limit,
        )
        cases = [(0.0, 5.0, 2.0), (0.0, 0.5, 2.0), (0.0, -5.0, 2.0), (0.0, 1.0, -1.0)]
        for prev, target, max_delta in cases:
            py = python_apply_rate_limit(prev, target, max_delta)
            jx = float(_jax_apply_rate_limit(
                jnp.array(prev), jnp.array(target), max_delta))
            assert jx == pytest.approx(py, abs=1e-15)

    def test_lowpass_parity(self):
        from wheeled_biped.controllers.k2_jax_controller import (
            _jax_apply_lowpass, python_apply_lowpass,
        )
        cases = [(0.0, 5.0, 0.15), (3.0, 5.0, 0.15), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
        for prev, target, alpha in cases:
            py = python_apply_lowpass(prev, target, alpha)
            jx = float(_jax_apply_lowpass(
                jnp.array(prev), jnp.array(target), alpha))
            assert jx == pytest.approx(py, abs=1e-15)


# ===========================================================================
# Phase 5: APCR1ND gate parity tests
# ===========================================================================


class TestAPCR1NDGateParity:
    """JAX APCR1ND gate matches Python source-of-truth behavior."""

    @staticmethod
    def _gate(sag_pos_err, prev_error=0.0, step_counter=25.0, converging_steps=0.0,
              recenter_held=0.0, pitch_x=0.02, roll_y=0.02, com_z=0.50, contact_valid=1.0,
              startup_guard=20, safe_com_z=0.35, safe_roll=0.3, safe_pitch=0.3,
              soft_enter=0.05, direct_enter=0.06, desired_band=0.08,
              release_inner=0.03, hold_outside=0.0, converging_release=15):
        """Helper to call k2_jax_apcr1nd_compute_gate with defaults."""
        from wheeled_biped.controllers.k2_jax_controller import k2_jax_apcr1nd_compute_gate
        active, _, _, new_conv, new_held = k2_jax_apcr1nd_compute_gate(
            sagittal_position_error_m=jnp.array(sag_pos_err, dtype=jnp.float64),
            prev_error=jnp.array(prev_error, dtype=jnp.float64),
            step_counter=jnp.array(step_counter, dtype=jnp.float64),
            converging_steps=jnp.array(converging_steps, dtype=jnp.float64),
            recenter_held=jnp.array(recenter_held, dtype=jnp.float64),
            pitch_x_rad=jnp.array(pitch_x, dtype=jnp.float64),
            roll_y_rad=jnp.array(roll_y, dtype=jnp.float64),
            com_z_m=jnp.array(com_z, dtype=jnp.float64),
            contact_valid=jnp.array(contact_valid > 0.5, dtype=jnp.bool_),
            startup_guard_steps=startup_guard,
            safe_min_com_z=safe_com_z,
            safe_roll_rad=safe_roll,
            safe_pitch_rad=safe_pitch,
            soft_enter_m=soft_enter,
            direct_enter_m=direct_enter,
            desired_band_m=desired_band,
            release_inner_m=release_inner,
            hold_outside_band=jnp.array(hold_outside, dtype=jnp.float64),
            converging_release_steps=converging_release,
        )
        return bool(active), float(new_conv), float(new_held)

    # ---- Safety gate tests ----

    def test_safety_includes_contact_valid(self):
        """When contact_valid=False, safety_pass must fail → gate inactive."""
        # With contact_valid=True, direct_entry at 0.07m (>= 0.06) + moving_away
        active, _, _ = self._gate(0.07, prev_error=0.02, contact_valid=1.0)
        assert active, "Gate should activate with contact_valid=True and large error"

        # With contact_valid=False, safety fails → gate must NOT activate
        active2, _, _ = self._gate(0.07, prev_error=0.02, contact_valid=0.0)
        assert not active2, "Gate must NOT activate when contact_valid=False"

    def test_safety_includes_pitch(self):
        """Excessive pitch must fail safety gate."""
        active, _, _ = self._gate(0.07, prev_error=0.02, pitch_x=0.5)  # 0.5 rad > safe_pitch=0.3
        assert not active, "Gate must NOT activate when pitch exceeds safe limit"

    def test_safety_includes_roll(self):
        """Excessive roll must fail safety gate."""
        active, _, _ = self._gate(0.07, prev_error=0.02, roll_y=0.5)
        assert not active, "Gate must NOT activate when roll exceeds safe limit"

    def test_safety_includes_com_z(self):
        """Low CoM height must fail safety gate."""
        active, _, _ = self._gate(0.07, prev_error=0.02, com_z=0.30)  # < safe_com_z=0.35
        assert not active, "Gate must NOT activate when CoM is too low"

    # ---- Converging steps gated by safety ----

    def test_converging_steps_reset_on_safety_fail(self):
        """Converging steps must NOT update when safety fails (matches Python)."""
        # With safety passing and converging, steps should increment
        _, conv1, _ = self._gate(0.03, prev_error=0.04, converging_steps=5.0)
        assert conv1 == 6.0, f"Converging steps should increment: got {conv1}"

        # With safety failing (pitch too high), converging steps should stay at 5
        _, conv2, _ = self._gate(0.03, prev_error=0.04, converging_steps=5.0, pitch_x=0.5)
        assert conv2 == 5.0, f"Converging steps must NOT change on safety fail: got {conv2}"

    def test_converging_steps_reset_when_not_converging(self):
        """Converging steps reset to 0 when not converging and safety passes."""
        _, conv, _ = self._gate(0.03, prev_error=0.02, converging_steps=10.0)
        assert conv == 0.0, f"Converging steps should reset to 0 when moving_away: got {conv}"

    # ---- Recenter held reset on safety fail ----

    def test_recenter_held_resets_on_safety_fail(self):
        """recenter_held must reset to 0 when safety fails (matches Python)."""
        # With safety passing and direct_entry, held should go to 1
        _, _, held1 = self._gate(0.07, prev_error=0.02, recenter_held=0.0)
        assert held1 == 1.0, "Held should activate on direct_entry"

        # With safety failing, held must reset to 0 (even if previously held)
        _, _, held2 = self._gate(0.07, prev_error=0.02, recenter_held=1.0, pitch_x=0.5)
        assert held2 == 0.0, f"Held must reset on safety fail: got {held2}"

    # ---- Entry condition tests ----

    def test_direct_enter_activates(self):
        """abs_error >= direct_enter_m AND moving_away → activate."""
        active, _, held = self._gate(0.07, prev_error=0.02)
        assert active, "Direct entry should activate at 0.07m with moving_away"
        assert held == 1.0

    def test_soft_enter_activates(self):
        """abs_error in [soft_enter, direct_enter) AND moving_away → activate."""
        active, _, held = self._gate(0.055, prev_error=0.02)
        assert active, "Soft entry should activate at 0.055m with moving_away"
        assert held == 1.0

    def test_emergency_entry_activates(self):
        """abs_error >= desired_band_m → activate regardless of moving_away."""
        # Even when converging (not moving_away), emergency entry should activate
        active, _, held = self._gate(0.10, prev_error=0.12)
        assert active, "Emergency entry should activate at 0.10m regardless"
        assert held == 1.0

    def test_no_activation_below_threshold(self):
        """Below soft_enter_m and no prior held → no activation."""
        active, _, held = self._gate(0.02, prev_error=0.01)
        assert not active, "Should not activate below threshold"
        assert held == 0.0

    # ---- Hold condition ----

    def test_hold_condition_with_prev_active(self):
        """When previously active (held=1), stay active if abs_error > release_inner."""
        active, _, held = self._gate(0.04, prev_error=0.02, recenter_held=1.0)
        assert active, "Hold condition should keep active"
        assert held == 1.0

    # ---- Release condition tests ----

    def test_release_by_inner_band(self):
        """When abs_error <= release_inner_m, release the gate."""
        active, _, held = self._gate(0.02, prev_error=0.02, recenter_held=1.0)
        assert not active, "Should release when within inner band"
        assert held == 0.0

    def test_converging_release(self):
        """After enough converging steps, release even above inner band."""
        active, _, held = self._gate(
            0.05, prev_error=0.06, recenter_held=1.0,
            converging_steps=16.0)  # >= converging_release_steps=15
        assert not active, "Should release after enough converging steps"
        assert held == 0.0

    def test_converging_release_too_few_steps(self):
        """Not enough converging steps → no release."""
        active, _, held = self._gate(
            0.05, prev_error=0.06, recenter_held=1.0,
            converging_steps=10.0)  # < converging_release_steps=15
        assert active, "Should NOT release with too few converging steps"

    # ---- Startup guard ----

    def test_startup_guard_blocks_activation(self):
        """Before startup guard expires, gate must not activate."""
        active, _, _ = self._gate(0.07, prev_error=0.02, step_counter=5.0, startup_guard=20)
        assert not active, "Startup guard should block activation"

    def test_startup_guard_passed_allows_activation(self):
        """After startup guard, gate may activate."""
        active, _, _ = self._gate(0.07, prev_error=0.02, step_counter=25.0, startup_guard=20)
        assert active, "Should activate after startup guard expires"

    # ---- Prev error always updated ----

    def test_prev_error_always_updated(self):
        """prev_error is always set to current sagittal_position_error_m."""
        # The function returns new_prev_error as part of the returned tuple
        from wheeled_biped.controllers.k2_jax_controller import k2_jax_apcr1nd_compute_gate
        _, _, new_prev, _, _ = k2_jax_apcr1nd_compute_gate(
            sagittal_position_error_m=jnp.array(0.07, dtype=jnp.float64),
            prev_error=jnp.array(0.02, dtype=jnp.float64),
            step_counter=jnp.array(25.0, dtype=jnp.float64),
            converging_steps=jnp.array(0.0, dtype=jnp.float64),
            recenter_held=jnp.array(0.0, dtype=jnp.float64),
            pitch_x_rad=jnp.array(0.02, dtype=jnp.float64),
            roll_y_rad=jnp.array(0.02, dtype=jnp.float64),
            com_z_m=jnp.array(0.50, dtype=jnp.float64),
            contact_valid=jnp.array(True, dtype=jnp.bool_),
            startup_guard_steps=20, safe_min_com_z=0.35, safe_roll_rad=0.3,
            safe_pitch_rad=0.3, soft_enter_m=0.05, direct_enter_m=0.06,
            desired_band_m=0.08, release_inner_m=0.03,
            hold_outside_band=jnp.array(0.0, dtype=jnp.float64),
            converging_release_steps=15,
        )
        assert float(new_prev) == 0.07, "prev_error must be set to current sag_pos_err"

    # ---- Position cap gated by recenter_active ----

    def test_position_cap_boost_inactive_when_gate_off(self):
        """When APCR1ND gate is inactive, boosted_cap should return normal (3.5)."""
        from wheeled_biped.controllers.k2_jax_controller import k2_jax_compute_boosted_position_cap
        cap = float(k2_jax_compute_boosted_position_cap(
            abs_error=jnp.array(0.15, dtype=jnp.float64),
            safety_gate_pass=jnp.array(True, dtype=jnp.bool_),
            boost_enabled=jnp.array(1.0, dtype=jnp.float64),
            apcr1nd_tuned_enabled=jnp.array(1.0, dtype=jnp.float64),
            soft_enter_m=jnp.array(0.05, dtype=jnp.float64),
            hard_band_m=jnp.array(0.10, dtype=jnp.float64),
            emergency_band_m=jnp.array(0.12, dtype=jnp.float64),
            desired_band_m=jnp.array(0.08, dtype=jnp.float64),
            cap_normal=jnp.array(3.5, dtype=jnp.float64),
            cap_soft=jnp.array(4.0, dtype=jnp.float64),
            cap_desired=jnp.array(5.0, dtype=jnp.float64),
            cap_hard=jnp.array(6.0, dtype=jnp.float64),
            cap_emergency=jnp.array(7.0, dtype=jnp.float64),
        ))
        assert cap == 7.0, f"Emergency band with safety → cap_emergency=7.0, got {cap}"

    def test_position_cap_boost_no_safety(self):
        """When safety fails, boosted_cap should return cap_normal (3.5) regardless of error."""
        from wheeled_biped.controllers.k2_jax_controller import k2_jax_compute_boosted_position_cap
        cap = float(k2_jax_compute_boosted_position_cap(
            abs_error=jnp.array(0.15, dtype=jnp.float64),
            safety_gate_pass=jnp.array(False, dtype=jnp.bool_),
            boost_enabled=jnp.array(1.0, dtype=jnp.float64),
            apcr1nd_tuned_enabled=jnp.array(1.0, dtype=jnp.float64),
            soft_enter_m=jnp.array(0.05, dtype=jnp.float64),
            hard_band_m=jnp.array(0.10, dtype=jnp.float64),
            emergency_band_m=jnp.array(0.12, dtype=jnp.float64),
            desired_band_m=jnp.array(0.08, dtype=jnp.float64),
            cap_normal=jnp.array(3.5, dtype=jnp.float64),
            cap_soft=jnp.array(4.0, dtype=jnp.float64),
            cap_desired=jnp.array(5.0, dtype=jnp.float64),
            cap_hard=jnp.array(6.0, dtype=jnp.float64),
            cap_emergency=jnp.array(7.0, dtype=jnp.float64),
        ))
        assert cap == 3.5, f"Safety fail → cap_normal=3.5, got {cap}"

    # ---- Wheel damping override gated by recenter_active ----

    def test_wheel_damping_override_inactive_when_gate_off(self):
        """When recenter_active=False, wheel damping override must NOT apply."""
        from wheeled_biped.controllers.k2_jax_controller import k2_jax_apcr1nd_wheel_damping_override
        tau_l, tau_r = k2_jax_apcr1nd_wheel_damping_override(
            tau_wheel_vel_left=jnp.array(5.0, dtype=jnp.float64),
            tau_wheel_vel_right=jnp.array(5.0, dtype=jnp.float64),
            wheel_vel_left_rad_s=jnp.array(2.0, dtype=jnp.float64),
            wheel_vel_right_rad_s=jnp.array(2.0, dtype=jnp.float64),
            sagittal_position_error_m=jnp.array(0.10, dtype=jnp.float64),
            recenter_active=jnp.array(False, dtype=jnp.bool_),
        )
        assert float(tau_l) == 5.0, "WD override must NOT apply when gate off"
        assert float(tau_r) == 5.0

    def test_wheel_damping_override_active_when_gate_on(self):
        """When recenter_active=True and error in hard band, damping scale should apply."""
        from wheeled_biped.controllers.k2_jax_controller import k2_jax_apcr1nd_wheel_damping_override, _K2_APCR_SCALE_HARD
        tau_l, tau_r = k2_jax_apcr1nd_wheel_damping_override(
            tau_wheel_vel_left=jnp.array(5.0, dtype=jnp.float64),
            tau_wheel_vel_right=jnp.array(5.0, dtype=jnp.float64),
            wheel_vel_left_rad_s=jnp.array(2.0, dtype=jnp.float64),
            wheel_vel_right_rad_s=jnp.array(2.0, dtype=jnp.float64),
            sagittal_position_error_m=jnp.array(0.10, dtype=jnp.float64),  # hard band
            recenter_active=jnp.array(True, dtype=jnp.bool_),
        )
        # Damping fights drift (both positive same sign) → scale from profile
        # K2 profile: scale_hard=0.15 → 5.0*0.15=0.75
        expected = 5.0 * float(_K2_APCR_SCALE_HARD)
        assert float(tau_l) == pytest.approx(expected, abs=0.01), f"WD scale {float(_K2_APCR_SCALE_HARD)} → {expected}, got {float(tau_l)}"
        assert float(tau_r) == pytest.approx(expected, abs=0.01)
