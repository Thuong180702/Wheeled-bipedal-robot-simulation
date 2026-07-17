"""Tests for hip_yaw_mode_math utilities.

The utilities are pure‑Python and operate on scalar values (or NumPy/JAX
compatible arrays).  The tests verify:

1. ``decompose`` / ``recompose`` round‑trip for arbitrary left/right values.
2. ``torque_decompose`` / ``torque_recompose`` round‑trip.
3. ``sign_for_divergence_correction`` returns a sign that matches the sign of
   the weighted sum ``div_error + div_rate`` (or +1 when that sum is zero).
"""

import numpy as np

from wheeled_biped.controllers.hip_yaw_mode_math import (
    decompose,
    recompose,
    torque_decompose,
    torque_recompose,
    sign_for_divergence_correction,
)


def test_decompose_recompose_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        left = rng.uniform(-1.0, 1.0)
        right = rng.uniform(-1.0, 1.0)
        common, divergence = decompose(left, right)
        left2, right2 = recompose(common, divergence)
        assert np.isclose(left2, left, atol=1e-12)
        assert np.isclose(right2, right, atol=1e-12)


def test_torque_decompose_recompose_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        tau_left = rng.uniform(-2.0, 2.0)
        tau_right = rng.uniform(-2.0, 2.0)
        tau_common, tau_div = torque_decompose(tau_left, tau_right)
        tau_left2, tau_right2 = torque_recompose(tau_common, tau_div)
        assert np.isclose(tau_left2, tau_left, atol=1e-12)
        assert np.isclose(tau_right2, tau_right, atol=1e-12)


def test_sign_for_divergence_correction_behaviour():
    rng = np.random.default_rng(2)
    for _ in range(200):
        div_error = rng.uniform(-1.0, 1.0)
        div_rate = rng.uniform(-1.0, 1.0)
        sign = sign_for_divergence_correction(div_error, div_rate)
        # sign should be either +1 or -1 (or +1 when sum is exactly zero)
        assert sign in (1.0, -1.0)
        weighted = div_error + div_rate
        # The sign should match the sign of the weighted sum (or +1 if zero)
        if weighted > 0:
            assert sign == 1.0
        elif weighted < 0:
            assert sign == -1.0
        else:
            assert sign == 1.0
