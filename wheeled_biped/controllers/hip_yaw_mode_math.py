"""Utility functions for hip‑yaw mode decomposition and torque handling.

This module provides a small, pure‑Python utility that is deliberately
independent of the rest of the code‑base so it can be unit‑tested in
isolation.  The functions operate on scalar values (or JAX arrays – the
operations are NumPy‑compatible) and implement the canonical definitions
used throughout the controller code:

* **Common mode** – the average of the left and right quantities.
* **Divergence mode** – the antisymmetric difference (left − right).

Both the position‑level and torque‑level APIs use the same math.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def decompose(left: float, right: float) -> Tuple[float, float]:
    """Decompose left/right values into common and divergence components.

    Args:
        left: Left‑side scalar (e.g. hip‑yaw position error).
        right: Right‑side scalar.

    Returns:
        A tuple ``(common, divergence)`` where ``common`` is the average of the
        two inputs and ``divergence`` is the antisymmetric difference
        ``left - right``.
    """
    common = 0.5 * (left + right)
    divergence = left - right
    return common, divergence


def recompose(common: float, divergence: float) -> Tuple[float, float]:
    """Recompose common/divergence back to left/right values.

    The inverse of :func:`decompose`:

    ``left  = common + divergence/2``
    ``right = common - divergence/2``
    """
    left = common + 0.5 * divergence
    right = common - 0.5 * divergence
    return left, right


def torque_decompose(tau_left: float, tau_right: float) -> Tuple[float, float]:
    """Decompose left/right torques into a common (symmetric) part and a
    divergence (antisymmetric) part.

    The mathematics are identical to :func:`decompose`; this wrapper exists for
    semantic clarity in the code‑base.
    """
    return decompose(tau_left, tau_right)


def torque_recompose(tau_common: float, tau_div: float) -> Tuple[float, float]:
    """Recompose torque components back to left/right torques.

    Mirrors :func:`recompose` for torque values.
    """
    return recompose(tau_common, tau_div)


def sign_for_divergence_correction(div_error: float, div_rate: float) -> float:
    """Return ``+1`` or ``-1`` indicating the sign to use for a corrective
    torque.

    The controller applies a torque of the form::

        torque = -sign * (Kp * div_error + Kd * div_rate)

    where ``Kp`` and ``Kd`` are positive gains.  To *reduce* the divergence the
    torque must be opposite in sign to the weighted sum ``Kp*error + Kd*rate``.
    Because the gains are positive we can simply look at the sign of the sum
    ``div_error + div_rate`` (the relative weighting does not affect the sign).

    If the sum is exactly zero we conservatively return ``+1`` – the resulting
    torque will be zero regardless of the sign.
    """
    # Compute the sign of the weighted error.  Using ``np.sign`` ensures we get
    # ``0`` when the argument is exactly zero; in that case we fall back to ``+1``.
    raw = div_error + div_rate
    s = np.sign(raw)
    if s == 0:
        return 1.0
    return float(s)
