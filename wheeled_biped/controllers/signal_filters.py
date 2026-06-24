"""Causal signal filters for controller-side online use.

Provides simple IIR biquad notch / band-stop filters that can be used
inside the control loop to attenuate specific frequency bands without
offline or non-causal processing.

All filters are causal (use only past and current input) and numerically
stable for the expected sample-rate and frequency ranges.

Design notes
============
- Biquad notch filter (Direct Form II Transposed) centered on a
  target frequency fc with quality factor Q.
- At Q >> 1 the notch is narrow; at Q ~ 1 the notch is wider.
- For a 100 Hz sample rate, a 2.5 Hz notch with Q=4–8 is appropriate
  — narrow enough to leave balance dynamics below 1 Hz untouched,
  wide enough to cover drift around 2.3–2.7 Hz.
- Phase response: near unity below 0.5 fc; ~0° at DC; phase wraps
  around fc.  Low-frequency group delay is well below one sample
  period for fs=100, fc=2.5, Q>=2.
"""

from __future__ import annotations

import math
from typing import Tuple


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


# ---------------------------------------------------------------------------
# Biquad notch filter (Direct Form II Transposed)
# ---------------------------------------------------------------------------

class BiquadNotchFilter:
    """Causal IIR biquad notch filter.

    Coefficients are computed from sampling frequency *fs_hz*, centre
    frequency *fc_hz*, and quality factor *Q* (or bandwidth *bw_hz*).

    Direct Form II Transposed (canonical):
        y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2]
                         - a1 * y[n-1] - a2 * y[n-2]

    Usage
    -----
    >>> nf = BiquadNotchFilter(fs_hz=100.0, fc_hz=2.5, Q=6.0)
    >>> y = nf.update(x)          # single sample
    >>> ys = [nf.update(xi) for xi in signal]   # stream
    >>> nf.reset()                # clear state
    """

    def __init__(
        self,
        fs_hz: float,
        fc_hz: float,
        Q: float | None = None,
        bw_hz: float | None = None,
    ):
        if Q is None and bw_hz is None:
            raise ValueError("Either Q or bw_hz must be provided")
        if Q is not None and bw_hz is not None:
            raise ValueError("Provide either Q or bw_hz, not both")
        _validate_finite(fs_hz, "fs_hz")
        _validate_finite(fc_hz, "fc_hz")
        if fs_hz <= 0:
            raise ValueError(f"fs_hz must be positive, got {fs_hz}")
        if fc_hz <= 0:
            raise ValueError(f"fc_hz must be positive, got {fc_hz}")
        if fc_hz >= fs_hz / 2.0:
            raise ValueError(
                f"fc_hz ({fc_hz}) must be < fs_hz/2 ({fs_hz/2})"
            )

        self.fs_hz = float(fs_hz)
        self.fc_hz = float(fc_hz)

        if Q is not None:
            _validate_finite(Q, "Q")
            if Q <= 0:
                raise ValueError(f"Q must be positive, got {Q}")
            self.Q = float(Q)
            self.bw_hz = float(fc_hz / Q)  # -3 dB bandwidth
        else:
            _validate_finite(bw_hz, "bw_hz")
            if bw_hz <= 0:
                raise ValueError(f"bw_hz must be positive, got {bw_hz}")
            self.bw_hz = float(bw_hz)
            self.Q = float(fc_hz / bw_hz)

        self._compute_coefficients()

        # State: x[n-1], x[n-2], y[n-1], y[n-2]
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0

    def _compute_coefficients(self) -> None:
        """Compute biquad notch coefficients (RBJ / Audio EQ Cookbook)."""
        w0 = 2.0 * math.pi * self.fc_hz / self.fs_hz
        alpha = math.sin(w0) / (2.0 * self.Q)
        cos_w0 = math.cos(w0)

        # Notch: b0 = b2 = 1, b1 = -2 * cos(w0)
        #        a0 = 1 + alpha, a1 = -2 * cos(w0), a2 = 1 - alpha
        # Normalised by a0:
        self._b0 = 1.0 / (1.0 + alpha)
        self._b1 = -2.0 * cos_w0 / (1.0 + alpha)
        self._b2 = 1.0 / (1.0 + alpha)
        self._a1 = -2.0 * cos_w0 / (1.0 + alpha)
        self._a2 = (1.0 - alpha) / (1.0 + alpha)

        # Stability check: pole magnitude should be < 1
        # For a notch with alpha > 0, poles are inside the unit circle.
        pole_mag_sq = self._a2 * self._a2  # approximate; real poles at -a1/2 ± sqrt(...)
        if not math.isfinite(self._b0) or not math.isfinite(self._a2):
            raise ValueError("Non-finite coefficients: check fc/Q/fs")
        if abs(self._a2) >= 1.0:
            raise ValueError(
                f"Unstable filter: |a2| = {abs(self._a2):.4f} >= 1.  "
                f"Check fc={self.fc_hz}, fs={self.fs_hz}, Q={self.Q}"
            )

    @property
    def b0(self) -> float:
        """Feedforward coefficient b0."""
        return self._b0

    @property
    def b1(self) -> float:
        """Feedforward coefficient b1."""
        return self._b1

    @property
    def b2(self) -> float:
        """Feedforward coefficient b2."""
        return self._b2

    @property
    def a1(self) -> float:
        """Feedback coefficient a1 (negated, as used in DF2T)."""
        return self._a1

    @property
    def a2(self) -> float:
        """Feedback coefficient a2 (negated, as used in DF2T)."""
        return self._a2

    def coefficients(self) -> Tuple[float, float, float, float, float]:
        """Return (b0, b1, b2, a1, a2) tuple."""
        return (self._b0, self._b1, self._b2, self._a1, self._a2)

    def reset(self) -> None:
        """Reset filter state (zero initial conditions)."""
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0

    def update(self, x: float) -> float:
        """Process one sample and return the filtered output.

        Parameters
        ----------
        x : float
            Input sample.

        Returns
        -------
        y : float
            Filtered output sample.
        """
        # Direct Form II Transposed
        y = (
            self._b0 * x
            + self._b1 * self._x1
            + self._b2 * self._x2
            - self._a1 * self._y1
            - self._a2 * self._y2
        )
        # Shift state
        self._x2 = self._x1
        self._x1 = x
        self._y2 = self._y1
        self._y1 = y
        return y

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return (x1, x2, y1, y2) state tuple."""
        return (self._x1, self._x2, self._y1, self._y2)

    def __repr__(self) -> str:
        return (
            f"BiquadNotchFilter(fs={self.fs_hz:.1f}, fc={self.fc_hz:.3f}, "
            f"Q={self.Q:.3f}, bw={self.bw_hz:.4f})"
        )


# ---------------------------------------------------------------------------
# Smoothstep gate helper for continuous filter activation
# ---------------------------------------------------------------------------

def smoothstep_gate(
    value: float,
    start: float,
    end: float,
) -> float:
    """Smooth Hermite interpolation gate in [start, end].

    Returns 0.0 when *value* <= *start*, 1.0 when *value* >= *end*,
    and smooth Hermite interpolation in between.
    """
    if end <= start:
        return 1.0 if value >= end else 0.0
    u = (value - start) / (end - start)
    u_clamped = max(0.0, min(1.0, u))
    return u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)
