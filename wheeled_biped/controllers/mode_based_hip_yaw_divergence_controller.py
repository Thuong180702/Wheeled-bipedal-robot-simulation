"""Mode-based hip yaw divergence controller.

Implements a small, testable component that computes antisymmetric torques to
counteract hip‑yaw divergence based on a simple proportional‑derivative law.
The controller is deliberately lightweight so it can be instantiated from a config
and used in the torque composition pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp


@dataclass(frozen=True)
class HipYawState:
    """State snapshot passed to the controller.

    Attributes
    ----------
    div_error: float
        Positional divergence error (left minus right). Positive means the left
        hip‑yaw is ahead of the right.
    div_rate: float
        Rate of divergence (left velocity minus right velocity).
    height: float
        Target CoM height (meters) – used for optional height gating.
    support_error: float
        Absolute support position error (meters) from sagittal controller.
    support_error_rate: float
        Rate of change of support error (m/s).
    """

    div_error: float
    div_rate: float
    height: float
    support_error: float = 0.0
    support_error_rate: float = 0.0


class ModeBasedHipYawDivergenceController:
    """Compute hip‑yaw divergence correction torques.

    The controller follows the spec in the task description. It can be disabled
    via the ``enabled`` flag. When disabled ``compute`` returns zero torques.

    Optional support-aware gating (opt-in, disabled by default):
    When ``support_gate_enabled`` is True, the height gate is multiplied by a
    continuous support-aware gate that attenuates mode-div torque when support
    position error or error rate exceeds thresholds. This prevents mode-div
    torque from re-exciting support/pitch dynamics at high heights.
    """

    def __init__(self, cfg: Dict):
        # Expected config keys (with defaults for robustness)
        self.enabled: bool = cfg.get("enabled", False)
        self.kp_div: float = cfg.get("kp_div", 0.0)
        self.kd_div: float = cfg.get("kd_div", 0.0)
        self.max_torque: float = cfg.get("max_torque", 0.0)
        # soft_limit_rad and soft_limit_gain define a simple height gate – we use a
        # smoothstep that linearly ramps from full authority at height <= soft_limit_rad
        # to zero authority at height >= soft_limit_rad + soft_limit_gain.
        self.soft_limit_rad: float = cfg.get("soft_limit_rad", 0.3)
        self.soft_limit_gain: float = cfg.get("soft_limit_gain", 0.5)
        self.ref_source: str = cfg.get("ref_source", "target")
        # Support-aware gating (opt-in, disabled by default)
        self.support_gate_enabled: bool = cfg.get("support_gate_enabled", False)
        self.support_threshold_m: float = cfg.get("support_threshold_m", 0.25)
        self.support_width_m: float = cfg.get("support_width_m", 0.10)
        self.support_min_gate: float = cfg.get("support_min_gate", 0.70)
        self.support_rate_threshold_mps: float = cfg.get("support_rate_threshold_mps", 0.05)
        self.support_rate_width_mps: float = cfg.get("support_rate_width_mps", 0.03)
        self.support_rate_min_gate: float = cfg.get("support_rate_min_gate", 0.70)

    @staticmethod
    def _smoothstep_down(x: float, low: float, high: float) -> float:
        """C1 smoothstep from 1 at x <= low down to 0 at x >= high."""
        if x <= low:
            return 1.0
        if x >= high:
            return 0.0
        u = (high - x) / (high - low)  # maps low->1, high->0
        return 3.0 * u ** 2 - 2.0 * u ** 3

    def _height_gate(self, height: float) -> float:
        """Compute a smooth gate based on height."""
        low = self.soft_limit_rad
        high = self.soft_limit_rad + self.soft_limit_gain
        return self._smoothstep_down(height, low, high)

    def _support_error_gate(self, support_error_abs: float) -> float:
        """Compute smooth support-error gate.

        Returns 1.0 when |support_error| <= threshold, decreases smoothly
        to support_min_gate when |support_error| >= threshold + width.
        Returns 1.0 when support_gate_enabled is False.
        """
        if not self.support_gate_enabled:
            return 1.0
        gate = self._smoothstep_down(support_error_abs, self.support_threshold_m,
                                     self.support_threshold_m + self.support_width_m)
        return self.support_min_gate + (1.0 - self.support_min_gate) * gate

    def _support_rate_gate(self, support_error_rate_abs: float) -> float:
        """Compute smooth support-error-rate gate.

        Returns 1.0 when |support_error_rate| <= threshold, decreases smoothly
        to support_rate_min_gate when |support_error_rate| >= threshold + width.
        Returns 1.0 when support_gate_enabled is False.
        """
        if not self.support_gate_enabled:
            return 1.0
        gate = self._smoothstep_down(support_error_rate_abs, self.support_rate_threshold_mps,
                                     self.support_rate_threshold_mps + self.support_rate_width_mps)
        return self.support_rate_min_gate + (1.0 - self.support_rate_min_gate) * gate

    def compute(self, state: HipYawState) -> Dict[str, float]:
        """Return a dict with ``tau_left`` and ``tau_right``.

        If the controller is disabled, both torques are zero.
        The sign convention matches the existing HY2‑DIV implementation:
        * Positive ``div_error`` (left ahead) → left torque negative,
          right torque positive.
        * ``div_rate`` contributes analogously.
        The raw torque is multiplied by a height gate (if ``ref_source`` is
        ``"target"``) and optionally a support-aware gate, then clipped
        to ``[-max_torque, max_torque]``.
        """
        if not self.enabled:
            return {"tau_left": 0.0, "tau_right": 0.0,
                    "tau_left_raw": 0.0, "tau_right_raw": 0.0,
                    "support_error_gate": 1.0, "support_rate_gate": 1.0,
                    "effective_support_gate": 1.0, "combined_gate": 1.0}

        # Proportional‑derivative law (antisymmetric)
        raw = -(self.kp_div * state.div_error + self.kd_div * state.div_rate)

        # Height gate
        gate = 1.0
        if self.ref_source == "target":
            gate = self._height_gate(state.height)

        # Support-aware gate (opt-in, multiplies height gate)
        support_error_gate = self._support_error_gate(abs(state.support_error))
        support_rate_gate = self._support_rate_gate(abs(state.support_error_rate))
        effective_support_gate = min(support_error_gate, support_rate_gate)
        combined_gate = gate * effective_support_gate

        torque = raw * combined_gate
        # Store pre-clip (raw) torque for telemetry / saturation analysis
        tau_left_raw = torque
        tau_right_raw = -torque
        # Clip to max torque magnitude
        torque_clipped = float(jnp.clip(torque, -self.max_torque, self.max_torque))
        # The antisymmetric torque is applied as left = torque, right = -torque
        tau_left = torque_clipped
        tau_right = -torque_clipped
        return {
            "tau_left": tau_left,
            "tau_right": tau_right,
            "tau_left_raw": tau_left_raw,
            "tau_right_raw": tau_right_raw,
            "support_error_gate": support_error_gate,
            "support_rate_gate": support_rate_gate,
            "effective_support_gate": effective_support_gate,
            "combined_gate": combined_gate,
        }
