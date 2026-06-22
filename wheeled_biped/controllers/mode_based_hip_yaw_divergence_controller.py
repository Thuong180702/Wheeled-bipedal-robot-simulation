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
    """

    div_error: float
    div_rate: float
    height: float


class ModeBasedHipYawDivergenceController:
    """Compute hip‑yaw divergence correction torques.

    The controller follows the spec in the task description. It can be disabled
    via the ``enabled`` flag. When disabled ``compute`` returns zero torques.
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

    def _height_gate(self, height: float) -> float:
        """Compute a smooth gate based on height.

        The gate is 1.0 when ``height`` <= ``soft_limit_rad`` and 0.0 when
        ``height`` >= ``soft_limit_rad + soft_limit_gain``. Between these values a
        smoothstep (C1 continuous) is used.
        """
        low = self.soft_limit_rad
        high = self.soft_limit_rad + self.soft_limit_gain
        if height <= low:
            return 1.0
        if height >= high:
            return 0.0
        u = (high - height) / (high - low)  # maps low->1, high->0
        # smoothstep C1: 3u^2 - 2u^3
        return 3.0 * u ** 2 - 2.0 * u ** 3

    def compute(self, state: HipYawState) -> Dict[str, float]:
        """Return a dict with ``tau_left`` and ``tau_right``.

        If the controller is disabled, both torques are zero.
        The sign convention matches the existing HY2‑DIV implementation:
        * Positive ``div_error`` (left ahead) → left torque negative,
          right torque positive.
        * ``div_rate`` contributes analogously.
        The raw torque is multiplied by a height gate (if ``ref_source`` is
        ``"target"``) and finally clipped to ``[-max_torque, max_torque]``.
        """
        if not self.enabled:
            return {"tau_left": 0.0, "tau_right": 0.0}

        # Proportional‑derivative law (antisymmetric)
        raw = -(self.kp_div * state.div_error + self.kd_div * state.div_rate)

        # Apply height gating based on ref_source; for now only "target" is used.
        gate = 1.0
        if self.ref_source == "target":
            gate = self._height_gate(state.height)
        # Additional sources could be added in the future.

        torque = raw * gate
        # Clip to max torque magnitude
        torque_clipped = float(jnp.clip(torque, -self.max_torque, self.max_torque))
        # Left gets negative of the computed torque (since raw already includes sign)
        # The antisymmetric torque is applied as left = torque, right = -torque
        tau_left = torque_clipped
        tau_right = -torque_clipped
        return {"tau_left": tau_left, "tau_right": tau_right}
