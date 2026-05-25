"""Stage 2B Direct Roll Controller.

Simple PD controller that directly commands hip_roll torques for lateral stabilization.
Does not route through contact force distributor or contact Jacobian.
"""

import jax.numpy as jnp
from jax import Array


class Stage2BRollDirectController:
    """Direct hip_roll torque controller for lateral stabilization."""

    def __init__(
        self,
        k_roll: float = 100.0,
        k_roll_rate: float = 20.0,
        k_roll_integral: float = 0.0,
        tau_hip_roll_max: float = 15.0,
        max_roll_moment: float = 30.0,
    ):
        """Initialize direct roll controller.

        Args:
            k_roll: Roll error gain (Nm/rad)
            k_roll_rate: Roll rate gain (Nm/(rad/s))
            k_roll_integral: Roll integral gain (Nm/(rad*s))
            tau_hip_roll_max: Maximum hip roll torque per side (Nm)
            max_roll_moment: Maximum total roll moment (Nm)
        """
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.k_roll_integral = k_roll_integral
        self.tau_hip_roll_max = tau_hip_roll_max
        self.max_roll_moment = max_roll_moment

        self.equilibrium_roll_y = 0.0

    def set_equilibrium_reference(self, roll_y: float):
        """Set equilibrium roll reference."""
        self.equilibrium_roll_y = roll_y

    def compute_roll_torques(
        self,
        roll_y: float,
        roll_rate_y: float,
        roll_integral: float = 0.0,
    ) -> tuple[Array, dict]:
        """Compute hip_roll torques for lateral stabilization.

        Sign convention verified by hip-roll micro-test:
        - Positive roll error (rolling right) requires negative moment (roll left correction)
        - Negative roll error (rolling left) requires positive moment (roll right correction)
        - Left hip_roll: tau = -M_roll / 2
        - Right hip_roll: tau = +M_roll / 2

        Args:
            roll_y: Current roll angle (rad)
            roll_rate_y: Current roll rate (rad/s)
            roll_integral: Accumulated roll error integral (rad*s)

        Returns:
            Tuple of (tau_roll_direct, diagnostics) where:
                - tau_roll_direct: (10,) torque vector, only [0,5] nonzero
                - diagnostics: dict with roll_error, m_roll_cmd, tau_hip_roll_left/right, saturation flags
        """
        roll_error = roll_y - self.equilibrium_roll_y

        # PD control: negative sign for restoring moment
        m_roll_cmd = (
            -self.k_roll * roll_error
            - self.k_roll_rate * roll_rate_y
            - self.k_roll_integral * roll_integral
        )

        # Clip total moment
        m_roll_clipped = jnp.clip(m_roll_cmd, -self.max_roll_moment, self.max_roll_moment)

        # Map to hip_roll torques
        tau_hip_roll_left = -m_roll_clipped / 2.0
        tau_hip_roll_right = m_roll_clipped / 2.0

        # Clip individual torques
        tau_hip_roll_left_clipped = jnp.clip(
            tau_hip_roll_left, -self.tau_hip_roll_max, self.tau_hip_roll_max
        )
        tau_hip_roll_right_clipped = jnp.clip(
            tau_hip_roll_right, -self.tau_hip_roll_max, self.tau_hip_roll_max
        )

        # Build full torque vector (only hip_roll joints nonzero)
        tau_roll_direct = jnp.zeros(10)
        tau_roll_direct = tau_roll_direct.at[0].set(tau_hip_roll_left_clipped)
        tau_roll_direct = tau_roll_direct.at[5].set(tau_hip_roll_right_clipped)

        # Diagnostics
        moment_saturated = jnp.abs(m_roll_cmd) > self.max_roll_moment
        left_saturated = jnp.abs(tau_hip_roll_left) > self.tau_hip_roll_max
        right_saturated = jnp.abs(tau_hip_roll_right) > self.tau_hip_roll_max

        diagnostics = {
            "roll_error": float(roll_error),
            "roll_rate": float(roll_rate_y),
            "m_roll_cmd": float(m_roll_cmd),
            "m_roll_clipped": float(m_roll_clipped),
            "tau_hip_roll_left": float(tau_hip_roll_left_clipped),
            "tau_hip_roll_right": float(tau_hip_roll_right_clipped),
            "moment_saturated": bool(moment_saturated),
            "left_saturated": bool(left_saturated),
            "right_saturated": bool(right_saturated),
        }

        return tau_roll_direct, diagnostics
