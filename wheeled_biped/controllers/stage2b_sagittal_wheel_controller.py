"""Stage 2B Sagittal Wheel Balance Controller.

Direct wheel torque controller for sagittal (pitch) stabilization.
Does not route through contact force distribution or contact Jacobian.
"""

import jax.numpy as jnp
from jax import Array


class Stage2BSagittalWheelController:
    """Direct wheel torque controller for sagittal stabilization."""

    def __init__(
        self,
        k_pitch: float = 10.0,
        k_pitch_rate: float = 2.0,
        k_cp: float = 4.0,
        k_com_y: float = 0.0,
        k_com_vy: float = 2.0,
        max_tau_wheel: float = 3.0,
    ):
        """Initialize sagittal wheel controller.

        Sign convention (verified by debug_wheel_sagittal_sign_simple.py):
        - Positive wheel torque moves robot backward (+Y)
        - Negative wheel torque moves robot forward (-Y)
        - Positive pitch_x means falling forward (-Y)
        - To oppose positive pitch_x, need positive torque (move backward)
        - Therefore: tau = -k_pitch * pitch_error

        Args:
            k_pitch: Pitch error gain (Nm/rad)
            k_pitch_rate: Pitch rate gain (Nm/(rad/s))
            k_cp: Capture point error gain (Nm/m)
            k_com_y: CoM Y position error gain (Nm/m)
            k_com_vy: CoM Y velocity gain (Nm/(m/s))
            max_tau_wheel: Maximum wheel torque magnitude (Nm)
        """
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.k_cp = k_cp
        self.k_com_y = k_com_y
        self.k_com_vy = k_com_vy
        self.max_tau_wheel = max_tau_wheel

        self.equilibrium_pitch_x = 0.0
        self.equilibrium_cp_y = 0.0
        self.equilibrium_com_y = 0.0

    def set_equilibrium_reference(
        self,
        pitch_x: float,
        cp_y: float,
        com_y: float,
    ):
        """Set equilibrium references."""
        self.equilibrium_pitch_x = pitch_x
        self.equilibrium_cp_y = cp_y
        self.equilibrium_com_y = com_y

    def compute_wheel_torques(
        self,
        pitch_x: float,
        pitch_rate_x: float,
        cp_y: float,
        com_y: float,
        com_vy: float,
    ) -> tuple[Array, dict]:
        """Compute wheel torques for sagittal stabilization.

        Args:
            pitch_x: Current pitch angle (rad)
            pitch_rate_x: Current pitch rate (rad/s)
            cp_y: Current capture point Y position (m)
            com_y: Current CoM Y position (m)
            com_vy: Current CoM Y velocity (m/s)

        Returns:
            Tuple of (tau_wheel, diagnostics) where:
                - tau_wheel: (10,) torque vector, only [4,9] nonzero
                - diagnostics: dict with errors, commands, saturation flags
        """
        # Compute errors relative to equilibrium
        pitch_error = pitch_x - self.equilibrium_pitch_x
        cp_error_y = cp_y - self.equilibrium_cp_y
        com_error_y = com_y - self.equilibrium_com_y

        # PD control (sign convention verified by debug_wheel_sagittal_sign_simple.py)
        # Positive pitch_x (falling forward) requires positive torque (move backward)
        # Positive torque → backward (+Y), Negative torque → forward (-Y)
        tau_wheel_cmd = (
            +self.k_pitch * pitch_error
            + self.k_pitch_rate * pitch_rate_x
            + self.k_cp * cp_error_y
            + self.k_com_y * com_error_y
            + self.k_com_vy * com_vy
        )

        # Clip torque
        tau_wheel_clipped = jnp.clip(tau_wheel_cmd, -self.max_tau_wheel, self.max_tau_wheel)

        # Build full torque vector (only wheel joints nonzero)
        tau_wheel = jnp.zeros(10)
        tau_wheel = tau_wheel.at[4].set(tau_wheel_clipped)  # l_wheel
        tau_wheel = tau_wheel.at[9].set(tau_wheel_clipped)  # r_wheel

        # Diagnostics
        saturated = jnp.abs(tau_wheel_cmd) > self.max_tau_wheel

        diagnostics = {
            "pitch_error": float(pitch_error),
            "pitch_rate_x": float(pitch_rate_x),
            "cp_error_y": float(cp_error_y),
            "com_error_y": float(com_error_y),
            "com_vy": float(com_vy),
            "tau_wheel_cmd": float(tau_wheel_cmd),
            "tau_wheel_clipped": float(tau_wheel_clipped),
            "saturated": bool(saturated),
        }

        return tau_wheel, diagnostics
