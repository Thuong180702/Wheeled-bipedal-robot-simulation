"""Stage 2C Sagittal State-Feedback Controller.

Full state-feedback controller for sagittal (pitch) stabilization with wheel velocity damping.
Addresses unbounded wheel velocity growth observed in Stage 2B.
"""

import jax.numpy as jnp
from jax import Array


class Stage2CSagittalStateFeedbackController:
    """State-feedback controller for sagittal stabilization with wheel velocity damping."""

    def __init__(
        self,
        k_pitch: float = 20.0,
        k_pitch_rate: float = 6.0,
        k_com_y: float = 0.0,
        k_com_vy: float = 0.0,
        k_cp_y: float = 8.0,
        k_wheel_vel: float = 0.3,
        max_tau_wheel: float = 8.0,
    ):
        """Initialize sagittal state-feedback controller.

        Sign convention (verified by debug_wheel_sagittal_sign_simple.py):
        - Positive wheel torque moves robot backward (+Y)
        - Negative wheel torque moves robot forward (-Y)
        - Positive pitch_x means falling forward (-Y)
        - To oppose positive pitch_x, need positive torque (move backward)
        - Positive wheel velocity means spinning forward, need negative torque to brake

        Args:
            k_pitch: Pitch error gain (Nm/rad)
            k_pitch_rate: Pitch rate gain (Nm/(rad/s))
            k_com_y: CoM Y position error gain (Nm/m)
            k_com_vy: CoM Y velocity gain (Nm/(m/s))
            k_cp_y: Capture point Y error gain (Nm/m)
            k_wheel_vel: Wheel velocity damping gain (Nm/(rad/s))
            max_tau_wheel: Maximum wheel torque magnitude (Nm)
        """
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.k_com_y = k_com_y
        self.k_com_vy = k_com_vy
        self.k_cp_y = k_cp_y
        self.k_wheel_vel = k_wheel_vel
        self.max_tau_wheel = max_tau_wheel

        self.equilibrium_pitch_x = 0.0
        self.equilibrium_com_y = 0.0
        self.equilibrium_cp_y = 0.0

    def set_equilibrium_reference(
        self,
        pitch_x: float,
        com_y: float,
        cp_y: float,
    ):
        """Set equilibrium references."""
        self.equilibrium_pitch_x = pitch_x
        self.equilibrium_com_y = com_y
        self.equilibrium_cp_y = cp_y

    def compute_wheel_torques(
        self,
        pitch_x: float,
        pitch_rate_x: float,
        com_y: float,
        com_vy: float,
        cp_y: float,
        wheel_vel_left: float,
        wheel_vel_right: float,
    ) -> tuple[Array, dict]:
        """Compute wheel torques for sagittal stabilization with velocity damping.

        Args:
            pitch_x: Current pitch angle (rad)
            pitch_rate_x: Current pitch rate (rad/s)
            com_y: Current CoM Y position (m)
            com_vy: Current CoM Y velocity (m/s)
            cp_y: Current capture point Y position (m)
            wheel_vel_left: Left wheel velocity (rad/s)
            wheel_vel_right: Right wheel velocity (rad/s)

        Returns:
            Tuple of (tau_wheel, diagnostics) where:
                - tau_wheel: (10,) torque vector, only [4,9] nonzero
                - diagnostics: dict with errors, terms, commands, saturation flags
        """
        # Compute errors relative to equilibrium
        pitch_error = pitch_x - self.equilibrium_pitch_x
        com_y_error = com_y - self.equilibrium_com_y
        cp_y_error = cp_y - self.equilibrium_cp_y

        # Mean wheel velocity for damping
        wheel_vel_mean = 0.5 * (wheel_vel_left + wheel_vel_right)

        # Compute individual control terms
        term_pitch = self.k_pitch * pitch_error
        term_pitch_rate = self.k_pitch_rate * pitch_rate_x
        term_com_y = self.k_com_y * com_y_error
        term_com_vy = self.k_com_vy * com_vy
        term_cp_y = self.k_cp_y * cp_y_error
        term_wheel_vel = self.k_wheel_vel * wheel_vel_mean

        # State-feedback control law
        # Sign convention: positive pitch_x (falling forward) requires positive torque (move backward)
        # Wheel velocity damping: positive wheel_vel requires negative torque (brake)
        tau_wheel_raw = (
            term_pitch
            + term_pitch_rate
            + term_com_y
            + term_com_vy
            + term_cp_y
            - term_wheel_vel  # Negative sign to oppose wheel velocity
        )

        # Clip torque
        tau_wheel_clipped = jnp.clip(tau_wheel_raw, -self.max_tau_wheel, self.max_tau_wheel)

        # Build full torque vector (only wheel joints nonzero)
        tau_wheel = jnp.zeros(10)
        tau_wheel = tau_wheel.at[4].set(tau_wheel_clipped)  # l_wheel
        tau_wheel = tau_wheel.at[9].set(tau_wheel_clipped)  # r_wheel

        # Diagnostics
        saturated = jnp.abs(tau_wheel_raw) > self.max_tau_wheel

        diagnostics = {
            "pitch_error": float(pitch_error),
            "pitch_rate_x": float(pitch_rate_x),
            "com_y_error": float(com_y_error),
            "com_vy": float(com_vy),
            "cp_y_error": float(cp_y_error),
            "wheel_vel_left": float(wheel_vel_left),
            "wheel_vel_right": float(wheel_vel_right),
            "wheel_vel_mean": float(wheel_vel_mean),
            "term_pitch": float(term_pitch),
            "term_pitch_rate": float(term_pitch_rate),
            "term_com_y": float(term_com_y),
            "term_com_vy": float(term_com_vy),
            "term_cp_y": float(term_cp_y),
            "term_wheel_vel": float(term_wheel_vel),
            "tau_wheel_raw": float(tau_wheel_raw),
            "tau_wheel_clipped": float(tau_wheel_clipped),
            "saturated": bool(saturated),
        }

        return tau_wheel, diagnostics
