"""Wheel balancing controller using inverted pendulum dynamics.

Treats wheels as horizontal thrusters (like aerial controller) to maintain balance.
Uses pitch angle and pitch rate feedback to compute wheel velocity commands.
Uses roll angle and roll rate feedback to compute hip roll torques.
"""

import jax.numpy as jnp
from jax import Array


class WheelBalancingController:
    """Wheel velocity controller for inverted pendulum balancing."""

    def __init__(
        self,
        k_pitch: float = 8.0,
        k_pitch_rate: float = 2.0,
        k_position: float = 0.5,
        max_wheel_velocity: float = 20.0,
        k_roll: float = 15.0,
        k_roll_rate: float = 3.0,
        max_hip_roll_torque: float = 30.0,
    ):
        """Initialize wheel balancing controller.

        Args:
            k_pitch: Pitch angle feedback gain (rad -> rad/s)
            k_pitch_rate: Pitch rate feedback gain (rad/s -> rad/s)
            k_position: Position drift feedback gain (m -> rad/s)
            max_wheel_velocity: Maximum wheel velocity command (rad/s)
            k_roll: Roll angle feedback gain (rad -> Nm)
            k_roll_rate: Roll rate feedback gain (rad/s -> Nm)
            max_hip_roll_torque: Maximum hip roll torque (Nm)
        """
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.k_position = k_position
        self.max_wheel_velocity = max_wheel_velocity
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.max_hip_roll_torque = max_hip_roll_torque

    def compute_wheel_velocity(
        self,
        pitch: float,
        pitch_rate: float,
        com_y: float,
        com_vy: float,
        roll: float = 0.0,
        roll_rate: float = 0.0,
        k_roll_wheel: float = 2.0,
    ) -> Array:
        """Compute wheel velocity commands for balancing.

        Args:
            pitch: Body pitch angle (rad, positive = forward tilt)
            pitch_rate: Body pitch rate (rad/s)
            com_y: CoM position in sagittal direction (m)
            com_vy: CoM velocity in sagittal direction (m/s)
            roll: Body roll angle (rad, positive = right tilt)
            roll_rate: Body roll rate (rad/s)
            k_roll_wheel: Roll feedback gain for differential wheel control

        Returns:
            Wheel velocity commands [left, right] (rad/s)
        """
        # Inverted pendulum control: wheel velocity opposes pitch
        # Positive pitch (forward tilt) -> negative wheel velocity (roll backward)
        v_pitch = -self.k_pitch * pitch
        v_pitch_rate = -self.k_pitch_rate * pitch_rate
        v_position = -self.k_position * com_y

        # Total wheel velocity command for pitch control
        v_wheel_base = v_pitch + v_pitch_rate + v_position

        # Differential wheel control for roll stabilization
        # Positive roll (right tilt) -> left wheel faster, right wheel slower
        v_roll_diff = -k_roll_wheel * roll

        # Left and right wheel velocities
        v_left = v_wheel_base + v_roll_diff
        v_right = v_wheel_base - v_roll_diff

        # Clip to maximum velocity
        v_left = jnp.clip(v_left, -self.max_wheel_velocity, self.max_wheel_velocity)
        v_right = jnp.clip(v_right, -self.max_wheel_velocity, self.max_wheel_velocity)

        return jnp.array([v_left, v_right])

    def compute_hip_roll_torques(
        self,
        roll: float,
        roll_rate: float,
    ) -> Array:
        """Compute hip roll torques for lateral stability.

        Args:
            roll: Body roll angle (rad, positive = right tilt)
            roll_rate: Body roll rate (rad/s)

        Returns:
            Hip roll torques [left, right] (Nm)
        """
        # Roll stabilization: torque opposes roll angle and rate
        # Positive roll (right tilt) -> negative torque (counteract)
        tau_roll = -self.k_roll * roll - self.k_roll_rate * roll_rate

        # Clip to maximum torque
        tau_roll = jnp.clip(tau_roll, -self.max_hip_roll_torque, self.max_hip_roll_torque)

        # Both hip rolls get same torque for roll stabilization
        return jnp.array([tau_roll, tau_roll])

    def compute_wheel_torques(
        self,
        wheel_velocity_cmd: Array,
        wheel_velocity_actual: Array,
        kp: float = 5.0,
        kd: float = 0.5,
        max_torque: float = 30.0,
    ) -> Array:
        """Convert wheel velocity commands to torques via PD control.

        Args:
            wheel_velocity_cmd: Desired wheel velocities [left, right] (rad/s)
            wheel_velocity_actual: Actual wheel velocities [left, right] (rad/s)
            kp: Proportional gain
            kd: Derivative gain (damping)
            max_torque: Maximum wheel torque (Nm)

        Returns:
            Wheel torques [left, right] (Nm)
        """
        velocity_error = wheel_velocity_cmd - wheel_velocity_actual
        tau_wheel = kp * velocity_error - kd * wheel_velocity_actual
        tau_wheel = jnp.clip(tau_wheel, -max_torque, max_torque)
        return tau_wheel
