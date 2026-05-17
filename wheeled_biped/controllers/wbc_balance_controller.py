"""
Whole-Body Control (WBC) Balance Controller

Torque-first architecture where WBC commands are primary control authority.
No dominant PID position control - only lightweight damping/tracking.

This addresses the root cause from Step 5.21: PID authority suppression.
"""

import numpy as np


class WBCBalanceController:
    """WBC balance controller with force/torque-based stabilization.

    Architecture:
        observation -> WBC torque computation -> actuators

    No position targets, no dominant PID, no authority suppression.
    """

    def __init__(
        self,
        k_roll: float = 20.0,
        k_roll_rate: float = 2.0,
        k_pitch: float = 5.0,
        k_pitch_rate: float = 0.5,
        allow_wheel_torque: bool = False,
        wheel_roll_gain: float = 0.0,
    ):
        """Initialize WBC balance controller.

        Args:
            k_roll: Roll stabilization gain.
            k_roll_rate: Roll rate damping gain.
            k_pitch: Pitch stabilization gain.
            k_pitch_rate: Pitch rate damping gain.
            allow_wheel_torque: Enable wheel torque for roll stabilization.
            wheel_roll_gain: Wheel torque gain for roll rate damping.
        """
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.allow_wheel_torque = allow_wheel_torque
        self.wheel_roll_gain = wheel_roll_gain

    def compute_torque(self, obs: np.ndarray) -> np.ndarray:
        """Compute WBC torque commands from observation.

        Args:
            obs: Observation vector (42-dim for BalanceEnv).
                [0:3] gravity_body
                [6:9] base_ang_vel

        Returns:
            Normalized torque commands in [-1, 1]^10.
        """
        # Extract state from observation
        gravity_body = obs[0:3]
        roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
        pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))

        angular_vel = obs[6:9]
        pitch_rate = float(angular_vel[0])
        roll_rate = float(angular_vel[1])

        # Compute torque commands
        roll_cmd = -(self.k_roll * roll + self.k_roll_rate * roll_rate)
        pitch_cmd = -(self.k_pitch * pitch + self.k_pitch_rate * pitch_rate)
        wheel_cmd = -self.wheel_roll_gain * roll_rate if self.allow_wheel_torque else 0.0

        # Build action vector
        action = np.zeros(10, dtype=np.float32)

        # Roll correction (hip roll joints)
        action[0] = np.clip(roll_cmd, -1.0, 1.0)  # L_HIP_ROLL
        action[5] = np.clip(roll_cmd, -1.0, 1.0)  # R_HIP_ROLL

        # Pitch correction (hip pitch and knee)
        action[2] = np.clip(0.5 * pitch_cmd, -1.0, 1.0)  # L_HIP_PITCH
        action[3] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)  # L_KNEE
        action[7] = np.clip(0.5 * pitch_cmd, -1.0, 1.0)  # R_HIP_PITCH
        action[8] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)  # R_KNEE

        # Wheel torque (optional)
        if self.allow_wheel_torque:
            action[4] = np.clip(wheel_cmd, -1.0, 1.0)  # L_WHEEL
            action[9] = np.clip(wheel_cmd, -1.0, 1.0)  # R_WHEEL

        return action

    def reset(self):
        """Reset controller state (stateless controller, no-op)."""
        pass
