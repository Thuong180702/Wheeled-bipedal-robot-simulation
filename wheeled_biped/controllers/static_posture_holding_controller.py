"""Static posture holding controller for Stage 2 correction-only WBC integration.

Maintains internal joint posture at calibrated equilibrium using PD control.
Designed to work with correction-only WBC: posture holding provides baseline
joint torques, WBC provides correction torques for balance/height tracking.

Key design principles:
- Holds joints at calibrated equilibrium posture (h=0.404m from keyframe)
- Uses joint-space PD control: tau = kp * (q_ref - q) - kd * qvel
- Applies only to internal support joints: hip_pitch/knee [2,3,7,8]
- Optional smaller gains for hip_roll/hip_yaw
- Wheel torques default to zero (wheels controlled by WBC)
- Respects actuator torque limits
"""

import jax.numpy as jnp
from jax import Array


class StaticPostureHoldingController:
    """Static posture holding controller using joint-space PD control."""

    def __init__(
        self,
        kp_hip_roll: float = 5.0,
        kd_hip_roll: float = 1.0,
        kp_hip_yaw: float = 5.0,
        kd_hip_yaw: float = 1.0,
        kp_hip_pitch: float = 30.0,
        kd_hip_pitch: float = 4.0,
        kp_knee: float = 40.0,
        kd_knee: float = 5.0,
        max_torque_hip_roll: float = 15.0,
        max_torque_hip_yaw: float = 15.0,
        max_torque_hip_pitch: float = 30.0,
        max_torque_knee: float = 30.0,
    ):
        """Initialize static posture holding controller.

        Args:
            kp_hip_roll: Proportional gain for hip_roll
            kd_hip_roll: Derivative gain for hip_roll
            kp_hip_yaw: Proportional gain for hip_yaw
            kd_hip_yaw: Derivative gain for hip_yaw
            kp_hip_pitch: Proportional gain for hip_pitch
            kd_hip_pitch: Derivative gain for hip_pitch
            kp_knee: Proportional gain for knee
            kd_knee: Derivative gain for knee
            max_torque_hip_roll: Maximum torque for hip_roll (Nm)
            max_torque_hip_yaw: Maximum torque for hip_yaw (Nm)
            max_torque_hip_pitch: Maximum torque for hip_pitch (Nm)
            max_torque_knee: Maximum torque for knee (Nm)
        """
        self.kp_hip_roll = kp_hip_roll
        self.kd_hip_roll = kd_hip_roll
        self.kp_hip_yaw = kp_hip_yaw
        self.kd_hip_yaw = kd_hip_yaw
        self.kp_hip_pitch = kp_hip_pitch
        self.kd_hip_pitch = kd_hip_pitch
        self.kp_knee = kp_knee
        self.kd_knee = kd_knee

        self.max_torque_hip_roll = max_torque_hip_roll
        self.max_torque_hip_yaw = max_torque_hip_yaw
        self.max_torque_hip_pitch = max_torque_hip_pitch
        self.max_torque_knee = max_torque_knee

        # Equilibrium reference (set via set_equilibrium_reference)
        self.equilibrium_joint_pos = None

    def set_equilibrium_reference(self, joint_pos: Array):
        """Set equilibrium joint positions from calibrated keyframe.

        Args:
            joint_pos: Equilibrium joint positions (10,)
        """
        self.equilibrium_joint_pos = joint_pos

    def compute_posture_holding_torque(
        self, joint_pos: Array, joint_vel: Array
    ) -> tuple[Array, dict]:
        """Compute posture holding torque using PD control.

        Args:
            joint_pos: Current joint positions (10,)
            joint_vel: Current joint velocities (10,)

        Returns:
            Tuple of (torque (10,), diagnostics dict)
        """
        if self.equilibrium_joint_pos is None:
            raise RuntimeError(
                "Equilibrium reference not set. Call set_equilibrium_reference() first."
            )

        # Compute position errors
        pos_error = self.equilibrium_joint_pos - joint_pos

        # Initialize torque array
        tau = jnp.zeros(10)

        # Joint indices
        # 0: l_hip_roll, 1: l_hip_yaw, 2: l_hip_pitch, 3: l_knee, 4: l_wheel
        # 5: r_hip_roll, 6: r_hip_yaw, 7: r_hip_pitch, 8: r_knee, 9: r_wheel

        # Hip roll (indices 0, 5)
        for idx in [0, 5]:
            tau_raw = self.kp_hip_roll * pos_error[idx] - self.kd_hip_roll * joint_vel[idx]
            tau = tau.at[idx].set(
                jnp.clip(tau_raw, -self.max_torque_hip_roll, self.max_torque_hip_roll)
            )

        # Hip yaw (indices 1, 6)
        for idx in [1, 6]:
            tau_raw = self.kp_hip_yaw * pos_error[idx] - self.kd_hip_yaw * joint_vel[idx]
            tau = tau.at[idx].set(
                jnp.clip(tau_raw, -self.max_torque_hip_yaw, self.max_torque_hip_yaw)
            )

        # Hip pitch (indices 2, 7)
        for idx in [2, 7]:
            tau_raw = self.kp_hip_pitch * pos_error[idx] - self.kd_hip_pitch * joint_vel[idx]
            tau = tau.at[idx].set(
                jnp.clip(tau_raw, -self.max_torque_hip_pitch, self.max_torque_hip_pitch)
            )

        # Knee (indices 3, 8)
        for idx in [3, 8]:
            tau_raw = self.kp_knee * pos_error[idx] - self.kd_knee * joint_vel[idx]
            tau = tau.at[idx].set(
                jnp.clip(tau_raw, -self.max_torque_knee, self.max_torque_knee)
            )

        # Wheels (indices 4, 9) - zero torque, controlled by WBC
        # tau[4] and tau[9] remain zero

        # Diagnostics
        hip_pitch_indices = jnp.array([2, 7])
        knee_indices = jnp.array([3, 8])

        diagnostics = {
            "posture_error_norm": float(jnp.linalg.norm(pos_error)),
            "posture_error_hip_pitch_max": float(
                jnp.max(jnp.abs(pos_error[hip_pitch_indices]))
            ),
            "posture_error_knee_max": float(jnp.max(jnp.abs(pos_error[knee_indices]))),
            "tau_posture_norm": float(jnp.linalg.norm(tau)),
            "tau_posture_hip_pitch_max": float(jnp.max(jnp.abs(tau[hip_pitch_indices]))),
            "tau_posture_knee_max": float(jnp.max(jnp.abs(tau[knee_indices]))),
        }

        return tau, diagnostics
