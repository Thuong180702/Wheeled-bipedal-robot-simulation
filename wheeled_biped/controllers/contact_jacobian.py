"""Contact Jacobian computation for force-to-torque mapping.

Maps Cartesian contact forces at wheel contact points to joint torques using
MuJoCo's built-in Jacobian computation.
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array


class ContactJacobian:
    """Computes contact Jacobians for wheel contact points."""

    def __init__(self, mj_model: mujoco.MjModel):
        """Initialize contact Jacobian computer.

        Args:
            mj_model: MuJoCo model with robot definition
        """
        self.mj_model = mj_model

        # Find wheel body IDs
        self.l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
        self.r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

        # Find hip roll joint IDs
        self.l_hip_roll_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "l_hip_roll")
        self.r_hip_roll_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_roll")

        # Preallocate Jacobian arrays (3 x nv for translation, 3 x nv for rotation)
        self.jacp = np.zeros((3, mj_model.nv))
        self.jacr = np.zeros((3, mj_model.nv))

    def compute_wheel_jacobians(self, mj_data: mujoco.MjData) -> tuple[Array, Array]:
        """Compute contact Jacobians for both wheels.

        Args:
            mj_data: MuJoCo data with current robot state

        Returns:
            Tuple of (J_left, J_right) where each is (3, 10) mapping contact forces to joint torques
        """
        # Left wheel Jacobian (translation only, we care about contact forces)
        mujoco.mj_jacBody(self.mj_model, mj_data, self.jacp, self.jacr, self.l_wheel_id)
        # Extract only joint DOFs (skip free joint: 6 DOFs for floating base)
        J_left = jnp.array(self.jacp[:, 6:16])  # (3, 10)

        # Right wheel Jacobian
        mujoco.mj_jacBody(self.mj_model, mj_data, self.jacp, self.jacr, self.r_wheel_id)
        J_right = jnp.array(self.jacp[:, 6:16])  # (3, 10)

        return J_left, J_right

    def compute_hip_roll_moment_contribution(self, tau_hip_roll: Array) -> float:
        """Compute roll moment (Mx) contribution from hip roll torques.

        Args:
            tau_hip_roll: Hip roll torques [left, right] (2,)

        Returns:
            Roll moment contribution (scalar)
        """
        # Hip roll torques directly contribute to roll moment about CoM
        # Both left and right hip roll torques add to roll moment
        mx = tau_hip_roll[0] + tau_hip_roll[1]
        return float(mx)

    def map_contact_forces_to_torques(
        self,
        mj_data: mujoco.MjData,
        f_left: Array,
        f_right: Array,
        tau_hip_roll: Array | None = None,
    ) -> Array:
        """Map contact forces and hip roll torques to joint torques.

        Args:
            mj_data: MuJoCo data with current robot state
            f_left: Left wheel contact force (3,) in world frame [fx, fy, fz]
            f_right: Right wheel contact force (3,) in world frame [fx, fy, fz]
            tau_hip_roll: Optional hip roll torques [left, right] (2,)

        Returns:
            Joint torques (10,) that produce the desired contact forces and hip roll torques
        """
        J_left, J_right = self.compute_wheel_jacobians(mj_data)

        # tau = J^T * f (virtual work principle)
        tau_left = J_left.T @ f_left  # (10,)
        tau_right = J_right.T @ f_right  # (10,)

        # Superposition: total torque is sum of contributions
        tau_total = tau_left + tau_right

        # Add hip roll torque contributions if provided
        if tau_hip_roll is not None:
            tau_hip_roll_array = jnp.asarray(tau_hip_roll)
            # Hip roll joints are indices 0 (left) and 5 (right)
            tau_total = tau_total.at[0].add(tau_hip_roll_array[0])
            tau_total = tau_total.at[5].add(tau_hip_roll_array[1])

        return tau_total
