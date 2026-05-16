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
        tau_hip_roll_array = jnp.asarray(tau_hip_roll)
        assert tau_hip_roll_array.shape == (2,), f"Expected shape (2,), got {tau_hip_roll_array.shape}"

        # Hip roll torques directly contribute to roll moment about CoM
        # Both left and right hip roll torques add to roll moment
        mx = tau_hip_roll_array[0] + tau_hip_roll_array[1]
        return float(mx)

    def build_wrench_matrix(
        self,
        mj_data: mujoco.MjData,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
    ) -> Array:
        """Build A_wrench matrix mapping decision variables to centroidal wrench.

        Decision variables: [f_left (3), f_right (3), tau_hip_roll_L, tau_hip_roll_R]
        Wrench: [Fx, Fy, Fz, Mx, My, Mz]

        Args:
            mj_data: MuJoCo data with current robot state
            wheel_pos_left: Left wheel position relative to CoM (3,) [x, y, z]
            wheel_pos_right: Right wheel position relative to CoM (3,) [x, y, z]

        Returns:
            A_wrench matrix (6, 8) mapping decision variables to wrench
        """
        # Get wheel Jacobians (3, 10) each
        J_left, J_right = self.compute_wheel_jacobians(mj_data)

        # Initialize wrench matrix (6, 8)
        A_wrench = jnp.zeros((6, 8))

        # Force rows (Fx, Fy, Fz): sum of wheel forces
        # Columns 0-2: left wheel forces, columns 3-5: right wheel forces
        A_wrench = A_wrench.at[0, 0].set(1.0)  # Fx from f_left_x
        A_wrench = A_wrench.at[0, 3].set(1.0)  # Fx from f_right_x
        A_wrench = A_wrench.at[1, 1].set(1.0)  # Fy from f_left_y
        A_wrench = A_wrench.at[1, 4].set(1.0)  # Fy from f_right_y
        A_wrench = A_wrench.at[2, 2].set(1.0)  # Fz from f_left_z
        A_wrench = A_wrench.at[2, 5].set(1.0)  # Fz from f_right_z

        # Moment rows: r × F for each wheel + hip roll contribution
        r_left = wheel_pos_left
        r_right = wheel_pos_right

        # Mx (roll moment) row
        # From left wheel: r_y * Fz - r_z * Fy
        A_wrench = A_wrench.at[3, 1].set(-r_left[2])  # -r_z * f_left_y
        A_wrench = A_wrench.at[3, 2].set(r_left[1])   # r_y * f_left_z
        # From right wheel: r_y * Fz - r_z * Fy
        A_wrench = A_wrench.at[3, 4].set(-r_right[2])  # -r_z * f_right_y
        A_wrench = A_wrench.at[3, 5].set(r_right[1])   # r_y * f_right_z
        # From hip roll torques: direct contribution
        A_wrench = A_wrench.at[3, 6].set(1.0)  # tau_hip_roll_L
        A_wrench = A_wrench.at[3, 7].set(1.0)  # tau_hip_roll_R

        # My (pitch moment) row
        # From left wheel: r_z * Fx - r_x * Fz
        A_wrench = A_wrench.at[4, 0].set(r_left[2])   # r_z * f_left_x
        A_wrench = A_wrench.at[4, 2].set(-r_left[0])  # -r_x * f_left_z
        # From right wheel: r_z * Fx - r_x * Fz
        A_wrench = A_wrench.at[4, 3].set(r_right[2])   # r_z * f_right_x
        A_wrench = A_wrench.at[4, 5].set(-r_right[0])  # -r_x * f_right_z

        # Mz (yaw moment) row
        # From left wheel: r_x * Fy - r_y * Fx
        A_wrench = A_wrench.at[5, 0].set(-r_left[1])  # -r_y * f_left_x
        A_wrench = A_wrench.at[5, 1].set(r_left[0])   # r_x * f_left_y
        # From right wheel: r_x * Fy - r_y * Fx
        A_wrench = A_wrench.at[5, 3].set(-r_right[1])  # -r_y * f_right_x
        A_wrench = A_wrench.at[5, 4].set(r_right[0])   # r_x * f_right_y

        return A_wrench

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

        Note:
            Hip roll torques are added at hardcoded indices 0 (left) and 5 (right).
            This assumes the joint ordering defined in the robot XML remains unchanged.
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
            assert tau_hip_roll_array.shape == (2,), f"Expected shape (2,), got {tau_hip_roll_array.shape}"
            # Hip roll joints are indices 0 (left) and 5 (right)
            tau_total = tau_total.at[0].add(tau_hip_roll_array[0])
            tau_total = tau_total.at[5].add(tau_hip_roll_array[1])

        return tau_total
