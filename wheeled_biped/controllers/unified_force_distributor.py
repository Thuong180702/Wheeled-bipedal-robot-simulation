"""Unified QP-based force distribution for wheeled biped.

Simultaneously optimizes wheel contact forces and hip roll torques to achieve
desired centroidal wrench while respecting contact and torque constraints.
"""

import jax.numpy as jnp
import mujoco
from jax import Array

from wheeled_biped.controllers.contact_jacobian import ContactJacobian


class UnifiedForceDistributor:
    """Unified QP force distributor using jaxopt.BoxOSQP."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        w_force: float = 0.01,
        w_torque: float = 0.1,
        w_smoothness: float = 0.5,
        tau_hip_roll_max: float = 10.0,
        max_iter: int = 10,
        eps_abs: float = 1e-3,
        eps_rel: float = 1e-3,
    ):
        """Initialize unified force distributor.

        Args:
            mj_model: MuJoCo model with robot definition
            w_force: Weight for contact force effort minimization
            w_torque: Weight for hip roll torque effort minimization
            w_smoothness: Weight for temporal smoothness (deviation from previous solution)
            tau_hip_roll_max: Maximum hip roll torque magnitude (Nm)
            max_iter: Maximum OSQP iterations (100Hz optimized)
            eps_abs: Absolute tolerance for OSQP convergence
            eps_rel: Relative tolerance for OSQP convergence
        """
        self.mj_model = mj_model
        self.w_force = w_force
        self.w_torque = w_torque
        self.w_smoothness = w_smoothness
        self.tau_hip_roll_max = tau_hip_roll_max
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        # Initialize contact Jacobian computer
        self.contact_jacobian = ContactJacobian(mj_model)

        # Previous solution for warm-starting (8D: [f_left(3), f_right(3), tau_hip_roll(2)])
        self.prev_solution = jnp.zeros(8)

    def _build_cost_matrix_p(self) -> Array:
        """Build quadratic cost matrix P for QP.

        P is diagonal with weights for effort minimization:
        - First 6 elements: w_force (wheel contact forces)
        - Last 2 elements: w_torque (hip roll torques)

        Returns:
            P matrix (8, 8) diagonal quadratic cost matrix
        """
        # Build diagonal weights
        diagonal = jnp.array([
            self.w_force, self.w_force, self.w_force,  # f_left
            self.w_force, self.w_force, self.w_force,  # f_right
            self.w_torque, self.w_torque               # tau_hip_roll
        ])

        # Create diagonal matrix
        P = jnp.diag(diagonal)

        return P

    def _build_linear_cost_q(self) -> Array:
        """Build linear cost vector q for QP.

        Implements smoothness penalty by penalizing deviation from previous solution:
        q = -2 * w_smoothness * P @ x_prev

        Returns:
            q vector (8,) linear cost vector
        """
        P = self._build_cost_matrix_p()
        q = -2.0 * self.w_smoothness * (P @ self.prev_solution)

        return q

    def distribute_wrench(
        self,
        mj_data: mujoco.MjData,
        desired_wrench: Array,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
    ) -> tuple[Array, Array, Array]:
        """Distribute desired wrench to wheel forces and hip roll torques.

        Args:
            mj_data: MuJoCo data with current robot state
            desired_wrench: Desired centroidal wrench (6,) [Fx, Fy, Fz, Mx, My, Mz]
            wheel_pos_left: Left wheel position relative to CoM (3,) [x, y, z]
            wheel_pos_right: Right wheel position relative to CoM (3,) [x, y, z]

        Returns:
            Tuple of (f_left, f_right, tau_hip_roll) where:
                - f_left: Left wheel contact force (3,) [fx, fy, fz]
                - f_right: Right wheel contact force (3,) [fx, fy, fz]
                - tau_hip_roll: Hip roll torques (2,) [left, right]
        """
        # TODO: Implement QP formulation and solving in Tasks 5-7
        raise NotImplementedError("QP solving not yet implemented")
