"""Unified QP-based force distribution for wheeled biped.

Simultaneously optimizes wheel contact forces and hip roll torques to achieve
desired centroidal wrench while respecting contact and torque constraints.
"""

import jax.numpy as jnp
import mujoco
from jax import Array
from jaxopt import BoxOSQP

from wheeled_biped.controllers.contact_jacobian import ContactJacobian


class UnifiedForceDistributor:
    """Unified QP force distributor using jaxopt.BoxOSQP."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        w_force: float = 1.0,
        w_torque: float = 1.0,
        w_smoothness: float = 0.1,
        w_wrench: float = 10000.0,
        tau_hip_roll_max: float = 10.0,
        max_iter: int = 200,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
    ):
        """Initialize unified force distributor.

        Args:
            mj_model: MuJoCo model with robot definition
            w_force: Weight for contact force effort minimization
            w_torque: Weight for hip roll torque effort minimization
            w_smoothness: Weight for temporal smoothness (deviation from previous solution)
            w_wrench: Weight for soft wrench tracking (replaces hard equality constraints)
            tau_hip_roll_max: Maximum hip roll torque magnitude (Nm)
            max_iter: Maximum OSQP iterations (increased for better convergence)
            eps_abs: Absolute tolerance for OSQP convergence (relaxed)
            eps_rel: Relative tolerance for OSQP convergence (relaxed)
        """
        self.mj_model = mj_model
        self.w_force = w_force
        self.w_torque = w_torque
        self.w_smoothness = w_smoothness
        self.w_wrench = w_wrench
        self.tau_hip_roll_max = tau_hip_roll_max
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        # Initialize contact Jacobian computer
        self.contact_jacobian = ContactJacobian(mj_model)

        # Previous solution for warm-starting (8D: [f_left(3), f_right(3), tau_hip_roll(2)])
        self.prev_solution = jnp.zeros(8)

    def _build_cost_matrix_p(
        self,
        mj_data: mujoco.MjData,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
    ) -> Array:
        """Build quadratic cost matrix P for QP.

        P combines effort minimization and soft wrench tracking:
        P = P_effort + 2 * w_wrench * A_wrench^T * A_wrench

        Args:
            mj_data: MuJoCo data with current robot state
            wheel_pos_left: Left wheel position relative to CoM (3,)
            wheel_pos_right: Right wheel position relative to CoM (3,)

        Returns:
            P matrix (8, 8) quadratic cost matrix
        """
        # Build effort minimization diagonal weights
        diagonal = jnp.array([
            self.w_force, self.w_force, self.w_force,  # f_left
            self.w_force, self.w_force, self.w_force,  # f_right
            self.w_torque, self.w_torque               # tau_hip_roll
        ])
        P_effort = jnp.diag(diagonal)

        # Build wrench matrix for soft constraint
        A_wrench = self.contact_jacobian.build_wrench_matrix(
            mj_data, wheel_pos_left, wheel_pos_right
        )

        # Add soft wrench tracking term: 2 * w_wrench * A^T * A
        # Factor of 2 accounts for QP form: minimize 0.5*x^T*P*x + q^T*x
        # To represent w*||Ax-b||^2, we need P = 2*w*A^T*A
        P_wrench = 2.0 * self.w_wrench * (A_wrench.T @ A_wrench)

        # Combine terms
        P = P_effort + P_wrench

        return P

    def _build_linear_cost_q(
        self,
        mj_data: mujoco.MjData,
        desired_wrench: Array,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
    ) -> Array:
        """Build linear cost vector q for QP.

        Combines smoothness penalty and soft wrench tracking:
        q = -2 * w_smoothness * P_effort @ x_prev - 2 * w_wrench * A_wrench^T @ b_wrench

        Args:
            mj_data: MuJoCo data with current robot state
            desired_wrench: Desired centroidal wrench (6,)
            wheel_pos_left: Left wheel position relative to CoM (3,)
            wheel_pos_right: Right wheel position relative to CoM (3,)

        Returns:
            q vector (8,) linear cost vector
        """
        # Smoothness term: penalize deviation from previous solution
        diagonal = jnp.array([
            self.w_force, self.w_force, self.w_force,
            self.w_force, self.w_force, self.w_force,
            self.w_torque, self.w_torque
        ])
        P_effort = jnp.diag(diagonal)
        q_smoothness = -2.0 * self.w_smoothness * (P_effort @ self.prev_solution)

        # Soft wrench tracking term: -2 * w_wrench * A^T @ b
        A_wrench = self.contact_jacobian.build_wrench_matrix(
            mj_data, wheel_pos_left, wheel_pos_right
        )
        q_wrench = -2.0 * self.w_wrench * (A_wrench.T @ desired_wrench)

        # Combine terms
        q = q_smoothness + q_wrench

        return q

    def _build_inequality_bounds(self) -> tuple[Array, Array]:
        """Build box constraint bounds for decision variables.

        Decision variables: [f_left(3), f_right(3), tau_hip_roll(2)]

        Constraints:
        - Contact forces: fz >= 0 (compressive), fx/fy unbounded
        - Hip roll torques: -tau_max <= tau <= tau_max

        Returns:
            Tuple of (lower, upper) where:
                - lower: Lower bounds (8,)
                - upper: Upper bounds (8,)
        """
        # Lower bounds
        lower = jnp.array([
            -jnp.inf, -jnp.inf, 0.0,  # f_left: fx/fy unbounded, fz >= 0
            -jnp.inf, -jnp.inf, 0.0,  # f_right: fx/fy unbounded, fz >= 0
            -self.tau_hip_roll_max,   # tau_hip_roll_L
            -self.tau_hip_roll_max,   # tau_hip_roll_R
        ])

        # Upper bounds
        upper = jnp.array([
            jnp.inf, jnp.inf, jnp.inf,  # f_left: no upper limit
            jnp.inf, jnp.inf, jnp.inf,  # f_right: no upper limit
            self.tau_hip_roll_max,      # tau_hip_roll_L
            self.tau_hip_roll_max,      # tau_hip_roll_R
        ])

        return lower, upper

    def distribute_wrench(
        self,
        mj_data: mujoco.MjData,
        desired_wrench: Array,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
    ) -> tuple[Array, Array, Array]:
        """Distribute desired wrench to wheel forces and hip roll torques.

        Uses soft constraint formulation for guaranteed feasibility:
        - Cost: effort minimization + soft wrench tracking + smoothness
        - Constraints: only box constraints (fz >= 0, |tau_hip_roll| <= tau_max)

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
        # Build QP cost matrices with soft wrench tracking
        P = self._build_cost_matrix_p(mj_data, wheel_pos_left, wheel_pos_right)
        q = self._build_linear_cost_q(mj_data, desired_wrench, wheel_pos_left, wheel_pos_right)

        # Build box constraints on decision variables (only constraints now)
        lower_box, upper_box = self._build_inequality_bounds()

        # BoxOSQP formulation with soft constraints:
        # min 0.5*x^T*P*x + q^T*x
        # s.t. l <= x <= u (box constraints only)

        # For BoxOSQP, we need A*x = z with l <= z <= u
        # With only box constraints, A = I, z = x, so l <= x <= u
        I = jnp.eye(8)
        A = I
        l = lower_box
        u = upper_box

        # Initialize BoxOSQP solver
        solver = BoxOSQP(
            maxiter=self.max_iter,
            tol=self.eps_abs,
        )

        # Solve QP
        result = solver.run(
            init_params=None,
            params_obj=(P, q),
            params_eq=A,
            params_ineq=(l, u),
        )

        # Extract solution from KKTSolution
        solution = result.params.primal[0]  # Extract x from (x, z) tuple

        # Update previous solution for next warm start
        self.prev_solution = solution

        # Extract decision variables
        f_left = solution[0:3]
        f_right = solution[3:6]
        tau_hip_roll = solution[6:8]

        return f_left, f_right, tau_hip_roll
