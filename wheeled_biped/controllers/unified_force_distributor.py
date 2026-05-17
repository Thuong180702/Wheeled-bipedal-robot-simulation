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
        w_force: float = 0.01,
        w_torque: float = 0.1,
        w_smoothness: float = 0.5,
        tau_hip_roll_max: float = 10.0,
        max_iter: int = 50,
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

    def _build_equality_constraints(
        self,
        mj_data: mujoco.MjData,
        desired_wrench: Array,
        wheel_pos_left: Array,
        wheel_pos_right: Array,
    ) -> tuple[Array, Array]:
        """Build equality constraint matrices for wrench matching.

        Constraint: A_eq @ x = b_eq
        where x = [f_left(3), f_right(3), tau_hip_roll(2)]

        Args:
            mj_data: MuJoCo data with current robot state
            desired_wrench: Desired centroidal wrench (6,) [Fx, Fy, Fz, Mx, My, Mz]
            wheel_pos_left: Left wheel position relative to CoM (3,)
            wheel_pos_right: Right wheel position relative to CoM (3,)

        Returns:
            Tuple of (A_eq, b_eq) where:
                - A_eq: Wrench matrix (6, 8)
                - b_eq: Desired wrench (6,)
        """
        # Build wrench matrix using ContactJacobian
        A_eq = self.contact_jacobian.build_wrench_matrix(
            mj_data, wheel_pos_left, wheel_pos_right
        )

        # Desired wrench is the equality constraint target
        b_eq = jnp.asarray(desired_wrench)

        return A_eq, b_eq

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
        # Build QP cost matrices
        P = self._build_cost_matrix_p()
        q = self._build_linear_cost_q()

        # Build equality constraints (wrench matching)
        A_eq, b_eq = self._build_equality_constraints(
            mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
        )

        # Build box constraints on decision variables
        lower_box, upper_box = self._build_inequality_bounds()

        # BoxOSQP formulation: min 0.5*x^T*P*x + q^T*x
        #                      s.t. A*x = z, l <= z <= u
        # We need to stack equality and box constraints:
        # A = [A_eq; I], l = [b_eq; lower_box], u = [b_eq; upper_box]

        # Stack constraint matrix: [A_eq (6x8); I (8x8)]
        I = jnp.eye(8)
        A = jnp.vstack([A_eq, I])  # (14, 8)

        # Stack bounds: equality constraints have l=u=b_eq, box constraints have l=lower, u=upper
        l = jnp.concatenate([b_eq, lower_box])  # (14,)
        u = jnp.concatenate([b_eq, upper_box])  # (14,)

        # Initialize BoxOSQP solver
        solver = BoxOSQP(
            maxiter=self.max_iter,
            tol=self.eps_abs,
        )

        # Solve QP (warm-starting will be added in future optimization)
        result = solver.run(
            init_params=None,  # Let solver initialize
            params_obj=(P, q),
            params_eq=A,
            params_ineq=(l, u),
        )

        # Extract solution from KKTSolution
        # result.params is a KKTSolution with primal=(x, z), dual_eq, dual_ineq
        # We need x, which is the decision variable
        solution = result.params.primal[0]  # Extract x from (x, z) tuple

        # Update previous solution for next warm start
        self.prev_solution = solution

        # Extract decision variables
        f_left = solution[0:3]
        f_right = solution[3:6]
        tau_hip_roll = solution[6:8]

        return f_left, f_right, tau_hip_roll
