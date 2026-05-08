"""Quadratic Programming allocator for hierarchical control (Phase B.7 Task 4).

Resolves conflicts between hierarchical control layers by formulating action
allocation as a constrained optimization problem:

    minimize: ||a - a_desired||^2 + regularization
    subject to: joint limits, velocity limits, priority constraints

This is useful when multiple control layers produce conflicting commands
(e.g., VMC wants to extend legs for CoM correction while height IK wants
to compress legs for height tracking).

References:
    - Hierarchical QP for humanoid control: Sentis & Khatib, "Synthesis of
      Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives",
      IJRR 2005
    - Lexicographic optimization: Kanoun et al., "Kinematic Control of Redundant
      Manipulators: Generalizing the Task-Priority Framework to Inequality Task",
      IEEE TRO 2011
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@dataclass
class QPAllocatorConfig:
    """Configuration for QP allocator."""

    # Joint limits (normalized [-1, 1])
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None

    # Priority weights for different control objectives
    # Higher weight = higher priority
    weight_height_ik: float = 1.0
    weight_com_vmc: float = 0.8
    weight_wheel_lqr: float = 1.2
    weight_roll_yaw: float = 0.6

    # Regularization
    regularization_weight: float = 0.01

    # Solver settings
    solver: str = "OSQP"  # OSQP, ECOS, SCS
    verbose: bool = False


class QPAllocator:
    """QP-based action allocator for hierarchical control.

    Resolves conflicts between control layers by solving:
        min ||a - a_desired||_W^2 + lambda * ||a||^2
        s.t. a_min <= a <= a_max

    where W is a diagonal weight matrix encoding layer priorities.
    """

    def __init__(self, config: QPAllocatorConfig):
        if not HAS_CVXPY:
            raise ImportError(
                "QP allocator requires cvxpy. Install with: pip install cvxpy"
            )

        self.config = config

        # Default joint limits if not provided
        if config.joint_limits_lower is None:
            self.config.joint_limits_lower = -np.ones(10)
        if config.joint_limits_upper is None:
            self.config.joint_limits_upper = np.ones(10)

    def allocate(
        self,
        desired_action: np.ndarray,
        layer_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Allocate action using QP optimization.

        Args:
            desired_action: Desired action from hierarchical controller, shape (10,).
            layer_weights: Optional per-joint priority weights, shape (10,).
                          If None, uses default weights from config.

        Returns:
            Allocated action satisfying constraints, shape (10,).
        """
        # Default weights: assign based on joint type
        if layer_weights is None:
            layer_weights = self._get_default_weights()

        # Decision variable
        a = cp.Variable(10)

        # Objective: weighted tracking + regularization
        W = np.diag(layer_weights)
        tracking_cost = cp.quad_form(a - desired_action, W)
        regularization_cost = self.config.regularization_weight * cp.sum_squares(a)
        objective = cp.Minimize(tracking_cost + regularization_cost)

        # Constraints: joint limits
        constraints = [
            a >= self.config.joint_limits_lower,
            a <= self.config.joint_limits_upper,
        ]

        # Solve QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.config.solver, verbose=self.config.verbose)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback: clip desired action
            return np.clip(
                desired_action,
                self.config.joint_limits_lower,
                self.config.joint_limits_upper,
            )

        return np.array(a.value)

    def _get_default_weights(self) -> np.ndarray:
        """Get default per-joint priority weights.

        Priority assignment:
            - Wheels (LQR): highest priority (balance-critical)
            - Hip pitch/knee (height IK + VMC): high priority
            - Hip roll (roll stabilization): medium priority
            - Hip yaw: lowest priority (least critical)
        """
        weights = np.array([
            self.config.weight_roll_yaw,      # l_hip_roll
            0.5,                               # l_hip_yaw (lowest)
            self.config.weight_height_ik,     # l_hip_pitch
            self.config.weight_height_ik,     # l_knee
            self.config.weight_wheel_lqr,     # l_wheel (highest)
            self.config.weight_roll_yaw,      # r_hip_roll
            0.5,                               # r_hip_yaw (lowest)
            self.config.weight_height_ik,     # r_hip_pitch
            self.config.weight_height_ik,     # r_knee
            self.config.weight_wheel_lqr,     # r_wheel (highest)
        ])
        return weights


def create_qp_allocator(
    weight_height_ik: float = 1.0,
    weight_com_vmc: float = 0.8,
    weight_wheel_lqr: float = 1.2,
    weight_roll_yaw: float = 0.6,
    regularization_weight: float = 0.01,
) -> QPAllocator:
    """Factory function to create QP allocator with custom weights."""
    config = QPAllocatorConfig(
        weight_height_ik=weight_height_ik,
        weight_com_vmc=weight_com_vmc,
        weight_wheel_lqr=weight_wheel_lqr,
        weight_roll_yaw=weight_roll_yaw,
        regularization_weight=regularization_weight,
    )
    return QPAllocator(config)
