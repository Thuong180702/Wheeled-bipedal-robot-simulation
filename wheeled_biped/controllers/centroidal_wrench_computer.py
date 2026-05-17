"""Centroidal wrench computation from control objectives.

Converts high-level control objectives (roll stabilization, CoM regulation,
capture point tracking) into a desired 6D wrench (force + moment) on the CoM.
"""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


class CentroidalWrenchComputer:
    """Computes desired centroidal wrench from control objectives."""

    def __init__(
        self,
        k_roll: float = 20.0,
        k_roll_rate: float = 4.0,
        k_com_lateral: float = 15.0,
        k_com_lateral_damping: float = 3.0,
        k_com_sagittal: float = 10.0,
        k_com_sagittal_damping: float = 2.0,
        k_cp_lateral: float = 25.0,
        k_cp_sagittal: float = 20.0,
        k_height: float = 5.0,
        robot_mass: float = 15.0,
        gravity: float = 9.81,
    ):
        """Initialize centroidal wrench computer.

        Args:
            k_roll: Roll stabilization gain
            k_roll_rate: Roll rate damping gain
            k_com_lateral: CoM lateral position gain
            k_com_lateral_damping: CoM lateral velocity damping
            k_com_sagittal: CoM sagittal position gain
            k_com_sagittal_damping: CoM sagittal velocity damping
            k_cp_lateral: Capture point lateral gain
            k_cp_sagittal: Capture point sagittal gain
            k_height: Height tracking gain
            robot_mass: Robot mass in kg
            gravity: Gravity constant
        """
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.k_com_lateral = k_com_lateral
        self.k_com_lateral_damping = k_com_lateral_damping
        self.k_com_sagittal = k_com_sagittal
        self.k_com_sagittal_damping = k_com_sagittal_damping
        self.k_cp_lateral = k_cp_lateral
        self.k_cp_sagittal = k_cp_sagittal
        self.k_height = k_height
        self.robot_mass = robot_mass
        self.gravity = gravity

    def compute_desired_wrench(
        self,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
    ) -> tuple[Array, Array]:
        """Compute desired 6D wrench on CoM from control objectives.

        Args:
            obs: Observation array with gravity_body at [0:3], base_ang_vel at [6:9]
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command

        Returns:
            Tuple of (desired_force, desired_moment) where:
                - desired_force: (3,) [fx, fy, fz] in world frame
                - desired_moment: (3,) [mx, my, mz] about CoM in world frame
        """
        # Extract state
        roll = jnp.arctan2(obs[1], obs[2])
        roll_rate = obs[6]
        com_pos = state.com_pos
        com_vel = state.com_vel
        cp = state.capture_point

        # === Force objectives ===

        # Gravity compensation: WBC must command forces to counteract gravity
        # The simulation applies gravity automatically, but the controller needs
        # to command upward forces to maintain equilibrium
        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])

        # Height tracking: additional vertical force to correct height error
        height_error = height_cmd - com_pos[2]
        f_height = jnp.array([0.0, 0.0, self.k_height * height_error])

        # CoM lateral regulation: lateral force to center CoM
        f_com_lateral = jnp.array([
            0.0,
            -self.k_com_lateral * com_pos[1] - self.k_com_lateral_damping * com_vel[1],
            0.0
        ])

        # CoM sagittal regulation: forward force to center CoM
        f_com_sagittal = jnp.array([
            -self.k_com_sagittal * com_pos[0] - self.k_com_sagittal_damping * com_vel[0],
            0.0,
            0.0
        ])

        # Capture point tracking: force to prevent divergence
        f_cp = jnp.array([
            -self.k_cp_sagittal * cp[0],
            -self.k_cp_lateral * cp[1],
            0.0
        ])

        # Total desired force (gravity compensation + control corrections)
        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # === Moment objectives ===

        # Roll stabilization: moment about x-axis to correct roll
        m_roll = -self.k_roll * roll - self.k_roll_rate * roll_rate

        # No pitch/yaw moments for now (wheels handle pitch via sagittal forces)
        desired_moment = jnp.array([m_roll, 0.0, 0.0])

        return desired_force, desired_moment

    def compute_desired_wrench_vector(
        self,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
    ) -> Array:
        """Compute desired 6D wrench vector for force distribution.

        This is a convenience method that concatenates force and moment
        into a single 6D vector for use with UnifiedForceDistributor.

        Args:
            obs: Observation array with gravity_body at [0:3], base_ang_vel at [6:9]
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command

        Returns:
            Desired wrench (6,) [Fx, Fy, Fz, Mx, My, Mz] in world frame
        """
        desired_force, desired_moment = self.compute_desired_wrench(obs, state, height_cmd)
        return jnp.concatenate([desired_force, desired_moment])
