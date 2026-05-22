"""Centroidal wrench computation from control objectives.

Converts high-level control objectives (roll stabilization, CoM regulation,
capture point tracking) into a desired 6D wrench (force + moment) on the CoM.
"""

import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
from wheeled_biped.controllers.orientation_utils import compute_robot_frame_orientation_from_gravity


class CentroidalWrenchComputer:
    """Computes desired centroidal wrench from control objectives."""

    def __init__(
        self,
        k_roll: float = 20.0,
        k_roll_rate: float = 4.0,
        k_roll_integral: float = 2.0,
        k_pitch: float = 5.0,
        k_pitch_rate: float = 1.0,
        k_com_lateral: float = 15.0,
        k_com_lateral_damping: float = 3.0,
        k_com_sagittal: float = 10.0,
        k_com_sagittal_damping: float = 2.0,
        k_cp_lateral: float = 25.0,
        k_cp_sagittal: float = 20.0,
        k_height: float = 5.0,
        robot_mass: float = 15.0,
        gravity: float = 9.81,
        max_roll_moment: float | None = None,
    ):
        """Initialize centroidal wrench computer.

        Args:
            k_roll: Roll stabilization gain (proportional)
            k_roll_rate: Roll rate damping gain (derivative)
            k_roll_integral: Roll integral gain (eliminates steady-state error)
            k_pitch: Pitch stabilization gain
            k_pitch_rate: Pitch rate damping gain
            k_com_lateral: CoM lateral position gain
            k_com_lateral_damping: CoM lateral velocity damping
            k_com_sagittal: CoM sagittal position gain
            k_com_sagittal_damping: CoM sagittal velocity damping
            k_cp_lateral: Capture point lateral gain
            k_cp_sagittal: Capture point sagittal gain
            k_height: Height tracking gain
            robot_mass: Robot mass in kg
            gravity: Gravity constant
            max_roll_moment: Optional roll moment clamp in Nm
        """
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.k_roll_integral = k_roll_integral
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.k_com_lateral = k_com_lateral
        self.k_com_lateral_damping = k_com_lateral_damping
        self.k_com_sagittal = k_com_sagittal
        self.k_com_sagittal_damping = k_com_sagittal_damping
        self.k_cp_lateral = k_cp_lateral
        self.k_cp_sagittal = k_cp_sagittal
        self.k_height = k_height
        self.robot_mass = robot_mass
        self.gravity = gravity
        self.max_roll_moment = max_roll_moment

    def _limit_roll_moment(self, m_roll: Array) -> Array:
        if self.max_roll_moment is None:
            return m_roll
        return jnp.clip(m_roll, -self.max_roll_moment, self.max_roll_moment)

    def compute_desired_wrench(
        self,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
        roll_integral: float = 0.0,
    ) -> tuple[Array, Array]:
        """Compute desired 6D wrench on CoM from control objectives.

        Args:
            obs: Observation array with gravity_body at [0:3], base_ang_vel at [6:9]
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command
            roll_integral: Accumulated roll error for integral control

        Returns:
            Tuple of (desired_force, desired_moment) where:
                - desired_force: (3,) [fx, fy, fz] in world frame
                - desired_moment: (3,) [mx, my, mz] about CoM in world frame
        """
        # Extract state using unified orientation computation
        gravity_body = obs[0:3]
        pitch_x, roll_y = compute_robot_frame_orientation_from_gravity(gravity_body)
        pitch_rate_x = obs[6]
        roll_rate_y = obs[7]
        com_pos = state.com_pos
        com_vel = state.com_vel
        cp = state.capture_point

        # === Force objectives ===

        # Gravity compensation: baseline vertical force to counteract gravity
        # Use total robot mass (CoM + legs) for proper physics-based compensation
        # Total mass = 8.1kg (from MuJoCo model includes all segments)
        # In static equilibrium, vertical force must equal total weight
        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])

        # Height tracking: additional vertical force to correct height error
        height_error = height_cmd - com_pos[2]
        f_height = jnp.array([0.0, 0.0, self.k_height * height_error])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # CoM lateral regulation: lateral force to center CoM on X-axis.
        f_com_lateral = jnp.array([
            -self.k_com_lateral * com_pos[0] - self.k_com_lateral_damping * com_vel[0],
            0.0,
            0.0
        ])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # CoM sagittal regulation: sagittal force to center CoM on Y-axis.
        f_com_sagittal = jnp.array([
            0.0,
            -self.k_com_sagittal * com_pos[1] - self.k_com_sagittal_damping * com_vel[1],
            0.0
        ])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # Capture-point corrections map lateral→Fx and sagittal→Fy.
        f_cp = jnp.array([
            -self.k_cp_lateral * cp[0],
            -self.k_cp_sagittal * cp[1],
            0.0
        ])

        # Total desired force (gravity compensation + control corrections)
        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # === Moment objectives ===

        # Roll stabilization: PID control to eliminate steady-state error
        # P term: proportional to current roll error
        # D term: damping based on roll rate
        # I term: accumulated error to eliminate bias/drift
        # CRITICAL FIX: Negative roll (left tilt) requires POSITIVE Mx to correct
        # The corrective moment must be opposite in sign to the roll error
        m_roll = -self.k_roll * roll_y - self.k_roll_rate * roll_rate_y - self.k_roll_integral * roll_integral
        m_roll = self._limit_roll_moment(m_roll)

        # Pitch stabilization: for wheeled biped, use inverted pendulum control
        # Sagittal force should be directly proportional to pitch angle (not scaled by height)
        # This follows inverted pendulum dynamics: F = m*g*theta for small angles
        # The k_pitch gain already encodes the appropriate force magnitude
        pitch_correction_force = -self.k_pitch * pitch_x - self.k_pitch_rate * pitch_rate_x

        # Add pitch correction force to sagittal force component
        # This will drive wheel motion through the contact Jacobian
        desired_force = desired_force.at[1].add(pitch_correction_force)

        # Keep pitch moment at zero since pitch control is handled by wheel motion
        m_pitch = 0.0

        desired_moment = jnp.array([m_roll, m_pitch, 0.0])

        return desired_force, desired_moment

    def compute_desired_wrench_from_state(
        self,
        state: CentroidalState,
        height_cmd: float,
        roll_integral: float = 0.0,
    ) -> tuple[Array, Array]:
        """Compute desired force and moment from explicit centroidal state.

        Args:
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command
            roll_integral: Accumulated roll error for integral control

        Returns:
            Tuple of (desired_force, desired_moment)
        """
        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])
        height_error = height_cmd - state.com_pos[2]
        f_height = jnp.array([0.0, 0.0, self.k_height * height_error])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        f_com_lateral = jnp.array([
            -self.k_com_lateral * state.com_pos[0]
            - self.k_com_lateral_damping * state.com_vel[0],
            0.0,
            0.0,
        ])
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        f_com_sagittal = jnp.array([
            0.0,
            -self.k_com_sagittal * state.com_pos[1]
            - self.k_com_sagittal_damping * state.com_vel[1],
            0.0,
        ])
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        f_cp = jnp.array([
            -self.k_cp_lateral * state.capture_point[0],
            -self.k_cp_sagittal * state.capture_point[1],
            0.0,
        ])

        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # Roll stabilization: PID control to eliminate steady-state error
        # CRITICAL FIX: Negative roll (left tilt) requires POSITIVE Mx to correct
        # The corrective moment must be opposite in sign to the roll error
        m_roll = -self.k_roll * state.body_roll_y - self.k_roll_rate * state.body_roll_rate_y - self.k_roll_integral * roll_integral
        m_roll = self._limit_roll_moment(m_roll)

        # Pitch stabilization: for wheeled biped, use inverted pendulum control
        # Sagittal force should be directly proportional to pitch angle (not scaled by height)
        # This follows inverted pendulum dynamics: F = m*g*theta for small angles
        # The k_pitch gain already encodes the appropriate force magnitude
        pitch_correction_force = -self.k_pitch * state.body_pitch_x - self.k_pitch_rate * state.body_pitch_rate_x
        desired_force = desired_force.at[1].add(pitch_correction_force)

        m_pitch = 0.0
        desired_moment = jnp.array([m_roll, m_pitch, 0.0])

        return desired_force, desired_moment

    def compute_desired_wrench_vector(
        self,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
        roll_integral: float = 0.0,
    ) -> Array:
        """Compute desired 6D wrench vector for force distribution.

        This is a convenience method that concatenates force and moment
        into a single 6D vector for use with UnifiedForceDistributor.

        Args:
            obs: Observation array with gravity_body at [0:3], base_ang_vel at [6:9]
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command
            roll_integral: Accumulated roll error for integral control

        Returns:
            Desired wrench (6,) [Fx, Fy, Fz, Mx, My, Mz] in world frame
        """
        desired_force, desired_moment = self.compute_desired_wrench(obs, state, height_cmd, roll_integral)
        return jnp.concatenate([desired_force, desired_moment])
