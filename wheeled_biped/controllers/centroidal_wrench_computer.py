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
        robot_mass: float,
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
        k_height_damping: float = 0.0,
        gravity: float = 9.81,
        max_roll_moment: float | None = None,
    ):
        """Initialize centroidal wrench computer.

        Args:
            robot_mass: Robot mass in kg (REQUIRED - must be derived from mj_model)
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
        self.k_height_damping = k_height_damping
        self.robot_mass = robot_mass
        self.gravity = gravity
        self.max_roll_moment = max_roll_moment

        # Equilibrium reference (set via set_equilibrium_reference)
        self.equilibrium_com_pos = None
        self.equilibrium_com_z = None
        self.equilibrium_pitch_x = None
        self.equilibrium_roll_y = None
        self.equilibrium_capture_point = None
        self.equilibrium_joint_pos = None

    def set_equilibrium_reference(
        self,
        com_pos: Array,
        com_z: float,
        pitch_x: float,
        roll_y: float,
        capture_point: Array,
        joint_pos: Array,
    ):
        """Set equilibrium reference for computing relative corrections.

        Must be called after calibrated initialization before computing corrections.

        Args:
            com_pos: Equilibrium CoM position (3,) [x, y, z]
            com_z: Equilibrium CoM z-position
            pitch_x: Equilibrium pitch angle (rad)
            roll_y: Equilibrium roll angle (rad)
            capture_point: Equilibrium capture point (2,) [x, y]
            joint_pos: Equilibrium joint positions (10,)
        """
        self.equilibrium_com_pos = com_pos
        self.equilibrium_com_z = com_z
        self.equilibrium_pitch_x = pitch_x
        self.equilibrium_roll_y = roll_y
        self.equilibrium_capture_point = capture_point
        self.equilibrium_joint_pos = joint_pos

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

        All correction terms are computed relative to calibrated equilibrium.

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
        # Verify equilibrium reference is set
        if self.equilibrium_com_pos is None:
            raise RuntimeError(
                "Equilibrium reference not set. Call set_equilibrium_reference() "
                "after calibrated initialization."
            )

        # Extract state using unified orientation computation
        gravity_body = obs[0:3]
        pitch_x, roll_y = compute_robot_frame_orientation_from_gravity(gravity_body)
        pitch_rate_x = obs[6]
        roll_rate_y = obs[7]
        com_pos = state.com_pos
        com_vel = state.com_vel
        cp = state.capture_point

        # CRITICAL: Compute equilibrium-relative errors
        com_error = com_pos - self.equilibrium_com_pos
        cp_error = cp - self.equilibrium_capture_point
        pitch_error = pitch_x - self.equilibrium_pitch_x
        roll_error = roll_y - self.equilibrium_roll_y
        height_error = self.equilibrium_com_z - com_pos[2]

        # === Force objectives (equilibrium-relative) ===

        # Gravity compensation: baseline vertical force to counteract gravity
        # Use total robot mass (CoM + legs) for proper physics-based compensation
        # Total mass = 8.1kg (from MuJoCo model includes all segments)
        # In static equilibrium, vertical force must equal total weight
        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])

        # Height tracking: proportional + vertical damping (equilibrium-relative)
        f_height = jnp.array([
            0.0,
            0.0,
            self.k_height * height_error - self.k_height_damping * com_vel[2],
        ])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # CoM lateral regulation (equilibrium-relative)
        f_com_lateral = jnp.array([
            -self.k_com_lateral * com_error[0] - self.k_com_lateral_damping * com_vel[0],
            0.0,
            0.0
        ])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # CoM sagittal regulation (equilibrium-relative)
        f_com_sagittal = jnp.array([
            0.0,
            -self.k_com_sagittal * com_error[1] - self.k_com_sagittal_damping * com_vel[1],
            0.0
        ])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # Capture-point corrections (equilibrium-relative)
        f_cp = jnp.array([
            -self.k_cp_lateral * cp_error[0],
            -self.k_cp_sagittal * cp_error[1],
            0.0
        ])

        # Total desired force (gravity compensation + control corrections)
        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # === Moment objectives (equilibrium-relative) ===

        # Roll stabilization: PID control (equilibrium-relative)
        # P term: proportional to roll error from equilibrium
        # D term: damping based on roll rate
        # I term: accumulated error to eliminate bias/drift
        # roll_y is rotation about body/world Y; correction must be applied on My channel.
        m_roll_y = -self.k_roll * roll_error - self.k_roll_rate * roll_rate_y - self.k_roll_integral * roll_integral
        m_roll_y = self._limit_roll_moment(m_roll_y)

        # Pitch stabilization: inverted pendulum control (equilibrium-relative)
        # Sagittal force must OPPOSE pitch error to restore balance
        # Forward pitch requires backward force, backward pitch requires forward force
        # This follows inverted pendulum dynamics: F = -k*theta for restoring force
        pitch_correction_force = -self.k_pitch * pitch_error - self.k_pitch_rate * pitch_rate_x

        # Add pitch correction force to sagittal force component
        # This will drive wheel motion through the contact Jacobian
        desired_force = desired_force.at[1].add(pitch_correction_force)

        # Keep Mx at zero in this model; pitch_x is handled through Fy.
        desired_moment = jnp.array([0.0, m_roll_y, 0.0])

        return desired_force, desired_moment

    def compute_desired_wrench_from_state(
        self,
        state: CentroidalState,
        height_cmd: float,
        roll_integral: float = 0.0,
    ) -> tuple[Array, Array]:
        """Compute desired force and moment from explicit centroidal state.

        All correction terms are computed relative to calibrated equilibrium.

        Args:
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command
            roll_integral: Accumulated roll error for integral control

        Returns:
            Tuple of (desired_force, desired_moment)
        """
        # Verify equilibrium reference is set
        if self.equilibrium_com_pos is None:
            raise RuntimeError(
                "Equilibrium reference not set. Call set_equilibrium_reference() "
                "after calibrated initialization."
            )

        # CRITICAL: Compute equilibrium-relative errors
        com_error = state.com_pos - self.equilibrium_com_pos
        cp_error = state.capture_point - self.equilibrium_capture_point
        pitch_error = state.body_pitch_x - self.equilibrium_pitch_x
        roll_error = state.body_roll_y - self.equilibrium_roll_y
        height_error = self.equilibrium_com_z - state.com_pos[2]

        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])
        f_height = jnp.array([
            0.0,
            0.0,
            self.k_height * height_error - self.k_height_damping * state.com_vel[2],
        ])

        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # CoM lateral regulation (equilibrium-relative)
        f_com_lateral = jnp.array([
            -self.k_com_lateral * com_error[0]
            - self.k_com_lateral_damping * state.com_vel[0],
            0.0,
            0.0,
        ])
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # CoM sagittal regulation (equilibrium-relative)
        f_com_sagittal = jnp.array([
            0.0,
            -self.k_com_sagittal * com_error[1]
            - self.k_com_sagittal_damping * state.com_vel[1],
            0.0,
        ])
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        # Capture point corrections (equilibrium-relative)
        f_cp = jnp.array([
            -self.k_cp_lateral * cp_error[0],
            -self.k_cp_sagittal * cp_error[1],
            0.0,
        ])

        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # Roll stabilization: PID control (equilibrium-relative)
        m_roll_y = -self.k_roll * roll_error - self.k_roll_rate * state.body_roll_rate_y - self.k_roll_integral * roll_integral
        m_roll_y = self._limit_roll_moment(m_roll_y)

        # Pitch stabilization: inverted pendulum control (equilibrium-relative)
        pitch_correction_force = -self.k_pitch * pitch_error - self.k_pitch_rate * state.body_pitch_rate_x
        desired_force = desired_force.at[1].add(pitch_correction_force)

        desired_moment = jnp.array([0.0, m_roll_y, 0.0])

        return desired_force, desired_moment

    def compute_desired_wrench_from_state_with_breakdown(
        self,
        state: CentroidalState,
        height_cmd: float,
        roll_integral: float = 0.0,
    ) -> tuple[Array, Array, dict]:
        """Compute desired wrench from state with detailed correction breakdown.

        Returns the same wrench as compute_desired_wrench_from_state(), plus a dictionary
        with individual correction components for telemetry.

        Args:
            state: CentroidalState
            height_cmd: Desired height command
            roll_integral: Accumulated roll error

        Returns:
            Tuple of (desired_force, desired_moment, breakdown)
        """
        # Verify equilibrium reference is set
        if self.equilibrium_com_pos is None:
            raise RuntimeError(
                "Equilibrium reference not set. Call set_equilibrium_reference() "
                "after calibrated initialization."
            )

        # Compute equilibrium-relative errors
        com_error = state.com_pos - self.equilibrium_com_pos
        cp_error = state.capture_point - self.equilibrium_capture_point
        pitch_error = state.body_pitch_x - self.equilibrium_pitch_x
        roll_error = state.body_roll_y - self.equilibrium_roll_y
        height_error = self.equilibrium_com_z - state.com_pos[2]

        # Compute individual correction components
        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])

        correction_Fz_height = self.k_height * height_error - self.k_height_damping * state.com_vel[2]
        f_height = jnp.array([0.0, 0.0, correction_Fz_height])

        correction_Fx_com = -self.k_com_lateral * com_error[0] - self.k_com_lateral_damping * state.com_vel[0]
        f_com_lateral = jnp.array([correction_Fx_com, 0.0, 0.0])

        correction_Fy_com = -self.k_com_sagittal * com_error[1] - self.k_com_sagittal_damping * state.com_vel[1]
        f_com_sagittal = jnp.array([0.0, correction_Fy_com, 0.0])

        correction_Fx_cp = -self.k_cp_lateral * cp_error[0]
        correction_Fy_cp = -self.k_cp_sagittal * cp_error[1]
        f_cp = jnp.array([correction_Fx_cp, correction_Fy_cp, 0.0])

        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # Roll moment
        correction_My_roll = -self.k_roll * roll_error - self.k_roll_rate * state.body_roll_rate_y - self.k_roll_integral * roll_integral
        correction_My_roll = self._limit_roll_moment(correction_My_roll)

        # Pitch correction force
        correction_Fy_pitch = -self.k_pitch * pitch_error - self.k_pitch_rate * state.body_pitch_rate_x
        desired_force = desired_force.at[1].add(correction_Fy_pitch)

        desired_moment = jnp.array([0.0, correction_My_roll, 0.0])

        # Build breakdown dictionary
        breakdown = {
            # Equilibrium-relative errors
            "com_error_x": float(com_error[0]),
            "com_error_y": float(com_error[1]),
            "com_error_z": float(com_error[2]),
            "cp_error_x": float(cp_error[0]),
            "cp_error_y": float(cp_error[1]),
            "pitch_error": float(pitch_error),
            "roll_error": float(roll_error),
            "height_error": float(height_error),

            # Individual correction force components
            "correction_Fx_com": float(correction_Fx_com),
            "correction_Fx_cp": float(correction_Fx_cp),
            "correction_Fy_com": float(correction_Fy_com),
            "correction_Fy_cp": float(correction_Fy_cp),
            "correction_Fy_pitch": float(correction_Fy_pitch),
            "correction_Fz_height": float(correction_Fz_height),

            # Individual correction moment components
            "correction_My_roll": float(correction_My_roll),

            # Total correction wrench (excluding baseline mg)
            "correction_wrench_Fx": float(desired_force[0]),
            "correction_wrench_Fy": float(desired_force[1] - f_gravity[1]),
            "correction_wrench_Fz": float(desired_force[2] - f_gravity[2]),
            "correction_wrench_My": float(correction_My_roll),
            "correction_wrench_norm": float(jnp.linalg.norm(jnp.array([
                desired_force[0],
                desired_force[1] - f_gravity[1],
                desired_force[2] - f_gravity[2],
                0.0,
                correction_My_roll,
                0.0
            ]))),

            # Absolute state values (for debugging)
            "pitch_x": float(state.body_pitch_x),
            "roll_y": float(state.body_roll_y),
            "com_pos_x": float(state.com_pos[0]),
            "com_pos_y": float(state.com_pos[1]),
            "com_pos_z": float(state.com_pos[2]),
            "cp_x": float(state.capture_point[0]),
            "cp_y": float(state.capture_point[1]),
        }

        return desired_force, desired_moment, breakdown

    def compute_desired_wrench_with_breakdown(
        self,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
        roll_integral: float = 0.0,
    ) -> tuple[Array, Array, dict]:
        """Compute desired wrench with detailed correction breakdown.

        Returns the same wrench as compute_desired_wrench(), plus a dictionary
        with individual correction components for telemetry.

        Args:
            obs: Observation array
            state: CentroidalState
            height_cmd: Desired height command
            roll_integral: Accumulated roll error

        Returns:
            Tuple of (desired_force, desired_moment, breakdown) where breakdown contains:
                - Individual correction force components
                - Individual correction moment components
                - Equilibrium-relative errors
        """
        # Verify equilibrium reference is set
        if self.equilibrium_com_pos is None:
            raise RuntimeError(
                "Equilibrium reference not set. Call set_equilibrium_reference() "
                "after calibrated initialization."
            )

        # Extract state
        gravity_body = obs[0:3]
        pitch_x, roll_y = compute_robot_frame_orientation_from_gravity(gravity_body)
        pitch_rate_x = obs[6]
        roll_rate_y = obs[7]
        com_pos = state.com_pos
        com_vel = state.com_vel
        cp = state.capture_point

        # Compute equilibrium-relative errors
        com_error = com_pos - self.equilibrium_com_pos
        cp_error = cp - self.equilibrium_capture_point
        pitch_error = pitch_x - self.equilibrium_pitch_x
        roll_error = roll_y - self.equilibrium_roll_y
        height_error = self.equilibrium_com_z - com_pos[2]

        # Compute individual correction components
        f_gravity = jnp.array([0.0, 0.0, self.robot_mass * self.gravity])

        correction_Fz_height = self.k_height * height_error - self.k_height_damping * com_vel[2]
        f_height = jnp.array([0.0, 0.0, correction_Fz_height])

        correction_Fx_com = -self.k_com_lateral * com_error[0] - self.k_com_lateral_damping * com_vel[0]
        f_com_lateral = jnp.array([correction_Fx_com, 0.0, 0.0])

        correction_Fy_com = -self.k_com_sagittal * com_error[1] - self.k_com_sagittal_damping * com_vel[1]
        f_com_sagittal = jnp.array([0.0, correction_Fy_com, 0.0])

        correction_Fx_cp = -self.k_cp_lateral * cp_error[0]
        correction_Fy_cp = -self.k_cp_sagittal * cp_error[1]
        f_cp = jnp.array([correction_Fx_cp, correction_Fy_cp, 0.0])

        desired_force = f_gravity + f_height + f_com_lateral + f_com_sagittal + f_cp

        # Roll moment
        correction_My_roll = -self.k_roll * roll_error - self.k_roll_rate * roll_rate_y - self.k_roll_integral * roll_integral
        correction_My_roll = self._limit_roll_moment(correction_My_roll)

        # Pitch correction force
        correction_Fy_pitch = -self.k_pitch * pitch_error - self.k_pitch_rate * pitch_rate_x
        desired_force = desired_force.at[1].add(correction_Fy_pitch)

        desired_moment = jnp.array([0.0, correction_My_roll, 0.0])

        # Build breakdown dictionary
        breakdown = {
            # Equilibrium-relative errors
            "com_error_x": float(com_error[0]),
            "com_error_y": float(com_error[1]),
            "com_error_z": float(com_error[2]),
            "cp_error_x": float(cp_error[0]),
            "cp_error_y": float(cp_error[1]),
            "pitch_error": float(pitch_error),
            "roll_error": float(roll_error),
            "height_error": float(height_error),

            # Individual correction force components
            "correction_Fx_com": float(correction_Fx_com),
            "correction_Fx_cp": float(correction_Fx_cp),
            "correction_Fy_com": float(correction_Fy_com),
            "correction_Fy_cp": float(correction_Fy_cp),
            "correction_Fy_pitch": float(correction_Fy_pitch),
            "correction_Fz_height": float(correction_Fz_height),

            # Individual correction moment components
            "correction_My_roll": float(correction_My_roll),

            # Total correction wrench (excluding baseline mg)
            "correction_wrench_Fx": float(desired_force[0]),
            "correction_wrench_Fy": float(desired_force[1] - f_gravity[1]),
            "correction_wrench_Fz": float(desired_force[2] - f_gravity[2]),
            "correction_wrench_My": float(correction_My_roll),
            "correction_wrench_norm": float(jnp.linalg.norm(jnp.array([
                desired_force[0],
                desired_force[1] - f_gravity[1],
                desired_force[2] - f_gravity[2],
                0.0,
                correction_My_roll,
                0.0
            ]))),

            # Absolute state values (for debugging)
            "pitch_x": float(pitch_x),
            "roll_y": float(roll_y),
            "com_pos_x": float(com_pos[0]),
            "com_pos_y": float(com_pos[1]),
            "com_pos_z": float(com_pos[2]),
            "cp_x": float(cp[0]),
            "cp_y": float(cp[1]),
        }

        return desired_force, desired_moment, breakdown

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
