"""Centroidal state estimation for dynamic balance control."""

import chex
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion


@chex.dataclass
class CentroidalStateEstimatorConfig:
    """Configuration for centroidal state estimator."""
    robot_mass: float  # Total robot mass (kg)
    torso_inertia: Array  # Torso inertia [Ixx, Iyy, Izz] (kg⋅m²)


@chex.dataclass(frozen=True)
class CentroidalState:
    """Centroidal state for dynamic balance control.

    Attributes:
        com_pos: Center of mass position [x, y, z] in world frame (m)
        com_vel: Center of mass velocity [vx, vy, vz] in world frame (m/s)
        capture_point: Capture point [x_cp, y_cp] in world frame (m)
        divergence: Divergent component [div_x, div_y] (m)
        linear_momentum: Linear momentum [px, py, pz] (kg⋅m/s)
        angular_momentum: Angular momentum [Lx, Ly, Lz] about CoM (kg⋅m²/s)
        left_wheel_contact: Left wheel contact state
        right_wheel_contact: Right wheel contact state
        left_wheel_force: Left wheel normal force (N)
        right_wheel_force: Right wheel normal force (N)
    """
    com_pos: Array  # shape: (3,)
    com_vel: Array  # shape: (3,)
    capture_point: Array  # shape: (2,)
    divergence: Array  # shape: (2,)
    linear_momentum: Array  # shape: (3,)
    angular_momentum: Array  # shape: (3,)
    left_wheel_contact: bool
    right_wheel_contact: bool
    left_wheel_force: float
    right_wheel_force: float
    base_quat: Array = jnp.array([1.0, 0.0, 0.0, 0.0])
    base_ang_vel: Array = jnp.zeros(3)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    left_contact_force_world: Array = jnp.zeros(3)
    right_contact_force_world: Array = jnp.zeros(3)
    total_contact_force_z: float = 0.0


class CentroidalStateEstimator:
    """Extracts centroidal state from MJX simulation data."""

    def __init__(self, config: CentroidalStateEstimatorConfig, mj_model=None):
        self.config = config
        self.dt = 0.02  # 50Hz control rate
        self.mj_model = mj_model

        if mj_model is not None:
            self.left_wheel_geom_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision"
            )
            self.right_wheel_geom_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision"
            )
            if self.left_wheel_geom_id == -1 or self.right_wheel_geom_id == -1:
                raise ValueError("Wheel collision geoms not found")
        else:
            self.left_wheel_geom_id = 5
            self.right_wheel_geom_id = 6

    def estimate(self, obs: Array, data, prev_com_pos: Array | None = None) -> tuple[CentroidalState, Array]:
        """Extract centroidal state from observation and MJX data.

        Args:
            obs: Observation vector (not used for CoM extraction)
            data: MJX data structure with subtree_com, qvel, contact info
            prev_com_pos: Previous CoM position for velocity computation (None on first call)

        Returns:
            (state, current_com_pos) - Return current CoM for next call
        """
        # Extract CoM position from MJX data
        # subtree_com[1] is the torso subtree CoM in world frame
        # IMPORTANT: Copy the value to avoid reference issues with in-place updates
        com_pos = jnp.array(data.subtree_com[1])

        # Compute CoM velocity via finite difference
        if prev_com_pos is None:
            com_vel = jnp.zeros(3)
        else:
            com_vel = (com_pos - prev_com_pos) / self.dt

        # Extract contact forces from contact data
        left_wheel_contact = False
        right_wheel_contact = False
        left_wheel_force = 0.0
        right_wheel_force = 0.0
        left_contact_force_world = jnp.zeros(3)
        right_contact_force_world = jnp.zeros(3)

        if self.mj_model is not None and hasattr(data, "ncon"):
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                force_contact = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, data, i, force_contact)
                frame = np.array(contact.frame).reshape(3, 3)
                force_world = frame.T @ force_contact[:3]

                if geom1 == self.left_wheel_geom_id or geom2 == self.left_wheel_geom_id:
                    left_wheel_contact = True
                    left_contact_force_world = left_contact_force_world + jnp.array(force_world)
                    left_wheel_force = float(left_contact_force_world[2])

                if geom1 == self.right_wheel_geom_id or geom2 == self.right_wheel_geom_id:
                    right_wheel_contact = True
                    right_contact_force_world = right_contact_force_world + jnp.array(force_world)
                    right_wheel_force = float(right_contact_force_world[2])
        elif hasattr(data, 'contact') and hasattr(data.contact, 'force'):
            for i in range(len(data.contact.geom1)):
                geom1 = int(data.contact.geom1[i])
                geom2 = int(data.contact.geom2[i])

                if geom1 == self.left_wheel_geom_id or geom2 == self.left_wheel_geom_id:
                    left_wheel_contact = True
                    left_wheel_force = float(abs(data.contact.force[i][2]))
                    left_contact_force_world = left_contact_force_world + jnp.array([0.0, 0.0, left_wheel_force])

                if geom1 == self.right_wheel_geom_id or geom2 == self.right_wheel_geom_id:
                    right_wheel_contact = True
                    right_wheel_force = float(abs(data.contact.force[i][2]))
                    right_contact_force_world = right_contact_force_world + jnp.array([0.0, 0.0, right_wheel_force])

        total_contact_force_z = float(left_contact_force_world[2] + right_contact_force_world[2])

        if hasattr(data, "qpos"):
            base_quat = jnp.array(data.qpos[3:7])
            roll, pitch, yaw = compute_orientation_from_quaternion(np.array(data.qpos[3:7]))
        else:
            base_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
            roll, pitch, yaw = 0.0, 0.0, 0.0

        if hasattr(data, "qvel") and len(data.qvel) >= 6:
            base_ang_vel = jnp.array(data.qvel[3:6])
        else:
            base_ang_vel = jnp.zeros(3)
        roll_rate = float(base_ang_vel[0])
        pitch_rate = float(base_ang_vel[1])
        yaw_rate = float(base_ang_vel[2])

        # Placeholder values for capture point (will be implemented in Task 4-5)
        capture_point = jnp.zeros(2)
        divergence = jnp.zeros(2)

        # Compute linear momentum
        linear_momentum = self.config.robot_mass * com_vel

        # Placeholder for angular momentum (simplified)
        angular_momentum = jnp.zeros(3)

        state = CentroidalState(
            com_pos=com_pos,
            com_vel=com_vel,
            capture_point=capture_point,
            divergence=divergence,
            linear_momentum=linear_momentum,
            angular_momentum=angular_momentum,
            left_wheel_contact=left_wheel_contact,
            right_wheel_contact=right_wheel_contact,
            left_wheel_force=left_wheel_force,
            right_wheel_force=right_wheel_force,
            base_quat=base_quat,
            base_ang_vel=base_ang_vel,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            left_contact_force_world=left_contact_force_world,
            right_contact_force_world=right_contact_force_world,
            total_contact_force_z=total_contact_force_z,
        )

        return state, com_pos
