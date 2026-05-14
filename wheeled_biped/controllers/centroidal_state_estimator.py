"""Centroidal state estimation for dynamic balance control."""

import chex
import jax.numpy as jnp
from jax import Array


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


class CentroidalStateEstimator:
    """Extracts centroidal state from MJX simulation data."""

    def __init__(self, config: CentroidalStateEstimatorConfig):
        self.config = config
        self.com_pos_prev = None
        self.dt = 0.02  # 50Hz control rate

    def estimate(self, obs: Array, data) -> CentroidalState:
        """Extract centroidal state from observation and MJX data.

        Args:
            obs: Observation vector (not used for CoM extraction)
            data: MJX data structure with subtree_com, qvel, contact info

        Returns:
            CentroidalState with all fields populated
        """
        # Extract CoM position from MJX data
        # subtree_com[1] is the torso subtree CoM in world frame
        com_pos = data.subtree_com[1]

        # Compute CoM velocity via finite difference
        if self.com_pos_prev is None:
            com_vel = jnp.zeros(3)
        else:
            com_vel = (com_pos - self.com_pos_prev) / self.dt

        self.com_pos_prev = com_pos

        # Placeholder values for other fields (will be implemented in later tasks)
        capture_point = jnp.zeros(2)
        divergence = jnp.zeros(2)
        linear_momentum = self.config.robot_mass * com_vel
        angular_momentum = jnp.zeros(3)
        left_wheel_contact = True
        right_wheel_contact = True
        left_wheel_force = 0.0
        right_wheel_force = 0.0

        return CentroidalState(
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
        )
