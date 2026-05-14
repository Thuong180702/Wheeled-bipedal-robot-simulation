"""Centroidal state estimation for dynamic balance control."""

from dataclasses import dataclass
import numpy as np


@dataclass
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
    com_pos: np.ndarray
    com_vel: np.ndarray
    capture_point: np.ndarray
    divergence: np.ndarray
    linear_momentum: np.ndarray
    angular_momentum: np.ndarray
    left_wheel_contact: bool
    right_wheel_contact: bool
    left_wheel_force: float
    right_wheel_force: float
