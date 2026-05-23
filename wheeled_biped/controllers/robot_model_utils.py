"""Robot model utilities for mass and weight computation.

Single source of truth for robot mass derived from MuJoCo model.
"""

import mujoco
import numpy as np


def get_total_robot_mass(mj_model: mujoco.MjModel) -> float:
    """Get total robot mass from MuJoCo model.

    Args:
        mj_model: MuJoCo model with body mass data

    Returns:
        Total robot mass in kg (sum of all body masses)
    """
    return float(np.sum(mj_model.body_mass))


def get_robot_weight(mj_model: mujoco.MjModel) -> float:
    """Get robot weight (mass * gravity) from MuJoCo model.

    Args:
        mj_model: MuJoCo model with body mass and gravity data

    Returns:
        Robot weight in N (total_mass * |gravity_z|)
    """
    total_mass = get_total_robot_mass(mj_model)
    gravity_z = float(mj_model.opt.gravity[2])
    return total_mass * abs(gravity_z)
