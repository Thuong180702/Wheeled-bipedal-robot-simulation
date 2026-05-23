"""Actuator sign and authority validation tests.

Verifies that each actuator produces force in the expected direction
and that support joints have sufficient authority.
"""

import numpy as np
import mujoco
import pytest


@pytest.fixture
def robot_at_keyframe():
    """Load robot at calibrated standing keyframe.

    Returns:
        Tuple of (mj_model, mj_data)
    """
    model_path = "assets/robot/wheeled_biped_real.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Reset to keyframe 0 if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    return model, data


def measure_contact_fz(mj_model, mj_data):
    """Measure total vertical contact force from MuJoCo contact solver.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        float: Total vertical contact force (Fz) in Newtons
    """
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    total_fz = 0.0

    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}

        if not (involves_floor and involves_wheel):
            continue

        # Use mj_contactForce to get the contact force in the contact frame
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)

        # Transform to world frame using contact frame
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return total_fz
