"""Force gap diagnostic script.

Runs one control cycle and prints force audit trail showing where
the 15-20N force gap occurs between desired and actual contact forces.

Usage:
    python scripts/debug_force_gap.py
"""

import argparse
import numpy as np
import mujoco
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def measure_wheel_floor_contact(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id):
    """Measure wheel-floor contact distance and force."""
    min_dist = None
    total_fz = 0.0
    contact_count = 0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        if not (involves_floor and involves_wheel):
            continue

        contact_count += 1
        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return {
        "min_dist": min_dist,
        "total_fz": total_fz,
        "contact_count": contact_count,
    }


def calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5):
    """Calibrate root_z to achieve target wheel-floor contact distance.

    Iteratively adjusts root_z position to achieve the target contact distance
    between wheels and floor. Uses mj_forward in the loop to update contact state.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_dist: Target contact distance in meters (default: -0.5mm penetration)
        max_iters: Maximum calibration iterations (default: 5)

    Returns:
        Dictionary with geom IDs for floor and wheels
    """
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        stats = measure_wheel_floor_contact(
            model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
        )
        min_dist = stats["min_dist"]
        if min_dist is None:
            break

        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break

        data.qpos[2] += delta_z
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

    mujoco.mj_forward(model, data)
    return {
        "floor_geom_id": floor_geom_id,
        "l_wheel_geom_id": l_wheel_geom_id,
        "r_wheel_geom_id": r_wheel_geom_id,
    }


def load_robot_at_keyframe():
    """Load robot at calibrated standing keyframe with proper initialization.

    Matches simulate_hierarchical_controller.py initialization:
    1. Reset to keyframe
    2. mj_forward
    3. Calibrate root_z for -0.5mm contact distance
    4. Zero velocities and accelerations
    5. mj_forward

    Returns:
        Tuple of (mj_model, mj_data)
    """
    # Step 1: Reset to keyframe
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Step 2: mj_forward
    mujoco.mj_forward(model, data)

    # Step 3: Calibrate root_z for -0.5mm contact distance
    calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4)

    # Step 4: Zero velocities and accelerations
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # Step 5: mj_forward
    mujoco.mj_forward(model, data)

    return model, data


def main():
    """Run force gap diagnostic."""
    parser = argparse.ArgumentParser(description="Force gap diagnostic")
    args = parser.parse_args()

    print("=" * 80)
    print("FORCE GAP DIAGNOSTIC")
    print("=" * 80)

    mj_model, mj_data = load_robot_at_keyframe()
    print(f"[OK] Robot loaded at keyframe 0")
    print(f"     Root z: {float(mj_data.qpos[2]):.6f}")
    print(f"     CoM z: {float(mj_data.subtree_com[1][2]):.6f}")

    # TODO: Add force audit trail
    # 1. Initialize controllers (WBC, estimators)
    # 2. Compute one control cycle
    # 3. Print force audit trail:
    #    - Desired contact forces from WBC
    #    - Contact forces from ContactJacobian
    #    - Actual contact forces from mj_contactForce
    #    - Force gap at each stage


if __name__ == "__main__":
    main()
