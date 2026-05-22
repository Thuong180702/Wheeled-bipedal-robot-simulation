import mujoco
import numpy as np

from scripts.simulate_hierarchical_controller import (
    calibrate_root_z_for_wheel_floor_contact,
    measure_wheel_floor_contact,
)

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def _wheel_floor_ids(model: mujoco.MjModel):
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    return floor_geom_id, l_wheel_geom_id, r_wheel_geom_id


def test_calibration_brings_wheel_floor_min_dist_near_target():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5)

    floor_geom_id, l_wheel_geom_id, r_wheel_geom_id = _wheel_floor_ids(model)
    stats = measure_wheel_floor_contact(
        model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
    )

    assert stats["min_dist"] is not None
    assert -0.0015 < float(stats["min_dist"]) < 0.0


def test_calibrated_keyframe_first_step_wheel_floor_fz_is_not_huge_impulse():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5)

    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

    floor_geom_id, l_wheel_geom_id, r_wheel_geom_id = _wheel_floor_ids(model)
    stats = measure_wheel_floor_contact(
        model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
    )

    weight = float(np.sum(model.body_mass) * abs(model.opt.gravity[2]))
    assert abs(float(stats["total_fz"])) < 3.0 * weight


def test_calibration_only_changes_root_z_not_joint_qpos():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_qpos_before = np.array(data.qpos[7:17], copy=True)

    calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5)

    joint_qpos_after = np.array(data.qpos[7:17], copy=True)
    assert np.allclose(joint_qpos_after, joint_qpos_before)
