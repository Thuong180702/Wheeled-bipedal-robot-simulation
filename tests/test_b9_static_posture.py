import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.phase_b9_posture_symmetry_fix import (
    build_symmetric_b9_posture as generate_symmetric_b9_posture,
    contact_forces_by_wheel,
    set_symmetric_pose,
    wheel_bottom_points,
)
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateConfig
from wheeled_biped.utils.config import get_model_path


HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def b9_config():
    return DualRateConfig.from_yaml(Path("configs/controllers/dual_rate_balance_controller_b9.yaml"))


def _rpy_from_quat(quat: np.ndarray) -> tuple[float, float, float]:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def _body_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    weighted = np.zeros(3)
    for body_id in range(model.nbody):
        weighted += model.body_mass[body_id] * data.xipos[body_id]
    return weighted / float(np.sum(model.body_mass))


def _pose_data(model: mujoco.MjModel, config: DualRateConfig, height: float) -> tuple[mujoco.MjData, object]:
    pose = generate_symmetric_b9_posture(height, model, config)
    data = mujoco.MjData(model)
    set_symmetric_pose(model, data, pose)
    return data, pose


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_left_right_joint_symmetry(mj_model, b9_config, height):
    data, pose = _pose_data(mj_model, b9_config, height)
    left_wheel = data.xpos[mj_model.body("l_wheel_link").id]
    right_wheel = data.xpos[mj_model.body("r_wheel_link").id]

    assert pose.hip_yaw_l == pytest.approx(0.0, abs=1e-9)
    assert pose.hip_yaw_r == pytest.approx(0.0, abs=1e-9)
    assert pose.hip_roll_l == pytest.approx(0.0, abs=1e-9)
    assert pose.hip_roll_r == pytest.approx(0.0, abs=1e-9)
    assert left_wheel[0] == pytest.approx(-right_wheel[0], abs=1e-3)
    assert left_wheel[1] == pytest.approx(right_wheel[1], abs=1e-3)
    assert left_wheel[2] == pytest.approx(right_wheel[2], abs=1e-4)


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_both_wheels_grounded(mj_model, b9_config, height):
    data, _ = _pose_data(mj_model, b9_config, height)
    left_bottom, right_bottom = wheel_bottom_points(mj_model, data)

    assert float(left_bottom[2]) <= 1e-8
    assert float(right_bottom[2]) <= 1e-8
    assert abs(float(left_bottom[2] - right_bottom[2])) < 1e-4


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_both_wheels_have_contact_force(mj_model, b9_config, height):
    data, _ = _pose_data(mj_model, b9_config, height)
    left_force, right_force = contact_forces_by_wheel(mj_model, data)

    assert math.isfinite(left_force)
    assert math.isfinite(right_force)
    assert left_force > 0.0
    assert right_force > 0.0


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_root_roll_near_zero(mj_model, b9_config, height):
    data, _ = _pose_data(mj_model, b9_config, height)
    roll, _, _ = _rpy_from_quat(data.qpos[3:7].copy())

    assert abs(roll) < 1e-3


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_com_lateral_centered(mj_model, b9_config, height):
    data, _ = _pose_data(mj_model, b9_config, height)
    com = _body_com(mj_model, data)

    assert abs(float(com[0])) < 0.03


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_knees_forward(mj_model, b9_config, height):
    data, _ = _pose_data(mj_model, b9_config, height)
    l_hip = data.xpos[mj_model.body("l_thigh").id]
    r_hip = data.xpos[mj_model.body("r_thigh").id]
    l_knee = data.xpos[mj_model.body("l_knee_link").id]
    r_knee = data.xpos[mj_model.body("r_knee_link").id]

    assert float(l_hip[1] - l_knee[1]) > 0.0
    assert float(r_hip[1] - r_knee[1]) > 0.0


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_posture_wheels_fore_aft_symmetric(mj_model, b9_config, height):
    data, _ = _pose_data(mj_model, b9_config, height)
    left_wheel = data.xpos[mj_model.body("l_wheel_link").id]
    right_wheel = data.xpos[mj_model.body("r_wheel_link").id]

    assert abs(float(left_wheel[1] - right_wheel[1])) < 1e-3


def test_b9_posture_no_bad_height_command(mj_model, b9_config):
    assert 0.70 not in b9_config.height_grid
    with pytest.raises(ValueError):
        generate_symmetric_b9_posture(0.70, mj_model, b9_config)


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_balanced_initialization_contact_force_symmetry(mj_model, b9_config, height):
    """Test that initialize_balanced_b9_posture produces t=0 contact-force balance."""
    from scripts.phase_b9_posture_symmetry_fix import initialize_balanced_b9_posture

    data = mujoco.MjData(mj_model)
    initialize_balanced_b9_posture(height, mj_model, data, b9_config)

    left_force, right_force = contact_forces_by_wheel(mj_model, data)
    force_diff = abs(left_force - right_force)

    assert left_force > 0.0, f"Left wheel must have positive contact force, got {left_force}"
    assert right_force > 0.0, f"Right wheel must have positive contact force, got {right_force}"
    assert force_diff < 20.0, f"Contact force asymmetry must be < 20 N, got {force_diff:.3f} N"


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_balanced_initialization_com_centered(mj_model, b9_config, height):
    """Test that initialize_balanced_b9_posture centers CoM laterally."""
    from scripts.phase_b9_posture_symmetry_fix import initialize_balanced_b9_posture

    data = mujoco.MjData(mj_model)
    initialize_balanced_b9_posture(height, mj_model, data, b9_config)

    com = _body_com(mj_model, data)
    assert abs(float(com[0])) < 0.35, f"CoM lateral offset must be < 35 cm, got {abs(float(com[0]))*100:.2f} cm"


@pytest.mark.parametrize("height", HEIGHTS)
def test_b9_balanced_initialization_load_symmetry(mj_model, b9_config, height):
    """Test that initialize_balanced_b9_posture produces balanced wheel loads."""
    from scripts.phase_b9_posture_symmetry_fix import initialize_balanced_b9_posture

    data = mujoco.MjData(mj_model)
    initialize_balanced_b9_posture(height, mj_model, data, b9_config)

    left_force, right_force = contact_forces_by_wheel(mj_model, data)
    left_bottom, right_bottom = wheel_bottom_points(mj_model, data)
    com = _body_com(mj_model, data)
    roll, _, _ = _rpy_from_quat(data.qpos[3:7].copy())

    force_diff = abs(left_force - right_force)
    clearance_diff = abs(float(left_bottom[2] - right_bottom[2]))
    com_lateral_offset = abs(float(com[0]))

    assert left_force > 0.0, f"Left wheel must have positive contact force, got {left_force}"
    assert right_force > 0.0, f"Right wheel must have positive contact force, got {right_force}"
    assert force_diff < 20.0, f"Contact force asymmetry must be < 20 N, got {force_diff:.3f} N"
    assert clearance_diff < 1e-3, f"Clearance difference must be < 1 mm, got {clearance_diff*1000:.3f} mm"
    assert com_lateral_offset < 0.35, f"CoM lateral offset must be < 35 cm, got {com_lateral_offset*100:.2f} cm"
    assert abs(roll) < 0.03, f"Root roll must be within optimized bound < 0.03 rad, got {abs(roll):.6f} rad"
