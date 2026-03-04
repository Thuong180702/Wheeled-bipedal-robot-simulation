"""
Tests cho MuJoCo MJCF model.

Kiểm tra:
  - Model load thành công
  - Đúng số lượng joints, actuators, sensors
  - Kinematics hợp lệ
  - Tư thế mặc định ổn định
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mj_model():
    """Tải MuJoCo model."""
    import mujoco
    from wheeled_biped.utils.config import get_model_path

    model_path = get_model_path()
    return mujoco.MjModel.from_xml_path(str(model_path))


@pytest.fixture
def mj_data(mj_model):
    """Tạo MuJoCo data và forward kinematics."""
    import mujoco

    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


class TestModelLoading:
    """Kiểm tra model load đúng."""

    def test_model_loads(self, mj_model):
        """Model phải load thành công."""
        assert mj_model is not None

    def test_correct_num_joints(self, mj_model):
        """Phải có 11 joints (1 freejoint + 10 hinge)."""
        # freejoint = 1, hinge joints = 10
        assert mj_model.njnt == 11

    def test_correct_num_actuators(self, mj_model):
        """Phải có 10 actuators."""
        assert mj_model.nu == 10

    def test_correct_qpos_size(self, mj_model):
        """qpos = 7 (freejoint) + 10 (hinges) = 17."""
        assert mj_model.nq == 17

    def test_correct_qvel_size(self, mj_model):
        """qvel = 6 (freejoint) + 10 (hinges) = 16."""
        assert mj_model.nv == 16


class TestModelStructure:
    """Kiểm tra cấu trúc model."""

    def test_has_torso_body(self, mj_model):
        """Phải có body 'torso'."""
        import mujoco

        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        assert body_id >= 0

    def test_has_all_joints(self, mj_model):
        """Phải có đủ 10 joint tên đúng."""
        import mujoco

        expected_joints = [
            "l_hip_roll",
            "l_hip_pitch",
            "l_knee",
            "l_ankle",
            "l_wheel",
            "r_hip_roll",
            "r_hip_pitch",
            "r_knee",
            "r_ankle",
            "r_wheel",
        ]
        for name in expected_joints:
            jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jnt_id >= 0, f"Thiếu joint: {name}"

    def test_has_all_actuators(self, mj_model):
        """Phải có đủ 10 actuator."""
        import mujoco

        expected_motors = [
            "l_hip_roll_motor",
            "l_hip_pitch_motor",
            "l_knee_motor",
            "l_ankle_motor",
            "l_wheel_motor",
            "r_hip_roll_motor",
            "r_hip_pitch_motor",
            "r_knee_motor",
            "r_ankle_motor",
            "r_wheel_motor",
        ]
        for name in expected_motors:
            act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert act_id >= 0, f"Thiếu actuator: {name}"

    def test_has_imu_sensors(self, mj_model):
        """Phải có cảm biến IMU."""
        import mujoco

        for name in ["imu_accel", "imu_gyro", "imu_quat"]:
            sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sensor_id >= 0, f"Thiếu sensor: {name}"

    def test_total_mass(self, mj_model, mj_data):
        """Tổng khối lượng ~7kg (±1kg tolerance)."""
        import mujoco

        total_mass = sum(mj_model.body_mass)
        assert 5.0 < total_mass < 9.0, f"Tổng khối lượng = {total_mass:.2f}kg"


class TestDefaultPose:
    """Kiểm tra tư thế mặc định."""

    def test_standing_height(self, mj_data):
        """Torso ở độ cao hợp lý (0.4 - 0.9m)."""
        torso_z = mj_data.qpos[2]
        assert 0.4 < torso_z < 0.9, f"Chiều cao torso = {torso_z:.3f}m"

    def test_upright_orientation(self, mj_data):
        """Quaternion gần [1,0,0,0] (đứng thẳng)."""
        quat = mj_data.qpos[3:7]
        # w nên gần 1, xyz nên gần 0
        assert abs(quat[0]) > 0.9, f"Quaternion w = {quat[0]:.3f}"
        assert np.linalg.norm(quat[1:]) < 0.3

    def test_keyframe_exists(self, mj_model):
        """Phải có ít nhất 1 keyframe."""
        assert mj_model.nkey >= 1

    def test_simulation_stable(self, mj_model, mj_data):
        """Mô phỏng 100 bước không crash."""
        import mujoco

        # Cho control = 0 và chạy 100 bước
        mj_data.ctrl[:] = 0
        for _ in range(100):
            mujoco.mj_step(mj_model, mj_data)

        # Kiểm tra không có NaN
        assert not np.any(np.isnan(mj_data.qpos)), "qpos chứa NaN!"
        assert not np.any(np.isnan(mj_data.qvel)), "qvel chứa NaN!"


class TestSymmetry:
    """Kiểm tra tính đối xứng của model."""

    def test_symmetric_masses(self, mj_model):
        """Chân trái và phải phải có khối lượng bằng nhau."""
        import mujoco

        left_bodies = [
            "l_hip_roll_link",
            "l_hip_pitch_link",
            "l_thigh",
            "l_knee_link",
            "l_ankle_link",
            "l_wheel_link",
        ]
        right_bodies = [
            "r_hip_roll_link",
            "r_hip_pitch_link",
            "r_thigh",
            "r_knee_link",
            "r_ankle_link",
            "r_wheel_link",
        ]

        for l_name, r_name in zip(left_bodies, right_bodies):
            l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, l_name)
            r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, r_name)
            if l_id >= 0 and r_id >= 0:
                l_mass = mj_model.body_mass[l_id]
                r_mass = mj_model.body_mass[r_id]
                assert (
                    abs(l_mass - r_mass) < 0.01
                ), f"{l_name}={l_mass:.3f}, {r_name}={r_mass:.3f}"
