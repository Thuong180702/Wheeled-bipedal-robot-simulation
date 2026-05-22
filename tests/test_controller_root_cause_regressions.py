import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer
from wheeled_biped.controllers.hierarchical_vmc_lqr import (
    HierarchicalVMCConfig,
    HierarchicalVMCController,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC


def _make_state(
    com_pos=(0.0, 0.0, 0.42),
    com_vel=(0.0, 0.0, 0.0),
    capture_point=(0.0, 0.0),
    pitch=0.0,
    pitch_rate=0.0,
):
    return CentroidalState(
        com_pos=jnp.array(com_pos),
        com_vel=jnp.array(com_vel),
        capture_point=jnp.array(capture_point),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,
        right_wheel_force=40.0,
        base_quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel=jnp.array([0.0, pitch_rate, 0.0]),
        roll=0.0,
        pitch=pitch,
        yaw=0.0,
        roll_rate=0.0,
        pitch_rate=pitch_rate,
        yaw_rate=0.0,
        left_contact_force_world=jnp.array([0.0, 0.0, 40.0]),
        right_contact_force_world=jnp.array([0.0, 0.0, 40.0]),
        total_contact_force_z=80.0,
    )


def _load_model_and_data_at_keyframe0():
    model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize(
    "com_pos, expected_force_idx, near_zero_idx",
    [
        ((0.1, 0.0, 0.42), 0, 1),
        ((0.0, 0.1, 0.42), 1, 0),
    ],
)
def test_com_axis_convention_for_desired_force(com_pos, expected_force_idx, near_zero_idx):
    computer = CentroidalWrenchComputer(
        k_com_lateral=10.0,
        k_com_lateral_damping=0.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=0.0,
        k_cp_lateral=0.0,
        k_cp_sagittal=0.0,
        k_pitch=0.0,
        k_pitch_rate=0.0,
    )
    state = _make_state(com_pos=com_pos)

    force, _ = computer.compute_desired_wrench_from_state(state, height_cmd=0.42)

    assert float(force[expected_force_idx]) < 0.0
    assert abs(float(force[near_zero_idx])) < 1e-8


@pytest.mark.parametrize(
    "capture_point, expected_force_idx, near_zero_idx",
    [
        ((0.1, 0.0), 0, 1),
        ((0.0, 0.1), 1, 0),
    ],
)
def test_capture_point_axis_convention_for_desired_force(capture_point, expected_force_idx, near_zero_idx):
    computer = CentroidalWrenchComputer(
        k_com_lateral=0.0,
        k_com_lateral_damping=0.0,
        k_com_sagittal=0.0,
        k_com_sagittal_damping=0.0,
        k_cp_lateral=10.0,
        k_cp_sagittal=10.0,
        k_pitch=0.0,
        k_pitch_rate=0.0,
    )
    state = _make_state(capture_point=capture_point)

    force, _ = computer.compute_desired_wrench_from_state(state, height_cmd=0.42)

    assert float(force[expected_force_idx]) < 0.0
    assert abs(float(force[near_zero_idx])) < 1e-8


def test_pitch_correction_enters_sagittal_force_component():
    computer = CentroidalWrenchComputer(
        k_com_lateral=0.0,
        k_com_lateral_damping=0.0,
        k_com_sagittal=0.0,
        k_com_sagittal_damping=0.0,
        k_cp_lateral=0.0,
        k_cp_sagittal=0.0,
        k_pitch=10.0,
        k_pitch_rate=0.0,
    )
    state = _make_state(pitch=0.1)

    force, _ = computer.compute_desired_wrench_from_state(state, height_cmd=0.42)

    assert abs(float(force[0])) < 1e-8
    assert float(force[1]) < 0.0


def test_wbc_contact_jacobian_sign_matches_direct_mapping():
    model, data = _load_model_and_data_at_keyframe0()
    wbc = IntegratedWBC(model)

    obs = jnp.zeros(42)
    state = _make_state()

    tau_wbc, diagnostics = wbc.compute_wbc_torque_with_diagnostics(
        data,
        obs,
        state,
        height_cmd=0.42,
    )

    tau_from_mapping = wbc.contact_jacobian.map_contact_forces_to_torques(
        data,
        diagnostics["f_left"],
        diagnostics["f_right"],
        tau_hip_roll=None,
    )

    tau_wbc_np = np.array(tau_wbc)
    tau_map_np = np.array(tau_from_mapping)

    nonzero_idx = np.where(np.abs(tau_map_np) > 1e-6)[0]
    assert nonzero_idx.size > 0

    same_sign = np.sign(tau_wbc_np[nonzero_idx]) == np.sign(tau_map_np[nonzero_idx])
    assert np.all(same_sign)


def test_hierarchical_vmc_requires_mjdata_when_sim_com_is_enabled():
    model, _ = _load_model_and_data_at_keyframe0()
    config = HierarchicalVMCConfig(com_use_sim=True, vmc_enabled=True)
    controller = HierarchicalVMCController(config, model)

    obs = np.zeros(42)
    obs[39] = 0.5

    with pytest.raises(ValueError, match="mj_data"):
        controller.compute_action(obs)


def test_hierarchical_vmc_default_ik_ranges_cover_real_standing_keyframe_family():
    config = HierarchicalVMCConfig()

    assert config.ik_hip_pitch_range[0] <= 0.926 <= config.ik_hip_pitch_range[1]
    assert config.ik_knee_range[0] <= 1.748 <= config.ik_knee_range[1]


def test_hierarchical_vmc_computation_uses_real_root_pose_from_mjdata():
    model, data = _load_model_and_data_at_keyframe0()
    controller = HierarchicalVMCController(HierarchicalVMCConfig(), model)

    com_y_1 = controller._compute_com_y(data)

    data.qpos[1] += 0.05
    mujoco.mj_forward(model, data)
    com_y_2 = controller._compute_com_y(data)

    assert abs(com_y_2 - com_y_1) > 1e-6
