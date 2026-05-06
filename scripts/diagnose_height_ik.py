"""Diagnostic script to verify height IK mapping correctness.

Tests whether the FK scan and polynomial fit produce correct height-to-joint mappings.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior


def measure_actual_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hip_pitch: float,
    knee: float,
) -> float:
    """Measure actual torso height for given joint configuration.

    Uses the same FK approach as the controller's FK scan.
    """
    mujoco.mj_resetData(model, data)

    # qpos layout: [base_pos(3), base_quat(4), joints(10)]
    L_HIP_PITCH_QPOS = 7 + 2
    L_KNEE_QPOS = 7 + 3
    R_HIP_PITCH_QPOS = 7 + 7
    R_KNEE_QPOS = 7 + 8

    # Set joint positions (symmetric left/right)
    data.qpos[L_HIP_PITCH_QPOS] = hip_pitch
    data.qpos[L_KNEE_QPOS] = knee
    data.qpos[R_HIP_PITCH_QPOS] = hip_pitch
    data.qpos[R_KNEE_QPOS] = knee

    # Initial base height guess
    data.qpos[2] = 0.6

    # Run forward kinematics
    mujoco.mj_kinematics(model, data)

    # Get left wheel position in world frame
    l_wheel_body_id = model.body("l_wheel_link").id
    wheel_z = data.xpos[l_wheel_body_id, 2]

    # Wheel radius (should match config)
    wheel_radius = 0.06

    # Adjust base z so wheel touches ground
    base_z_adjustment = wheel_radius - wheel_z
    data.qpos[2] += base_z_adjustment

    # Recompute kinematics with adjusted base height
    mujoco.mj_kinematics(model, data)

    # Return base z position (torso height)
    return data.qpos[2]


def main():
    print("=" * 80)
    print("Height IK Mapping Diagnostic")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Create LQR/IK prior
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    print("\n1. Height IK mapping verification:")
    print("-" * 80)
    print(f"{'Height cmd (m)':<15} {'Hip pitch (rad)':<18} {'Knee (rad)':<15} {'Actual height (m)':<20} {'Error (m)'}")
    print("-" * 80)

    test_heights = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    for h_cmd in test_heights:
        # Get IK solution
        hip_pitch_des, knee_des = controller.height_ik(h_cmd)

        # Measure actual height
        h_actual = measure_actual_height(model, data, hip_pitch_des, knee_des)

        error = h_actual - h_cmd

        print(f"{h_cmd:<15.3f} {hip_pitch_des:<18.4f} {knee_des:<15.4f} {h_actual:<20.4f} {error:+.4f}")

    print("\n2. Joint limit check:")
    print("-" * 80)
    hip_pitch_limits = controller.config.joint_limits["hip_pitch"]
    knee_limits = controller.config.joint_limits["knee"]
    print(f"Hip pitch limits: [{hip_pitch_limits[0]:.3f}, {hip_pitch_limits[1]:.3f}] rad")
    print(f"Knee limits:      [{knee_limits[0]:.3f}, {knee_limits[1]:.3f}] rad")

    print("\n3. IK polynomial coefficients:")
    print("-" * 80)
    print(f"Hip pitch poly: {controller.height_ik.hip_pitch_poly}")
    print(f"Knee poly:      {controller.height_ik.knee_poly}")

    print("\n4. Height range from FK scan:")
    print("-" * 80)
    print(f"Height range: [{controller.height_ik.height_range[0]:.3f}, {controller.height_ik.height_range[1]:.3f}] m")

    print("\n5. Test standing pose at h=0.65m:")
    print("-" * 80)
    hip_pitch_des, knee_des = controller.height_ik(0.65)
    h_actual = measure_actual_height(model, data, hip_pitch_des, knee_des)

    print(f"Commanded height: 0.65 m")
    print(f"IK solution: hip_pitch={hip_pitch_des:.4f} rad, knee={knee_des:.4f} rad")
    print(f"Actual height: {h_actual:.4f} m")
    print(f"Error: {h_actual - 0.65:+.4f} m")

    # Now test with full reset and simulation
    print("\n6. Test with full simulation reset:")
    print("-" * 80)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"After mj_resetData + mj_forward:")
    print(f"  Torso z: {data.qpos[2]:.4f} m")
    print(f"  L hip_pitch: {data.qpos[model.jnt_qposadr[model.joint('l_hip_pitch').id]]:.4f} rad")
    print(f"  L knee: {data.qpos[model.jnt_qposadr[model.joint('l_knee').id]]:.4f} rad")
    print(f"  R hip_pitch: {data.qpos[model.jnt_qposadr[model.joint('r_hip_pitch').id]]:.4f} rad")
    print(f"  R knee: {data.qpos[model.jnt_qposadr[model.joint('r_knee').id]]:.4f} rad")

    print("\n" + "=" * 80)
    print("Diagnostic complete")
    print("=" * 80)

    print("\nInterpretation:")
    print("- If actual height matches commanded height (error < 0.01m), IK is correct")
    print("- If actual height is much higher, FK scan or polynomial fit is wrong")
    print("- If joints are outside limits, IK is producing invalid solutions")
    print("- If reset pose has zero joint angles, initial pose is not configured")


if __name__ == "__main__":
    main()
