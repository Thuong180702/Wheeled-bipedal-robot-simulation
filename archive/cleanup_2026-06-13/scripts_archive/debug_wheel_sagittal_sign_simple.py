"""Simplified wheel sagittal sign diagnostic.

Applies isolated wheel torques without other controllers to determine sign convention.
"""

import mujoco
import numpy as np
from pathlib import Path


def test_wheel_torque_sign():
    """Test wheel torque sign convention with isolated torques."""
    print("=" * 80)
    print("SIMPLIFIED WHEEL SAGITTAL SIGN TEST")
    print("=" * 80)

    # Load model
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Initialize
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"\n[Model Info]")
    print(f"  nq (position DOF): {model.nq}")
    print(f"  nv (velocity DOF): {model.nv}")
    print(f"  nu (actuators): {model.nu}")

    # Print joint names and indices
    print(f"\n[Joint Configuration]")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        qpos_addr = model.jnt_qposadr[i]
        dof_addr = model.jnt_dofadr[i]
        print(f"  Joint {i}: {joint_name}, type={joint_type}, qpos_addr={qpos_addr}, dof_addr={dof_addr}")

    # Print actuator names and indices
    print(f"\n[Actuator Configuration]")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Actuator {i}: {act_name}")

    # Find wheel actuator indices
    l_wheel_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "l_wheel_motor")
    r_wheel_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_wheel_motor")

    print(f"\n[Wheel Actuators]")
    print(f"  l_wheel actuator index: {l_wheel_act_id}")
    print(f"  r_wheel actuator index: {r_wheel_act_id}")

    # Get initial state
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    print(f"\n[Initial State]")
    print(f"  Root position: [{data.qpos[0]:.6f}, {data.qpos[1]:.6f}, {data.qpos[2]:.6f}]")
    print(f"  Root velocity: [{data.qvel[0]:.6f}, {data.qvel[1]:.6f}, {data.qvel[2]:.6f}]")

    # Test cases
    test_cases = [
        (10.0, "positive_10Nm"),
        (-10.0, "negative_10Nm"),
    ]

    control_dt = 0.01
    n_substeps = 10

    for tau_wheel, test_name in test_cases:
        print(f"\n{'=' * 80}")
        print(f"[Test: {test_name}] tau_wheel = {tau_wheel:.1f} Nm")
        print(f"{'=' * 80}")

        # Reset to initial state
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)

        # Apply only wheel torques (all other actuators zero)
        data.ctrl[:] = 0.0
        data.ctrl[l_wheel_act_id] = tau_wheel
        data.ctrl[r_wheel_act_id] = tau_wheel

        print(f"\n  Applied ctrl: {data.ctrl}")

        # Run simulation
        for step in range(50):
            mujoco.mj_step(model, data)

            if step in [0, 4, 9, 19, 49]:
                # Get wheel joint velocities
                l_wheel_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_wheel")
                r_wheel_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_wheel")

                l_wheel_dof = model.jnt_dofadr[l_wheel_jnt_id]
                r_wheel_dof = model.jnt_dofadr[r_wheel_jnt_id]

                l_wheel_vel = data.qvel[l_wheel_dof]
                r_wheel_vel = data.qvel[r_wheel_dof]

                # Get root velocity
                root_vel_y = data.qvel[1]  # Y velocity

                print(f"  Step {step:2d}: wheel_vel=[{l_wheel_vel:+.3f}, {r_wheel_vel:+.3f}] rad/s, root_vy={root_vel_y:+.6f} m/s")

        # Final state
        final_root_y = data.qpos[1]
        final_root_vy = data.qvel[1]
        delta_y = final_root_y - initial_qpos[1]

        print(f"\n  Final state:")
        print(f"    Root Y: {final_root_y:+.6f} m (delta={delta_y:+.6f} m)")
        print(f"    Root Vy: {final_root_vy:+.6f} m/s")

        if delta_y < -0.001:
            direction = "FORWARD (-Y)"
        elif delta_y > 0.001:
            direction = "BACKWARD (+Y)"
        else:
            direction = "STATIONARY"

        print(f"    -> Robot moved: {direction}")

    print(f"\n{'=' * 80}")
    print(f"[Conclusion]")
    print(f"{'=' * 80}")
    print(f"Determine from the results:")
    print(f"  1. Positive wheel torque moves robot in which direction?")
    print(f"  2. Negative wheel torque moves robot in which direction?")
    print(f"  3. To move forward (-Y), use which sign?")
    print(f"  4. To move backward (+Y), use which sign?")


if __name__ == "__main__":
    test_wheel_torque_sign()
