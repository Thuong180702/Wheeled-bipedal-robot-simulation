"""Correct controller architecture for wheeled biped.

Architecture:
1. Wheels: Velocity PD based on pitch error (Segway balance)
2. Leg joints: Position PD for height/posture
3. Roll stabilization: Antisymmetric hip_roll torques

This is the CORRECT approach - not force-based WBC.
"""

import mujoco
import numpy as np


def get_pitch_roll_from_quat(quat):
    """Extract pitch and roll from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    pitch = float(np.arctan2(2 * (w * y + x * z), 1 - 2 * (y**2 + z**2)))
    roll = float(2 * np.arcsin(np.clip(2 * (w * x - y * z), -1, 1)))
    return pitch, roll


def main():
    print("=" * 80)
    print("Correct Wheeled Biped Controller")
    print("=" * 80)

    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize from keyframe
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized from keyframe 0")

    mujoco.mj_forward(mj_model, mj_data)

    # Verify wheels are on ground
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    l_wheel_z = mj_data.xpos[l_wheel_id][2]
    print(f"Left wheel z: {l_wheel_z:.3f}m (ground=0.060m)")

    if l_wheel_z > 0.1:
        print(f"ERROR: Wheels are {l_wheel_z - 0.06:.3f}m above ground!")
        return

    # Target leg configuration - UPDATED to match new keyframe with bent legs
    # Hip_roll will be actively controlled for roll stabilization
    target_leg_pos = np.array([
        0.0, 0.85, 1.9,  # left: hip_yaw, hip_pitch, knee (BENT LEGS)
        0.0, 0.85, 1.9,  # right: hip_yaw, hip_pitch, knee
    ])
    LEG_INDICES = [1, 2, 3, 6, 7, 8]  # exclude hip_roll (0, 5) and wheels (4, 9)

    # Controller gains
    # Leg position PD - for hip_yaw, hip_pitch, knee only
    kp_leg = np.array([100, 200, 200, 100, 200, 200])
    kd_leg = np.array([10, 20, 20, 10, 20, 20])

    # Active roll stabilization through hip_roll (like Segway for pitch)
    K_roll = 150.0       # Roll proportional gain
    K_roll_rate = 25.0   # Roll rate damping gain

    # Segway balance (from LQR baseline)
    K_pitch = 87.7
    K_pitch_rate = 20.6
    K_fwd_vel = 5.2
    K_fwd_pos = 0.6

    # Wheel velocity tracking
    kp_wheel_vel = 5.0  # Nm/(rad/s)

    print(f"\nController gains:")
    print(f"  Segway: K_pitch={K_pitch:.1f}, K_pitch_rate={K_pitch_rate:.1f}")
    print(f"  Leg position: kp_hip_pitch={kp_leg[2]:.1f}, kp_knee={kp_leg[3]:.1f}")
    print(f"  Wheel velocity: kp={kp_wheel_vel:.1f}")

    print(f"\nRunning simulation for 3000 steps (60 seconds)")
    print(f"{'Step':>6} {'h':>7} {'pitch':>8} {'roll':>8} {'wheel_vel':>10} {'max_leg':>9}")
    print("-" * 65)

    max_steps = 3000
    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    fwd_pos_drift = 0.0

    for step in range(max_steps):
        # Current state
        joint_pos = mj_data.qpos[7:17]
        joint_vel = mj_data.qvel[6:16]
        quat = mj_data.qpos[3:7]

        pitch, roll = get_pitch_roll_from_quat(quat)
        pitch_rate = mj_data.qvel[4]
        roll_rate = mj_data.qvel[3]
        fwd_vel = -mj_data.qvel[1]
        fwd_pos_drift += fwd_vel * control_dt

        # --- Leg position PD control (hip_yaw, hip_pitch, knee only) ---
        tau = np.zeros(10)
        for i, idx in enumerate(LEG_INDICES):
            pos_err = target_leg_pos[i] - joint_pos[idx]
            tau[idx] = kp_leg[i] * pos_err - kd_leg[i] * joint_vel[idx]

        # --- Active roll stabilization through hip_roll (antisymmetric) ---
        # Positive roll (leaning right) -> push left leg out, pull right leg in
        roll_torque = K_roll * roll + K_roll_rate * roll_rate
        tau[0] = -roll_torque  # l_hip_roll (negative = push left leg out when leaning right)
        tau[5] = roll_torque   # r_hip_roll (positive = pull right leg in when leaning right)

        # --- Segway wheel velocity control ---
        wheel_vel_cmd = -(K_pitch * pitch + K_pitch_rate * pitch_rate +
                         K_fwd_vel * fwd_vel + K_fwd_pos * fwd_pos_drift)

        l_wheel_vel = joint_vel[4]
        r_wheel_vel = joint_vel[9]

        tau[4] = kp_wheel_vel * (wheel_vel_cmd - l_wheel_vel)
        tau[9] = kp_wheel_vel * (wheel_vel_cmd - r_wheel_vel)

        mj_data.ctrl[:] = tau

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        com_z = mj_data.qpos[2]

        if step % 50 == 0:
            max_leg_tau = float(np.max(np.abs(tau[LEG_INDICES])))
            print(f"{step:6d} {com_z:7.3f} {np.rad2deg(pitch):8.2f} {np.rad2deg(roll):8.2f} {wheel_vel_cmd:10.2f} {max_leg_tau:9.2f}")

        if com_z < 0.35:
            print(f"\n[TERMINATED] step {step}: height_too_low (h={com_z:.3f}m)")
            break
        if abs(pitch) > 0.785 or abs(roll) > 0.785:
            print(f"\n[TERMINATED] step {step}: orientation_fail (pitch={np.rad2deg(pitch):.1f}deg, roll={np.rad2deg(roll):.1f}deg)")
            break
    else:
        print(f"\n[SUCCESS] Completed {max_steps} steps!")

    print("=" * 80)


if __name__ == "__main__":
    main()
