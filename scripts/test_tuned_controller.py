"""Properly tuned controller based on actual torque requirements.

Based on torque calculations:
- Hip pitch needs: 6.9 Nm (not 245 Nm!)
- Knee needs: 2.1 Nm
- Hip roll needs: 6.1 Nm
- Wheel needs: 6.7 Nm

Current 60 Nm motors are 8-10x stronger than needed.

Solution: Reduce PD gains by 15x to prevent oscillations.
"""

import mujoco
import numpy as np


def get_pitch_roll_from_quat(quat):
    """Extract pitch and roll from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    pitch = float(np.arctan2(2 * (w * y + x * z), 1 - 2 * (y**2 + z**2)))
    roll = float(2 * np.arcsin(np.clip(2 * (w * x - y * z), -1, 1)))
    return pitch, roll


def check_ground_contact(mj_model, mj_data):
    """Check if wheels are in contact with ground."""
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    l_contacts = 0
    r_contacts = 0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        if (geom1 == l_wheel_geom_id and geom2 == floor_geom_id) or \
           (geom2 == l_wheel_geom_id and geom1 == floor_geom_id):
            l_contacts += 1

        if (geom1 == r_wheel_geom_id and geom2 == floor_geom_id) or \
           (geom2 == r_wheel_geom_id and geom1 == floor_geom_id):
            r_contacts += 1

    return l_contacts > 0, r_contacts > 0, l_contacts, r_contacts


def main():
    print("=" * 80)
    print("Properly Tuned Controller (100Hz)")
    print("=" * 80)

    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize from keyframe
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized from keyframe 0")

    mujoco.mj_forward(mj_model, mj_data)

    # Get sensor IDs
    imu_accel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
    imu_gyro_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    imu_quat_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")

    # Target leg configuration - match keyframe
    target_leg_pos = np.array([
        0.0, 0.4, 0.4,  # left: hip_yaw, hip_pitch, knee
        0.0, 0.4, 0.4,  # right: hip_yaw, hip_pitch, knee
    ])
    LEG_INDICES = [1, 2, 3, 6, 7, 8]  # exclude hip_roll (0, 5) and wheels (4, 9)

    # PROPERLY TUNED GAINS (reduced by 15x from previous)
    # Previous: kp=[100, 200, 200], kd=[10, 20, 20] -> caused 700°/s oscillations
    # New: kp=[10, 15, 15], kd=[1, 2, 2] -> should be stable
    kp_leg = np.array([10, 15, 15, 10, 15, 15])  # hip_yaw, hip_pitch, knee (L/R)
    kd_leg = np.array([1, 2, 2, 1, 2, 2])

    # Active roll stabilization (also reduced)
    K_roll = 20.0       # Reduced from 150
    K_roll_rate = 5.0   # Reduced from 25

    # Segway balance (from LQR baseline - keep these)
    K_pitch = 87.7
    K_pitch_rate = 20.6
    K_fwd_vel = 5.2
    K_fwd_pos = 0.6

    # Wheel velocity tracking
    kp_wheel_vel = 5.0

    print(f"\nController gains (TUNED based on torque requirements):")
    print(f"  Segway: K_pitch={K_pitch:.1f}, K_pitch_rate={K_pitch_rate:.1f}")
    print(f"  Roll: K_roll={K_roll:.1f}, K_roll_rate={K_roll_rate:.1f}")
    print(f"  Leg position: kp_hip_pitch={kp_leg[2]:.1f}, kp_knee={kp_leg[3]:.1f}")
    print(f"  Leg damping: kd_hip_pitch={kd_leg[2]:.1f}, kd_knee={kd_leg[3]:.1f}")
    print(f"  Wheel velocity: kp={kp_wheel_vel:.1f}")

    print(f"\nRunning simulation at 100Hz for 6000 steps (60 seconds)")
    print(f"{'Step':>6} {'h':>7} {'pitch':>8} {'roll':>8} {'contact':>8} {'max_tau':>9} {'max_vel':>9}")
    print("-" * 75)

    max_steps = 6000
    control_dt = 0.01  # 100Hz
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    fwd_pos_drift = 0.0

    for step in range(max_steps):
        # Read IMU sensors
        imu_accel_adr = mj_model.sensor_adr[imu_accel_id]
        imu_gyro_adr = mj_model.sensor_adr[imu_gyro_id]
        imu_quat_adr = mj_model.sensor_adr[imu_quat_id]

        imu_quat = mj_data.sensordata[imu_quat_adr:imu_quat_adr+4]
        imu_gyro = mj_data.sensordata[imu_gyro_adr:imu_gyro_adr+3]

        # Extract orientation from IMU
        pitch, roll = get_pitch_roll_from_quat(imu_quat)
        pitch_rate = imu_gyro[1]
        roll_rate = imu_gyro[0]

        # Read joint encoders
        joint_pos = mj_data.qpos[7:17]
        joint_vel = mj_data.qvel[6:16]

        # Forward velocity
        fwd_vel = -mj_data.qvel[1]
        fwd_pos_drift += fwd_vel * control_dt

        # Check ground contact
        l_contact, r_contact, l_contacts, r_contacts = check_ground_contact(mj_model, mj_data)
        contact_status = f"{int(l_contact)}{int(r_contact)}"

        # --- Leg position PD control ---
        tau = np.zeros(10)
        for i, idx in enumerate(LEG_INDICES):
            pos_err = target_leg_pos[i] - joint_pos[idx]
            tau[idx] = kp_leg[i] * pos_err - kd_leg[i] * joint_vel[idx]

        # --- Active roll stabilization ---
        roll_torque = K_roll * roll + K_roll_rate * roll_rate
        tau[0] = -roll_torque  # l_hip_roll
        tau[5] = roll_torque   # r_hip_roll

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

        if step % 1 == 0 and step < 25:  # Detailed output for first 25 steps
            max_tau = float(np.max(np.abs(tau)))
            max_vel = float(np.max(np.abs(np.rad2deg(joint_vel))))
            actual_tau = mj_data.actuator_force.copy()

            print(f"\n[STEP {step}] pitch={np.rad2deg(pitch):6.2f}°, roll={np.rad2deg(roll):6.2f}°, contact={contact_status}")
            print(f"  Commanded torques: l_hip_pitch={tau[2]:6.2f} Nm, l_knee={tau[3]:6.2f} Nm, l_hip_roll={tau[0]:6.2f} Nm")
            print(f"  Actual torques:    l_hip_pitch={actual_tau[2]:6.2f} Nm, l_knee={actual_tau[3]:6.2f} Nm, l_hip_roll={actual_tau[0]:6.2f} Nm")
            print(f"  Joint velocities:  l_hip_pitch={np.rad2deg(joint_vel[2]):6.1f}°/s, l_knee={np.rad2deg(joint_vel[3]):6.1f}°/s")
            print(f"  Max commanded tau: {max_tau:.2f} Nm, Max joint vel: {max_vel:.1f}°/s")

        # Termination conditions
        if com_z < 0.35:
            print(f"\n[TERMINATED] step {step}: height_too_low (h={com_z:.3f}m)")
            break
        if abs(pitch) > 0.785 or abs(roll) > 0.785:
            print(f"\n[TERMINATED] step {step}: orientation_fail (pitch={np.rad2deg(pitch):.1f}deg, roll={np.rad2deg(roll):.1f}deg)")
            break
    else:
        print(f"\n[SUCCESS] Completed {max_steps} steps (60 seconds)!")

    print("=" * 80)


if __name__ == "__main__":
    main()
