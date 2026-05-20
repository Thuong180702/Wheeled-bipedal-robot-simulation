"""Sensor-based controller using IMU and torque feedback at 100Hz.

Uses actual robot sensors:
- IMU (accelerometer, gyro, quaternion) for orientation
- Wheel torque sensors for ground contact detection
- Joint encoders for position/velocity feedback

Control architecture:
1. Wheels: Velocity PD based on pitch error (Segway balance)
2. Leg joints: Position PD for height/posture
3. Roll stabilization: Antisymmetric hip_roll torques
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
    """Check if wheels are in contact with ground.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        Tuple of (left_contact, right_contact, num_left_contacts, num_right_contacts)
    """
    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    l_contacts = 0
    r_contacts = 0

    # Check all active contacts
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        # Check if this is a wheel-floor contact
        if (geom1 == l_wheel_geom_id and geom2 == floor_geom_id) or \
           (geom2 == l_wheel_geom_id and geom1 == floor_geom_id):
            l_contacts += 1

        if (geom1 == r_wheel_geom_id and geom2 == floor_geom_id) or \
           (geom2 == r_wheel_geom_id and geom1 == floor_geom_id):
            r_contacts += 1

    l_contact = l_contacts > 0
    r_contact = r_contacts > 0

    return l_contact, r_contact, l_contacts, r_contacts


def main():
    print("=" * 80)
    print("Sensor-Based Controller (100Hz)")
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

    print(f"\nSensor IDs:")
    print(f"  IMU accel: {imu_accel_id}")
    print(f"  IMU gyro: {imu_gyro_id}")
    print(f"  IMU quat: {imu_quat_id}")
    print(f"\n[NOTE] Contact forces computed after first physics step")

    # Target leg configuration - match keyframe (hip_pitch=0.4, knee=0.4)
    target_leg_pos = np.array([
        0.0, 0.4, 0.4,  # left: hip_yaw, hip_pitch, knee
        0.0, 0.4, 0.4,  # right: hip_yaw, hip_pitch, knee
    ])
    LEG_INDICES = [1, 2, 3, 6, 7, 8]  # exclude hip_roll (0, 5) and wheels (4, 9)

    # Controller gains
    # Leg position PD
    kp_leg = np.array([100, 200, 200, 100, 200, 200])
    kd_leg = np.array([10, 20, 20, 10, 20, 20])

    # Active roll stabilization
    K_roll = 150.0
    K_roll_rate = 25.0

    # Segway balance (from LQR baseline)
    K_pitch = 87.7
    K_pitch_rate = 20.6
    K_fwd_vel = 5.2
    K_fwd_pos = 0.6

    # Wheel velocity tracking
    kp_wheel_vel = 5.0

    print(f"\nController gains:")
    print(f"  Segway: K_pitch={K_pitch:.1f}, K_pitch_rate={K_pitch_rate:.1f}")
    print(f"  Roll: K_roll={K_roll:.1f}, K_roll_rate={K_roll_rate:.1f}")
    print(f"  Leg position: kp_hip_pitch={kp_leg[2]:.1f}, kp_knee={kp_leg[3]:.1f}")
    print(f"  Wheel velocity: kp={kp_wheel_vel:.1f}")

    print(f"\nRunning simulation at 100Hz for 6000 steps (60 seconds)")
    print(f"{'Step':>6} {'h':>7} {'pitch':>8} {'roll':>8} {'contact':>8} {'wheel_vel':>10} {'l_hip_p':>9} {'r_hip_p':>9}")
    print("-" * 85)

    max_steps = 6000
    control_dt = 0.01  # 100Hz control
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    warmup_steps = 10  # Skip contact checking during warmup

    fwd_pos_drift = 0.0

    for step in range(max_steps):
        # Read IMU sensors
        imu_accel_adr = mj_model.sensor_adr[imu_accel_id]
        imu_gyro_adr = mj_model.sensor_adr[imu_gyro_id]
        imu_quat_adr = mj_model.sensor_adr[imu_quat_id]

        imu_quat = mj_data.sensordata[imu_quat_adr:imu_quat_adr+4]  # [w, x, y, z]
        imu_gyro = mj_data.sensordata[imu_gyro_adr:imu_gyro_adr+3]  # [wx, wy, wz]
        imu_accel = mj_data.sensordata[imu_accel_adr:imu_accel_adr+3]  # [ax, ay, az]

        # Debug: print IMU quaternion on first step
        if step == 0:
            print(f"\n[DEBUG] IMU quaternion: w={imu_quat[0]:.4f}, x={imu_quat[1]:.4f}, y={imu_quat[2]:.4f}, z={imu_quat[3]:.4f}")

        # Extract orientation from IMU
        pitch, roll = get_pitch_roll_from_quat(imu_quat)
        pitch_rate = imu_gyro[1]  # angular velocity around y-axis (pitch)
        roll_rate = imu_gyro[0]   # angular velocity around x-axis (roll)

        # Read joint encoders
        joint_pos = mj_data.qpos[7:17]
        joint_vel = mj_data.qvel[6:16]

        # Forward velocity (from body velocity, not IMU)
        fwd_vel = -mj_data.qvel[1]
        fwd_pos_drift += fwd_vel * control_dt

        # Check ground contact
        l_contact, r_contact, l_contacts, r_contacts = check_ground_contact(mj_model, mj_data)
        contact_status = f"{int(l_contact)}{int(r_contact)}"

        # Debug: check wheel positions and contacts on first few steps
        if step < 3:
            l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
            r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
            l_wheel_z = mj_data.xpos[l_wheel_id][2]
            r_wheel_z = mj_data.xpos[r_wheel_id][2]
            print(f"[DEBUG] step {step}: ncon={mj_data.ncon}, l_wheel_z={l_wheel_z:.4f}, r_wheel_z={r_wheel_z:.4f}, l_contacts={l_contacts}, r_contacts={r_contacts}")
            if mj_data.ncon > 0:
                for i in range(min(3, mj_data.ncon)):
                    c = mj_data.contact[i]
                    g1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
                    g2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
                    print(f"[DEBUG]   contact {i}: {g1_name} <-> {g2_name}")

        # --- Leg position PD control (hip_yaw, hip_pitch, knee only) ---
        tau = np.zeros(10)
        for i, idx in enumerate(LEG_INDICES):
            pos_err = target_leg_pos[i] - joint_pos[idx]
            tau[idx] = kp_leg[i] * pos_err - kd_leg[i] * joint_vel[idx]

        # --- Active roll stabilization through hip_roll (antisymmetric) ---
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

        if step % 1 == 0 and step < 20:  # Print every step for first 20 steps
            # Read actual motor torques from MuJoCo (after saturation)
            actual_tau = mj_data.actuator_force.copy()

            print(f"\n[STEP {step}] pitch={np.rad2deg(pitch):6.2f}°, roll={np.rad2deg(roll):6.2f}°, contact={contact_status}")
            print(f"  Joint positions: l_hip_pitch={np.rad2deg(joint_pos[2]):6.2f}°, l_knee={np.rad2deg(joint_pos[3]):6.2f}°")
            print(f"  Joint velocities: l_hip_pitch={np.rad2deg(joint_vel[2]):6.2f}°/s, l_knee={np.rad2deg(joint_vel[3]):6.2f}°/s")
            print(f"  Commanded torques:")
            print(f"    l_hip_roll={tau[0]:7.2f} Nm, l_hip_yaw={tau[1]:7.2f} Nm, l_hip_pitch={tau[2]:7.2f} Nm, l_knee={tau[3]:7.2f} Nm, l_wheel={tau[4]:7.2f} Nm")
            print(f"    r_hip_roll={tau[5]:7.2f} Nm, r_hip_yaw={tau[6]:7.2f} Nm, r_hip_pitch={tau[7]:7.2f} Nm, r_knee={tau[8]:7.2f} Nm, r_wheel={tau[9]:7.2f} Nm")
            print(f"  ACTUAL motor torques (after saturation):")
            print(f"    l_hip_roll={actual_tau[0]:7.2f} Nm, l_hip_yaw={actual_tau[1]:7.2f} Nm, l_hip_pitch={actual_tau[2]:7.2f} Nm, l_knee={actual_tau[3]:7.2f} Nm, l_wheel={actual_tau[4]:7.2f} Nm")
            print(f"    r_hip_roll={actual_tau[5]:7.2f} Nm, r_hip_yaw={actual_tau[6]:7.2f} Nm, r_hip_pitch={actual_tau[7]:7.2f} Nm, r_knee={actual_tau[8]:7.2f} Nm, r_wheel={actual_tau[9]:7.2f} Nm")

            # Show saturation
            saturated = []
            motor_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
                          'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']
            for i in range(10):
                if abs(tau[i]) > 60.0:
                    saturated.append(f"{motor_names[i]}({tau[i]:.0f}Nm)")
            if saturated:
                print(f"  [SATURATED] {', '.join(saturated)}")

        # Termination conditions
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
