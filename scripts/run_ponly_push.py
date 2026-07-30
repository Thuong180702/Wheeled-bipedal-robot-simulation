#!/usr/bin/env python3
"""Run P-only simulation: wheel PD + leg torque PD, NO WBC, NO anchor."""
import json, os, numpy as np, mujoco

ROOT = "/Users/admin/Wheeled-bipedal-robot-simulation"
MODEL_PATH = os.path.join(ROOT, "assets/robot/wheeled_biped_real.xml")
OUT_PATH = os.path.join(ROOT, "outputs/ringdown_ponly.json")

DT = 0.002; CTRL_DEC = 5
KP_WHEEL = 50.0; KD_WHEEL = 6.0

# High-gain leg posture PD (direct torque, no WBC)
KP_HIP_PITCH = 300.0; KD_HIP_PITCH = 20.0
KP_KNEE = 300.0; KD_KNEE = 20.0
KP_HIP_ROLL = 150.0; KD_HIP_ROLL = 10.0

def get_pitch(d):
    q = d.qpos[3:7]
    return float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    # Keyframe standing pose
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    q_des = data.qpos[7:17].copy()

    ACT = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j): j
           for j in range(model.nu)}

    # Joint name → qpos/qvel index mapping
    JNT = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            JNT[name] = j

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    total_s = 20.0; n_steps = int(total_s / DT)
    push_start = int(3.0 / DT); push_dur = int(0.07 / DT)

    n_log = n_steps // CTRL_DEC + 1
    times = np.zeros(n_log); pitches = np.zeros(n_log)
    log_idx = 0

    for step in range(n_steps):
        if push_start <= step < push_start + push_dur:
            data.xfrc_applied[torso_id, 0] = 90.0

        if step % CTRL_DEC == 0:
            pitch = get_pitch(data)
            pitch_rate = float(data.qvel[3])
            times[log_idx] = step * DT
            pitches[log_idx] = pitch
            log_idx += 1

            # Wheel torque
            tau_w = KP_WHEEL * pitch + KD_WHEEL * pitch_rate
            tau_w = np.clip(tau_w, -25, 25)

            # Leg torque via name-based assignment
            for name, act_idx in ACT.items():
                if 'wheel' in name:
                    data.ctrl[act_idx] = tau_w
                elif 'hip_pitch' in name:
                    jname = name.replace('_motor', '')
                    jidx = JNT.get(jname)
                    if jidx is not None:
                        err = q_des[jidx] - data.qpos[7 + jidx]
                        derr = -data.qvel[6 + jidx]
                        data.ctrl[act_idx] = np.clip(KP_HIP_PITCH * err + KD_HIP_PITCH * derr, -150, 150)
                elif 'knee' in name:
                    jname = name.replace('_motor', '')
                    jidx = JNT.get(jname)
                    if jidx is not None:
                        err = q_des[jidx] - data.qpos[7 + jidx]
                        derr = -data.qvel[6 + jidx]
                        data.ctrl[act_idx] = np.clip(KP_KNEE * err + KD_KNEE * derr, -150, 150)
                elif 'hip_roll' in name:
                    jname = name.replace('_motor', '')
                    jidx = JNT.get(jname)
                    if jidx is not None:
                        err = q_des[jidx] - data.qpos[7 + jidx]
                        derr = -data.qvel[6 + jidx]
                        data.ctrl[act_idx] = np.clip(KP_HIP_ROLL * err + KD_HIP_ROLL * derr, -60, 60)
                # hip_yaw: leave at 0

            if abs(pitch) > 0.8 or data.qpos[2] < 0.15:
                pitches[log_idx:] = pitch
                times[log_idx:] = step * DT
                break

        mujoco.mj_step(model, data)

    times = times[:log_idx]; pitches = pitches[:log_idx]
    print(f"P-only: {len(times)} samples, t=[{times[0]:.1f}, {times[-1]:.1f}]s")
    print(f"  Pitch range: [{np.min(pitches):.4f}, {np.max(pitches):.4f}] rad")
    print(f"  Pitch RMS: {np.sqrt(np.mean(pitches**2)):.4f} rad")
    print(f"  Base height range: [{data.qpos[2]:.3f}]m")

    result = {"time_s": times.tolist(), "pitch_rad": pitches.tolist(),
              "metadata": {"controller": "P-only", "kp_wheel": KP_WHEEL, "push_N": 90}}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
