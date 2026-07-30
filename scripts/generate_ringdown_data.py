#!/usr/bin/env python3
"""Generate real ringdown data: ACC vs P-only under 90N forward push at 3s.

Uses MuJoCo model + a minimal-but-functional balance controller.
The "ACC" variant has scheduled kp + anchor integral + damping boost.
The "P-only" variant has fixed kp=50, no integral, no boost.
Both share the same leg posture PD controller.
"""
import json, os, sys
import numpy as np
import mujoco

ROOT = "/Users/admin/Wheeled-bipedal-robot-simulation"
MODEL_PATH = os.path.join(ROOT, "assets/robot/wheeled_biped_real.xml")
OUT_PATH = os.path.join(ROOT, "outputs/ringdown_data.json")

DT = 0.002            # 500 Hz sim
CTRL_DEC = 5          # ctrl at 100 Hz
G = 9.81

# ── Balance params ──
KP_STIFF = 50.0; KP_SOFT = 35.0
KD_PITCH = 6.0
K_VEL = 2.5
K_POS = 40.0
KI = 15.0; K_BOOST = 25.0

# ── Leg posture PD (holds default standing pose) ──
KP_HIP_PITCH = 80.0; KD_HIP_PITCH = 5.0
KP_KNEE = 100.0; KD_KNEE = 6.0
KP_HIP_ROLL = 40.0; KD_HIP_ROLL = 3.0

def get_pitch(d):
    q = d.qpos[3:7]
    return float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))

def get_roll(d):
    q = d.qpos[3:7]
    return float(np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)))

def smoothstep(x, lo, hi):
    t = np.clip((x - lo)/(hi - lo), 0, 1)
    return t*t*(3 - 2*t)

def run_sim(mode, model, data, push_N=90, total_s=20, push_start_s=3, push_dur_s=0.07):
    """mode: 'acc' or 'ponly'"""
    n_steps = int(total_s / DT)
    push_start = int(push_start_s / DT)
    push_dur = int(push_dur_s / DT)

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.54
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # Get default joint angles (standing pose) from keyframe or qpos0
    q_des = data.qpos[7:17].copy()  # joint reference = reset pose
    data.qpos[7:17] = q_des

    # Actuator indices
    ACT = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j): j
           for j in range(model.nu)}

    n_log = n_steps // CTRL_DEC + 1
    pitch_log = np.zeros(n_log); time_log = np.zeros(n_log)
    log_idx = 0

    integral = 0.0; ema_vel = 0.0
    home_x = float(data.qpos[0])

    for step in range(n_steps):
        if push_start <= step < push_start + push_dur:
            data.xfrc_applied[torso_id, 0] = push_N

        if step % CTRL_DEC == 0:
            pitch = get_pitch(data)
            roll = get_roll(data)
            pitch_rate = float(data.qvel[3])
            roll_rate = float(data.qvel[4])
            vx = float(data.qvel[0])
            dx = float(data.qpos[0] - home_x)

            pitch_log[log_idx] = pitch
            time_log[log_idx] = step * DT
            log_idx += 1

            # ── Wheel torque (balance) ──
            if mode == 'acc':
                g_prox = 1 - smoothstep(abs(dx), 0.05, 0.15)
                if abs(vx) > ema_vel:
                    ema_vel = 0.35*abs(vx) + 0.65*ema_vel
                else:
                    ema_vel = 0.007*abs(vx) + 0.993*ema_vel
                g_env = 1 - smoothstep(ema_vel, 0.18, 0.30)
                g_kp = 1 - g_env * g_prox
                kp = KP_STIFF - (KP_STIFF - KP_SOFT)*g_kp
                integral += g_prox*g_env*(-dx)*DT*CTRL_DEC
                integral = np.clip(integral, -10, 10)
                tau_anchor = KI*integral
                tau_boost = g_prox*g_env*K_BOOST*(-vx)
            else:
                kp = KP_STIFF
                tau_anchor = 0.0
                tau_boost = 0.0

            tau_balance = kp*pitch + KD_PITCH*pitch_rate - K_VEL*vx - K_POS*np.clip(dx, -0.1, 0.1)
            tau_wheel = np.clip(tau_balance + tau_anchor + tau_boost, -25, 25)

            # ── Leg posture torque ──
            tau_leg = np.zeros(10)
            for side, hip_pitch_idx, knee_idx, hip_roll_idx, hip_yaw_idx in [
                ('l', 2, 3, 0, 1), ('r', 7, 8, 5, 6)]:
                tau_leg[hip_pitch_idx] = KP_HIP_PITCH*(q_des[hip_pitch_idx] - data.qpos[7+hip_pitch_idx]) - KD_HIP_PITCH*data.qvel[6+hip_pitch_idx]
                tau_leg[knee_idx] = KP_KNEE*(q_des[knee_idx] - data.qpos[7+knee_idx]) - KD_KNEE*data.qvel[6+knee_idx]
                tau_leg[hip_roll_idx] = KP_HIP_ROLL*(q_des[hip_roll_idx] - data.qpos[7+hip_roll_idx]) - KD_HIP_ROLL*data.qvel[6+hip_roll_idx]

            # ── Assign torques ──
            for name, j in ACT.items():
                if 'wheel' in name:
                    data.ctrl[j] = tau_wheel
                elif name == 'l_hip_pitch_motor':
                    data.ctrl[j] = tau_leg[2]
                elif name == 'r_hip_pitch_motor':
                    data.ctrl[j] = tau_leg[7]
                elif name == 'l_knee_motor':
                    data.ctrl[j] = tau_leg[3]
                elif name == 'r_knee_motor':
                    data.ctrl[j] = tau_leg[8]
                elif name == 'l_hip_roll_motor':
                    data.ctrl[j] = tau_leg[0]
                elif name == 'r_hip_roll_motor':
                    data.ctrl[j] = tau_leg[5]

            # Fall check
            if abs(pitch) > 0.8 or data.qpos[2] < 0.15:
                pitch_log[log_idx:] = pitch
                time_log[log_idx:] = step*DT
                break

        mujoco.mj_step(model, data)

    return time_log[:log_idx].tolist(), pitch_log[:log_idx].tolist()


def main():
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    # Set initial qpos to default keyframe
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"Default standing height: z={data.qpos[2]:.3f}m, pitch={get_pitch(data):.3f}rad")
    print(f"Joint refs: {np.round(data.qpos[7:17], 3)}")

    print("\nRunning ACC (90N forward push, 20s)...")
    data1 = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data1, 0)
    t_acc, p_acc = run_sim('acc', model, data1)
    print(f"  ACC: {len(t_acc)} samples, t=[{t_acc[0]:.1f}, {t_acc[-1]:.1f}]s, "
          f"final pitch={p_acc[-1]:.3f}rad")

    print("Running P-only (90N forward push, 20s)...")
    data2 = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data2, 0)
    t_ponly, p_ponly = run_sim('ponly', model, data2)
    print(f"  P-only: {len(t_ponly)} samples, t=[{t_ponly[0]:.1f}, {t_ponly[-1]:.1f}]s, "
          f"final pitch={p_ponly[-1]:.3f}rad")

    result = {
        "metadata": {"push_N": 90, "push_start_s": 3, "push_dur_s": 0.07,
                     "model": MODEL_PATH, "sim_dt": DT, "ctrl_hz": 1/(DT*CTRL_DEC)},
        "acc": {"time_s": t_acc, "pitch_rad": p_acc},
        "ponly": {"time_s": t_ponly, "pitch_rad": p_ponly},
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
