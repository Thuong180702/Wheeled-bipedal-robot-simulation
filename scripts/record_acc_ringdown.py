#!/usr/bin/env python3
"""Record ACC (V3_ANCHOR) ringdown with push_sweep_paper.py approach."""
import json, os, sys, numpy as np, mujoco

sys.path.insert(0, '/Users/admin/Wheeled-bipedal-robot-simulation')
import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.teleop_shaper import HeightPosture

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
OUT_PATH = "outputs/acc_ringdown_v3_anchor.json"
DT = 0.01; SUBSTEPS = 5
TOTAL_S = 20.0; PUSH_START_S = 3.0; PUSH_DUR_S = 0.07; PUSH_N = 90.0

def get_pitch(d):
    q = d.qpos[3:7]
    return float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))

def get_roll(d):
    q = d.qpos[3:7]
    return float(np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)))

def get_yaw(d):
    q = d.qpos[3:7]
    return float(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)))

def main():
    print(f"V3_ANCHOR ringdown: {PUSH_N}N forward push at {PUSH_START_S}s")

    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # Setup posture
    hp = HeightPosture()
    nom = json.load(open(
        "archive/cleanup_2026-06-13/output_summaries/"
        "balance_core_true_height_variants/"
        "variant_nominal__variant_setup.json"))
    h0 = float(nom["target_com_z_m"])
    posture = np.array([
        nom["hip_roll_left"], nom["hip_yaw_left"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
        nom["hip_roll_right"], nom["hip_yaw_right"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])

    data = mujoco.MjData(model)
    data.qpos[7:17] = posture
    data.qpos[2] = float(nom["calibrated_root_z_m"])
    mujoco.mj_forward(model, data)

    # Init V3_ANCHOR
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture, height_ref=h0)

    # Settle for 3s
    print("Settling 3s...")
    for _ in range(int(3.0/DT)):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    home_x, home_y = float(data.qpos[0]), float(data.qpos[1])
    print(f"Home: x={home_x:.4f} y={home_y:.4f} z={data.qpos[2]:.4f}")

    # Main run
    total_steps = int(TOTAL_S / DT)
    push_start = int(PUSH_START_S / DT)
    push_dur = int(PUSH_DUR_S / DT)

    t_log = np.zeros(total_steps)
    pitch_log = np.zeros(total_steps); roll_log = np.zeros(total_steps)
    yaw_log = np.zeros(total_steps)
    com_x_log = np.zeros(total_steps); com_y_log = np.zeros(total_steps)

    for step in range(total_steps):
        # Push: clear then apply during window
        data.xfrc_applied[torso_id, :3] = 0.0
        if push_start <= step < push_start + push_dur:
            data.xfrc_applied[torso_id, 0] = PUSH_N

        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        t_log[step] = step * DT
        pitch_log[step] = get_pitch(data)
        roll_log[step] = get_roll(data)
        yaw_log[step] = get_yaw(data)
        com_x_log[step] = float(data.qpos[0] - home_x)
        com_y_log[step] = float(data.qpos[1] - home_y)

        if abs(pitch_log[step]) > 0.8 or data.qpos[2] < 0.15:
            print(f"FALL at {step*DT:.1f}s")
            t_log = t_log[:step+1]; pitch_log = pitch_log[:step+1]
            roll_log = roll_log[:step+1]; yaw_log = yaw_log[:step+1]
            com_x_log = com_x_log[:step+1]; com_y_log = com_y_log[:step+1]
            break

    print(f"Done: {len(t_log)} steps, t=[{t_log[0]:.1f}, {t_log[-1]:.1f}]s")
    print(f"Pitch: [{np.degrees(np.min(pitch_log)):.3f}, {np.degrees(np.max(pitch_log)):.3f}] deg")
    print(f"Roll:  [{np.degrees(np.min(roll_log)):.3f}, {np.degrees(np.max(roll_log)):.3f}] deg")
    print(f"Yaw:   [{np.degrees(np.min(yaw_log)):.3f}, {np.degrees(np.max(yaw_log)):.3f}] deg")
    print(f"X drift: {com_x_log[0]:.4f} -> {com_x_log[-1]:.4f} m")
    print(f"Y drift: {com_y_log[0]:.4f} -> {com_y_log[-1]:.4f} m")

    result = {
        "metadata": {"profile": PROFILE, "push_N": PUSH_N, "push_start_s": PUSH_START_S},
        "time_s": t_log.tolist(), "pitch_rad": pitch_log.tolist(),
        "roll_rad": roll_log.tolist(), "yaw_rad": yaw_log.tolist(),
        "com_x_m": com_x_log.tolist(), "com_y_m": com_y_log.tolist(),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
