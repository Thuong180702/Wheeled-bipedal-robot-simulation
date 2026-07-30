#!/usr/bin/env python3
"""Record P-only (V3 base, no anchor) standing pitch for FFT."""
import json, os, sys, numpy as np, mujoco

sys.path.insert(0, '/Users/admin/Wheeled-bipedal-robot-simulation')
import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.teleop_shaper import HeightPosture

# Use V3 base (no anchor) for P-only
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_BASE"
OUT_PATH = "outputs/ponly_standing_pitch.json"
DT = 0.01; SUBSTEPS = 5
TOTAL_S = 15.0  # 15s of standing for good FFT resolution

def get_pitch(d):
    q = d.qpos[3:7]
    return float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))

def main():
    print(f"Recording P-only standing pitch ({PROFILE})...")
    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))

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

    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture, height_ref=h0)

    # Settle 3s
    for _ in range(int(3.0/DT)):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    # Record
    total_steps = int(TOTAL_S / DT)
    pitch = np.zeros(total_steps)
    t = np.zeros(total_steps)

    for step in range(total_steps):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        t[step] = step * DT
        pitch[step] = get_pitch(data)

        if abs(pitch[step]) > 0.8 or data.qpos[2] < 0.15:
            print(f"FALL at {step*DT:.1f}s")
            t = t[:step+1]; pitch = pitch[:step+1]
            break

    print(f"Done: {len(t)} steps, pitch RMS={np.std(pitch)*1000:.2f} mrad, "
          f"range=[{np.degrees(np.min(pitch)):.2f}, {np.degrees(np.max(pitch)):.2f}] deg")

    result = {"time_s": t.tolist(), "pitch_rad": pitch.tolist(),
              "metadata": {"profile": PROFILE, "dt": DT, "total_s": TOTAL_S}}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
