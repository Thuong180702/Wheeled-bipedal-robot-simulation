#!/usr/bin/env python3
"""Robustness sweep: noise × delay factorial. Uses subprocess pattern
(same as replicate_ablation_n5.py) to avoid JAX/mjpython init issues.

4 noise levels × 3 delay levels, N=5 trials each cell.
Metrics: idle CoM X RMS (mm), F_max forward push (N).

Usage:
  .venv/bin/mjpython scripts/collect_robustness_sweep.py
  .venv/bin/mjpython scripts/collect_robustness_sweep.py --n-trials 5
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Measurement script (runs in subprocess, same pattern as replicate) ──
MEASURE_SCRIPT = r'''
import json, sys, time, os
import numpy as np
import mujoco as mj
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig)
from wheeled_biped.utils.config import get_model_path

DT = 0.01; SUBSTEPS = 5
TOTAL_IDLE_S = 25.0; SETTLE_S = 5.0
PUSH_DUR = 7; PUSH_START = 300; POST_PUSH_S = 17.0
POST_PUSH_STEPS = int(POST_PUSH_S / DT)
PITCH_LIMIT = 0.8; HEIGHT_LIMIT = 0.30
FORCE_MIN, FORCE_MAX = 10.0, 200.0
N_BISECT = 8; TOLERANCE = 5.0
N_TRIALS = 5

cfg = json.loads(sys.argv[1])  # {noise_name, noise_cfg, delay_steps, n_trials}
noise_name = cfg["noise_name"]
noise_cfg = cfg["noise_cfg"]
delay_steps = cfg["delay_steps"]
n_trials = cfg.get("n_trials", 5)

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])
JOINT_NAMES = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
               "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
BASE_SEED = 20260728

model = mj.MjModel.from_xml_path(str(get_model_path()))
l_wheel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "r_wheel_link")
torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso")
robot_mass = float(np.sum(model.body_mass))
torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
est_cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
est = CentroidalStateEstimator(est_cfg, mj_model=model)
CTX = {"centroidal_estimator": est, "initial_yaw_z": 0.0,
       "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
       "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}

def _delay_line(delay_steps):
    """FIFO of length delay_steps, zero-filled so the warm-up is a real hold.

    Until 2026-08-02 this was a SINGLE slot overwritten every control step, so
    every delay_steps > 0 applied exactly one control period of lag regardless
    of the value requested -- the "30/50/100/150 ms" cells were all 10 ms.  The
    archived robustness_sweep.json / delay_stability_sweep.json predate this
    fix and are reported in the paper as the 10 ms cells they actually are;
    re-running this script now produces genuinely distinct delays.
    """
    from collections import deque
    return deque([np.zeros(model.nu)] * delay_steps, maxlen=max(delay_steps, 1))

def _delayed(buf, tau, delay_steps):
    """Read BEFORE append: buf[0] is the command written delay_steps ago."""
    if delay_steps <= 0:
        return tau
    out = buf[0]
    buf.append(tau.copy())
    return out

def settle(data, v3, steps=300):
    for _ in range(steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], CTX, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mj.mj_step(model, data)

def run_idle_trial(data, v3, noise_cfg, delay_steps, seed):
    """Run idle standing. Returns com_x_rms_mm or None if fell."""
    rng = np.random.default_rng(seed)
    n_steps = int(TOTAL_IDLE_S / DT)
    settle_start = int(SETTLE_S / DT)
    window_samples = int(20.0 / DT)
    home_x = float(data.qpos[0])
    com_x_vals = np.zeros(window_samples)
    delay_buf = _delay_line(delay_steps)

    for step in range(n_steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], CTX, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        tau = np.array(r["tau_v3"])

        data.ctrl[:] = _delayed(delay_buf, tau, delay_steps)

        if noise_cfg.get("gyro", 0) > 0:
            data.qvel[3:6] += rng.normal(0, noise_cfg["gyro"], 3)
        if noise_cfg.get("joint", 0) > 0:
            data.qpos[7:17] += rng.normal(0, noise_cfg["joint"], 10)

        for _ in range(SUBSTEPS):
            mj.mj_step(model, data)

        idx = step - settle_start
        if 0 <= idx < window_samples:
            com_x_vals[idx] = (data.qpos[0] - home_x) * 1000

        q = data.qpos[3:7]
        pitch = np.arcsin(-2*(q[1]*q[3] - q[0]*q[2]))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.15:
            return None

    return float(np.sqrt(np.mean((com_x_vals - np.mean(com_x_vals))**2)))

def run_push_bisect(data, v3, noise_cfg, delay_steps, seed):
    """Binary search max forward push. Returns force_N."""
    rng = np.random.default_rng(seed)
    lo, hi = FORCE_MIN, FORCE_MAX
    best = lo

    for _ in range(N_BISECT):
        mid = (lo + hi) / 2
        d = mj.MjData(model)
        d.qpos[:] = data.qpos[:]
        d.qvel[:] = data.qvel[:]
        mj.mj_forward(model, d)
        v3c = {"jax_step_fn": v3["jax_step_fn"],
               "jax_state": v3["jax_state"],
               "jax_params": v3["jax_params"]}
        delay_buf = _delay_line(delay_steps)
        survived = True

        for step in range(POST_PUSH_STEPS + PUSH_DUR):
            r = compute_v3_torque_for_state(
                d, model, v3c["jax_step_fn"], v3c["jax_state"],
                v3c["jax_params"], CTX, teleop=None)
            v3c["jax_state"] = r["next_jax_state"]
            tau = np.array(r["tau_v3"])

            d.ctrl[:] = _delayed(delay_buf, tau, delay_steps)

            if noise_cfg.get("gyro", 0) > 0:
                d.qvel[3:6] += rng.normal(0, noise_cfg["gyro"], 3)
            if noise_cfg.get("joint", 0) > 0:
                d.qpos[7:17] += rng.normal(0, noise_cfg["joint"], 10)

            d.xfrc_applied[torso_id, :3] = 0.0
            if step < PUSH_DUR:
                d.xfrc_applied[torso_id, 0] = mid

            for _ in range(SUBSTEPS):
                mj.mj_step(model, d)

            q = d.qpos[3:7]
            pitch = np.arcsin(-2*(q[1]*q[3] - q[0]*q[2]))
            if abs(pitch) > PITCH_LIMIT or d.subtree_com[0][2] < HEIGHT_LIMIT:
                survived = False
                break

        if survived:
            best = mid
            lo = mid + TOLERANCE / 2
        else:
            hi = mid - TOLERANCE / 2

    return float(best)


t0 = time.time()
idle_vals = []
push_vals = []
fell_count = 0

for trial in range(n_trials):
    seed = BASE_SEED * 100 + delay_steps * 10 + trial

    # Idle
    rng_i = np.random.default_rng(seed)
    data_i = mj.MjData(model)
    perturbed = POSTURE + rng_i.normal(0.0, 0.005, size=10)
    for j, jname in enumerate(JOINT_NAMES):
        jid = model.joint(jname).id
        lo_j, hi_j = model.jnt_range[jid]
        perturbed[j] = float(np.clip(perturbed[j], lo_j, hi_j))
    data_i.qpos[7:17] = perturbed
    data_i.qpos[2] = ROOT_Z + rng_i.normal(0.0, 0.001)
    mj.mj_forward(model, data_i)
    v3_i = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3_i["jax_state"] = pack_state_k2()
    settle(data_i, v3_i, steps=PUSH_START)
    idle_rms = run_idle_trial(data_i, v3_i, noise_cfg, delay_steps, seed + 10000)
    if idle_rms is not None:
        idle_vals.append(idle_rms)
    else:
        fell_count += 1

    # Push
    seed_p = seed + 50000
    rng_p = np.random.default_rng(seed_p)
    data_p = mj.MjData(model)
    perturbed_p = POSTURE + rng_p.normal(0.0, 0.005, size=10)
    for j, jname in enumerate(JOINT_NAMES):
        jid = model.joint(jname).id
        lo_j, hi_j = model.jnt_range[jid]
        perturbed_p[j] = float(np.clip(perturbed_p[j], lo_j, hi_j))
    data_p.qpos[7:17] = perturbed_p
    data_p.qpos[2] = ROOT_Z + rng_p.normal(0.0, 0.001)
    mj.mj_forward(model, data_p)
    v3_p = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3_p["jax_state"] = pack_state_k2()
    settle(data_p, v3_p, steps=PUSH_START)
    f_max = run_push_bisect(data_p, v3_p, noise_cfg, delay_steps, seed_p + 100000)
    push_vals.append(f_max)

elapsed = time.time() - t0
idle_arr = np.array(idle_vals) if idle_vals else np.array([])
push_arr = np.array(push_vals)

print(json.dumps({
    "noise": noise_name,
    "delay_ms": delay_steps * 10,
    "delay_steps": delay_steps,
    "n_trials": n_trials,
    "n_fell": fell_count,
    "idle_rms_mm_mean": float(np.mean(idle_arr)) if len(idle_arr) > 0 else None,
    "idle_rms_mm_std": float(np.std(idle_arr, ddof=1)) if len(idle_arr) > 1 else 0.0,
    "f_max_N_mean": float(np.mean(push_arr)),
    "f_max_N_std": float(np.std(push_arr, ddof=1)) if len(push_arr) > 1 else 0.0,
    "idle_vals": [float(v) for v in idle_vals],
    "push_vals": [float(v) for v in push_vals],
    "elapsed_min": elapsed / 60.0,
}))
'''

NOISE_LEVELS = {
    "clean": {"gyro": 0.0, "accel": 0.0, "joint": 0.0},
    "low":   {"gyro": 0.01, "accel": 0.01, "joint": 0.001},
    "med":   {"gyro": 0.05, "accel": 0.05, "joint": 0.005},
    "high":  {"gyro": 0.10, "accel": 0.10, "joint": 0.010},
}
DELAY_STEPS = [0, 1, 3]  # 0, 10, 30 ms at 100Hz


def run_cell(noise_name: str, noise_cfg: dict, delay_steps: int,
             n_trials: int) -> dict | None:
    """Run one noise×delay cell via subprocess. Returns parsed result or None."""
    cfg_json = json.dumps({
        "noise_name": noise_name,
        "noise_cfg": noise_cfg,
        "delay_steps": delay_steps,
        "n_trials": n_trials,
    })
    try:
        result = subprocess.run(
            [sys.executable, "-c", MEASURE_SCRIPT, cfg_json],
            capture_output=True, text=True, timeout=3600,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"    FAILED (rc={result.returncode})")
            if result.stderr:
                # Show last lines of stderr
                lines = result.stderr.strip().split("\n")
                for line in lines[-3:]:
                    print(f"    stderr: {line[:200]}")
            return None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith('{"'):
                return json.loads(line)
        print(f"    No JSON in output. stdout tail: {result.stdout[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        print("    TIMEOUT (60 min)")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    total = len(NOISE_LEVELS) * len(DELAY_STEPS)
    print("=" * 72)
    print(f"ROBUSTNESS SWEEP — {total} cells × N={args.n_trials}")
    print(f"Est. time: ~{total * 3} min")
    print("=" * 72)

    results = {}
    t_start = time.time()
    idx = 0

    for noise_name, noise_cfg in NOISE_LEVELS.items():
        for delay_steps in DELAY_STEPS:
            delay_ms = delay_steps * 10
            label = f"{noise_name}_{delay_ms}ms"
            idx += 1
            print(f"\n[{idx}/{total}] {label} ...", end=" ", flush=True)

            t0 = time.time()
            data = run_cell(noise_name, noise_cfg, delay_steps, args.n_trials)
            dt = time.time() - t0

            if data:
                if data["idle_rms_mm_mean"] is not None:
                    print(f"OK ({dt:.0f}s) Idle={data['idle_rms_mm_mean']:.2f}±{data['idle_rms_mm_std']:.2f}mm "
                          f"F_max={data['f_max_N_mean']:.0f}±{data['f_max_N_std']:.0f}N")
                else:
                    print(f"OK ({dt:.0f}s) ALL FELL (fell={data['n_fell']})")
                results[label] = data
            else:
                print(f"FAILED ({dt:.0f}s)")
                results[label] = {"error": "measurement_failed"}

            # Save incrementally
            (OUT_DIR / "robustness_partial.json").write_text(
                json.dumps(results, indent=2))

    # Final save
    results["_metadata"] = {
        "n_trials": args.n_trials,
        "noise_levels": list(NOISE_LEVELS.keys()),
        "delay_ms": [d * 10 for d in DELAY_STEPS],
        "total_elapsed_min": (time.time() - t_start) / 60.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = OUT_DIR / "robustness_sweep.json"
    json.dump(results, out_path.open("w"), indent=2)

    # Summary table
    print(f"\n{'='*72}")
    print("ROBUSTNESS SWEEP RESULTS")
    print(f"{'='*72}")
    print(f"{'Condition':<22} {'Idle RMS (mm)':>18} {'F_max (N)':>18}")
    print("-" * 60)
    for label in sorted(results):
        if label.startswith("_"):
            continue
        r = results[label]
        if "error" not in r and r.get("idle_rms_mm_mean") is not None:
            print(f"{label:<22} {r['idle_rms_mm_mean']:>8.2f} ± {r['idle_rms_mm_std']:>5.2f}  "
                  f"{r['f_max_N_mean']:>8.0f} ± {r['f_max_N_std']:>5.0f}")
        else:
            print(f"{label:<22} {'--- (fell/error)':>18} {'---':>18}")

    print(f"\nSaved → {out_path}")
    print(f"Total time: {results['_metadata']['total_elapsed_min']:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
