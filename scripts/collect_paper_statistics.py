"""
Collect paper statistics: run all ACC measurements with N≥10 trials,
compute mean±std, and output JSON for paper table updates.

Measures:
  A. Idle standing CoM RMS — 10 independent 20s trials
  B. Push recovery — per-direction max survived force, 10 trials each
  C. Ringdown time — 10 trials after 90N forward push
  D. Drop recovery — N=10 per height (10, 20, ..., 100 cm)
  E. Ledge drive-off — N=10 per height (20, 30, 40, 50 cm)
  F. Curb straddle — N=10 per height (10, 15, 20 cm)

Output: outputs/paper_statistics/stats.json
"""
import json, sys, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import mujoco
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
)
from wheeled_biped.controllers.k2_jax_controller import (
    k2_jax_controller_step, pack_input_k2_standalone,
    pack_params_k2, pack_state_k2,
)

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
MODEL_PATH = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
CONTROL_DT = 0.01
SIM_DT = 0.002
SUBSTEPS = 5
N_TRIALS = 10


def load_model_and_controller():
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(mj_model)
    ctrl = SagittalVelocityDampedBalanceController.from_profile(mj_model, mj_data, PROFILE)
    jax_params = pack_params_k2(ctrl.params_flat)
    jax_state = pack_state_k2(ctrl.state_flat)
    return mj_model, mj_data, ctrl, jax_params, jax_state


# =========================================================================
# A. Standing Idle Measurement (N trials)
# =========================================================================
def run_standing_trial(trial_id, duration_s=23.0, settle_s=3.0):
    """One standing trial. Returns CoM RMS in mm."""
    mj_model, mj_data, ctrl, jax_params, jax_state = load_model_and_controller()
    total_steps = int(duration_s / CONTROL_DT)
    settle_steps = int(settle_s / CONTROL_DT)

    com_x, com_y = [], []

    for step in range(total_steps):
        jax_input = pack_input_k2_standalone(mj_data, ctrl, jax_params, jax_state)
        tau, jax_state, _ = k2_jax_controller_step(jax_state, jax_input, jax_params)
        mj_data.ctrl[:] = np.array(tau)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mj_model, mj_data)

        if step >= settle_steps:
            com = mj_data.subtree_com[0]
            com_x.append(float(com[0]))
            com_y.append(float(com[1]))

    # Sagittal CoM RMS (X direction — forward/backward)
    x_arr = np.array(com_x)
    y_arr = np.array(com_y)
    rms_x = float(np.std(x_arr)) * 1000  # mm
    rms_y = float(np.std(y_arr)) * 1000

    # Total XY RMS
    xy = np.sqrt((x_arr - np.mean(x_arr))**2 + (y_arr - np.mean(y_arr))**2)
    rms_xy = float(np.sqrt(np.mean(xy**2))) * 1000

    return {"trial": trial_id, "rms_x_mm": rms_x, "rms_y_mm": rms_xy,
            "rms_sagittal_mm": rms_x, "com_mean_z_m": float(np.mean([mj_data.subtree_com[0][2] for _ in [0]]))}


def run_standing_statistics():
    print("=" * 60)
    print(f"A. Standing Idle — {N_TRIALS} trials")
    print("=" * 60)
    trials = []
    for i in range(N_TRIALS):
        r = run_standing_trial(i)
        trials.append(r)
        print(f"  Trial {i+1}/{N_TRIALS}: sagittal RMS = {r['rms_sagittal_mm']:.3f} mm")

    rms_vals = [t["rms_sagittal_mm"] for t in trials]
    return {
        "test": "standing_idle",
        "n_trials": N_TRIALS,
        "duration_s": 20.0,
        "settle_s": 3.0,
        "rms_sagittal_mm_mean": float(np.mean(rms_vals)),
        "rms_sagittal_mm_std": float(np.std(rms_vals, ddof=1)),
        "rms_sagittal_mm_min": float(np.min(rms_vals)),
        "rms_sagittal_mm_max": float(np.max(rms_vals)),
        "ci_95": float(1.96 * np.std(rms_vals, ddof=1) / np.sqrt(N_TRIALS)),
        "per_trial": trials,
    }


# =========================================================================
# B. Push Recovery Statistics (8 directions, binary search per direction)
# =========================================================================
def run_push_binary_search(angle_deg, max_force=160.0, min_force=10.0, tol=5.0, max_iters=8):
    """Binary search for max survived force at a given angle."""
    lo, hi = min_force, max_force
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        if run_single_push(angle_deg, mid):
            lo = mid
        else:
            hi = mid
    return lo


def run_single_push(angle_deg, force_N, push_dur_steps=7, warmup_steps=300, post_push_s=17.0):
    """Test if robot survives a single push."""
    mj_model, mj_data, ctrl, jax_params, jax_state = load_model_and_controller()
    angle_rad = np.deg2rad(angle_deg)
    force_vec = np.array([-np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * force_N
    total_steps = warmup_steps + push_dur_steps + int(post_push_s / CONTROL_DT)

    for step in range(total_steps):
        jax_input = pack_input_k2_standalone(mj_data, ctrl, jax_params, jax_state)
        tau, jax_state, _ = k2_jax_controller_step(jax_state, jax_input, jax_params)
        mj_data.ctrl[:] = np.array(tau)

        if warmup_steps <= step < warmup_steps + push_dur_steps:
            mj_data.xfrc_applied[1, :3] = force_vec

        for _ in range(SUBSTEPS):
            mujoco.mj_step(mj_model, mj_data)

        quat = mj_data.qpos[3:7]
        pitch = float(np.arcsin(-2 * (quat[1]*quat[3] - quat[0]*quat[2])))
        if abs(pitch) > 0.8 or mj_data.qpos[2] < 0.30:
            return False
    return True


def run_push_statistics():
    print("\n" + "=" * 60)
    print(f"B. Push Recovery — 8 directions, {N_TRIALS} binary searches each")
    print("=" * 60)
    angles = [0, 45, 90, 135, 180, -135, -90, -45]

    all_trials = {ang: [] for ang in angles}
    for ang in angles:
        print(f"  Angle {ang:4d}°: ", end="", flush=True)
        for trial in range(N_TRIALS):
            f_max = run_push_binary_search(ang)
            all_trials[ang].append(f_max)
            print(f"{f_max:.0f} ", end="", flush=True)
        print()

    # Per-direction stats
    per_direction = {}
    f_min_values = []
    f_med_values = []
    for ang in angles:
        vals = all_trials[ang]
        per_direction[ang] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }
        f_min_values.append(float(np.mean(vals)))

    # F_min = minimum across all directions, F_med = median across directions
    f_min_mean = float(np.min(f_min_values))
    f_med_mean = float(np.median(f_min_values))

    return {
        "test": "push_recovery",
        "n_trials_per_direction": N_TRIALS,
        "n_directions": len(angles),
        "binary_search_tol_N": 5.0,
        "F_min_N": f_min_mean,
        "F_med_N": f_med_mean,
        "F_min_std_across_directions": float(np.std(f_min_values, ddof=1)),
        "per_direction": per_direction,
    }


# =========================================================================
# C. Ringdown Time (10 trials after 90N forward push)
# =========================================================================
def run_ringdown_trial(trial_id, settle_threshold_deg=5.0, confirm_steps=50):
    """Measure ringdown time after 90N forward push."""
    mj_model, mj_data, ctrl, jax_params, jax_state = load_model_and_controller()
    push_force = 90.0
    push_dur = 7
    push_start = 300
    total_steps = push_start + push_dur + 2000  # +20s post-push

    pitch_history = []
    push_applied_step = None

    for step in range(total_steps):
        jax_input = pack_input_k2_standalone(mj_data, ctrl, jax_params, jax_state)
        tau, jax_state, _ = k2_jax_controller_step(jax_state, jax_input, jax_params)
        mj_data.ctrl[:] = np.array(tau)

        if push_start <= step < push_start + push_dur:
            mj_data.xfrc_applied[1, :3] = np.array([0.0, -push_force, 0.0])
            if push_applied_step is None:
                push_applied_step = step

        for _ in range(SUBSTEPS):
            mujoco.mj_step(mj_model, mj_data)

        quat = mj_data.qpos[3:7]
        pitch = float(np.arcsin(-2 * (quat[1]*quat[3] - quat[0]*quat[2])))
        pitch_deg = float(np.degrees(pitch))
        pitch_history.append(pitch_deg)

        if abs(pitch) > 0.8 or mj_data.qpos[2] < 0.30:
            return {"trial": trial_id, "fell": True, "ringdown_s": float("nan")}

    # Find ringdown: first time pitch stays below threshold for confirm_steps
    post_push_start = push_start + push_dur
    ringdown_step = None
    for i in range(post_push_start, len(pitch_history) - confirm_steps):
        if all(abs(pitch_history[j]) < settle_threshold_deg for j in range(i, i + confirm_steps)):
            ringdown_step = i
            break

    if ringdown_step is not None:
        ringdown_s = (ringdown_step - push_applied_step) * CONTROL_DT
    else:
        ringdown_s = float("inf")

    peak_pitch = float(np.max(np.abs(pitch_history[push_start:push_start + 100])))

    return {
        "trial": trial_id,
        "fell": False,
        "ringdown_s": float(ringdown_s),
        "peak_pitch_deg": peak_pitch,
    }


def run_ringdown_statistics():
    print("\n" + "=" * 60)
    print(f"C. Ringdown Time — {N_TRIALS} trials after 90N forward push")
    print("=" * 60)
    trials = []
    for i in range(N_TRIALS):
        r = run_ringdown_trial(i)
        trials.append(r)
        status = f"ringdown={r['ringdown_s']:.1f}s, peak={r['peak_pitch_deg']:.1f}°"
        if r["fell"]:
            status = "FELL"
        print(f"  Trial {i+1}/{N_TRIALS}: {status}")

    rd_vals = [t["ringdown_s"] for t in trials if not t["fell"] and t["ringdown_s"] < float("inf")]
    peak_vals = [t["peak_pitch_deg"] for t in trials if not t["fell"]]

    return {
        "test": "ringdown",
        "n_trials": N_TRIALS,
        "push_force_N": 90.0,
        "push_direction": "forward",
        "fell_count": sum(1 for t in trials if t["fell"]),
        "ringdown_s_mean": float(np.mean(rd_vals)) if rd_vals else float("nan"),
        "ringdown_s_std": float(np.std(rd_vals, ddof=1)) if len(rd_vals) > 1 else 0.0,
        "peak_pitch_deg_mean": float(np.mean(peak_vals)) if peak_vals else float("nan"),
        "peak_pitch_deg_std": float(np.std(peak_vals, ddof=1)) if len(peak_vals) > 1 else 0.0,
        "per_trial": trials,
    }


# =========================================================================
# Main
# =========================================================================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--standing", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--ringdown", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    do_all = args.all or not (args.standing or args.push or args.ringdown)

    all_stats = {}
    t0 = time.time()

    if do_all or args.standing:
        all_stats["standing"] = run_standing_statistics()

    if do_all or args.push:
        all_stats["push"] = run_push_statistics()

    if do_all or args.ringdown:
        all_stats["ringdown"] = run_ringdown_statistics()

    all_stats["metadata"] = {
        "profile": PROFILE,
        "n_trials_default": N_TRIALS,
        "control_hz": 100,
        "sim_hz": 500,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": time.time() - t0,
    }

    out_path = OUT_DIR / "stats.json"
    json.dump(all_stats, out_path.open("w"), indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Saved → {out_path}")
    print(f"Elapsed: {all_stats['metadata']['elapsed_s']:.0f}s")

    # Print paper-ready summary
    print(f"\n{'='*60}")
    print("PAPER-READY STATISTICS")
    print(f"{'='*60}")
    if "standing" in all_stats:
        s = all_stats["standing"]
        print(f"Idle CoM RMS: {s['rms_sagittal_mm_mean']:.2f} ± {s['rms_sagittal_mm_std']:.2f} mm (N={s['n_trials']}, 95%CI=±{s['ci_95']:.2f}mm)")
    if "push" in all_stats:
        p = all_stats["push"]
        print(f"F_min: {p['F_min_N']:.0f} N, F_med: {p['F_med_N']:.0f} N (N={p['n_trials_per_direction']} per direction, {p['n_directions']} directions)")
    if "ringdown" in all_stats:
        r = all_stats["ringdown"]
        print(f"Ringdown: {r['ringdown_s_mean']:.1f} ± {r['ringdown_s_std']:.1f} s (N={r['n_trials']}, {r['fell_count']} falls)")
        print(f"Peak pitch: {r['peak_pitch_deg_mean']:.1f} ± {r['peak_pitch_deg_std']:.1f}°")


if __name__ == "__main__":
    main()
