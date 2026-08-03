#!/usr/bin/env python3
"""
N=10 replication for ablation rows L2 and S1–S5 (R4 fix from Round 11 review).

Patches k2_jax_controller.py for each config, runs 8-direction push binary
search with N=10 independent reps, then restores the original file.

Each config: ~20 min. All 6: ~120 min.

Usage:
  .venv/bin/python scripts/replicate_ablation_n10.py
  .venv/bin/python scripts/replicate_ablation_n10.py --configs L2 S2 S3  # selective
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTRL_FILE = ROOT / "wheeled_biped/controllers/k2_jax_controller.py"
BACKUP = str(CTRL_FILE) + ".n5_backup"
OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Push measurement script (runs in subprocess to pick up patches) ──────────
# Based on push_ci.py logic, adapted for subprocess execution.
MEASURE_SCRIPT = r'''
import json, sys, time
import numpy as np
import mujoco as mj
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig)
from wheeled_biped.utils.config import get_model_path

DT = 0.01; SUBSTEPS = 5
PUSH_DUR = 7; PUSH_START = 300; POST_PUSH_S = 17.0
POST_PUSH_STEPS = int(POST_PUSH_S / DT)
PITCH_LIMIT = 0.8; HEIGHT_LIMIT = 0.30
FORCE_MIN, FORCE_MAX = 10.0, 160.0
N_BISECT = 8; TOLERANCE = 5.0
N_REPS = 10
ANGLES = [0, 45, 90, 135, 180, -135, -90, -45]

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
param_overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

model = mj.MjModel.from_xml_path(str(get_model_path()))
l_wheel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "r_wheel_link")
robot_mass = float(np.sum(model.body_mass))
torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)

def make_ctx():
    """One context per trial.

    compute_v3_torque_for_state MUTATES this dict: it keeps prev_com_pos and
    the airborne_mode / airborne_count / ground_count flight-mode latch there.
    Sharing one context across bisection iterations lets a trial that ended in
    the air hand the next trial a latched flight mode, so that trial starts
    airborne and dies at almost any force.
    """
    return {"centroidal_estimator": CentroidalStateEstimator(cfg, mj_model=model),
            "initial_yaw_z": 0.0,
            "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
            "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}

def settle(data, v3, steps=300):
    for _ in range(steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], v3["ctx"], teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mj.mj_step(model, data)

def run_push(data, v3, force_N, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    force = np.array([force_N * np.cos(angle_rad), force_N * np.sin(angle_rad), 0.0])
    torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso")
    for step in range(POST_PUSH_STEPS + PUSH_DUR):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], v3["ctx"], teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PUSH_DUR:
            data.xfrc_applied[torso_id, :3] = force
        for _ in range(SUBSTEPS):
            mj.mj_step(model, data)
        quat = data.qpos[3:7].copy()
        pitch = np.arcsin(-2*(quat[1]*quat[3] - quat[0]*quat[2]))
        if abs(pitch) > PITCH_LIMIT or data.subtree_com[0][2] < HEIGHT_LIMIT:
            return False
    return True

def bisect_one_direction(angle_deg, seed):
    rng = np.random.default_rng(seed)
    lo, hi = FORCE_MIN, FORCE_MAX
    best = lo
    for i in range(N_BISECT):
        mid = (lo + hi) / 2
        data = mj.MjData(model)
        perturbed = POSTURE + rng.normal(0.0, 0.005, size=10)
        for j, jname in enumerate(JOINT_NAMES):
            jid = model.joint(jname).id
            lo_j, hi_j = model.jnt_range[jid]
            perturbed[j] = float(np.clip(perturbed[j], lo_j, hi_j))
        data.qpos[7:17] = perturbed
        data.qpos[2] = ROOT_Z + rng.normal(0.0, 0.001)
        mj.mj_forward(model, data)
        v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
        v3["jax_state"] = pack_state_k2()
        v3["ctx"] = make_ctx()
        # Apply parameter overrides
        params = v3["jax_params"]
        for idx, val in param_overrides.items():
            params = params.at[int(idx)].set(float(val))
        v3["jax_params"] = params
        settle(data, v3, steps=PUSH_START)
        if run_push(data, v3, mid, angle_deg):
            best = mid
            lo = mid + TOLERANCE / 2
        else:
            hi = mid - TOLERANCE / 2
    return best

t0 = time.time()
all_reps = {ang: [] for ang in ANGLES}
for rep in range(N_REPS):
    seed = 20260728 * 100 + rep
    for ang in ANGLES:
        th = bisect_one_direction(ang, seed + int(ang))
        all_reps[ang].append(th)
    elapsed = time.time() - t0
    if rep < N_REPS - 1:
        eta = elapsed / (rep+1) * (N_REPS - rep - 1)

# Per-rep F_min and F_med
f_min_vals = []
f_med_vals = []
for rep in range(N_REPS):
    rep_vals = [all_reps[ang][rep] for ang in ANGLES]
    f_min_vals.append(min(rep_vals))
    f_med_vals.append(sorted(rep_vals)[len(rep_vals)//2])

f_min_mean = float(np.mean(f_min_vals))
f_min_std = float(np.std(f_min_vals, ddof=1))
f_med_mean = float(np.mean(f_med_vals))
f_med_std = float(np.std(f_med_vals, ddof=1))
ci95_min = float(2.262 * f_min_std / np.sqrt(N_REPS))  # t-dist for N=10, df=9
ci95_med = float(2.262 * f_med_std / np.sqrt(N_REPS))
elapsed = time.time() - t0

print(json.dumps({
    "n_reps": N_REPS, "n_directions": len(ANGLES),
    "F_min_mean": f_min_mean, "F_min_std": f_min_std,
    "F_med_mean": f_med_mean, "F_med_std": f_med_std,
    "F_min_ci95": ci95_min, "F_med_ci95": ci95_med,
    "f_min_per_rep": f_min_vals, "f_med_per_rep": f_med_vals,
    "all_reps": {str(k): v for k, v in all_reps.items()},
    "elapsed_min": elapsed / 60.0,
}))
'''

# ── Config definitions ───────────────────────────────────────────────────────
# Each config: (id, name, patches, param_overrides)
# Patches are (old_str, new_str) pairs applied to k2_jax_controller.py.
# param_overrides are {param_index: value} applied after controller init.
# Patch strings verified against k2_jax_controller.py @ f92640c.

CONFIGS = [
    # ── L2: L1 + global integral (no prox gate, no envelope) ──
    {
        "id": "L2",
        "name": "L2: +Integral K_i (no gate)",
        "patches": [
            # Remove proximity gate on integral application
            ("_anchor_integ_applied = _anchor_integ * _anchor_prox",
             "_anchor_integ_applied = _anchor_integ  # N5: GLOBAL I (no prox)"),
            # Disable envelope gate (anchor_quiet = always 1.0)
            ('_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))',
             '_anchor_quiet = 1.0  # N5: no envelope gate'),
        ],
        # Paper-era low damping + no boost + fixed k_p=50
        "param_overrides": {32: 1.0, 54: 6.0, 91: 0.0, 94: 50.0},
    },
    # ── L2.5: L2 + proximity gate only (decomposes L2→L3 additive confound) ──
    # Isolates the proximity gate's contribution: integral is gated by spatial
    # proximity (±15cm) but envelope follower and damping boost are disabled.
    # Differs from L2: keeps _anchor_prox gate on integral.
    # Differs from L3: no envelope gate, no damping boost.
    {
        "id": "L2_5",
        "name": "L2.5: +Proximity gate only",
        "patches": [
            # Keep proximity gate on integral (DON'T patch it out — key diff vs L2)
            # Only disable envelope gate (anchor_quiet = always 1.0)
            ('_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))',
             '_anchor_quiet = 1.0  # L2.5: no envelope gate (prox gate only)'),
        ],
        # Same as L2: paper-era low damping + no boost + fixed k_p=50
        "param_overrides": {32: 1.0, 54: 6.0, 91: 0.0, 94: 50.0},
    },
    # ── S0: Full ACC with scheduled k_p (default V3_ANCHOR, rejected variant) ──
    {
        "id": "S0",
        "name": "S0: ACC + scheduled k_p (rejected)",
        "patches": [],
        "param_overrides": {},  # default scheduled k_p
    },
    # ── S1: Full ACC minus scheduled k_p (canonical, fixed k_p=50, = L3) ──
    {
        "id": "S1",
        "name": "S1: -Scheduled k_p (fixed k_p=50)",
        "patches": [],
        "param_overrides": {94: 50.0},  # force k_p=50 regardless of gates
    },
    # ── S2: Full ACC minus damping boost ──
    {
        "id": "S2",
        "name": "S2: -Damping boost",
        "patches": [],
        "param_overrides": {91: 0.0},  # zero anchor velocity boost
    },
    # ── S3: Full ACC with symmetric EMA ──
    {
        "id": "S3",
        "name": "S3: -Asym. EMA (sym. EMA)",
        "patches": [
            ("jnp.where(_act_dev > 0.0, 0.35, 0.0067) * _act_dev",
             "0.0067 * _act_dev  # N5: symmetric EMA (slow release only)"),
        ],
        "param_overrides": {},
    },
    # ── S4: Full ACC minus prox gate (global I, keeps EMA + boost + sched k_p) ──
    {
        "id": "S4",
        "name": "S4: -Prox. gate (global I, envelope-gated)",
        "patches": [
            ("_anchor_integ_applied = _anchor_integ * _anchor_prox",
             "_anchor_integ_applied = _anchor_integ  # N5: no prox gate"),
        ],
        "param_overrides": {},
    },
    # ── S5: Full ACC minus pitch stability gate g_θ ──
    {
        "id": "S5",
        "name": "S5: -Pitch gate g_θ",
        "patches": [
            ("* _anchor_stab)",
             "* 1.0)  # N5: was _anchor_stab (pitch gate removed)"),
        ],
        "param_overrides": {},
    },
]


def _clear_cache():
    """Remove cached .pyc files for k2 modules so subprocess picks up patches."""
    pkg = ROOT / "wheeled_biped"
    for root, dirs, files in os.walk(str(pkg)):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            for f in os.listdir(cache_dir):
                if any(k in f for k in ["k2_jax_controller", "offline_three_arm",
                                         "sagittal_velocity", "centroidal"]):
                    os.remove(os.path.join(cache_dir, f))


def apply_patches(patches: list[tuple[str, str]]):
    """Apply string-replacement patches to k2_jax_controller.py."""
    shutil.copy2(str(CTRL_FILE), BACKUP)
    with open(CTRL_FILE) as f:
        content = f.read()
    for old, new in patches:
        if old not in content:
            shutil.copy2(BACKUP, str(CTRL_FILE))
            os.remove(BACKUP)
            raise ValueError(f"Patch string not found:\n  {old[:120]}...")
        content = content.replace(old, new)
    with open(CTRL_FILE, "w") as f:
        f.write(content)
    _clear_cache()


def restore_original():
    """Restore k2_jax_controller.py from backup."""
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, str(CTRL_FILE))
        os.remove(BACKUP)
        _clear_cache()


def run_measurement(param_overrides: dict) -> dict | None:
    """Run N=5 push sweep in subprocess. Returns dict or None."""
    param_json = json.dumps(param_overrides)
    try:
        result = subprocess.run(
            [sys.executable, "-c", MEASURE_SCRIPT, param_json],
            capture_output=True, text=True, timeout=1800,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"    FAILED (rc={result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr[-500:]}")
            return None
        # Find the JSON line in output
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith('{"'):
                return json.loads(line)
        print(f"    No JSON in output. stdout tail: {result.stdout[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        print("    TIMEOUT (30 min)")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="N≥5 replication for ablation rows L2 and S1–S5")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config IDs to run (default: all)")
    parser.add_argument("--tag", default="",
                        help="suffix for the output filenames, so a partial "
                             "re-run does not clobber a full previous result")
    args = parser.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    configs = [c for c in CONFIGS
               if args.configs is None or c["id"] in args.configs]

    print("=" * 72)
    print(f"R1 FIX: N≥5 Ablation Replication — {len(configs)} configs")
    print(f"N=10 reps × 8 dirs × 8 bisect iters per config")
    print(f"Est. time: ~{len(configs) * 20} min")
    print("=" * 72)

    results = {}
    t_start = time.time()

    for i, cfg in enumerate(configs):
        print(f"\n{'─'*72}")
        print(f"[{i+1}/{len(configs)}] {cfg['name']}")
        print(f"  Patches: {len(cfg['patches'])}, Params: {cfg['param_overrides']}")

        # Apply patches
        patch_error = False
        if cfg["patches"]:
            try:
                apply_patches(cfg["patches"])
                print("  Patches applied OK")
            except ValueError as e:
                print(f"  PATCH FAILED: {e}")
                patch_error = True

        if patch_error:
            restore_original()
            results[cfg["id"]] = {"error": "patch_failed"}
            continue

        # Run measurement
        print(f"  Running push sweep (N=10)...", end=" ", flush=True)
        t0 = time.time()
        data = run_measurement(cfg["param_overrides"])
        dt = time.time() - t0

        # Restore original immediately
        if cfg["patches"]:
            restore_original()

        if data:
            print(f"OK ({dt/60:.1f}m)")
            print(f"  F_min = {data['F_min_mean']:.1f} ± {data['F_min_std']:.1f} N "
                  f"(95%CI ±{data['F_min_ci95']:.1f})")
            print(f"  F_med = {data['F_med_mean']:.1f} ± {data['F_med_std']:.1f} N "
                  f"(95%CI ±{data['F_med_ci95']:.1f})")
            results[cfg["id"]] = data
        else:
            print(f"FAILED ({dt/60:.1f}m)")
            results[cfg["id"]] = {"error": "measurement_failed"}

        # Save incrementally
        out_path = OUT_DIR / f"ablation_n10_partial{tag}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Final restore (belt-and-suspenders)
    restore_original()

    # ── Final save ──
    out_path = OUT_DIR / f"ablation_n10_results{tag}.json"
    results["_metadata"] = {
        "n_reps": 10,
        "n_directions": 8,
        "binary_search_tol_N": 5.0,
        "binary_search_iters": 8,
        "force_range_N": [10.0, 160.0],
        "total_elapsed_min": (time.time() - t_start) / 60.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": "f92640c",
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Summary ──
    print(f"\n{'='*72}")
    print("N=10 ABLATION RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"{'Config':<6} {'F_min':>8} {'±std':>8} {'95%CI':>8}  {'F_med':>8} {'±std':>8} {'95%CI':>8}")
    print("-" * 72)
    for cfg in configs:
        cid = cfg["id"]
        if cid in results and "error" not in results[cid]:
            r = results[cid]
            print(f"{cid:<6} {r['F_min_mean']:>8.1f} {r['F_min_std']:>8.1f} "
                  f"{r['F_min_ci95']:>8.1f}  {r['F_med_mean']:>8.1f} "
                  f"{r['F_med_std']:>8.1f} {r['F_med_ci95']:>8.1f}")
        else:
            print(f"{cid:<6} {'FAILED':>40}")

    print(f"\nSaved → {out_path}")
    print(f"Total time: {results['_metadata']['total_elapsed_min']:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
