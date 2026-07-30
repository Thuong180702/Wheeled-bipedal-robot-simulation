#!/usr/bin/env python3
"""
R4: Decompose the L2→L3 bundled ablation.

Creates intermediate configurations between L2 (global integral, no gate)
and L3 (full anchor: prox gate + asym EMA + damping boost + scheduled k_p)
to attribute the 55mm→0.3mm idle RMS improvement to specific sub-components.

Configurations measured:
  M1: L2 + Prox gate only        (no envelope, no boost, fixed k_p=50)
  M2: L2 + Prox gate + sym EMA   (no boost, fixed k_p=50)
  M3: L2 + Prox gate + asym EMA  (no boost, fixed k_p=50)
  M4: Full ACC minus pitch stability gate g_θ

Approach: patches k2_jax_controller.py, runs measurement in a clean subprocess,
then restores. Same architecture as scripts/run_structural_ablations.py.

Usage:
  python scripts/run_r4_decompose_ablation.py
  python scripts/run_r4_decompose_ablation.py --push  # also measure push thresholds (SLOW)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTRL_FILE = ROOT / "wheeled_biped/controllers/k2_jax_controller.py"
BACKUP = str(CTRL_FILE) + ".r4_backup"
OUT_DIR = ROOT / "outputs" / "r4_decompose_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Measurement script template ──────────────────────────────────────────────────
# Runs in a subprocess to pick up patched k2_jax_controller.py cleanly.
MEASURE_SCRIPT = r'''
import json, os, sys
import numpy as np
import mujoco
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.utils.config import get_model_path

DT = 0.01; SUBSTEPS = 5; TOTAL_S = 25.0; SETTLE_S = 5.0
N_TRIALS = 5
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"

# Load nominal
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
h0 = float(nom["target_com_z_m"])
posture = np.array([
    nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
root_z = float(nom["calibrated_root_z_m"])

model = mujoco.MjModel.from_xml_path(str(get_model_path()))

# Build context
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig)
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
robot_mass = float(np.sum(model.body_mass))
torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
centroidal_config = CentroidalStateEstimatorConfig(
    robot_mass=robot_mass, torso_inertia=torso_inertia)
centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model=model)
ctx = {
    "centroidal_estimator": centroidal_estimator,
    "initial_yaw_z": 0.0,
    "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
    "eq_joint": posture, "height_ref": h0, "prev_com_pos": None,
}

# Param overrides from argv
param_overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

n_steps = int(TOTAL_S / DT)
settle_start = int(SETTLE_S / DT)
_joint_names = [
    "l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
    "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]
trial_rms_values = []

for trial in range(N_TRIALS):
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    params = v3["jax_params"]
    for idx, val in param_overrides.items():
        params = params.at[int(idx)].set(float(val))
    v3["jax_params"] = params

    data = mujoco.MjData(model)
    rng = np.random.default_rng(42 + trial)
    perturbed = posture + rng.normal(0.0, 0.005, size=10)
    for j, jname in enumerate(_joint_names):
        jid = model.joint(jname).id
        lo, hi = model.jnt_range[jid]
        perturbed[j] = float(np.clip(perturbed[j], lo, hi))
    data.qpos[7:17] = perturbed
    data.qpos[2] = root_z + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)

    com_x = np.zeros(n_steps)
    for step in range(n_steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"],
            v3["jax_state"], v3["jax_params"],
            ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        com_x[step] = float(data.subtree_com[0][0])

    settled = com_x[settle_start:]
    rms = float(np.sqrt(np.mean((settled - settled.mean())**2)) * 1000.0)
    trial_rms_values.append(rms)

mean_rms = float(np.mean(trial_rms_values))
std_rms = float(np.std(trial_rms_values, ddof=1)) if N_TRIALS > 1 else 0.0
try:
    from scipy import stats
    ci = stats.t.interval(0.95, N_TRIALS-1, loc=mean_rms, scale=std_rms/np.sqrt(N_TRIALS))
    ci_lo, ci_hi = float(ci[0]), float(ci[1])
except ImportError:
    ci_lo = mean_rms - 2.776*std_rms/np.sqrt(N_TRIALS)
    ci_hi = mean_rms + 2.776*std_rms/np.sqrt(N_TRIALS)

print(json.dumps({
    "mean_rms_mm": mean_rms, "std_rms_mm": std_rms,
    "ci95_lo_mm": ci_lo, "ci95_hi_mm": ci_hi,
    "trials": trial_rms_values, "n_trials": N_TRIALS,
}))
'''


def _clear_cache():
    """Remove cached .pyc files for k2 modules so subprocess picks up patches."""
    for root, dirs, files in os.walk(str(ROOT / "wheeled_biped")):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            for f in os.listdir(cache_dir):
                if "k2_jax_controller" in f or "offline_three_arm" in f or "sagittal_velocity" in f:
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
            raise ValueError(f"Patch string not found:\n  {old[:100]}...")
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
    """Run idle RMS measurement in subprocess. Returns dict or None."""
    param_json = json.dumps(param_overrides)
    try:
        result = subprocess.run(
            [sys.executable, "-c", MEASURE_SCRIPT, param_json],
            capture_output=True, text=True, timeout=600,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"    FAILED (rc={result.returncode}): {result.stderr[-300:]}")
            return None
        # Output may have both print() and the final JSON; take last line
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        print(f"    No JSON in output. stdout: {result.stdout[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        print("    TIMEOUT")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def run_push_sweep(param_overrides: dict) -> dict | None:
    """Run quick 8-direction push sweep. Returns F_min, F_med or None."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        sys.executable,
        str(ROOT / "scripts/push_sweep_paper.py"),
        "--quick", "--output", tmp_path,
    ]
    for idx, val in param_overrides.items():
        cmd.extend(["--param", str(idx), str(val)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"    Push sweep FAILED: {result.stderr[-300:]}")
            return None
        with open(tmp_path) as f:
            data = json.load(f)
        os.unlink(tmp_path)
        return {"F_min_N": data["F_min_N"], "F_med_N": data["F_med_N"], "F_max_N": data["F_max_N"]}
    except Exception as e:
        print(f"    Push sweep ERROR: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


# ── Configuration definitions ────────────────────────────────────────────────────
def define_configs():
    """Return ordered list of (id, name, description, patches, param_overrides)."""
    # Damping levels:
    #   V3_ANCHOR:     velocity_damping_scale (idx 32) = 1.5,  drift_k_vel (idx 54) = 15.0
    #   V3_HOMING:     velocity_damping_scale = 1.10,  drift_k_vel = 10.0
    #   V3 base:       velocity_damping_scale = 1.10,  drift_k_vel = 10.0
    #   Paper L0-L2:   velocity_damping_scale ≈ 1.0 (pre-V3), drift_k_vel = 6.0 (V1)
    #
    # The paper's L2 (~55mm limit cycle) was measured on a pre-V3 profile with
    # lower base velocity damping. Modern V3_HOMING has higher damping (1.10) that
    # already suppresses most of the limit cycle. We measure both regimes.
    PAPER_L2_PARAMS = {32: 1.0, 54: 6.0}     # Pre-V3 damping (paper's L0-L2 baseline)
    HOMING_L2_PARAMS = {32: 1.10, 54: 10.0}  # V3_HOMING-level damping

    return [
        # ═══════════════════════════════════════════════════════════════════════
        # L2 BASELINES (paper-era low damping vs modern V3_HOMING damping)
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "L2_paper_level",
            "name": "L2: Global integral (paper damping: vd=1.0)",
            "desc": "Global integral, no gate/envelope/boost, k_p=50, paper-era low base damping (vd=1.0). Reproduces ~55mm limit cycle.",
            "patches": [
                ("_anchor_integ_applied = _anchor_integ * _anchor_prox",
                 "_anchor_integ_applied = _anchor_integ  # R4: GLOBAL I"),
                ("_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))",
                 "_anchor_quiet = 1.0  # R4: no envelope"),
            ],
            "param_overrides": {**PAPER_L2_PARAMS, 91: 0.0, 94: 50.0},
        },
        {
            "id": "L2_homing_level",
            "name": "L2: Global integral (V3_HOMING damping: vd=1.10)",
            "desc": "Global integral, no gate/envelope/boost, k_p=50, V3_HOMING-level damping (vd=1.10). Modern damping largely suppresses the limit cycle.",
            "patches": [
                ("_anchor_integ_applied = _anchor_integ * _anchor_prox",
                 "_anchor_integ_applied = _anchor_integ  # R4: GLOBAL I"),
                ("_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))",
                 "_anchor_quiet = 1.0  # R4: no envelope"),
            ],
            "param_overrides": {**HOMING_L2_PARAMS, 91: 0.0, 94: 50.0},
        },
        # ═══════════════════════════════════════════════════════════════════════
        # M1: Proximity gate in isolation (paper-era damping to show contrast)
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "M1_prox_gate_only",
            "name": "M1: Prox gate only (paper vd=1.0, no envelope/boost/k_p sched)",
            "desc": "Prox-gated integral (±5-15cm), no EMA, no boost, k_p=50, paper-era damping.",
            "patches": [
                ("_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))",
                 "_anchor_quiet = 1.0  # R4: no envelope"),
            ],
            "param_overrides": {**PAPER_L2_PARAMS, 91: 0.0, 94: 50.0},
        },
        # ═══════════════════════════════════════════════════════════════════════
        # M2: Prox gate + symmetric EMA
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "M2_prox_sym_ema",
            "name": "M2: Prox gate + sym EMA, no boost, k_p=50",
            "desc": "Prox-gated integral + symmetric (slow) EMA, no boost, k_p=50.",
            "patches": [
                ("jnp.where(_act_dev > 0.0, 0.35, 0.0067) * _act_dev",
                 "0.0067 * _act_dev  # R4: symmetric EMA"),
            ],
            "param_overrides": {**HOMING_L2_PARAMS, 91: 0.0, 94: 50.0},
        },
        # ═══════════════════════════════════════════════════════════════════════
        # M3: Prox gate + asymmetric EMA (no boost = what EMA alone contributes)
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "M3_prox_asym_ema",
            "name": "M3: Prox gate + asym EMA, no boost, k_p=50",
            "desc": "Prox-gated integral + asymmetric EMA, no damping boost, k_p=50.",
            "patches": [],
            "param_overrides": {**HOMING_L2_PARAMS, 91: 0.0, 94: 50.0},
        },
        # ═══════════════════════════════════════════════════════════════════════
        # M4: Full ACC minus pitch stability gate g_θ (critical test)
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "M4_no_pitch_stab",
            "name": "M4: Full ACC minus pitch stability gate g_θ",
            "desc": "All anchor components EXCEPT pitch/pitch-rate gate. Boost active during oscillation → parametric amplification.",
            "patches": [
                ("* _anchor_stab)",
                 "* 1.0)  # R4: was _anchor_stab"),
            ],
            "param_overrides": {},
        },
        # ═══════════════════════════════════════════════════════════════════════
        # M5: Damping boost without k_p schedule (isolates boost contribution)
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "M5_boost_fixed_kp",
            "name": "M5: Full ACC minus k_p schedule (boost active, k_p=50)",
            "desc": "Prox gate + asym EMA + damping boost active + fixed k_p=50. Isolates the boost's idle RMS contribution. Equivalent to paper S1.",
            "patches": [],
            "param_overrides": {94: 50.0},  # fixed k_p, keep boost at anchor level (5.0)
        },
        # ═══════════════════════════════════════════════════════════════════════
        # L3: Full ACC reference
        # ═══════════════════════════════════════════════════════════════════════
        {
            "id": "L3_full_acc",
            "name": "L3: Full ACC (reference)",
            "desc": "Full anchor: prox gate + asym EMA + damping boost + scheduled k_p.",
            "patches": [],
            "param_overrides": {},
        },
    ]


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="R4: Decompose L2→L3 bundled ablation")
    parser.add_argument("--push", action="store_true",
                        help="Also measure push thresholds (8-dir sweep, ~10 min/config)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config IDs to run (default: all)")
    args = parser.parse_args()

    all_configs = define_configs()
    configs = [c for c in all_configs if args.configs is None or c["id"] in args.configs]

    print("=" * 72)
    print("R4: Decompose L2→L3 Bundled Ablation")
    print(f"Configs: {len(configs)} | Trials: 5 × 20s | Push: {'YES' if args.push else 'NO'}")
    print("=" * 72)

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'─'*72}")
        print(f"[{i+1}/{len(configs)}] {cfg['name']}")
        print(f"  Patches: {len(cfg['patches'])}, Params: {cfg['param_overrides']}")

        # Apply patches
        if cfg["patches"]:
            try:
                apply_patches(cfg["patches"])
                print("  Patches applied OK")
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

        # Measure idle RMS (subprocess picks up patched controller)
        print("  Measuring idle CoM RMS...", end=" ", flush=True)
        idle = run_measurement(cfg["param_overrides"])
        if idle:
            print(f"{idle['mean_rms_mm']:.3f} ± {idle['std_rms_mm']:.3f} mm "
                  f"(N={idle['n_trials']}, 95%CI [{idle['ci95_lo_mm']:.3f}, {idle['ci95_hi_mm']:.3f}])")
        else:
            print("FAILED")

        # Restore original
        restore_original()

        # Optionally measure push thresholds
        push = None
        if args.push:
            print("  Measuring push thresholds...", end=" ", flush=True)
            push = run_push_sweep(cfg["param_overrides"])
            if push:
                print(f"F_min={push['F_min_N']:.0f}N F_med={push['F_med_N']:.0f}N")
            else:
                print("FAILED")

        results.append({
            "id": cfg["id"], "name": cfg["name"], "desc": cfg["desc"],
            "idle_rms": idle, "push": push, "n_patches": len(cfg["patches"]),
        })

    # Final restore (belt-and-suspenders)
    restore_original()

    # ── Summary ──
    print(f"\n{'='*72}")
    print("R4 ABLATION SUMMARY — Idle CoM X RMS (mm)")
    print(f"{'='*72}")
    print(f"{'Config':<48} {'RMS':>8} {'±std':>8} {'95% CI':>24}")
    print("-" * 72)
    for r in results:
        ir = r["idle_rms"]
        if ir:
            print(f"{r['name']:<48} {ir['mean_rms_mm']:>7.3f} {ir['std_rms_mm']:>7.3f}  "
                  f"[{ir['ci95_lo_mm']:.3f}, {ir['ci95_hi_mm']:.3f}]")
        else:
            print(f"{r['name']:<48} {'FAILED':>16}")

    print("-" * 72)

    # Attribution
    l2 = next((r for r in results if r["id"] == "L2_baseline" and r["idle_rms"]), None)
    l3 = next((r for r in results if r["id"] == "L3_full_acc" and r["idle_rms"]), None)
    if l2 and l3:
        l2_rms = l2["idle_rms"]["mean_rms_mm"]
        l3_rms = l3["idle_rms"]["mean_rms_mm"]
        print(f"\nΔ Attribution (L2→L3 total: {l2_rms:.2f} → {l3_rms:.2f} mm, "
              f"Δ = {l2_rms - l3_rms:.2f} mm, {(l2_rms/l3_rms):.0f}× improvement)")
        for r in results:
            if r["id"] not in ("L2_baseline", "L3_full_acc") and r["idle_rms"]:
                m = r["idle_rms"]["mean_rms_mm"]
                delta = l2_rms - m
                pct = delta / l2_rms * 100 if l2_rms > 0 else 0
                print(f"  {r['id']}: {m:.2f} mm → captures {pct:.0f}% of "
                      f"{l2_rms - l3_rms:.2f} mm total improvement")

    # Save
    out_path = OUT_DIR / "r4_decompose_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
