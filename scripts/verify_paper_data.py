#!/usr/bin/env python3
"""Comprehensive paper data verification — generates fresh JSON for all tables.

Usage:
  mjpython scripts/verify_paper_data.py --all
  mjpython scripts/verify_paper_data.py --drop
  mjpython scripts/verify_paper_data.py --ledge
  mjpython scripts/verify_paper_data.py --curb
  mjpython scripts/verify_paper_data.py --standing
  mjpython scripts/verify_paper_data.py --push  (re-runs push sweeps)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import existing test functions ──
sys.path.insert(0, str(ROOT))
import scripts.drop_recovery_tests as drop_mod
import scripts.ramp_step_tests as ramp_mod


def run_drop_suite():
    """Run drop recovery tests for all heights in paper (Table IV)."""
    print("\n" + "="*70)
    print("DROP RECOVERY TESTS (Table IV)")
    print("="*70)
    results = []
    # Heights from paper: stress test (100, 60cm) + envelope
    heights_cm = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    t0 = time.time()
    for h_cm in heights_cm:
        h_m = h_cm / 100.0
        r = drop_mod.run_drop(h_m, tilt_pitch_deg=0.0, duration_s=12.0)
        results.append(r)
        if r.get("fell"):
            print(f"  {h_cm:3d}cm: FALL at {r['fall_t']:.1f}s  td_vz={r.get('touchdown_vz',0):.1f}m/s")
        else:
            st = f"{r['settle_s']:.1f}s" if r['settle_s'] else "never"
            print(f"  {h_cm:3d}cm: {r['verdict']:>4s}  peak_pitch={r['peak_pitch']:5.1f}°  "
                  f"settle={st}  drift={r['drift_m']:.3f}m")
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s")

    # Save JSON
    out = {"test": "drop_recovery", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "num_heights": len(heights_cm), "results": results}
    path = OUT_DIR / "drop_recovery.json"
    json.dump(out, path.open("w"), indent=2, default=str)
    print(f"  Saved: {path}")
    return results


def run_ledge_suite():
    """Run ramp-step (ledge drive-off) tests for paper Table IV."""
    print("\n" + "="*70)
    print("LEDGE DRIVE-OFF TESTS (Table IV)")
    print("="*70)
    results = []
    heights_cm = [20, 30, 40, 50]
    t0 = time.time()
    for h_cm in heights_cm:
        h_m = h_cm / 100.0
        r = ramp_mod.run_ramp_step(h_m, duration_s=30.0, course="up_off")
        results.append(r)
        if r.get("fell"):
            print(f"  {h_cm:2d}cm: FALL at {r['fall_t']:.1f}s")
        else:
            st = f"{r['settle_s']:.1f}s" if r['settle_s'] else "never"
            tr = f"{r['t_release']:.1f}s" if r['t_release'] else "—"
            print(f"  {h_cm:2d}cm: {r['verdict']:>4s}  peak_pitch_land={r['peak_pitch_land']:5.1f}°  "
                  f"settle={st}  t_release={tr}")
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s")

    out = {"test": "ledge_drive_off", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "num_heights": len(heights_cm), "results": results}
    path = OUT_DIR / "ledge_drive_off.json"
    json.dump(out, path.open("w"), indent=2, default=str)
    print(f"  Saved: {path}")
    return results


def run_curb_suite():
    """Run one-wheel curb tests for paper Table V."""
    print("\n" + "="*70)
    print("CURB/TERRAIN TESTS (Table V)")
    print("="*70)
    results = []
    heights_cm = [10, 15, 20]
    t0 = time.time()
    for h_cm in heights_cm:
        h_m = h_cm / 100.0
        r = ramp_mod.run_curb(h_m, duration_s=30.0)
        results.append(r)
        if r.get("fell"):
            print(f"  {h_cm:2d}cm: FALL at {r['fall_t']:.1f}s")
        else:
            st = f"{r['settle_s']:.1f}s" if r['settle_s'] else "never"
            print(f"  {h_cm:2d}cm: {r['verdict']:>4s}  roll_curb_max={r['roll_curb_max']:5.1f}°  "
                  f"d_max={r['d_max']*100:4.1f}cm  settle={st}")
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s")

    out = {"test": "curb_terrain", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "num_heights": len(heights_cm), "results": results}
    path = OUT_DIR / "curb_terrain.json"
    json.dump(out, path.open("w"), indent=2, default=str)
    print(f"  Saved: {path}")
    return results


def _run_one_idle_trial(profile, model, nom, posture, h0, P, trial_id, seed):
    """Single standing idle trial. Returns result dict or None if fell."""
    import mujoco
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        compute_v3_torque_for_state, init_v3_controller)
    from wheeled_biped.controllers.k2_jax_controller import pack_state_k2

    DT = 0.01; SUBSTEPS = 5; TOTAL_S = 25.0
    rng = np.random.default_rng(seed)

    v3 = dict(init_v3_controller(profile_name=profile, model=model))
    v3["jax_state"] = pack_state_k2()
    data = mujoco.MjData(model)
    # Randomize initial joint positions ±0.005 rad (paper methodology)
    perturbed = posture + rng.normal(0.0, 0.005, size=10)
    joint_names = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
                   "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]
    for j, jname in enumerate(joint_names):
        jid = model.joint(jname).id
        lo, hi = model.jnt_range[jid]
        perturbed[j] = float(np.clip(perturbed[j], lo, hi))
    data.qpos[7:17] = perturbed
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)
    ctx = P._build_v3_controller_context(
        model, data, v3, eq_joint=posture, height_ref=h0)

    n_steps = int(TOTAL_S / DT)
    home_x = float(data.qpos[0])
    home_y = float(data.qpos[1])

    logs = {"com_x_mm": np.zeros(n_steps), "com_y_mm": np.zeros(n_steps),
            "com_z_m": np.zeros(n_steps), "pitch_deg": np.zeros(n_steps),
            "vx": np.zeros(n_steps)}

    for step in range(n_steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        q = data.qpos[3:7]
        pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2]-q[3]*q[1]), -1, 1)))
        logs["com_x_mm"][step] = (data.qpos[0] - home_x) * 1000
        logs["com_y_mm"][step] = (data.qpos[1] - home_y) * 1000
        logs["com_z_m"][step] = data.subtree_com[0][2]
        logs["pitch_deg"][step] = pitch
        logs["vx"][step] = data.qvel[0]

        if abs(pitch) > 46 or data.qpos[2] < 0.15:
            return None  # fell

    # Anchor needs ~5s for EMA init transient to decay; 20s measurement window
    settle_start = int(5.0 / DT)
    window_samples = int(20.0 / DT)
    x_win = logs["com_x_mm"][settle_start:settle_start + window_samples]
    y_win = logs["com_y_mm"][settle_start:settle_start + window_samples]
    pitch_win = logs["pitch_deg"][settle_start:settle_start + window_samples]

    return {
        "trial": trial_id,
        "survived": True,
        "com_x_rms_mm": float(np.sqrt(np.mean((x_win - np.mean(x_win))**2))),
        "com_y_rms_mm": float(np.sqrt(np.mean((y_win - np.mean(y_win))**2))),
        "com_x_p2p_mm": float(np.max(x_win) - np.min(x_win)),
        "pitch_rms_deg": float(np.sqrt(np.mean(pitch_win**2))),
    }


def run_standing_idle(n_trials=1):
    """Run standing idle precision test for both ACC and P-only (Table II).

    Args:
        n_trials: Number of independent trials per profile (paper uses N=10).
    """
    import mujoco
    import scripts.promote_v3_vs_assist as P
    from wheeled_biped.teleop_shaper import HeightPosture

    print("\n" + "="*70)
    print(f"STANDING IDLE TESTS (Table II) — N={n_trials} per profile")
    print("="*70)

    profiles = {
        "ACC": "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR",
        "P-only": "K2_JAX_DEDICATED_DEFAULT_V3",
    }

    DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
    nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
    h0 = float(nom["target_com_z_m"])
    posture = np.array([
        nom["hip_roll_left"], nom["hip_yaw_left"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
        nom["hip_roll_right"], nom["hip_yaw_right"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])

    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    base_seed = 20260727  # paper-documented seed

    all_results = {}

    for name, profile in profiles.items():
        print(f"\n  {name} ({profile})...")
        trials = []
        for trial_id in range(n_trials):
            seed = base_seed + trial_id
            result = _run_one_idle_trial(profile, model, nom, posture, h0, P,
                                         trial_id, seed)
            if result is None:
                print(f"    Trial {trial_id}: FELL")
                trials.append({"trial": trial_id, "survived": False})
            else:
                print(f"    Trial {trial_id}: CoM X RMS={result['com_x_rms_mm']:.2f}mm  "
                      f"P2P={result['com_x_p2p_mm']:.1f}mm  "
                      f"Pitch RMS={result['pitch_rms_deg']:.2f}°")
                trials.append(result)

        # Aggregate across trials
        survived = [t for t in trials if t.get("survived")]
        if survived:
            x_rms_vals = np.array([t["com_x_rms_mm"] for t in survived])
            x_p2p_vals = np.array([t["com_x_p2p_mm"] for t in survived])
            pitch_vals = np.array([t["pitch_rms_deg"] for t in survived])
            all_results[name] = {
                "profile": profile,
                "n_trials": n_trials,
                "n_survived": len(survived),
                "com_x_rms_mm_mean": float(np.mean(x_rms_vals)),
                "com_x_rms_mm_std": float(np.std(x_rms_vals, ddof=1)),
                "com_x_rms_mm_ci95": float(2.262 * np.std(x_rms_vals, ddof=1) / np.sqrt(len(survived))) if len(survived) > 1 else None,
                "com_x_p2p_mm_mean": float(np.mean(x_p2p_vals)),
                "com_x_p2p_mm_std": float(np.std(x_p2p_vals, ddof=1)),
                "pitch_rms_deg_mean": float(np.mean(pitch_vals)),
                "pitch_rms_deg_std": float(np.std(pitch_vals, ddof=1)),
                "trials": trials,
            }
            s = all_results[name]
            print(f"    AGGREGATE: {s['com_x_rms_mm_mean']:.2f}±{s['com_x_rms_mm_std']:.2f}mm  "
                  f"95%CI ±{s['com_x_rms_mm_ci95']:.2f}mm  P2P={s['com_x_p2p_mm_mean']:.1f}mm")
        else:
            all_results[name] = {"profile": profile, "n_trials": n_trials,
                                 "n_survived": 0, "trials": trials}
            print(f"    ALL FELL ({n_trials}/{n_trials})")

    out = {"test": "standing_idle", "n_trials": n_trials,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "results": all_results}
    path = OUT_DIR / "standing_idle.json"
    json.dump(out, path.open("w"), indent=2, default=str)
    print(f"\n  Saved: {path}")
    return all_results


def run_push_verification():
    """Verify existing push sweep data and re-run if needed."""
    print("\n" + "="*70)
    print("PUSH SWEEP VERIFICATION (Table III)")
    print("="*70)

    existing = sorted(ROOT.glob("outputs/push_sweep_*.json"))
    configs = {
        "push_sweep_acc_final2.json": ("Full ACC", 80, 113),
        "push_sweep_homing_v2.json": ("V3_HOMING", 84, 134),
        "push_sweep_v3base_v2.json": ("V3 base (P-only)", 80, 136),
        "push_sweep_no_integral.json": ("No integral (Ki=0)", 82, 124),
        "push_sweep_no_boost.json": ("No damping boost", 81, 118),
        "push_sweep_fixed_kp50.json": ("Fixed kp=50", 81, 113),
        "push_sweep_sym_ema.json": ("Symmetric EMA", 81, 109),
        "push_sweep_global_i.json": ("Global I", 80, 118),
    }

    verified = 0
    missing = 0
    for fname, (label, expected_fmin, expected_fmed) in configs.items():
        path = ROOT / "outputs" / fname
        if path.exists():
            d = json.load(path.open())
            fmin = d["F_min_N"]
            fmed = d["F_med_N"]
            fmin_ok = abs(fmin - expected_fmin) < 2  # within 2N rounding
            fmed_ok = abs(fmed - expected_fmed) < 2
            status = "✅" if (fmin_ok and fmed_ok) else "⚠️"
            print(f"  {status} {label:25s}: F_min={fmin:.0f}N (paper:{expected_fmin}N)  "
                  f"F_med={fmed:.0f}N (paper:{expected_fmed}N)")
            if fmin_ok and fmed_ok:
                verified += 1
        else:
            print(f"  ❌ {label:25s}: FILE MISSING: {fname}")
            missing += 1

    print(f"\n  Verified: {verified}/{len(configs)}  Missing: {missing}")
    return verified, missing


def compare_with_paper():
    """Cross-check all fresh results against paper claims."""
    print("\n" + "="*70)
    print("PAPER COMPARISON")
    print("="*70)

    issues = []

    # ── Table II: Standing ──
    standing_path = OUT_DIR / "standing_idle.json"
    if standing_path.exists():
        d = json.load(standing_path.open())
        acc = d["results"].get("ACC", {})
        ponly = d["results"].get("P-only", {})

        if acc.get("n_survived", 0) > 0:
            rms = acc.get("com_x_rms_mm_mean", acc.get("com_x_rms_mm", 0))
            rms_std = acc.get("com_x_rms_mm_std", 0)
            if rms < 1.0:
                print(f"  ✅ ACC idle CoM RMS: {rms:.2f}±{rms_std:.2f}mm (paper: 0.56±0.08mm)")
            else:
                print(f"  ⚠️  ACC idle CoM RMS: {rms:.2f}±{rms_std:.2f}mm (paper: 0.56±0.08mm)")
                issues.append(f"ACC idle RMS: measured {rms:.2f}mm vs paper 0.56mm")

        if ponly.get("n_survived", 0) > 0:
            rms = ponly.get("com_x_rms_mm_mean", ponly.get("com_x_rms_mm", 0))
            if 20 < rms < 80:
                print(f"  ✅ P-only idle CoM RMS: {rms:.0f}mm (paper: ~39mm RMS)")
            else:
                print(f"  ⚠️  P-only idle CoM RMS: {rms:.0f}mm (paper: ~39mm RMS)")
                issues.append(f"P-only idle RMS: measured {rms:.0f}mm vs paper ~39mm RMS")

    # ── Table IV: Drop ──
    drop_path = OUT_DIR / "drop_recovery.json"
    if drop_path.exists():
        d = json.load(drop_path.open())
        for r in d["results"]:
            h = r.get("h_cm", 0)
            if h == 100 and not r.get("fell"):
                pp = r.get("peak_pitch", 0)
                if 20 < pp < 30:
                    print(f"  ✅ Drop 100cm peak pitch: {pp:.1f}° (paper: 23.5°, within range)")
                else:
                    print(f"  ⚠️  Drop 100cm peak pitch: {pp:.1f}° (paper: 23.5°)")
                    issues.append(f"Drop 100cm peak pitch: {pp:.1f}° vs paper 23.5°")
            if h == 60 and not r.get("fell"):
                pp = r.get("peak_pitch", 0)
                print(f"     Drop 60cm peak pitch: {pp:.1f}° (paper: 16.8°)")

    # ── Table IV: Ledge ──
    ledge_path = OUT_DIR / "ledge_drive_off.json"
    if ledge_path.exists():
        d = json.load(ledge_path.open())
        for r in d["results"]:
            h = r.get("h_cm", 0)
            if h in [20, 50] and not r.get("fell"):
                pp = r.get("peak_pitch_land", 0)
                paper_val = {20: 20.5, 50: 27.4}.get(h, 0)
                print(f"     Ledge {h}cm peak pitch: {pp:.1f}° (paper: {paper_val}°)")

    # ── Table V: Curb ──
    curb_path = OUT_DIR / "curb_terrain.json"
    if curb_path.exists():
        d = json.load(curb_path.open())
        paper_curb = {10: 4.8, 15: 6.2, 20: 6.1}
        for r in d["results"]:
            h = int(r.get("h_cm", 0))
            if h in paper_curb and not r.get("fell"):
                roll = r.get("roll_curb_max", 0)
                expected = paper_curb[h]
                if abs(roll - expected) < 2.0:
                    print(f"  ✅ Curb {h}cm roll: {roll:.1f}° (paper: {expected}°)")
                else:
                    print(f"  ⚠️  Curb {h}cm roll: {roll:.1f}° (paper: {expected}°)")
                    issues.append(f"Curb {h}cm roll: {roll:.1f}° vs paper {expected}°")

    if issues:
        print(f"\n  {len(issues)} ISSUES FOUND:")
        for i in issues:
            print(f"    - {i}")
    else:
        print(f"\n  ✅ All checks passed!")

    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Run all tests")
    p.add_argument("--drop", action="store_true")
    p.add_argument("--ledge", action="store_true")
    p.add_argument("--curb", action="store_true")
    p.add_argument("--standing", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--compare", action="store_true", help="Only compare against paper")
    p.add_argument("--n-trials", type=int, default=1,
                   help="Number of trials for standing idle (paper: N=10)")
    args = p.parse_args()

    run_all = args.all or not any([args.drop, args.ledge, args.curb,
                                    args.standing, args.push, args.compare])

    if run_all or args.drop:
        run_drop_suite()
    if run_all or args.ledge:
        run_ledge_suite()
    if run_all or args.curb:
        run_curb_suite()
    if run_all or args.standing:
        run_standing_idle(n_trials=args.n_trials)
    if run_all or args.push:
        run_push_verification()

    if run_all or args.compare:
        compare_with_paper()

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
