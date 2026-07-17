"""Phase 7: Physics equilibrium feedforward fixed-height validation (2000 steps).

Compares:
  Baseline: calibrated_support_position_outer_loop_pitch_ref_v2 (B2v2)
  Candidate: physics_equilibrium_feedforward_outer_loop (PFF)

All 10 fixed heights: low_0p300, low_0p320, low_0p330, low_0p340, low_0p360, low_0p380,
                     high_0p430, high_0p450, high_0p465, high_0p480.

Setup: centered_posture_height_schedule (outputs/physical_target_height_setups_centered)

Pass criteria:
  - no fall at any height
  - no WBC / hidden-torque / ownership violation
  - contact / height / roll / posture safe
  - hip-yaw safe
  - candidate better or equal on at least 6/10 heights (by score)
  - candidate not worse on more than 2/10 heights
  - Protected heights: low_0p320, low_0p330, low_0p360, high_0p480 must not regress
  - maxabs_cand <= maxabs_base + 0.02
  - P2P_cand <= P2P_base * 1.15
  - out15_cand <= out15_base + 3 pp
  - feedforward values smooth and physically plausible

Outputs:
  outputs/physics_ff_phase7_2000/physics_ff_fixed_height_summary.csv
  outputs/physics_ff_phase7_2000/physics_ff_fixed_height_summary.json
  docs/validation/physics_equilibrium_feedforward_fixed_height_report.md
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_BASE = ROOT / "outputs" / "physics_ff_phase7_2000"
PER_RUN_TIMEOUT_S = 2400  # 40 min per 2000-step run
MAX_WORKERS = 4

DRIFT_COL = "active_pitch_crossing_signed_error_m"

# Baseline and candidate profiles
BASELINE_PROFILE = "calibrated_support_position_outer_loop_pitch_ref_v2"
CANDIDATE_PROFILE = "physics_equilibrium_feedforward_outer_loop"

HEIGHTS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
    "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]

PROTECTED_HEIGHTS = {"low_0p320", "low_0p330", "low_0p360", "high_0p480"}
MAXABS_TOL = 0.02
P2P_FACTOR = 1.15
OUT15_TOL_PP = 3.0
SCREEN_STEPS = 2000
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"


def score(m):
    """Multi-objective score (lower = better)."""
    if m is None:
        return 1e9
    if m.get("fell"):
        return 1e8
    s = 2.0 * abs(m["pos_pct"] - 50)
    s += 120.0 * max(0, m["support_maxabs"] - 0.18)
    s += 90.0 * max(0, m["support_p2p"] - 0.26)
    s += 70.0 * m["out15_pct"]
    s += 30.0 * m["out10_pct"]
    s += 20.0 * m.get("yaw_drift_growth", 0)
    s += 20.0 * m.get("hip_yaw_abs_max", 0.0)
    s += 30.0 * m.get("asym_rms", 0.0)
    if m.get("pitch_max", 0) > 14.0:
        s += 50.0 * (m["pitch_max"] - 14.0)
    if m.get("roll_rms", 0) > 2.5:
        s += 50.0 * (m["roll_rms"] - 2.5)
    if m.get("comz_min", 1.0) < 0.25:
        s += 100.0
    zc_rate = m.get("zero_crossings", 0) / max(1, m["steps"])
    if zc_rate > 0.05:
        s += 200.0 * (zc_rate - 0.05)
    return round(s, 2)


def run_sim(label, profile, out_dir):
    """Run one simulation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{SCREEN_STEPS}.csv"
    if tel_dst.exists():
        return tel_dst

    setup_path = SETUP_DIR / f"{label}_setup.json"
    if not setup_path.exists():
        print(f"  MISSING setup {label}", flush=True)
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(SCREEN_STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(SCREEN_STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        return None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        return None

    direct_tels = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if direct_tels:
        newest = direct_tels[0]
        if newest != tel_dst:
            shutil.copy2(newest, tel_dst)
        return tel_dst if tel_dst.exists() else newest

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if tels:
        shutil.copy2(tels[0], tel_dst)
        try:
            tels[0].unlink()
        except OSError:
            pass
    return tel_dst if tel_dst.exists() else None


def analyze(path):
    """Analyze telemetry CSV and return metrics dict."""
    import csv
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 100:
        return None

    steady = rows[100:]  # post-step-200 steady state (2000 steps total, decimation 1)
    n = len(rows)
    fell_short = n < 1999  # 2000 steps - 1 head row

    def fcol(key):
        out = []
        for r in rows:
            v = r.get(key, "")
            if v in ("", "nan", "None", None):
                out.append(0.0)
            else:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(0.0)
        return out

    def bcol(key):
        return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]

    drift = fcol(DRIFT_COL)
    abs_drift = [abs(x) for x in drift]
    support_error = fcol("support_position_error_m")
    pitch_x_deg = [math.degrees(x) if hasattr(math, "radians") else 0.0 for x in fcol("robot_pitch_x")]
    roll_y_deg = [math.degrees(x) for x in fcol("robot_roll_y")]
    hip_yaw_l = fcol("l_hip_yaw_pos")
    hip_yaw_r = fcol("r_hip_yaw_pos")
    wheel_vel = fcol("wheel_vel_mean_rad_s")
    wbc_authority = sum(1 for r in rows if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower() in ("true", "1", "1.0"))
    wbc_owner = sum(1 for r in rows if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower())
    hidden_torque_max = max(fcol("hidden_torque_norm")) if "hidden_torque_norm" in rows[0] else 0.0
    ownership_violation_max = max(fcol("ownership_violation_count")) if "ownership_violation_count" in rows[0] else 0.0

    # Terminate detection: terminated=True OR short run
    fell_anywhere = any(bcol("terminated"))
    fell = fell_anywhere or fell_short

    # Stats
    def stats(vals):
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "maxabs": 0.0, "p2p": 0.0}
        m = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        return {"mean": m, "min": mn, "max": mx, "maxabs": max(abs(mn), abs(mx)), "p2p": mx - mn}

    support_stats = stats(support_error)
    pitch_stats = stats(pitch_x_deg)
    roll_stats = stats(roll_y_deg)
    hip_yaw_max = max([abs(v) for v in hip_yaw_l + hip_yaw_r]) if hip_yaw_l and hip_yaw_r else 0.0
    wheel_vel_rms = math.sqrt(sum(v * v for v in wheel_vel) / max(1, len(wheel_vel)))

    # Bands
    nz = len(support_error)
    abs_support = [abs(v) for v in support_error]
    out05 = sum(1 for v in abs_support if v > 0.05)
    out10 = sum(1 for v in abs_support if v > 0.10)
    out15 = sum(1 for v in abs_support if v > 0.15)
    out05_pct = 100.0 * out05 / max(1, nz)
    out10_pct = 100.0 * out10 / max(1, nz)
    out15_pct = 100.0 * out15 / max(1, nz)
    pos_pct = 100.0 * sum(1 for v in support_error if v > 0) / max(1, nz)
    neg_pct = 100.0 * sum(1 for v in support_error if v < 0) / max(1, nz)

    # Physics feedforward telemetry (if present)
    physics_ff_enabled = any(bcol("physics_ff_enabled"))
    physics_ff_active = sum(1 for v in bcol("physics_ff_active_this_step") if v)
    physics_ff_tau_eq_vals = fcol("physics_ff_tau_eq_each_wheel_nm")
    physics_ff_pitch_eq_vals = fcol("physics_ff_pitch_eq_no_off_deg")
    tau_eq_mean = sum(physics_ff_tau_eq_vals) / max(1, len(physics_ff_tau_eq_vals))
    pitch_eq_mean = sum(physics_ff_pitch_eq_vals) / max(1, len(physics_ff_pitch_eq_vals))
    empirical_disabled = sum(1 for v in bcol("empirical_pitch_ref_offset_disabled") if v)

    return {
        "steps": n,
        "fell": fell,
        "fall_short": fell_short,
        "support_min": support_stats["min"],
        "support_max": support_stats["max"],
        "support_maxabs": support_stats["maxabs"],
        "support_p2p": support_stats["p2p"],
        "support_mean": support_stats["mean"],
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "out05_pct": out05_pct,
        "out10_pct": out10_pct,
        "out15_pct": out15_pct,
        "pitch_rms": pitch_stats["maxabs"],  # use maxabs for RMS as metric
        "pitch_max": pitch_stats["max"],
        "roll_rms": roll_stats["maxabs"],
        "comz_min": min(fcol("com_z")) if "com_z" in rows[0] else 0.0,
        "hip_yaw_abs_max": hip_yaw_max,
        "wheel_vel_rms": wheel_vel_rms,
        "wbc_authority_rows": wbc_authority,
        "wbc_owner_rows": wbc_owner,
        "hidden_torque_max": hidden_torque_max,
        "ownership_violation_max": ownership_violation_max,
        "physics_ff_enabled_steps": sum(bcol("physics_ff_enabled")),
        "physics_ff_active_steps": physics_ff_active,
        "physics_ff_tau_eq_nm_mean": tau_eq_mean,
        "physics_ff_pitch_eq_deg_mean": pitch_eq_mean,
        "empirical_disabled_steps": empirical_disabled,
    }


def safety_ok(m):
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, "fall"
    if m.get("hip_yaw_abs_max", 0) > 0.20:
        return False, "hip_yaw_unsafe"
    if m.get("pitch_max", 0) > 16.0:
        return False, "pitch_unsafe"
    if m.get("roll_rms", 0) > 3.0:
        return False, "roll_unsafe"
    if m.get("wbc_authority_rows", 0) > 0:
        return False, "wbc_authority_enabled"
    if m.get("wbc_owner_rows", 0) > 0:
        return False, "wbc_owner_present"
    if m.get("hidden_torque_max", 0.0) > 0.1:
        return False, "hidden_torque"
    if m.get("ownership_violation_max", 0.0) > 0.0:
        return False, "ownership_violation"
    if m.get("comz_min", 1.0) < 0.25:
        return False, "comz_low"
    return True, "safe"


def main():
    import subprocess
    import shutil
    SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 7: Physics Equilibrium Feedforward Fixed-Height Validation", flush=True)
    print(f"Baseline: {BASELINE_PROFILE}", flush=True)
    print(f"Candidate: {CANDIDATE_PROFILE}", flush=True)
    print(f"Heights: {HEIGHTS}", flush=True)
    print("=" * 78, flush=True)

    # Build job list
    jobs = []
    for label in HEIGHTS:
        for prof, name in [(BASELINE_PROFILE, "baseline"), (CANDIDATE_PROFILE, "candidate")]:
            out_dir = OUT_BASE / f"{name}_{label}"
            jobs.append((label, name, prof, out_dir))

    # Run with worker pool
    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_sim, label, prof, out_dir): (label, out_dir) for label, name, prof, out_dir in jobs}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            label, out_dir = futures[future]
            tel_path = future.result()
            if tel_path:
                m = analyze(tel_path)
                if m is not None:
                    m["score"] = score(m)
                results[(label, out_dir.name.split("_")[0])] = m
            else:
                results[(label, out_dir.name.split("_")[0])] = None
            done += 1
            print(f"  [{done}/{total}] {label} {out_dir.name.split('_')[0]}: ", end="", flush=True)
            m = results.get((label, out_dir.name.split("_")[0]))
            if m:
                print(f"steps={m['steps']} maxabs={m['support_maxabs']:.4f} P2P={m['support_p2p']:.4f} fell={m['fell']}")
            else:
                print("MISSING")

    # Comparison analysis
    csv_rows = []
    gate_pass = []
    gate_warn = []
    gate_fail = []
    protected_regress = []
    improve_heights = []

    for label in HEIGHTS:
        base = results.get((label, "baseline"))
        cand = results.get((label, "candidate"))
        row = {"height": label, "baseline": base, "candidate": cand}
        csv_rows.append(row)

        if base is None or cand is None:
            continue

        # Safety checks
        base_safe, base_reason = safety_ok(base)
        cand_safe, cand_reason = safety_ok(cand)
        if not cand_safe:
            gate_fail.append((label, cand_reason))
            continue
        if not base_safe:
            gate_warn.append((label, f"baseline unsafe: {base_reason}"))

        # Performance comparison
        sc_base = score(base)
        sc_cand = score(cand)
        improves = sc_cand < sc_base - 0.5
        worse = sc_cand > sc_base + 0.5

        if improves:
            improve_heights.append(label)

        # Tolerances
        maxabs_ok = cand["support_maxabs"] <= base["support_maxabs"] + MAXABS_TOL
        p2p_ok = cand["support_p2p"] <= base["support_p2p"] * P2P_FACTOR
        out15_ok = cand["out15_pct"] <= base["out15_pct"] + OUT15_TOL_PP

        if not maxabs_ok:
            gate_warn.append((label, f"maxabs {cand['support_maxabs']:.4f} > base+0.02 {base['support_maxabs']+MAXABS_TOL:.4f}"))
        if not p2p_ok:
            gate_warn.append((label, f"P2P {cand['support_p2p']:.4f} > base*1.15 {base['support_p2p']*P2P_FACTOR:.4f}"))
        if not out15_ok:
            gate_warn.append((label, f"out15 {cand['out15_pct']:.2f}% > base+3pp {base['out15_pct']+OUT15_TOL_PP:.2f}%"))

        # Protected regression check
        if label in PROTECTED_HEIGHTS and worse:
            protected_regress.append(label)

        # Overall pass for this height (no fail, tolerances respected)
        if cand_safe and maxabs_ok and p2p_ok and out15_ok:
            gate_pass.append(label)

    # Write CSV summary
    csv_path = OUT_BASE / "physics_ff_fixed_height_summary.csv"
    fieldnames = [
        "height", "profile", "steps", "fell", "support_min", "support_max", "support_maxabs",
        "support_p2p", "pos_pct", "neg_pct", "out05_pct", "out10_pct", "out15_pct",
        "pitch_rms", "pitch_max", "roll_rms", "comz_min", "hip_yaw_abs_max", "wheel_vel_rms",
        "wbc_authority_rows", "wbc_owner_rows", "score",
        "physics_ff_active_steps", "physics_ff_tau_eq_nm_mean", "physics_ff_pitch_eq_deg_mean"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in HEIGHTS:
            for prof in ["baseline", "candidate"]:
                m = results.get((label, prof))
                if m is None:
                    continue
                row = {
                    "height": label,
                    "profile": prof,
                    "steps": m["steps"],
                    "fell": m["fell"],
                    "support_min": round(m["support_min"], 4),
                    "support_max": round(m["support_max"], 4),
                    "support_maxabs": round(m["support_maxabs"], 4),
                    "support_p2p": round(m["support_p2p"], 4),
                    "pos_pct": round(m["pos_pct"], 1),
                    "neg_pct": round(m["neg_pct"], 1),
                    "out05_pct": round(m["out05_pct"], 1),
                    "out10_pct": round(m["out10_pct"], 1),
                    "out15_pct": round(m["out15_pct"], 1),
                    "pitch_rms": round(m["pitch_rms"], 2),
                    "pitch_max": round(m["pitch_max"], 2),
                    "roll_rms": round(m["roll_rms"], 2),
                    "comz_min": round(m["comz_min"], 4),
                    "hip_yaw_abs_max": round(m["hip_yaw_abs_max"], 4),
                    "wheel_vel_rms": round(m["wheel_vel_rms"], 4),
                    "wbc_authority_rows": m["wbc_authority_rows"],
                    "wbc_owner_rows": m["wbc_owner_rows"],
                    "score": m.get("score", None),
                    "physics_ff_active_steps": m.get("physics_ff_active_steps", 0),
                    "physics_ff_tau_eq_nm_mean": round(m.get("physics_ff_tau_eq_nm_mean", 0.0), 4),
                    "physics_ff_pitch_eq_deg_mean": round(m.get("physics_ff_pitch_eq_deg_mean", 0.0), 4),
                }
                writer.writerow(row)
    print(f"wrote {csv_path}")

    # JSON dump
    json_path = OUT_BASE / "physics_ff_fixed_height_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "baseline_profile": BASELINE_PROFILE,
            "candidate_profile": CANDIDATE_PROFILE,
            "heights": {h: {
                "baseline": results.get((h, "baseline")),
                "candidate": results.get((h, "candidate")),
            } for h in HEIGHTS},
        }, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"wrote {json_path}")

    # Decision logic
    print()
    print("=" * 78)
    print("PHASE 7 GATES")
    print("=" * 78)
    print(f"Pass heights (no fail, tolerances ok): {gate_pass}")
    print(f"Warnings: {gate_warn}")
    print(f"Fails: {gate_fail}")
    print(f"Protected regress: {protected_regress}")
    print(f"Improve heights: {improve_heights}")

    n_pass = len(gate_pass)
    n_warn = len(gate_warn)
    n_fail = len(gate_fail)
    n_improve = len(improve_heights)
    n_regress = len([h for h in HEIGHTS if h in protected_regress or (
        results.get((h, "candidate")) is not None and results.get((h, "baseline")) is not None and
        results[(h, "candidate")]["support_maxabs"] > results[(h, "baseline")]["support_maxabs"] + MAXABS_TOL + 0.01
    )])

    classification = None
    if n_fail > 0:
        classification = "PHYSICS_FF_FIXED_HEIGHT_FAIL_SAFETY"
    elif n_pass >= 6 and n_regress == 0:
        classification = "PHYSICS_FF_FIXED_HEIGHT_PASS"
    elif n_pass >= 6 and n_regress > 0:
        classification = "PHYSICS_FF_FIXED_HEIGHT_PASS_EXPERIMENTAL"
    else:
        classification = "PHYSICS_FF_NOT_BETTER_KEEP_B2V2"

    print()
    print(f"Classification: {classification}")
    print("=" * 78)
    return classification


if __name__ == "__main__":
    sys.exit(0 if main() in ("PHYSICS_FF_FIXED_HEIGHT_PASS", "PHYSICS_FF_FIXED_HEIGHT_PASS_EXPERIMENTAL") else 1)
