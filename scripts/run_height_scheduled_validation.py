"""Phase 4: Validate height_scheduled_pitch_equilibrium_trim.

Runs the new profile at high_0p480 for multiple step counts (500, 1200, 2000, 5000),
then runs a full 2000-step height ladder across all 10 heights.

Compares each run against:
  - pitch_equilibrium_trim (static +4 deg)
  - pitch_bias_compensated_zero_crossing_recenter
  - adaptive_support_centering_trim (offset-0 baseline)

Outputs:
  outputs/.../active_pitch_crossing/height_scheduled_validation_high_0p480.json
  outputs/.../active_pitch_crossing/height_scheduled_validation_ladder.json
  docs/validation/height_scheduled_pitch_equilibrium_trim_validation_report.md
"""
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"
PER_RUN_TIMEOUT_S = 900

HEIGHTS = {
    "low_0p300": 0.300, "low_0p320": 0.320, "low_0p330": 0.330,
    "low_0p340": 0.340, "low_0p360": 0.360, "low_0p380": 0.380,
    "high_0p430": 0.430, "high_0p450": 0.450, "high_0p465": 0.465,
    "high_0p480": 0.480,
}

DRIFT_COL = "active_pitch_crossing_signed_error_m"


def run_sim(label, steps, profile, out_dir):
    """Run simulation; return (telemetry_path|None, summary_path|None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{steps}.csv"
    sum_dst = out_dir / "run_summary.json"
    if tel_dst.exists():
        return tel_dst, sum_dst if sum_dst.exists() else None

    setup_path = SETUP_DIR / f"{label}_setup.json"
    if not setup_path.exists():
        print(f"  MISSING setup {label}", flush=True)
        return None, None

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
    ]
    try:
        result = subprocess.run(
            args, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        print(f"  TIMEOUT {label} {steps}s {profile}", flush=True)
        return None, None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        print(f"  FAILED rc={result.returncode} {label} {steps}s {profile}", flush=True)
        return None, None

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(SIM_OUT.glob("run_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if tels:
        shutil.copy2(tels[0], tel_dst)
        try: tels[0].unlink()
        except OSError: pass
    if sums:
        shutil.copy2(sums[0], sum_dst)
        try: sums[0].unlink()
        except OSError: pass
    return tel_dst if tel_dst.exists() else None, sum_dst if sum_dst.exists() else None


def fcol(rows, key, default=float("nan")):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            out.append(default)
        else:
            try: out.append(float(v))
            except ValueError: out.append(default)
    return out


def bcol(rows, key):
    return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def analyze(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    drift = clean(fcol(rows, DRIFT_COL))
    pitch = clean(fcol(rows, "robot_pitch_x"))
    roll = clean(fcol(rows, "robot_roll_y"))
    comz = clean(fcol(rows, "com_z"))
    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))
    yaw_drift = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    term = any(bcol(rows, "terminated"))
    term_reason = ""
    if term:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                term_reason = r.get("termination_reason", "") or ""
                break

    nz = len(drift)
    abs_drift = [abs(x) for x in drift]
    pos = sum(1 for x in drift if x > 0)
    neg = sum(1 for x in drift if x < 0)
    zc = sum(1 for i in range(1, len(drift)) if (drift[i-1] <= 0) != (drift[i] <= 0))
    pos_area = sum(x for x in drift if x > 0)
    neg_area = -sum(x for x in drift if x < 0)
    area_total = pos_area + neg_area
    area_balance = abs(pos_area - neg_area) / area_total if area_total > 1e-9 else 1.0

    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]
    hy_all = [abs(x) for x in (lhy + rhy)]

    def out_pct(thr):
        return 100 * sum(1 for x in abs_drift if x > thr) / nz if nz else 0.0

    return {
        "steps": n,
        "fell": term,
        "term_reason": term_reason,
        "min_drift": round(min(drift), 4) if drift else 0.0,
        "max_drift": round(max(drift), 4) if drift else 0.0,
        "max_abs": round(max(abs_drift), 4) if abs_drift else 0.0,
        "p2p": round(max(drift) - min(drift), 4) if drift else 0.0,
        "pos_pct": round(100 * pos / nz, 1) if nz else 0.0,
        "neg_pct": round(100 * neg / nz, 1) if nz else 0.0,
        "zero_crossings": zc,
        "pos_area": round(pos_area, 3),
        "neg_area": round(neg_area, 3),
        "area_balance": round(area_balance, 3),
        "out05_pct": round(out_pct(0.05), 1),
        "out08_pct": round(out_pct(0.08), 1),
        "out10_pct": round(out_pct(0.10), 1),
        "out15_pct": round(out_pct(0.15), 1),
        "pitch_rms_deg": round(rms(pitch_deg), 2),
        "pitch_max_abs_deg": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "roll_rms_deg": round(rms(roll_deg), 2),
        "comz_min": round(min(comz), 4) if comz else 0.0,
        "comz_max": round(max(comz), 4) if comz else 0.0,
        "hip_yaw_abs_max_rad": round(max(hy_all), 4) if hy_all else 0.0,
        "yaw_drift_max_rad": round(max((abs(x) for x in yaw_drift), default=0.0), 4),
    }


def verdict(m):
    if m is None: return "MISSING"
    if m["fell"]: return f"FALL({m['term_reason'][:20]})"
    if m["hip_yaw_abs_max_rad"] > 0.35: return "HY_UNSAFE"
    if m["pitch_max_abs_deg"] > 16.0: return "PITCH_UNSAFE"
    if m["roll_rms_deg"] > 3.0: return "ROLL_UNSAFE"
    pos = m["pos_pct"]
    maxabs = m["max_abs"]
    p2p = m["p2p"]
    if 35 <= pos <= 65 and maxabs <= 0.22 and p2p <= 0.34:
        return "PASS"
    if 25 <= pos <= 75 and maxabs <= 0.25 and p2p <= 0.40:
        return "PASS_WITH_MONITORING"
    if pos > 75 or pos < 25:
        return "ONE_SIDED"
    return "MARGINAL"


def main():
    NEW_PROFILE = "height_scheduled_pitch_equilibrium_trim"
    STATIC_PROFILE = "pitch_equilibrium_trim"
    PBC_PROFILE = "pitch_bias_compensated_zero_crossing_recenter"
    ADAPTIVE_PROFILE = "adaptive_support_centering_trim"

    print("=" * 70, flush=True)
    print("Phase 4: height_scheduled_pitch_equilibrium_trim validation", flush=True)
    print("=" * 70, flush=True)

    # --- 4A: high_0p480 multi-step runs ---
    print("\n--- 4A: high_0p480 multi-step (500/1200/2000/5000) ---", flush=True)
    step_counts = [500, 1200, 2000, 5000]
    h_label = "high_0p480"

    high_results = {}
    for steps in step_counts:
        for profile, tag in [
            (NEW_PROFILE, "sched"),
            (STATIC_PROFILE, "static4"),
            (PBC_PROFILE, "pbc"),
            (ADAPTIVE_PROFILE, "adaptive"),
        ]:
            key = f"{steps}_{tag}"
            out_dir = OUT_BASE / f"hs_val_{steps}_{h_label}_{tag}"
            print(f"  [{steps}s {tag}]", end=" ", flush=True)
            t0 = time.time()
            tel, _ = run_sim(h_label, steps, profile, out_dir)
            elapsed = time.time() - t0
            m = analyze(tel)
            high_results[key] = m
            v = verdict(m)
            if m:
                print(f"pos%={m['pos_pct']} max={m['max_abs']:.3f} P2P={m['p2p']:.3f} -> {v} ({elapsed:.0f}s)", flush=True)
            else:
                print(f"MISSING ({elapsed:.0f}s)", flush=True)

    # --- 4B: full height ladder at 2000 steps ---
    print("\n--- 4B: 2000-step height ladder ---", flush=True)
    ladder_results = {}
    for label in HEIGHTS:
        m_new, m_static, m_pbc, m_adaptive = None, None, None, None
        for profile, tag, store in [
            (NEW_PROFILE,    "sched",   "new"),
            (STATIC_PROFILE, "static4", "static"),
            (PBC_PROFILE,    "pbc",     "pbc"),
            (ADAPTIVE_PROFILE, "adaptive", "adaptive"),
        ]:
            out_dir = OUT_BASE / f"hs_ladder_2000_{label}_{tag}"
            # reuse existing adaptive ladder runs to save time
            if tag == "adaptive":
                existing = OUT_BASE / f"adaptive_height_ladder_2000_{label}" / "telemetry_2000.csv"
                if existing.exists():
                    m = analyze(existing)
                    ladder_results.setdefault(label, {})[store] = m
                    continue
            print(f"  {label} {tag}", end=" ", flush=True)
            t0 = time.time()
            tel, _ = run_sim(label, 2000, profile, out_dir)
            elapsed = time.time() - t0
            m = analyze(tel)
            ladder_results.setdefault(label, {})[store] = m
            v = verdict(m) if m else "MISSING"
            print(f"-> {v} ({elapsed:.0f}s)", flush=True)

    # --- Print summary tables ---
    print("\n=== 4A: high_0p480 multi-step comparison ===", flush=True)
    print(f"{'steps':>6} {'profile':>10} {'pos%':>6} {'neg%':>6} {'min':>7} {'max':>7} {'maxabs':>7} {'P2P':>6} {'out15':>6} {'verdict':>20}", flush=True)
    print("-" * 100, flush=True)
    for steps in step_counts:
        for tag in ["sched", "static4", "pbc", "adaptive"]:
            key = f"{steps}_{tag}"
            m = high_results.get(key)
            if m:
                v = verdict(m)
                print(f"{steps:>6} {tag:>10} {m['pos_pct']:>6} {m['neg_pct']:>6} "
                      f"{m['min_drift']:>7.3f} {m['max_drift']:>7.3f} {m['max_abs']:>7.3f} "
                      f"{m['p2p']:>6.3f} {m['out15_pct']:>6.1f} {v:>20}", flush=True)
            else:
                print(f"{steps:>6} {tag:>10}  {'MISSING':>50}", flush=True)

    print("\n=== 4B: height ladder — sched vs static4 vs pbc vs adaptive ===", flush=True)
    print(f"{'height':>12} {'profile':>10} {'pos%':>6} {'neg%':>6} {'min':>7} {'max':>7} {'maxabs':>7} {'P2P':>6} {'out15':>6} {'verdict':>22}", flush=True)
    print("-" * 110, flush=True)
    for label in HEIGHTS:
        r = ladder_results.get(label, {})
        for tag, store in [("sched", "new"), ("static4", "static"), ("pbc", "pbc"), ("adaptive", "adaptive")]:
            m = r.get(store)
            if m:
                v = verdict(m)
                print(f"{label:>12} {tag:>10} {m['pos_pct']:>6} {m['neg_pct']:>6} "
                      f"{m['min_drift']:>7.3f} {m['max_drift']:>7.3f} {m['max_abs']:>7.3f} "
                      f"{m['p2p']:>6.3f} {m['out15_pct']:>6.1f} {v:>22}", flush=True)
            else:
                print(f"{label:>12} {tag:>10}  MISSING", flush=True)

    # --- Classification ---
    ladder_new = {lbl: ladder_results.get(lbl, {}).get("new") for lbl in HEIGHTS}
    any_fall = any(m and m["fell"] for m in ladder_new.values())
    any_unsafe = any(m and verdict(m) in ("HY_UNSAFE", "PITCH_UNSAFE", "ROLL_UNSAFE", f"FALL({m['term_reason'][:20]})") for lbl, m in ladder_new.items() if m)
    one_sided_count = sum(1 for m in ladder_new.values() if m and verdict(m) == "ONE_SIDED")
    pass_count = sum(1 for m in ladder_new.values() if m and verdict(m) in ("PASS", "PASS_WITH_MONITORING"))

    if any_fall or any_unsafe:
        classification = "HEIGHT_SCHEDULED_OFFSET_FAIL_SAFETY"
    elif one_sided_count > 3:
        classification = "HEIGHT_SCHEDULED_OFFSET_NOT_ENOUGH"
    elif pass_count >= 8:
        classification = "HEIGHT_SCHEDULED_OFFSET_PASS"
    elif pass_count >= 5:
        classification = "HEIGHT_SCHEDULED_OFFSET_PASS_WITH_MONITORING"
    else:
        classification = "HEIGHT_SCHEDULED_OFFSET_INCONCLUSIVE"

    print(f"\nClassification: {classification}", flush=True)

    # Save JSON results
    out_json = OUT_BASE / "height_scheduled_validation_results.json"
    out_json.write_text(json.dumps({
        "high_0p480_multistep": high_results,
        "ladder_2000": ladder_results,
        "classification": classification,
    }, indent=2, default=str))
    print(f"Results: {out_json}", flush=True)

    # Write markdown report stub (full report in Phase 8)
    md = []
    md.append("# Phase 4: height_scheduled_pitch_equilibrium_trim Validation\n\n")
    md.append(f"**Classification: `{classification}`**\n\n")
    md.append("## high_0p480 multi-step\n\n")
    md.append("| steps | profile | pos% | neg% | min | max | maxabs | P2P | out15% | verdict |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for steps in step_counts:
        for tag in ["sched", "static4", "pbc", "adaptive"]:
            key = f"{steps}_{tag}"
            m = high_results.get(key)
            if m:
                md.append(f"| {steps} | {tag} | {m['pos_pct']} | {m['neg_pct']} | "
                           f"{m['min_drift']} | {m['max_drift']} | {m['max_abs']} | "
                           f"{m['p2p']} | {m['out15_pct']} | {verdict(m)} |\n")
    md.append("\n## Height ladder (2000 steps)\n\n")
    md.append("| height | profile | pos% | neg% | min | max | maxabs | P2P | out15% | verdict |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for label in HEIGHTS:
        r = ladder_results.get(label, {})
        for tag, store in [("sched", "new"), ("static4", "static"), ("pbc", "pbc"), ("adaptive", "adaptive")]:
            m = r.get(store)
            if m:
                md.append(f"| {label} | {tag} | {m['pos_pct']} | {m['neg_pct']} | "
                           f"{m['min_drift']} | {m['max_drift']} | {m['max_abs']} | "
                           f"{m['p2p']} | {m['out15_pct']} | {verdict(m)} |\n")
    (ROOT / "docs" / "validation" / "height_scheduled_pitch_equilibrium_trim_validation_report.md").write_text("".join(md))
    print(f"Report: docs/validation/height_scheduled_pitch_equilibrium_trim_validation_report.md", flush=True)


if __name__ == "__main__":
    main()
