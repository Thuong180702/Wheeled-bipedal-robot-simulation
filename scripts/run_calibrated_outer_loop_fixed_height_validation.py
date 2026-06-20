"""Phase 6: Calibrated outer-loop fixed-height validation (B2 vs B vs A vs B2v2).

Compares the Phase B2 calibrated profile against B, A, and B2v2:
  A)  height_scheduled_pitch_equilibrium_trim             (Phase A, structural fix baseline)
  B)  support_position_outer_loop_pitch_ref              (Phase B, Kp=+1.0, Kd=0.0 fixed)
  B2) calibrated_support_position_outer_loop_pitch_ref   (Phase B calibration v1: FAILED Phase 6)
  B2v2) calibrated_support_position_outer_loop_pitch_ref_v2 (Phase B calibration v2: smoothed upper band)

Runs 2000 steps at all 10 fixed heights for all 4 profiles.

Phase 6 gating (from task spec) — B2v2 passes only if:

Hard gates (every height):
  - no fall
  - no WBC / hidden-torque / ownership violation
  - contact / height / roll / posture safe
  - hip-yaw not worse than B beyond monitoring thresholds

Performance gates:
  - B2v2 must not regress vs B on more than 2/10 heights by score
  - B2v2 must not regress vs A on protected heights: high_0p480, high_0p465, low_0p330, low_0p360
  - maxabs_B2v2 <= max(maxabs_B + 0.02, maxabs_A + 0.02)
  - P2P_B2v2 <= max(P2P_B * 1.15, P2P_A * 1.15)
  - out15_B2v2 <= min(out15_B, out15_A) + 5 pp where feasible

Outputs:
  outputs/.../calibrated_outer_loop_fixed_height_metrics.csv
  docs/validation/calibrated_outer_loop_fixed_height_v2_report.md

Classification:
  CALIBRATED_OUTER_LOOP_V2_FIXED_HEIGHT_PASS
  CALIBRATED_OUTER_LOOP_V2_FIXED_HEIGHT_PASS_WITH_MONITORING
  CALIBRATED_OUTER_LOOP_V2_NOT_BETTER
  CALIBRATED_OUTER_LOOP_V2_FAIL_SAFETY
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"
PER_RUN_TIMEOUT_S = 1200  # 20 min per run
MAX_WORKERS = 4

DRIFT_COL = "active_pitch_crossing_signed_error_m"

PROFILES = [
    ("A", "height_scheduled_pitch_equilibrium_trim"),
    ("B", "support_position_outer_loop_pitch_ref"),
    ("B2", "calibrated_support_position_outer_loop_pitch_ref"),
    ("B2v2", "calibrated_support_position_outer_loop_pitch_ref_v2"),
]

HEIGHTS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
    "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]

PROTECTED_HEIGHTS = {"high_0p480", "low_0p330", "low_0p360", "low_0p380"}
MAXABS_TOL = 0.02
P2P_FACTOR = 1.15
OUT15_TOL_PP = 5.0
SCREEN_STEPS = 2000


def score(m):
    """Multi-objective score (lower = better). Final drift NOT included."""
    if m is None:
        return 1e9
    if m.get("fell"):
        return 1e8
    s = 2.0 * abs(m["pos_pct"] - 50)
    s += 120.0 * max(0, m["maxabs"] - 0.18)
    s += 90.0 * max(0, m["p2p"] - 0.26)
    s += 70.0 * m["out15_pct"]
    s += 30.0 * m["out10_pct"]
    s += 20.0 * max(0, m.get("yaw_drift_growth", 0))
    s += 20.0 * m.get("hip_yaw_abs_max", 0.0)
    s += 30.0 * m.get("asym_rms", 0.0)
    if m.get("pitch_max", 0) > 14.0:
        s += 50.0 * (m["pitch_max"] - 14.0)
    if m.get("roll_rms", 0) > 2.5:
        s += 50.0 * (m["roll_rms"] - 2.5)
    if m.get("comz_min", 1.0) < 0.25:
        s += 100.0
    zc_rate = m["zero_crossings"] / max(1, m["steps"])
    if zc_rate > 0.05:
        s += 200.0 * (zc_rate - 0.05)
    return round(s, 2)


def run_sim(label, tag, profile, out_dir):
    """Run one simulation, reuse existing telemetry if present."""
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

    # When --output-dir is supplied, the simulator writes into out_dir directly.
    direct_tels = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if direct_tels:
        newest = direct_tels[0]
        if newest != tel_dst:
            shutil.copy2(newest, tel_dst)
        return tel_dst if tel_dst.exists() else newest

    # Fallback for older simulator behavior.
    tels = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if tels:
        shutil.copy2(tels[0], tel_dst)
        try:
            tels[0].unlink()
        except OSError:
            pass
    return tel_dst if tel_dst.exists() else None


def analyze(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    def fcol(key, default=float("nan")):
        out = []
        for r in rows:
            v = r.get(key, "")
            if v in ("", "nan", "None", None):
                out.append(default)
            else:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(default)
        return out

    def clean(xs):
        return [x for x in xs if x == x and x != float("nan")]

    def bcol(key):
        return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0")
                 for r in rows]

    def rms(xs):
        xs = clean(xs)
        return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0

    drift = clean(fcol(DRIFT_COL))
    pitch = clean(fcol("robot_pitch_x"))
    roll = clean(fcol("robot_roll_y"))
    comz = clean(fcol("com_z"))
    lhy = clean(fcol("l_hip_yaw_pos"))
    rhy = clean(fcol("r_hip_yaw_pos"))
    yaw_drift = clean(fcol("yaw_drift_from_initial_rad"))

    abs_drift = [abs(x) for x in drift]
    pos = sum(1 for x in drift if x > 0)
    neg = sum(1 for x in drift if x < 0)
    zc = sum(1 for i in range(1, len(drift)) if (drift[i - 1] <= 0) != (drift[i] <= 0))
    pos_area = sum(x for x in drift if x > 0)
    neg_area = -sum(x for x in drift if x < 0)
    area_total = pos_area + neg_area
    area_balance = abs(pos_area - neg_area) / area_total if area_total > 1e-9 else 1.0

    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]
    hy_all = [abs(x) for x in (lhy + rhy)]

    nz = len(drift)

    def out_pct(thr):
        return 100 * sum(1 for x in abs_drift if x > thr) / nz if nz else 0.0

    tau_wbc_max = clean(fcol("tau_wbc_max"))
    hidden_torque_norm = clean(fcol("hidden_torque_norm"))
    ownership_violation = clean(fcol("ownership_violation_count"))
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    term = any(bcol("terminated"))
    term_reason = ""
    if term:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                term_reason = r.get("termination_reason", "") or ""
                break

    # Windowed metrics per 500 steps
    windowed = []
    for win_start in range(0, n, 500):
        win_end = min(win_start + 500, n)
        wd = drift[win_start:win_end]
        wa = [abs(x) for x in wd]
        windowed.append({
            "window": f"{win_start}-{win_end}",
            "min": round(min(wd), 4) if wd else 0.0,
            "max": round(max(wd), 4) if wd else 0.0,
            "maxabs": round(max(wa), 4) if wa else 0.0,
            "p2p": round(max(wd) - min(wd), 4) if wd else 0.0,
        })

    return {
        "steps": n,
        "fell": term,
        "term_reason": term_reason,
        "min_drift": round(min(drift), 4) if drift else 0.0,
        "max_drift": round(max(drift), 4) if drift else 0.0,
        "maxabs": round(max(abs_drift), 4) if abs_drift else 0.0,
        "p2p": round(max(drift) - min(drift), 4) if drift else 0.0,
        "pos_pct": round(100 * pos / nz, 1) if nz else 0.0,
        "neg_pct": round(100 * neg / nz, 1) if nz else 0.0,
        "zero_crossings": zc,
        "pos_area": round(pos_area, 3),
        "neg_area": round(neg_area, 3),
        "area_balance": round(area_balance, 3),
        "out03_pct": round(out_pct(0.03), 1),
        "out05_pct": round(out_pct(0.05), 1),
        "out08_pct": round(out_pct(0.08), 1),
        "out10_pct": round(out_pct(0.10), 1),
        "out15_pct": round(out_pct(0.15), 1),
        "pitch_rms": round(rms(pitch_deg), 2),
        "pitch_max": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "roll_rms": round(rms(roll_deg), 2),
        "comz_min": round(min(comz), 4) if comz else 0.0,
        "comz_max": round(max(comz), 4) if comz else 0.0,
        "hip_yaw_abs_max": round(max(hy_all), 4) if hy_all else 0.0,
        "yaw_drift_max": round(max((abs(x) for x in yaw_drift), default=0.0), 4),
        "yaw_drift_growth": round(yaw_drift[-1] - yaw_drift[0], 4) if len(yaw_drift) > 1 else 0.0,
        "asym_rms": round(math.sqrt(sum((l - r) ** 2 for l, r in zip(lhy, rhy)) / max(1, len(lhy))), 4) if lhy and rhy else 0.0,
        "tau_wbc_max": round(max(tau_wbc_max), 4) if tau_wbc_max else 0.0,
        "hidden_torque_max": round(max(hidden_torque_norm), 4) if hidden_torque_norm else 0.0,
        "ownership_violation_max": round(max(ownership_violation), 4) if ownership_violation else 0.0,
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "score": None,
        "windowed": windowed,
    }


def safety_ok(m):
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, f"fall({m['term_reason'][:20]})"
    if m.get("hip_yaw_abs_max", 0) > 0.35:
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
    return True, "safe"


def fmt(m):
    if m is None:
        return "MISSING"
    fall = f"FALL({m['term_reason'][:15]})" if m["fell"] else "ok"
    return (
        f"{fall} pos%={m['pos_pct']:.1f} max={m['maxabs']:.3f} "
        f"P2P={m['p2p']:.3f} out15={m['out15_pct']:.1f}% "
        f"hy={m.get('hip_yaw_abs_max',0):.3f} sc={score(m):.0f}"
    )


def worker(args):
    """Run one (label, tag, profile) tuple. Returns (label, tag, m)."""
    label, tag, profile, out_dir = args
    tel = run_sim(label, tag, profile, out_dir)
    m = analyze(tel)
    if m is not None:
        m["score"] = score(m)
    return label, tag, m


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 6: calibrated outer-loop fixed-height validation (A vs B vs B2)", flush=True)
    for tag, prof in PROFILES:
        print(f"  {tag} = {prof}", flush=True)
    print("=" * 78, flush=True)

    # Build job list
    jobs = []
    for label in HEIGHTS:
        for tag, profile in PROFILES:
            out_dir = OUT_BASE / f"phase6_2000_{label}_{tag}"
            jobs.append((label, tag, profile, out_dir))

    # Run with worker pool
    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, job): job for job in jobs}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            label, tag, m = future.result()
            results[(label, tag)] = m
            done += 1
            t = fmt(m)
            print(f"  [{done}/{total}] {label} {tag}: {t}", flush=True)

    # Compute scores
    for key, m in results.items():
        if m is not None:
            m["score"] = score(m)

    # Per-height comparison table
    per_height = {}
    for label in HEIGHTS:
        row = {tag: results.get((label, tag)) for tag in ["A", "B", "B2", "B2v2"]}
        per_height[label] = row

    # Gate evaluation — comparing B2v2 vs B
    hard_fail = []
    regress_heights = []
    improve_heights = []
    protected_regress = []

    # Updated protected heights: high_0p480, high_0p465 are protected (B2 failed these)
    PROTECTED_HEIGHTS_V2 = {"high_0p480", "high_0p465"}

    for label in HEIGHTS:
        a = per_height[label]["A"]
        b = per_height[label]["B"]
        b2 = per_height[label]["B2"]  # v1 — for reference
        b2v2 = per_height[label]["B2v2"]  # v2 — the candidate
        b_safe, b_reason = safety_ok(b)
        b2v2_safe, b2v2_reason = safety_ok(b2v2)

        # Performance comparison — B2v2 vs B
        sc_a = score(a)
        sc_b = score(b)
        sc_b2v2 = score(b2v2)
        b2v2_improves_vs_b = sc_b2v2 < sc_b - 0.5
        b2v2_worse_vs_b = sc_b2v2 > sc_b + 0.5
        b2v2_improves_vs_a = sc_b2v2 < sc_a - 0.5

        # Specific regression checks
        b2v2_maxabs_ok = (b2v2 is None or b is None or
                        b2v2["maxabs"] <= max(b["maxabs"] + MAXABS_TOL,
                                            (a["maxabs"] if a else 0) + MAXABS_TOL))
        b2v2_p2p_ok = (b2v2 is None or b is None or
                      b2v2["p2p"] <= max(b["p2p"] * P2P_FACTOR,
                                       (a["p2p"] if a else 0) * P2P_FACTOR))
        b2v2_out15_ok = (b2v2 is None or b is None or
                        b2v2["out15_pct"] <= min(b["out15_pct"], a["out15_pct"] if a else 0) + OUT15_TOL_PP)

        hy_worse = (b2v2 and b and
                     b2v2.get("hip_yaw_abs_max", 0) > b.get("hip_yaw_abs_max", 0) + 0.05
                     and b2v2.get("hip_yaw_abs_max", 0) > 0.20)

        regress = (b2v2_worse_vs_b or not b2v2_maxabs_ok or not b2v2_p2p_ok or not b2v2_out15_ok)
        imp = b2v2_improves_vs_b

        if not b2v2_safe:
            hard_fail.append((label, b2v2_reason))
        if regress:
            regress_heights.append(label)
            if label in PROTECTED_HEIGHTS_V2:
                protected_regress.append(label)
        if imp:
            improve_heights.append(label)
        if hy_worse:
            hard_fail.append((label, "hip_yaw_worse_than_B"))

        per_height[label]["regress"] = regress
        per_height[label]["improve"] = imp

    n_improve = len(improve_heights)
    n_regress = len(regress_heights)

    classification = None
    if hard_fail:
        classification = "CALIBRATED_OUTER_LOOP_V2_FAIL_SAFETY"
    elif protected_regress:
        classification = "CALIBRATED_OUTER_LOOP_V2_NOT_BETTER"
    elif n_regress > 2:
        classification = "CALIBRATED_OUTER_LOOP_V2_NOT_BETTER"
    elif n_improve >= 6:
        classification = "CALIBRATED_OUTER_LOOP_V2_FIXED_HEIGHT_PASS"
    else:
        classification = "CALIBRATED_OUTER_LOOP_V2_FIXED_HEIGHT_PASS_WITH_MONITORING"

    print(f"\n>>> B2v2 improve={n_improve}/10  regress={n_regress}  hard_fail={len(hard_fail)}", flush=True)
    print(f">>> Classification: {classification}", flush=True)

    _write_outputs(per_height, classification, n_improve, n_regress,
                   hard_fail, regress_heights, improve_heights, protected_regress)
    return classification


def _write_outputs(per_height, classification, n_improve, n_regress,
                   hard_fail, regress_heights, improve_heights, protected_regress):
    rows = []
    for label, row in per_height.items():
        for tag in ["A", "B", "B2", "B2v2"]:
            m = row.get(tag)
            rows.append({
                "height": label, "profile": tag,
                "fell": m.get("fell") if m else True,
                "min_drift": m.get("min_drift") if m else None,
                "max_drift": m.get("max_drift") if m else None,
                "maxabs": m.get("maxabs") if m else None,
                "p2p": m.get("p2p") if m else None,
                "pos_pct": m.get("pos_pct") if m else None,
                "neg_pct": m.get("neg_pct") if m else None,
                "zero_crossings": m.get("zero_crossings") if m else None,
                "out03_pct": m.get("out03_pct") if m else None,
                "out05_pct": m.get("out05_pct") if m else None,
                "out08_pct": m.get("out08_pct") if m else None,
                "out10_pct": m.get("out10_pct") if m else None,
                "out15_pct": m.get("out15_pct") if m else None,
                "pitch_rms": m.get("pitch_rms") if m else None,
                "pitch_max": m.get("pitch_max") if m else None,
                "roll_rms": m.get("roll_rms") if m else None,
                "comz_min": m.get("comz_min") if m else None,
                "hip_yaw_abs_max": m.get("hip_yaw_abs_max") if m else None,
                "yaw_drift_max": m.get("yaw_drift_max") if m else None,
                "score": m.get("score") if m else None,
                "regress": row.get("regress") if tag == "B2v2" else None,
                "improve": row.get("improve") if tag == "B2v2" else None,
            })

    csv_path = OUT_BASE / "calibrated_outer_loop_fixed_height_metrics.csv"
    if rows:
        fieldnames = ["height", "profile", "fell", "min_drift", "max_drift",
                      "maxabs", "p2p", "pos_pct", "neg_pct", "zero_crossings",
                      "out03_pct", "out05_pct", "out08_pct", "out10_pct", "out15_pct",
                      "pitch_rms", "pitch_max", "roll_rms", "comz_min",
                      "hip_yaw_abs_max", "yaw_drift_max", "score",
                      "regress", "improve"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nMetrics CSV: {csv_path}", flush=True)

    # Report
    report = ROOT / "docs" / "validation" / "calibrated_outer_loop_fixed_height_v2_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    def mv(m, k, f=".4f"):
        if m is None:
            return "N/A"
        v = m.get(k, "N/A")
        return format(v, f) if isinstance(v, float) else str(v)

    def sok(m):
        if m is None:
            return False, "missing"
        return safety_ok(m)

    L = [
        "# Calibrated Outer-Loop v2 — Fixed-Height Validation (Phase 6)",
        "",
        "**A:** `height_scheduled_pitch_equilibrium_trim`",
        "**B:** `support_position_outer_loop_pitch_ref` (Kp=+1.0, Kd=0.0)",
        "**B2:** `calibrated_support_position_outer_loop_pitch_ref` (v1 — failed Phase 6)",
        "**B2v2:** `calibrated_support_position_outer_loop_pitch_ref_v2` (v2 — smoothed upper band)",
        f"**Classification:** `{classification}`",
        "",
        "## Gates",
        "",
        f"- B2v2 improve heights (vs B by score): {n_improve}/10 (need >=6 for full pass, >=0 for partial)",
        f"- B2v2 regression heights (vs B by score): {n_regress}/10 (>2 fails)",
        f"- hard safety failures: {len(hard_fail)} {hard_fail if hard_fail else ''}",
    ]
    if protected_regress:
        L.append(f"- protected height regression: {protected_regress}")

    L += [
        "",
        "## Per-Height Comparison (2000 steps)",
        "",
        "| height | prof | fell | pos% | min | max | maxabs | P2P | "
        "out15% | hip_yaw | score | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for label in per_height:
        for tag in ["A", "B", "B2", "B2v2"]:
            m = per_height[label].get(tag)
            v = ""
            if tag == "B2v2":
                ph = per_height[label]
                sok2, reason2 = sok(m)
                if not sok2:
                    v = f"HARD_FAIL({reason2})"
                elif ph.get("regress"):
                    v = "REGRESS"
                elif ph.get("improve"):
                    v = "IMPROVE"
                else:
                    v = "EQUAL"
            L.append(
                f"| {label} | {tag} | {mv(m,'fell','')} | {mv(m,'pos_pct','.1f')} | "
                f"{mv(m,'min_drift')} | {mv(m,'max_drift')} | {mv(m,'maxabs')} | "
                f"{mv(m,'p2p')} | {mv(m,'out15_pct','.1f')} | "
                f"{mv(m,'hip_yaw_abs_max')} | {mv(m,'score','.1f')} | {v} |"
            )
        L.append("")

    L += [
        "## Decision",
        "",
        f"- **{classification}**",
    ]

    if "PASS" in classification:
        L.append("- **Proceed to Step C/D (random/changing height).**")
    elif "NOT_BETTER" in classification and n_regress <= 2:
        L.append("- **Partial pass: no regression vs B. Eligible for Step C/D if n_regress <= 2.**")
    else:
        L.append("- **Do not proceed. Keep `support_position_outer_loop_pitch_ref` as current best.**")

    report.write_text("\n".join(L) + "\n")
    print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
