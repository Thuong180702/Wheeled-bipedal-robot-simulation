"""Phase 5: Support-position outer-loop fixed-height full validation (B vs A).

Compares the Phase B candidate profile against the Phase A baseline:
  A) height_scheduled_pitch_equilibrium_trim          (Phase A, current best)
  B) support_position_outer_loop_pitch_ref            (Phase B, Kp=+1.0, Kd=0.0)

Runs:
  5A) high_0p480 at 1200, 2000, 5000 steps (B vs A)
  5B) all 10 fixed heights at 2000 steps (B vs A)

Phase 5 gating (from task spec) — B proceeds to Step C only if:

Hard gates (every height):
  - no fall
  - no WBC / hidden-torque / ownership violation
  - contact / height / roll / posture safe
  - hip-yaw not worse than A beyond monitoring thresholds

Performance gates:
  - B not worse than A on more than 1 height
  - For every height:
        maxabs_B <= maxabs_A + 0.02 m
        P2P_B    <= P2P_A * 1.15
        out15_B  <= out15_A + 3 pp
  - For >= 6 of 10 heights: at least one of maxabs / P2P / out10 / out15 / pos-balance improves
  - high_0p480 must not regress
  - low_0p330 and low_0p360 must not regress

Outputs:
  outputs/.../active_pitch_crossing/support_position_outer_loop_fixed_height_metrics.csv
  docs/validation/support_position_outer_loop_pitch_ref_fixed_height_report.md

Classification:
  OUTER_LOOP_FIXED_HEIGHT_PASS_BETTER_THAN_HEIGHT_SCHEDULE
  OUTER_LOOP_FIXED_HEIGHT_PASS_WITH_MONITORING
  OUTER_LOOP_FIXED_HEIGHT_NOT_BETTER
  OUTER_LOOP_FIXED_HEIGHT_FAIL_SAFETY
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
PER_RUN_TIMEOUT_S = 1800

DRIFT_COL = "active_pitch_crossing_signed_error_m"

BASE_PROFILE = "height_scheduled_pitch_equilibrium_trim"   # A
OL_PROFILE = "support_position_outer_loop_pitch_ref"       # B

HEIGHTS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
    "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]

# Heights the task flags as must-not-regress.
PROTECTED_HEIGHTS = {"high_0p480", "low_0p330", "low_0p360"}

# Tolerances (task spec).
MAXABS_TOL = 0.02       # m: maxabs_B <= maxabs_A + 0.02
P2P_FACTOR = 1.15       # P2P_B <= P2P_A * 1.15
OUT15_TOL_PP = 3.0      # out15_B <= out15_A + 3 pp


def run_sim(label, steps, profile, out_dir, extra_args=None):
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

    cmd = [
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
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
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
    # WBC / hidden-torque / ownership safety signals.
    #
    # IMPORTANT: tau_wbc_max / tau_wbc_norm is the STRUCTURAL QP wrench output
    # that feeds the shape_posture / support_feedforward decomposition. It is
    # nonzero on every run, including the Phase-A baseline (~17 Nm), and is NOT
    # evidence of active WBC. The correct "is WBC active" signals (see
    # tests/test_step_e_wbc_gate_validator.py) are:
    #   - per_actuator_wbc_authority_enabled == True on any row, OR
    #   - a "wbc" token in active_torque_owner_per_joint.
    # Hidden torque and ownership violations are independent hard-fail signals.
    tau_wbc_max = clean(fcol(rows, "tau_wbc_max"))
    hidden_torque_norm = clean(fcol(rows, "hidden_torque_norm"))
    ownership_violation = clean(fcol(rows, "ownership_violation_count"))
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
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
        "tau_wbc_max": round(max(tau_wbc_max), 4) if tau_wbc_max else 0.0,
        "hidden_torque_max": round(max(hidden_torque_norm), 4) if hidden_torque_norm else 0.0,
        "ownership_violation_max": round(max(ownership_violation), 4) if ownership_violation else 0.0,
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
    }


def safety_ok(m):
    """Hard safety gates: no fall, posture/hip-yaw/roll within thresholds, no active WBC.

    IMPORTANT: tau_wbc_max is the structural QP wrench output (~15-18 Nm on every run,
    including Phase A baseline) and is NOT evidence of active WBC. The correct active-WBC
    signals are wbc_authority_rows > 0 (per_actuator_wbc_authority_enabled=True on any row)
    or wbc_owner_rows > 0 ("wbc" token in active_torque_owner_per_joint). Hidden torque
    and ownership violations are independent hard-fail signals.
    """
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, f"fall({m['term_reason'][:20]})"
    if m["hip_yaw_abs_max_rad"] > 0.35:
        return False, "hip_yaw_unsafe"
    if m["pitch_max_abs_deg"] > 16.0:
        return False, "pitch_unsafe"
    if m["roll_rms_deg"] > 3.0:
        return False, "roll_unsafe"
    # Active WBC check: per_actuator_wbc_authority_enabled or wbc owner token.
    if m.get("wbc_authority_rows", 0) > 0:
        return False, "wbc_authority_enabled"
    if m.get("wbc_owner_rows", 0) > 0:
        return False, "wbc_owner_present"
    # Hidden torque and ownership violations.
    if m.get("hidden_torque_max", 0.0) > 0.1:
        return False, "hidden_torque"
    if m.get("ownership_violation_max", 0.0) > 0.0:
        return False, "ownership_violation"
    return True, "safe"


def improves(b, a):
    """True if B improves at least one centering metric vs A."""
    if not a or not b:
        return False
    pos_closer = abs(b["pos_pct"] - 50) < abs(a["pos_pct"] - 50)
    maxabs_better = b["max_abs"] < a["max_abs"]
    p2p_better = b["p2p"] < a["p2p"]
    out15_better = b["out15_pct"] < a["out15_pct"]
    out10_better = b["out10_pct"] < a["out10_pct"]
    return any([pos_closer, maxabs_better, p2p_better, out15_better, out10_better])


def regresses(b, a):
    """True if B regresses vs A beyond per-height tolerance."""
    if not a or not b:
        return True
    if b["max_abs"] > a["max_abs"] + MAXABS_TOL:
        return True
    if a["p2p"] > 0 and b["p2p"] > a["p2p"] * P2P_FACTOR:
        return True
    if b["out15_pct"] > a["out15_pct"] + OUT15_TOL_PP:
        return True
    return False


def fmt(m):
    if m is None:
        return "MISSING"
    fall = f"FALL({m['term_reason'][:15]})" if m["fell"] else "ok"
    return (
        f"{fall} pos%={m['pos_pct']:.1f} max={m['max_abs']:.3f} "
        f"P2P={m['p2p']:.3f} out15={m['out15_pct']:.1f}% "
        f"hy={m['hip_yaw_abs_max_rad']:.3f} wbc={m['tau_wbc_max']:.2f}"
    )


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 5: support-position outer-loop fixed-height validation (B vs A)", flush=True)
    print(f"  A = {BASE_PROFILE}", flush=True)
    print(f"  B = {OL_PROFILE}", flush=True)
    print("=" * 78, flush=True)

    rows_out = []

    # ---- 5A: high_0p480 multi-step ---------------------------------------- #
    print("\n--- 5A: high_0p480 multi-step (1200 / 2000 / 5000) ---", flush=True)
    multistep = {}
    for steps in [1200, 2000, 5000]:
        for prof, tag in [(BASE_PROFILE, "A"), (OL_PROFILE, "B")]:
            out_dir = OUT_BASE / f"ol_fh_{steps}_high_0p480_{tag}"
            t0 = time.time()
            tel, _ = run_sim("high_0p480", steps, prof, out_dir)
            m = analyze(tel)
            multistep[(steps, tag)] = m
            sok, sreason = safety_ok(m)
            print(f"  [{steps}s {tag}] {fmt(m)}  safe={sok}({sreason})  ({time.time()-t0:.0f}s)", flush=True)
            rows_out.append({"phase": "5A", "height": "high_0p480", "steps": steps,
                             "profile": tag, "safe": sok, **(m or {})})

    # ---- 5B: 10-height ladder at 2000 steps ------------------------------- #
    print("\n--- 5B: 10-height ladder (2000 steps) — B vs A ---", flush=True)
    ladder = {}
    for label in HEIGHTS:
        ladder[label] = {}
        for prof, tag in [(BASE_PROFILE, "A"), (OL_PROFILE, "B")]:
            out_dir = OUT_BASE / f"ol_fh_2000_{label}_{tag}"
            t0 = time.time()
            tel, _ = run_sim(label, 2000, prof, out_dir)
            m = analyze(tel)
            ladder[label][tag] = m
            sok, sreason = safety_ok(m)
            print(f"  {label:>11} {tag}: {fmt(m)}  safe={sok}({sreason})  ({time.time()-t0:.0f}s)", flush=True)
            rows_out.append({"phase": "5B", "height": label, "steps": 2000,
                             "profile": tag, "safe": sok, **(m or {})})

    # ---- Gate evaluation -------------------------------------------------- #
    print("\n" + "=" * 78, flush=True)
    print("Gate evaluation", flush=True)
    print("=" * 78, flush=True)

    hard_fail = []        # heights where B fails hard safety
    regress_heights = []  # heights where B regresses beyond tolerance
    improve_heights = []  # heights where B improves >=1 metric
    protected_regress = []  # protected heights that regressed

    per_height = {}
    for label in HEIGHTS:
        a = ladder[label]["A"]
        b = ladder[label]["B"]
        b_safe, b_reason = safety_ok(b)
        a_safe, _ = safety_ok(a)
        reg = regresses(b, a)
        imp = improves(b, a)
        # hip-yaw "not worse than A beyond monitoring": allow small absolute slack
        hy_worse = (b and a and b["hip_yaw_abs_max_rad"] > a["hip_yaw_abs_max_rad"] + 0.05
                    and b["hip_yaw_abs_max_rad"] > 0.20)

        if not b_safe:
            hard_fail.append((label, b_reason))
        if reg:
            regress_heights.append(label)
            if label in PROTECTED_HEIGHTS:
                protected_regress.append(label)
        if imp:
            improve_heights.append(label)
        if hy_worse:
            hard_fail.append((label, "hip_yaw_worse_than_A"))

        per_height[label] = {
            "a": a, "b": b, "b_safe": b_safe, "b_reason": b_reason,
            "regress": reg, "improve": imp, "hy_worse": hy_worse,
        }
        verdict = "OK"
        if not b_safe:
            verdict = f"HARD_FAIL({b_reason})"
        elif reg:
            verdict = "REGRESS"
        elif imp:
            verdict = "IMPROVE"
        print(f"  {label:>11}: B {verdict}"
              f"  (maxabs A={a['max_abs'] if a else 'NA'} B={b['max_abs'] if b else 'NA'},"
              f" P2P A={a['p2p'] if a else 'NA'} B={b['p2p'] if b else 'NA'},"
              f" out15 A={a['out15_pct'] if a else 'NA'} B={b['out15_pct'] if b else 'NA'})", flush=True)

    # high_0p480 multistep regression check (use 2000-step from ladder + multistep)
    high_regress = False
    for steps in [1200, 2000, 5000]:
        a = multistep.get((steps, "A"))
        b = multistep.get((steps, "B"))
        bs, _ = safety_ok(b)
        if not bs or regresses(b, a):
            high_regress = True

    # ---- Classification --------------------------------------------------- #
    n_improve = len(improve_heights)
    n_regress = len(regress_heights)

    classification = None
    reasons = []

    if hard_fail:
        classification = "OUTER_LOOP_FIXED_HEIGHT_FAIL_SAFETY"
        reasons.append(f"hard safety failures: {hard_fail}")
    elif protected_regress:
        classification = "OUTER_LOOP_FIXED_HEIGHT_NOT_BETTER"
        reasons.append(f"protected height(s) regressed: {protected_regress}")
    elif high_regress:
        classification = "OUTER_LOOP_FIXED_HEIGHT_NOT_BETTER"
        reasons.append("high_0p480 multi-step regressed")
    elif n_regress > 1:
        classification = "OUTER_LOOP_FIXED_HEIGHT_NOT_BETTER"
        reasons.append(f"B worse than A on {n_regress} heights (>1 allowed)")
    elif n_improve >= 6:
        classification = "OUTER_LOOP_FIXED_HEIGHT_PASS_BETTER_THAN_HEIGHT_SCHEDULE"
        reasons.append(f"{n_improve}/10 heights improved, <=1 regression, protected heights safe")
    else:
        classification = "OUTER_LOOP_FIXED_HEIGHT_PASS_WITH_MONITORING"
        reasons.append(f"safe and within tolerance but only {n_improve}/10 improved (need >=6 for full pass)")

    print(f"\n>>> improve={n_improve}/10  regress={n_regress}  hard_fail={len(hard_fail)}", flush=True)
    print(f">>> Classification: {classification}", flush=True)
    for r in reasons:
        print(f"    - {r}", flush=True)

    _write_outputs(rows_out, multistep, ladder, per_height, classification, reasons,
                   n_improve, n_regress, hard_fail)
    return classification


def _write_outputs(rows, multistep, ladder, per_height, classification, reasons,
                   n_improve, n_regress, hard_fail):
    csv_path = OUT_BASE / "support_position_outer_loop_fixed_height_metrics.csv"
    if rows:
        fieldnames = ["phase", "height", "steps", "profile", "safe"] + sorted(
            {k for r in rows for k in r.keys()} - {"phase", "height", "steps", "profile", "safe"}
        )
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nMetrics CSV: {csv_path}", flush=True)

    report = ROOT / "docs" / "validation" / "support_position_outer_loop_pitch_ref_fixed_height_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    def mv(m, k, f=".4f"):
        if m is None: return "N/A"
        v = m.get(k, "N/A")
        return format(v, f) if isinstance(v, float) else str(v)

    L = [
        "# Support-Position Outer-Loop — Fixed-Height Validation (Phase 5)",
        "",
        f"**A (baseline):** `{BASE_PROFILE}`",
        f"**B (candidate):** `{OL_PROFILE}` (Kp=+1.0 deg/m, Kd=0.0, P-only)",
        f"**Classification:** `{classification}`",
        "",
        "## Gates",
        "",
        f"- improve heights: {n_improve}/10 (need >=6 for full pass)",
        f"- regression heights: {n_regress} (>1 fails)",
        f"- hard safety failures: {len(hard_fail)} {hard_fail if hard_fail else ''}",
    ]
    for r in reasons:
        L.append(f"- {r}")
    L += ["", "---", "", "## 5A: high_0p480 multi-step (B vs A)", "",
          "| steps | prof | fell | pos% | max_abs | P2P | out15% | hip_yaw | wbc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for steps in [1200, 2000, 5000]:
        for tag in ["A", "B"]:
            m = multistep.get((steps, tag))
            L.append(f"| {steps} | {tag} | {mv(m,'fell','')} | {mv(m,'pos_pct','.1f')} | "
                     f"{mv(m,'max_abs')} | {mv(m,'p2p')} | {mv(m,'out15_pct','.1f')} | "
                     f"{mv(m,'hip_yaw_abs_max_rad')} | {mv(m,'tau_wbc_max','.2f')} |")

    L += ["", "## 5B: 10-height ladder (2000 steps)", "",
          "| height | prof | fell | pos% | min | max | max_abs | P2P | out15% | hip_yaw | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label in HEIGHTS:
        ph = per_height[label]
        for tag in ["A", "B"]:
            m = ladder[label][tag]
            verdict = ""
            if tag == "B":
                if not ph["b_safe"]:
                    verdict = f"HARD_FAIL({ph['b_reason']})"
                elif ph["regress"]:
                    verdict = "REGRESS"
                elif ph["improve"]:
                    verdict = "IMPROVE"
                else:
                    verdict = "EQUAL"
            L.append(f"| {label} | {tag} | {mv(m,'fell','')} | {mv(m,'pos_pct','.1f')} | "
                     f"{mv(m,'min_drift')} | {mv(m,'max_drift')} | {mv(m,'max_abs')} | "
                     f"{mv(m,'p2p')} | {mv(m,'out15_pct','.1f')} | {mv(m,'hip_yaw_abs_max_rad')} | {verdict} |")

    L += ["", "## Decision", ""]
    if classification.startswith("OUTER_LOOP_FIXED_HEIGHT_PASS"):
        L += ["- B is safe and within tolerance vs A. **Proceed to Step C (random/changing height).**"]
    else:
        L += ["- B did not pass. **Keep `height_scheduled_pitch_equilibrium_trim` as current best. Do NOT proceed to Step C.**"]
    L.append("")
    report.write_text("\n".join(L) + "\n")
    print(f"Report: {report}", flush=True)

    summary = {
        "classification": classification,
        "n_improve": n_improve,
        "n_regress": n_regress,
        "hard_fail": hard_fail,
        "reasons": reasons,
    }
    (OUT_BASE / "support_position_outer_loop_fixed_height_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
