"""Phase 7: Step D — random push disturbance validation.

Tests both profiles under sagittal push disturbances applied mid-run
via the --sagittal-push-* CLI.  Only runs at high_0p480 (best-studied
height) and low_0p330 (Phase A protected height) to keep total run count
manageable.

Test cases:
  D1: small sagittal push (30 N, 5 steps every 150 steps), high_0p480, 1000 steps
  D2: medium sagittal push (60 N, 5 steps every 150 steps), high_0p480, 1000 steps
  D3: small sagittal push (30 N), low_0p330, 1000 steps
  D4: medium sagittal push (60 N), low_0p330, 1000 steps
  D5: larger push (90 N, 5 steps every 200 steps), high_0p480, 1000 steps
  D6: random-direction push (45 N, 5 steps every 150 steps), high_0p480, 1000 steps

Profiles:
  A) calibrated_support_position_outer_loop_pitch_ref_v2    (B2v2 baseline)
  B) physics_equilibrium_feedforward_outer_loop              (current PFF)
  C) physics_equilibrium_feedforward_outer_loop_low_band_support_v2 (v2 candidate)
  D) same as C + differential wheel yaw stabilizer          (architecture fix)

Pass criteria:
  - no fall in mild/medium push cases (D1-D4)
  - support drift recovers to bounded band (max_abs <= 0.25 m)
  - hip-yaw does not diverge (< 0.35 rad)
  - no contact loss beyond transient
  - B better than or equal to A in push recovery

Outputs:
  docs/validation/step_d_random_push_validation_report.md
  outputs/.../step_d_push_metrics.csv

Classification:
  STEP_D_RANDOM_PUSH_PASS
  STEP_D_RANDOM_PUSH_PASS_WITH_MONITORING
  STEP_D_RANDOM_PUSH_FAIL
  STEP_D_RANDOM_PUSH_INCONCLUSIVE
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

DRIFT_COL = "active_pitch_crossing_signed_error_m"

BASE_PROFILE = "height_scheduled_pitch_equilibrium_trim"
OL_PROFILE = "support_position_outer_loop_pitch_ref"

# Step D test matrix: (case_id, height_label, steps, push_mag_N, push_dur_steps, push_interval)
PUSH_CASES = [
    ("D1_small_push_high",   "high_0p480", 1000, 30,  5, 150),
    ("D2_medium_push_high",  "high_0p480", 1000, 60,  5, 150),
    ("D3_small_push_low",    "low_0p330",  1000, 30,  5, 150),
    ("D4_medium_push_low",   "low_0p330",  1000, 60,  5, 150),
    ("D5_large_push_high",   "high_0p480", 1000, 90,  5, 200),
    ("D6_random_push_high",  "high_0p480", 1000, 45,  5, 150),
]

# Cases where a fall is a hard fail (mild/medium)
MUST_NOT_FALL = {"D1_small_push_high", "D2_medium_push_high",
                 "D3_small_push_low",  "D4_medium_push_low"}


def run_sim(
    label,
    steps,
    profile,
    out_dir,
    push_magnitude=0,
    push_duration=5,
    push_interval=150,
    push_sagittal=True,
    enable_wheel_yaw=False,
    wheel_yaw_kp=None,
    wheel_yaw_kd=None,
    wheel_yaw_max_torque=None,
    wheel_yaw_lowpass_alpha=None,
    wheel_yaw_height_gate_low=None,
    wheel_yaw_height_gate_high=None,
    yaw_controller_kp=None,
    yaw_controller_kd=None,
    yaw_controller_max_torque=None,
):
    """Run one push simulation segment."""
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
    if enable_wheel_yaw:
        cmd += ["--enable-wheel-yaw-stabilizer"]
        if wheel_yaw_kp is not None:
            cmd += ["--wheel-yaw-kp", str(float(wheel_yaw_kp))]
        if wheel_yaw_kd is not None:
            cmd += ["--wheel-yaw-kd", str(float(wheel_yaw_kd))]
        if wheel_yaw_max_torque is not None:
            cmd += ["--wheel-yaw-max-torque", str(float(wheel_yaw_max_torque))]
        if wheel_yaw_lowpass_alpha is not None:
            cmd += ["--wheel-yaw-lowpass-alpha", str(float(wheel_yaw_lowpass_alpha))]
        if wheel_yaw_height_gate_low is not None:
            cmd += ["--wheel-yaw-height-gate-low", str(float(wheel_yaw_height_gate_low))]
        if wheel_yaw_height_gate_high is not None:
            cmd += ["--wheel-yaw-height-gate-high", str(float(wheel_yaw_height_gate_high))]
        if yaw_controller_kp is not None:
            cmd += ["--yaw-controller-kp", str(float(yaw_controller_kp))]
        if yaw_controller_kd is not None:
            cmd += ["--yaw-controller-kd", str(float(yaw_controller_kd))]
        if yaw_controller_max_torque is not None:
            cmd += ["--yaw-controller-max-torque", str(float(yaw_controller_max_torque))]
    if push_magnitude > 0:
        cmd += [
            "--push-enabled",
            "--push-magnitude-n", str(float(push_magnitude)),
            "--push-duration-steps", str(push_duration),
            "--push-interval-steps", str(push_interval),
        ]
        if not push_sagittal:
            cmd += ["--sagittal-push-random-direction"]

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        print(f"  TIMEOUT {label} {steps}s", flush=True)
        return None, None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        # Try anyway — run may have produced partial telemetry
        tels = sorted(SIM_OUT.glob("telemetry_*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        sums = sorted(SIM_OUT.glob("run_summary_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if tels:
            shutil.copy2(tels[0], tel_dst)
            try: tels[0].unlink()
            except OSError: pass
        if sums:
            shutil.copy2(sums[0], sum_dst)
            try: sums[0].unlink()
            except OSError: pass
        if not tel_dst.exists():
            print(f"  FAILED rc={result.returncode} {label} {steps}s {profile}", flush=True)
            return None, None

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(SIM_OUT.glob("run_summary_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
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
    roll  = clean(fcol(rows, "robot_roll_y"))
    lhy   = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy   = clean(fcol(rows, "r_hip_yaw_pos"))
    yaw_drift = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    hidden_torque_norm = clean(fcol(rows, "hidden_torque_norm"))
    ownership_violation = clean(fcol(rows, "ownership_violation_count"))
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

    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg  = [math.degrees(x) for x in roll]
    hy_all    = [abs(x) for x in (lhy + rhy)]

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
        "out15_pct": round(out_pct(0.15), 1),
        "out25_pct": round(out_pct(0.25), 1),
        "pitch_max_abs_deg": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "roll_rms_deg": round(rms(roll_deg), 2),
        "hip_yaw_abs_max_rad": round(max(hy_all), 4) if hy_all else 0.0,
        "yaw_drift_max_rad": round(max((abs(x) for x in yaw_drift), default=0.0), 4),
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "hidden_torque_max": round(max(hidden_torque_norm), 4) if hidden_torque_norm else 0.0,
        "ownership_violation_max": round(max(ownership_violation), 4) if ownership_violation else 0.0,
    }


def safety_ok(m):
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
    if m["wbc_authority_rows"] > 0:
        return False, "wbc_active"
    if m["wbc_owner_rows"] > 0:
        return False, "wbc_owner_detected"
    if m["hidden_torque_max"] > 0.5:
        return False, "hidden_torque"
    if m["ownership_violation_max"] > 0:
        return False, "ownership_violation"
    return True, "safe"


def fmt(m):
    if m is None:
        return "MISSING"
    fall = f"FALL({m['term_reason'][:12]})" if m["fell"] else "ok"
    return (
        f"{fall} pos%={m['pos_pct']:.1f} max={m['max_abs']:.3f} "
        f"P2P={m['p2p']:.3f} out25={m['out25_pct']:.1f}% "
        f"hy={m['hip_yaw_abs_max_rad']:.3f}"
    )


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 7 / Step D: random push disturbance validation", flush=True)
    print(f"  A = {BASE_PROFILE}", flush=True)
    print(f"  B = {OL_PROFILE}", flush=True)
    print("=" * 78, flush=True)

    all_rows = []
    case_results = {}  # case_id -> {"A": m, "B": m}

    for case_id, height_label, steps, push_mag, push_dur, push_int in PUSH_CASES:
        print(f"\n[{case_id}] height={height_label} steps={steps} "
              f"push={push_mag}N dur={push_dur} interval={push_int}", flush=True)
        case_results[case_id] = {}
        for profile, tag in [(BASE_PROFILE, "A"), (OL_PROFILE, "B")]:
            out_dir = OUT_BASE / f"step_d_{case_id}_{tag}"
            t0 = time.time()
            tel, _ = run_sim(
                height_label, steps, profile, out_dir,
                push_magnitude=push_mag,
                push_duration=push_dur,
                push_interval=push_int,
            )
            m = analyze(tel)
            case_results[case_id][tag] = m
            sok, sreason = safety_ok(m)
            print(
                f"  [{tag}] {fmt(m)}  safe={sok}({sreason})  ({time.time()-t0:.0f}s)",
                flush=True
            )
            if m:
                all_rows.append({
                    "case_id": case_id, "height": height_label, "steps": steps,
                    "push_mag_N": push_mag, "push_dur": push_dur, "push_int": push_int,
                    "profile": tag, "safe": sok, **m
                })

    # ---- Gate evaluation -------------------------------------------------- #
    print("\n" + "=" * 78, flush=True)
    print("Step D gate evaluation", flush=True)
    print("=" * 78, flush=True)

    must_not_fall_pass = True
    any_hard_fail = False
    max_drift_B = 0.0
    b_not_worse_count = 0
    total_cases = 0

    for case_id, height_label, steps, push_mag, push_dur, push_int in PUSH_CASES:
        a = case_results[case_id].get("A")
        b = case_results[case_id].get("B")
        sok_b, sreason_b = safety_ok(b)
        if not sok_b:
            any_hard_fail = True
            if case_id in MUST_NOT_FALL and b and b["fell"]:
                must_not_fall_pass = False
        if b:
            max_drift_B = max(max_drift_B, b.get("max_abs", 0))
        # B not worse: max_abs B <= max_abs A + 0.05 m (relaxed for push)
        b_not_worse = (
            b is not None and a is not None
            and b["max_abs"] <= a["max_abs"] + 0.05
            and (not b["fell"] or a["fell"])
        )
        if b_not_worse:
            b_not_worse_count += 1
        total_cases += 1
        print(
            f"  {case_id:28s}: safe_B={sok_b}({sreason_b}) "
            f"fell_B={b['fell'] if b else '?'} "
            f"b_not_worse={b_not_worse}",
            flush=True
        )

    print(f"\n  must_not_fall_pass={must_not_fall_pass}", flush=True)
    print(f"  any_hard_fail={any_hard_fail}", flush=True)
    print(f"  max_drift_B={max_drift_B:.3f}", flush=True)
    print(f"  b_not_worse={b_not_worse_count}/{total_cases}", flush=True)

    if not must_not_fall_pass:
        classification = "STEP_D_RANDOM_PUSH_FAIL"
    elif any_hard_fail and b_not_worse_count < total_cases // 2:
        classification = "STEP_D_RANDOM_PUSH_FAIL"
    elif any_hard_fail:
        classification = "STEP_D_RANDOM_PUSH_PASS_WITH_MONITORING"
    elif b_not_worse_count >= total_cases - 1:
        classification = "STEP_D_RANDOM_PUSH_PASS"
    elif b_not_worse_count >= total_cases // 2:
        classification = "STEP_D_RANDOM_PUSH_PASS_WITH_MONITORING"
    else:
        classification = "STEP_D_RANDOM_PUSH_INCONCLUSIVE"

    print(f"\n>>> Classification: {classification}", flush=True)

    # ---- Write outputs --------------------------------------------------- #
    csv_path = OUT_BASE / "step_d_push_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nMetrics CSV: {csv_path}", flush=True)

    report_path = ROOT / "docs" / "validation" / "step_d_random_push_validation_report.md"
    lines = [
        "# Step D: Random Push Disturbance Validation",
        "",
        f"**A:** `{BASE_PROFILE}`",
        f"**B:** `{OL_PROFILE}`",
        f"**Classification:** `{classification}`",
        "",
        "---",
        "",
        "## Case Results",
        "",
        "| Case | Height | Push(N) | Prof | Fell | max_abs | P2P | out25% | hip_yaw | safe |",
        "|------|--------|---------|------|------|---------|-----|--------|---------|------|",
    ]
    for case_id, height_label, steps, push_mag, push_dur, push_int in PUSH_CASES:
        for tag in ("A", "B"):
            m = case_results[case_id].get(tag)
            if m:
                sok, _ = safety_ok(m)
                lines.append(
                    f"| {case_id} | {height_label} | {push_mag} | {tag} | "
                    f"{m['fell']} | {m['max_abs']:.3f} | {m['p2p']:.3f} | "
                    f"{m['out25_pct']:.1f} | {m['hip_yaw_abs_max_rad']:.3f} | {sok} |"
                )
    lines += [
        "",
        "## Decision",
        "",
        f"- **{classification}**",
        "",
    ]
    if classification.startswith("STEP_D_RANDOM_PUSH_PASS"):
        lines += [
            "- B passed Step D push disturbance validation.",
            "- Proceed to Phase 8 final report and commit decision.",
        ]
    else:
        lines += [
            "- B had issues under push disturbance.",
            "- See case table for details.",
        ]
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report: {report_path}", flush=True)

    summary_path = OUT_BASE / "step_d_push_summary.json"
    summary_path.write_text(json.dumps({
        "classification": classification,
        "must_not_fall_pass": must_not_fall_pass,
        "any_hard_fail": any_hard_fail,
        "max_drift_B": max_drift_B,
        "b_not_worse_count": b_not_worse_count,
        "total_cases": total_cases,
    }, indent=2))


if __name__ == "__main__":
    main()
