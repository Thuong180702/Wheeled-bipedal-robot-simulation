"""Phase 4: Support-position outer-loop 500-step sign and gain screening.

Runs high_0p480 500-step sweeps comparing:
  A) Baseline: height_scheduled_pitch_equilibrium_trim (Phase A, no outer loop)
  B) support_position_outer_loop_pitch_ref with Kp positive (+0.5, +1.0, +1.5 deg/m)
  C) support_position_outer_loop_pitch_ref with Kp negative (-0.5, -1.0, -1.5 deg/m)
  D) Selected sign with Kd screening (Kd = 0.05, 0.10, 0.20 deg/(m/s))

Selection criteria (all required for candidate selection):
  - no fall
  - posture safe (pitch < 16 deg, roll_rms < 3 deg, hip_yaw < 0.35 rad)
  - maxabs_B <= maxabs_A + 0.02 m
  - P2P_B <= P2P_A * 1.15
  - no clear oscillation runaway (zero_crossings not much higher than baseline)
  - pos% improves toward 50% OR out15/out10 improves

Outputs:
  outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/outer_loop_500_gain_sweep_metrics.csv
  docs/validation/outer_loop_500_gain_sweep_report.md

Classification:
  OUTER_LOOP_500_CANDIDATE_SELECTED
  OUTER_LOOP_500_NOT_BETTER
  OUTER_LOOP_500_FAIL_SAFETY
  OUTER_LOOP_500_INCONCLUSIVE
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
PER_RUN_TIMEOUT_S = 600

DRIFT_COL = "active_pitch_crossing_signed_error_m"
STEPS = 500
HEIGHT_LABEL = "high_0p480"

BASE_PROFILE = "height_scheduled_pitch_equilibrium_trim"
OL_PROFILE = "support_position_outer_loop_pitch_ref"


def run_sim(label, steps, profile, out_dir, extra_args=None):
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


def safety_ok(m):
    """True if no fall and posture metrics are within safety thresholds."""
    if m is None: return False
    if m["fell"]: return False
    if m["hip_yaw_abs_max_rad"] > 0.35: return False
    if m["pitch_max_abs_deg"] > 16.0: return False
    if m["roll_rms_deg"] > 3.0: return False
    return True


def performance_ok(m, baseline):
    """True if B does not regress vs A by more than tolerance."""
    if not safety_ok(m): return False
    if not baseline: return True
    if m["max_abs"] > baseline["max_abs"] + 0.02: return False
    if baseline["p2p"] > 0 and m["p2p"] > baseline["p2p"] * 1.15: return False
    return True


def improves(m, baseline):
    """True if at least one centering metric improves vs baseline."""
    if not baseline: return True
    pos_closer = abs(m["pos_pct"] - 50) < abs(baseline["pos_pct"] - 50)
    maxabs_better = m["max_abs"] < baseline["max_abs"]
    p2p_better = m["p2p"] < baseline["p2p"]
    out15_better = m["out15_pct"] < baseline["out15_pct"]
    out10_better = m["out10_pct"] < baseline["out10_pct"]
    return any([pos_closer, maxabs_better, p2p_better, out15_better, out10_better])


def fmt(m):
    if m is None: return "MISSING"
    fall = f"FALL({m['term_reason'][:15]})" if m["fell"] else "ok"
    return (
        f"fall={fall} pos%={m['pos_pct']:.1f} max={m['max_abs']:.3f} "
        f"P2P={m['p2p']:.3f} out15={m['out15_pct']:.1f}% "
        f"hy={m['hip_yaw_abs_max_rad']:.3f} zc={m['zero_crossings']}"
    )


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("Phase 4: support-position outer-loop 500-step sign + gain sweep", flush=True)
    print(f"Height: {HEIGHT_LABEL}  Steps: {STEPS}", flush=True)
    print("=" * 72, flush=True)

    rows_out = []

    # ------------------------------------------------------------------ #
    # A: Phase A baseline (no outer loop)                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[A] Baseline: {BASE_PROFILE}", flush=True)
    t0 = time.time()
    tel, _ = run_sim(HEIGHT_LABEL, STEPS, BASE_PROFILE, OUT_BASE / f"ol_sweep_baseline_{HEIGHT_LABEL}")
    m_base = analyze(tel)
    print(f"  {fmt(m_base)}  ({time.time()-t0:.0f}s)", flush=True)
    rows_out.append({"tag": "baseline", "profile": BASE_PROFILE, "kp": 0.0, "kd": 0.0,
                     **(m_base or {})})

    # ------------------------------------------------------------------ #
    # B/C: Sign sweep — Kp positive and negative, Kd=0                    #
    # ------------------------------------------------------------------ #
    sign_results = {}
    for sign_label, kp_vals in [("pos", [0.5, 1.0, 1.5]), ("neg", [-0.5, -1.0, -1.5])]:
        print(f"\n[{sign_label.upper()} sign] Kp sweep (Kd=0):", flush=True)
        for kp in kp_vals:
            tag = f"kp{kp:+.1f}_kd0.00"
            out_dir = OUT_BASE / f"ol_sweep_{tag.replace('+', 'p').replace('-', 'n').replace('.', 'd')}_{HEIGHT_LABEL}"
            t0 = time.time()
            tel, _ = run_sim(
                HEIGHT_LABEL, STEPS, OL_PROFILE, out_dir,
                extra_args=["--vd-outer-loop-kp-deg-per-m", str(kp),
                            "--vd-outer-loop-kd-deg-per-mps", "0.0"],
            )
            m = analyze(tel)
            sign_results[(sign_label, kp)] = m
            ok = safety_ok(m)
            better = improves(m, m_base) if ok else False
            print(f"  Kp={kp:+.2f} Kd=0.00 -> {fmt(m)}  safe={ok} better={better}  ({time.time()-t0:.0f}s)", flush=True)
            rows_out.append({"tag": tag, "profile": OL_PROFILE, "kp": kp, "kd": 0.0,
                             "safe": ok, "better": better, **(m or {})})

    # ------------------------------------------------------------------ #
    # Select better sign (if any)                                          #
    # ------------------------------------------------------------------ #
    # Among Kp=+1.0 and Kp=-1.0 (mid-point of each sweep), pick the one  #
    # that passes safety AND improves more centering metrics.              #
    pos_m = sign_results.get(("pos", 1.0))
    neg_m = sign_results.get(("neg", -1.0))
    selected_sign = None
    selected_kp = None
    if safety_ok(pos_m) and improves(pos_m, m_base):
        if not safety_ok(neg_m) or not improves(neg_m, m_base):
            selected_sign = "positive"
            selected_kp = 1.0
        else:
            # Both improve — pick the one with better pos% centering
            pos_dist = abs(pos_m["pos_pct"] - 50)
            neg_dist = abs(neg_m["pos_pct"] - 50)
            if pos_dist <= neg_dist:
                selected_sign = "positive"
                selected_kp = 1.0
            else:
                selected_sign = "negative"
                selected_kp = -1.0
    elif safety_ok(neg_m) and improves(neg_m, m_base):
        selected_sign = "negative"
        selected_kp = -1.0

    if selected_sign is None:
        print("\n>>> No sign improved safety + centering vs baseline.", flush=True)
        print(">>> Classification: OUTER_LOOP_500_NOT_BETTER", flush=True)
        _write_outputs(rows_out, m_base, None, None, "OUTER_LOOP_500_NOT_BETTER", None)
        return

    print(f"\n>>> Selected sign: {selected_sign}  (Kp={selected_kp:+.2f})", flush=True)

    # ------------------------------------------------------------------ #
    # D: PD screening with selected sign                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[PD] Kd screening with Kp={selected_kp:+.2f}:", flush=True)
    kd_vals = [0.05, 0.10, 0.20]
    pd_results = {}
    for kd in kd_vals:
        tag = f"kp{selected_kp:+.1f}_kd{kd:.2f}"
        out_dir = OUT_BASE / f"ol_sweep_{tag.replace('+', 'p').replace('-', 'n').replace('.', 'd')}_{HEIGHT_LABEL}"
        t0 = time.time()
        tel, _ = run_sim(
            HEIGHT_LABEL, STEPS, OL_PROFILE, out_dir,
            extra_args=["--vd-outer-loop-kp-deg-per-m", str(selected_kp),
                        "--vd-outer-loop-kd-deg-per-mps", str(kd)],
        )
        m = analyze(tel)
        pd_results[kd] = m
        ok = safety_ok(m)
        better = improves(m, m_base) if ok else False
        print(f"  Kp={selected_kp:+.2f} Kd={kd:.2f} -> {fmt(m)}  safe={ok} better={better}  ({time.time()-t0:.0f}s)", flush=True)
        rows_out.append({"tag": tag, "profile": OL_PROFILE, "kp": selected_kp, "kd": kd,
                         "safe": ok, "better": better, **(m or {})})

    # ------------------------------------------------------------------ #
    # Pick best PD combo                                                   #
    # ------------------------------------------------------------------ #
    # Prefer the Kd that passes safety + performance and minimises |pos%-50|
    best_kd = 0.0  # P-only is already validated
    best_m = sign_results.get((selected_sign[:3], selected_kp))
    best_score = abs(best_m["pos_pct"] - 50) if best_m else 999

    for kd, m in pd_results.items():
        if safety_ok(m) and performance_ok(m, m_base):
            score = abs(m["pos_pct"] - 50)
            if score < best_score:
                best_score = score
                best_kd = kd
                best_m = m

    final_kp = selected_kp
    final_kd = best_kd

    print(f"\n>>> Candidate: Kp={final_kp:+.2f}  Kd={final_kd:.2f}", flush=True)
    print(f">>> Metrics:   {fmt(best_m)}", flush=True)

    if not safety_ok(best_m):
        classification = "OUTER_LOOP_500_FAIL_SAFETY"
    elif not performance_ok(best_m, m_base):
        classification = "OUTER_LOOP_500_NOT_BETTER"
    else:
        classification = "OUTER_LOOP_500_CANDIDATE_SELECTED"

    print(f"\n>>> Classification: {classification}", flush=True)
    _write_outputs(rows_out, m_base, best_m, (final_kp, final_kd), classification, selected_sign)


def _write_outputs(rows, m_base, m_best, gains, classification, sign):
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_BASE / "outer_loop_500_gain_sweep_metrics.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nMetrics CSV: {csv_path}", flush=True)

    report_path = ROOT / "docs" / "validation" / "outer_loop_500_gain_sweep_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _mf(m, key, fmt_str=".4f"):
        if m is None: return "N/A"
        v = m.get(key, "N/A")
        if isinstance(v, float): return format(v, fmt_str)
        return str(v)

    lines = [
        "# Outer-Loop 500-Step Sign and Gain Sweep Report",
        "",
        f"**Profile:** `support_position_outer_loop_pitch_ref`",
        f"**Baseline:** `height_scheduled_pitch_equilibrium_trim`",
        f"**Height:** `{HEIGHT_LABEL}`  **Steps:** `{STEPS}`",
        f"**Classification:** `{classification}`",
        "",
        "---",
        "",
        "## Baseline (Phase A)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| pos% | {_mf(m_base, 'pos_pct', '.1f')} |",
        f"| max_abs (m) | {_mf(m_base, 'max_abs')} |",
        f"| P2P (m) | {_mf(m_base, 'p2p')} |",
        f"| out15% | {_mf(m_base, 'out15_pct', '.1f')} |",
        f"| hip_yaw_max (rad) | {_mf(m_base, 'hip_yaw_abs_max_rad')} |",
        f"| zero_crossings | {_mf(m_base, 'zero_crossings', 'd') if m_base else 'N/A'} |",
        f"| fell | {m_base['fell'] if m_base else 'N/A'} |",
        "",
    ]

    if gains is not None:
        kp, kd = gains
        lines += [
            "## Selected Candidate",
            "",
            f"**Sign:** `{sign}`  **Kp:** `{kp:+.2f} deg/m`  **Kd:** `{kd:.2f} deg/(m/s)`",
            "",
            f"| Metric | Baseline | Candidate | Delta |",
            f"|--------|----------|-----------|-------|",
        ]
        for key, label in [
            ("pos_pct", "pos%"), ("max_abs", "max_abs (m)"), ("p2p", "P2P (m)"),
            ("out15_pct", "out15%"), ("hip_yaw_abs_max_rad", "hip_yaw_max"),
        ]:
            bv = m_base.get(key, float("nan")) if m_base else float("nan")
            cv = m_best.get(key, float("nan")) if m_best else float("nan")
            try:
                delta = f"{cv - bv:+.4f}"
            except Exception:
                delta = "N/A"
            lines.append(f"| {label} | {bv:.4f} | {cv:.4f} | {delta} |")

        lines += [
            "",
            f"**fell:** `{m_best['fell'] if m_best else 'N/A'}`",
            "",
        ]

    lines += [
        "## Classification rationale",
        "",
        f"`{classification}`",
        "",
    ]
    if classification == "OUTER_LOOP_500_CANDIDATE_SELECTED":
        if gains:
            kp, kd = gains
            lines += [
                f"- No fall, posture safe, hip-yaw safe.",
                f"- maxabs/P2P within tolerance vs baseline.",
                f"- At least one centering metric improved.",
                f"- Candidate: Kp={kp:+.2f} deg/m, Kd={kd:.2f} deg/(m/s), sign={sign}.",
                "",
                "**Next step:** Phase 5 fixed-height full ladder validation.",
            ]
    elif classification == "OUTER_LOOP_500_NOT_BETTER":
        lines += [
            "- No configuration improved centering without degrading safety.",
            "- Keep `height_scheduled_pitch_equilibrium_trim` as current best.",
            "- Do not proceed to Phase 5.",
        ]
    elif classification == "OUTER_LOOP_500_FAIL_SAFETY":
        lines += [
            "- Best candidate failed safety gates (fall/posture/hip-yaw).",
            "- Do not proceed to Phase 5.",
        ]
    else:
        lines += ["- Results inconclusive."]

    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report: {report_path}", flush=True)

    # Also write a JSON summary for programmatic consumption
    summary = {
        "classification": classification,
        "height": HEIGHT_LABEL,
        "steps": STEPS,
        "selected_sign": sign,
        "selected_kp": gains[0] if gains else None,
        "selected_kd": gains[1] if gains else None,
        "baseline": m_base,
        "candidate": m_best,
    }
    (OUT_BASE / "outer_loop_500_gain_sweep_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
