"""Phase 1: Height x pitch_ref_offset sweep for the structural sagittal centering fix.

Runs adaptive_support_centering_trim (all safety machinery intact) while varying
ONLY the pitch reference offset via --vd-pitch-ref-offset-deg. At offset 0 this is
identical to adaptive_support_centering_trim, so the existing offset-0 height-ladder
telemetry is reused instead of re-running.

For every (height, offset) it computes the full task metric set (NOT final drift)
and a score, then selects the best offset per height.

Design notes / correctness:
- Drift column: active_pitch_crossing_signed_error_m (signed support drift).
- Scoring deliberately EXCLUDES final drift, per task spec.
- Resumable: if a run's telemetry already exists it is reused, so a crash never
  loses completed work and the analysis can be re-run standalone.

Outputs:
- outputs/.../active_pitch_crossing/height_scheduled_pitch_offset_sweep_metrics.csv
- outputs/.../active_pitch_crossing/height_scheduled_pitch_offset_sweep_summary.json
- docs/validation/height_scheduled_pitch_offset_sweep_report.md
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
SWEEP_DIR = OUT_BASE / "pitch_offset_sweep"
ADAPTIVE_LADDER_DIR = OUT_BASE  # adaptive_height_ladder_2000_{label}/telemetry_2000.csv
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

STEPS = 2000
PROFILE = "adaptive_support_centering_trim"
PER_RUN_TIMEOUT_S = 600

HEIGHTS = {
    "low_0p300": 0.300, "low_0p320": 0.320, "low_0p330": 0.330,
    "low_0p340": 0.340, "low_0p360": 0.360, "low_0p380": 0.380,
    "high_0p430": 0.430, "high_0p450": 0.450, "high_0p465": 0.465,
    "high_0p480": 0.480,
}
LABELS = list(HEIGHTS.keys())
# Extended negative range: the offset-0 equilibrium seed shows the low band
# (0.320-0.360 m) settles at negative equilibrium pitch and needs offsets down to
# ~-2.9 deg to center drift. A grid floored at -2 deg would report "best at
# boundary" for those heights and force a re-run, so the floor is -4 deg.
OFFSETS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

DRIFT_COL = "active_pitch_crossing_signed_error_m"
PITCH_COL = "robot_pitch_x"
ROLL_COL = "robot_roll_y"
COMZ_COL = "com_z"


# --------------------------------------------------------------------------- #
# Run helpers
# --------------------------------------------------------------------------- #
def off_tag(off):
    s = f"{off:+.0f}"  # -2 -> "-2", +0 -> "+0"
    return s.replace("+", "p").replace("-", "m")


def run_dir(label, off):
    return SWEEP_DIR / f"{label}_off{off_tag(off)}"


def telemetry_path(label, off):
    return run_dir(label, off) / "telemetry_2000.csv"


def reuse_offset0(label):
    """Reuse the existing adaptive (offset-0) ladder telemetry if present."""
    src = ADAPTIVE_LADDER_DIR / f"adaptive_height_ladder_2000_{label}" / "telemetry_2000.csv"
    if not src.exists():
        return False
    dst = telemetry_path(label, 0.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def run_one(label, off):
    """Run a single (height, offset). Returns True on success/reuse."""
    dst = telemetry_path(label, off)
    if dst.exists():
        return True  # resume: already done
    if abs(off) < 1e-9 and reuse_offset0(label):
        return True

    setup_path = SETUP_DIR / f"{label}_setup.json"
    if not setup_path.exists():
        print(f"  MISSING setup {label}", flush=True)
        return False

    out_dir = run_dir(label, off)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--vd-pitch-ref-offset-deg", str(off),
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
    ]
    try:
        result = subprocess.run(
            args, cwd=str(ROOT), capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        print(f"  TIMEOUT {label} off={off:+.0f}", flush=True)
        return False

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        print(f"  FAILED {label} off={off:+.0f} rc={result.returncode}", flush=True)
        return False

    tel = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tel:
        print(f"  NO TELEMETRY {label} off={off:+.0f}", flush=True)
        return False
    shutil.copy2(tel[0], dst)
    try:
        tel[0].unlink()
    except OSError:
        pass
    return True


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def fcol(rows, key, default=float("nan")):
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


def bcol(rows, key):
    return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def analyze(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    drift = clean(fcol(rows, DRIFT_COL))
    pitch = clean(fcol(rows, PITCH_COL))
    roll = clean(fcol(rows, ROLL_COL))
    comz = clean(fcol(rows, COMZ_COL))
    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))
    hy_err_rms = clean(fcol(rows, "hip_yaw_error_rms"))
    yaw_drift = clean(fcol(rows, "yaw_drift_from_initial_rad"))

    term = bcol(rows, "terminated")
    fell = any(term)
    term_reason = ""
    if fell:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1", "1.0"):
                term_reason = r.get("termination_reason", "") or ""
                break

    nz = len(drift)
    abs_drift = [abs(x) for x in drift]
    pos = sum(1 for x in drift if x > 0)
    neg = sum(1 for x in drift if x < 0)
    pos_pct = 100 * pos / nz if nz else 0.0
    neg_pct = 100 * neg / nz if nz else 0.0

    zc = 0
    for i in range(1, len(drift)):
        if (drift[i - 1] <= 0) != (drift[i] <= 0):
            zc += 1

    pos_area = sum(x for x in drift if x > 0)
    neg_area = -sum(x for x in drift if x < 0)
    area_total = pos_area + neg_area
    # area balance score: 0 = perfectly balanced, 1 = fully one-sided
    area_balance = abs(pos_area - neg_area) / area_total if area_total > 1e-9 else 1.0

    def out_pct(thr):
        return 100 * sum(1 for x in abs_drift if x > thr) / nz if nz else 0.0

    d_min = min(drift) if drift else 0.0
    d_max = max(drift) if drift else 0.0
    d_maxabs = max(abs_drift) if abs_drift else 0.0
    p2p = d_max - d_min

    # hip yaw
    hy_all = [abs(x) for x in (lhy + rhy)]
    hy_abs_max = max(hy_all) if hy_all else 0.0
    lr_asym = (rms(lhy) - rms(rhy)) if (lhy and rhy) else 0.0
    yaw_drift_max = max((abs(x) for x in yaw_drift), default=0.0)

    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]

    out15 = out_pct(0.15)
    sym_score = abs(pos_pct - 50.0)
    posture_penalty = 0.0
    # pitch posture: flag if |pitch| ever exceeds ~12 deg sustained (danger ~0.10rad=5.7deg gate;
    # equilibrium settles 3-5deg, transients to ~8.5deg are seen — treat >14deg as unsafe posture)
    pitch_abs_max = max((abs(p) for p in pitch_deg), default=0.0)
    roll_rms = rms(roll_deg)
    comz_min = min(comz) if comz else 0.0

    if pitch_abs_max > 14.0:
        posture_penalty += 200.0
    if roll_rms > 3.0:
        posture_penalty += 100.0
    # Hip-yaw penalty is now ONLY the absolute yaw-drift growth gate. The former
    # absolute hy_abs_max > 0.20 rad gate was removed: the accepted offset-0
    # adaptive baseline itself exceeds 0.20 rad at low heights (0.30 m -> 0.203,
    # 0.38 m -> 0.271), so it flagged shipped behavior, not behavior the offset
    # introduces. Hip-yaw is judged baseline-relative in main() instead.
    yaw_drift_penalty = 100.0 if yaw_drift_max > 0.25 else 0.0
    fall_penalty = 1000.0 if fell else 0.0

    score = (
        2.0 * sym_score
        + 100.0 * max(0.0, d_maxabs - 0.20)
        + 80.0 * max(0.0, p2p - 0.30)
        + 50.0 * out15
        + posture_penalty
        + yaw_drift_penalty
        + fall_penalty
    )

    return {
        "steps": n,
        "fell": fell,
        "term_reason": term_reason,
        "min_drift": round(d_min, 4),
        "max_drift": round(d_max, 4),
        "max_abs": round(d_maxabs, 4),
        "p2p": round(p2p, 4),
        "pos_pct": round(pos_pct, 1),
        "neg_pct": round(neg_pct, 1),
        "zero_crossings": zc,
        "pos_area": round(pos_area, 3),
        "neg_area": round(neg_area, 3),
        "area_balance": round(area_balance, 3),
        "sym_score": round(sym_score, 1),
        "out05_pct": round(out_pct(0.05), 1),
        "out08_pct": round(out_pct(0.08), 1),
        "out10_pct": round(out_pct(0.10), 1),
        "out15_pct": round(out15, 1),
        "pitch_min_deg": round(min(pitch_deg), 2) if pitch_deg else None,
        "pitch_max_deg": round(max(pitch_deg), 2) if pitch_deg else None,
        "pitch_rms_deg": round(rms(pitch_deg), 2),
        "roll_rms_deg": round(roll_rms, 2),
        "comz_min": round(comz_min, 4),
        "comz_mean": round(sum(comz) / len(comz), 4) if comz else None,
        "comz_max": round(max(comz), 4) if comz else None,
        "hip_yaw_abs_max_rad": round(hy_abs_max, 4),
        "hip_yaw_lr_asym": round(lr_asym, 4),
        "hip_yaw_err_rms": round(rms(hy_err_rms), 4) if hy_err_rms else None,
        "yaw_drift_max_rad": round(yaw_drift_max, 4),
        "posture_penalty": round(posture_penalty, 1),
        "yaw_drift_penalty": round(yaw_drift_penalty, 1),
        "fall_penalty": round(fall_penalty, 1),
        "score": round(score, 2),
    }


# Baseline-relative hip-yaw margin (rad). A (height,offset) is hip-yaw-safe if it
# is not materially worse than the accepted offset-0 adaptive baseline AT THAT
# HEIGHT. The absolute 0.20 rad gate was wrong: the accepted baseline itself
# exceeds it at low heights (0.30 m -> 0.203, 0.38 m -> 0.271), so an absolute
# gate flags behavior that is already shipped, not behavior the offset introduces.
HY_REL_MARGIN_RAD = 0.03


def safe(prof_safe_metrics, baseline_hy_rad):
    """Posture/contact/hip-yaw safety verdict, hip-yaw gated relative to baseline.

    baseline_hy_rad is the offset-0 hip_yaw_abs_max at the SAME height.
    """
    m = prof_safe_metrics
    if m["fell"]:
        return False
    if m["posture_penalty"] > 0:
        return False
    hy_gate = max(0.20, baseline_hy_rad + HY_REL_MARGIN_RAD)
    if m["hip_yaw_abs_max_rad"] > hy_gate:
        return False
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    grid = [(lbl, off) for lbl in LABELS for off in OFFSETS]
    total = len(grid)

    print(f"Sweep: {len(LABELS)} heights x {len(OFFSETS)} offsets = {total} runs", flush=True)
    t0 = time.time()
    for i, (lbl, off) in enumerate(grid):
        done = telemetry_path(lbl, off).exists()
        tag = "reuse" if done else "run"
        print(f"[{i+1}/{total}] {lbl} off={off:+.0f} ({tag})", flush=True)
        if not done:
            ok = run_one(lbl, off)
            el = time.time() - t0
            print(f"    {'ok' if ok else 'FAIL'}  cumulative={el/60:.1f}min", flush=True)

    # Analyze. Two passes: (1) compute raw metrics for every (height, offset),
    # (2) derive the offset-0 hip-yaw baseline per height, then apply the
    # baseline-relative hip-yaw safety verdict.
    raw = {}  # (lbl, off) -> metrics
    for lbl in LABELS:
        for off in OFFSETS:
            p = telemetry_path(lbl, off)
            if not p.exists():
                continue
            m = analyze(p)
            if m is None:
                continue
            raw[(lbl, off)] = m

    # offset-0 hip-yaw baseline per height (the accepted adaptive profile).
    baseline_hy = {}
    for lbl in LABELS:
        m0 = raw.get((lbl, 0.0))
        baseline_hy[lbl] = m0["hip_yaw_abs_max_rad"] if m0 else 0.20

    rows_out = []
    per_height = {}
    for lbl in LABELS:
        for off in OFFSETS:
            m = raw.get((lbl, off))
            if m is None:
                continue
            rec = {"label": lbl, "height_m": HEIGHTS[lbl], "offset_deg": off,
                   "safe": safe(m, baseline_hy[lbl]),
                   "baseline_hy_rad": round(baseline_hy[lbl], 4), **m}
            rows_out.append(rec)
            per_height.setdefault(lbl, []).append(rec)

    # Select best per height: no fall -> safe -> lowest score -> lower p2p -> lower max_abs
    selection = {}
    for lbl, recs in per_height.items():
        candidates = [r for r in recs if not r["fell"]]
        safe_c = [r for r in candidates if r["safe"]]
        pool = safe_c if safe_c else candidates if candidates else recs
        pool.sort(key=lambda r: (r["score"], r["p2p"], r["max_abs"]))
        best = pool[0]
        selection[lbl] = {
            "height_m": HEIGHTS[lbl],
            "best_offset_deg": best["offset_deg"],
            "best_at_boundary": best["offset_deg"] in (OFFSETS[0], OFFSETS[-1]),
            "score": best["score"],
            "safe": best["safe"],
            "fell": best["fell"],
            "pos_pct": best["pos_pct"],
            "neg_pct": best["neg_pct"],
            "min_drift": best["min_drift"],
            "max_drift": best["max_drift"],
            "max_abs": best["max_abs"],
            "p2p": best["p2p"],
            "out15_pct": best["out15_pct"],
        }

    # Write metrics CSV
    metrics_csv = OUT_BASE / "height_scheduled_pitch_offset_sweep_metrics.csv"
    if rows_out:
        keys = list(rows_out[0].keys())
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)

    # Classification
    n_expected = total
    n_done = len(rows_out)
    any_safety_fail = False
    inconclusive = n_done < n_expected
    boundary_flags = [lbl for lbl, s in selection.items() if s["best_at_boundary"] and s["best_offset_deg"] == OFFSETS[-1]]
    for lbl, s in selection.items():
        # a height where even the best choice falls or is unsafe is a safety concern
        if s["fell"] or not s["safe"]:
            any_safety_fail = True

    if any_safety_fail:
        classification = "HEIGHT_OFFSET_SWEEP_FAIL_SAFETY"
    elif inconclusive or boundary_flags:
        classification = "HEIGHT_OFFSET_SWEEP_INCONCLUSIVE"
    else:
        classification = "HEIGHT_OFFSET_SWEEP_READY"

    summary = {
        "profile": PROFILE,
        "steps": STEPS,
        "offsets_deg": OFFSETS,
        "heights": HEIGHTS,
        "runs_expected": n_expected,
        "runs_analyzed": n_done,
        "selection": selection,
        "boundary_at_max_offset": boundary_flags,
        "classification": classification,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    summary_json = OUT_BASE / "height_scheduled_pitch_offset_sweep_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    # Markdown report
    md = []
    md.append("# Phase 1: Height x Pitch-Offset Sweep Report\n\n")
    md.append(f"**Profile:** `{PROFILE}` + `--vd-pitch-ref-offset-deg` | "
              f"**Steps:** {STEPS} | **Runs:** {n_done}/{n_expected}\n\n")
    md.append(f"**Classification: `{classification}`**\n\n")
    md.append("Scoring (final drift deliberately excluded): "
              "`2*|pos%-50| + 100*max(0,maxabs-0.20) + 80*max(0,p2p-0.30) + "
              "50*out15% + posture + hip_yaw + fall`.\n\n")

    md.append("## Selected offset per height\n\n")
    md.append("| Height (m) | Best offset | pos% | neg% | min | max | maxabs | P2P | out15% | safe | score |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for lbl in LABELS:
        if lbl not in selection:
            continue
        s = selection[lbl]
        md.append(f"| {s['height_m']:.3f} | {s['best_offset_deg']:+.0f}° | {s['pos_pct']} | "
                  f"{s['neg_pct']} | {s['min_drift']} | {s['max_drift']} | {s['max_abs']} | "
                  f"{s['p2p']} | {s['out15_pct']} | {'Y' if s['safe'] else 'N'} | {s['score']} |\n")
    md.append("\n")

    md.append("## Full grid\n\n")
    md.append("| Height | off | pos% | neg% | min | max | maxabs | P2P | ZC | out15% | "
              "pitchRMS° | rollRMS° | comZmin | HYmax | fall | score |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows_out:
        md.append(f"| {r['label']} | {r['offset_deg']:+.0f} | {r['pos_pct']} | {r['neg_pct']} | "
                  f"{r['min_drift']} | {r['max_drift']} | {r['max_abs']} | {r['p2p']} | "
                  f"{r['zero_crossings']} | {r['out15_pct']} | {r['pitch_rms_deg']} | "
                  f"{r['roll_rms_deg']} | {r['comz_min']} | {r['hip_yaw_abs_max_rad']} | "
                  f"{'Y' if r['fell'] else ''} | {r['score']} |\n")
    if boundary_flags:
        md.append(f"\n**Note:** best offset hit the +6° grid boundary for: "
                  f"{', '.join(boundary_flags)} — a +7° run may be warranted.\n")

    report_md = ROOT / "docs" / "validation" / "height_scheduled_pitch_offset_sweep_report.md"
    report_md.write_text("".join(md))

    print(f"\nClassification: {classification}", flush=True)
    print(f"Metrics CSV: {metrics_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Report MD:   {report_md}", flush=True)
    print("\nSelected schedule (height_m -> offset_deg):", flush=True)
    for lbl in LABELS:
        if lbl in selection:
            print(f"  {selection[lbl]['height_m']:.3f} -> {selection[lbl]['best_offset_deg']:+.0f}", flush=True)


if __name__ == "__main__":
    main()
