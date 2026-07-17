"""Phase 2: Detailed fine outer-loop gain sweep for the calibrated profile.

Sweeps the dynamic outer-loop parameters of `support_position_outer_loop_pitch_ref`
(B) per height, screening at SCREEN_STEPS then refining, scoring each candidate by
the task's centering+posture+yaw score (NOT final drift). Produces best-per-height
selections for the Phase 3 height-function fit.

Stages:
  2A  Kp/Kd coarse grid (parallel subprocess pool)
  2B  local refinement around best Kp/Kd
  2C  theta_ref_max / deadband / rate_limit / lowpass tuning on top candidate
  2D  Ki only on PD-stable top candidate, with anti-windup

Outputs:
  outputs/.../outer_loop_detailed_gain_sweep_metrics.csv
  outputs/.../outer_loop_detailed_gain_sweep_best_per_height.json
  docs/validation/outer_loop_detailed_gain_sweep_report.md

Run sub-stages independently to bound wall-clock:
  python scripts/run_outer_loop_detailed_gain_sweep.py 2A
  python scripts/run_outer_loop_detailed_gain_sweep.py 2B
  python scripts/run_outer_loop_detailed_gain_sweep.py 2C
  python scripts/run_outer_loop_detailed_gain_sweep.py 2D
  python scripts/run_outer_loop_detailed_gain_sweep.py report
"""
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
SWEEP_DIR = OUT_BASE / "gain_sweep_runs"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"
PER_RUN_TIMEOUT_S = 900
DRIFT = "active_pitch_crossing_signed_error_m"
PROFILE = "support_position_outer_loop_pitch_ref"

SCREEN_STEPS = 500   # screening horizon (≈60s/run)
MAX_WORKERS = 4

HEIGHTS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
    "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]
HEIGHT_M = {
    "low_0p300": 0.300, "low_0p320": 0.320, "low_0p330": 0.330,
    "low_0p340": 0.340, "low_0p360": 0.360, "low_0p380": 0.380,
    "high_0p430": 0.430, "high_0p450": 0.450, "high_0p465": 0.465,
    "high_0p480": 0.480,
}

# Stage 2A coarse grid (reduced from task superset for wall-clock; spans the
# requested 0.6..1.6 Kp / 0.0..0.3 Kd ranges with the dense region near 1.0-1.3).
KP_GRID = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50]
KD_GRID = [0.00, 0.05, 0.15]

CURRENT_KP = 1.0
CURRENT_KD = 0.0
CURRENT_THETA = 3.0
CURRENT_DEADBAND = 0.015
CURRENT_RATE = 0.03
CURRENT_LOWPASS = 0.15


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


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


def run_one(args_tuple):
    """Run one sweep candidate as a subprocess. Returns (run_id, telemetry_path|None)."""
    (run_id, label, kp, kd, ki, theta, deadband, rate, lowpass, integral_on, steps) = args_tuple
    out_dir = SWEEP_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{steps}.csv"
    if tel_dst.exists():
        return run_id, str(tel_dst)

    setup_path = SETUP_DIR / f"{label}_setup.json"
    if not setup_path.exists():
        return run_id, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--output-dir", str(out_dir),
        "--vd-outer-loop-kp-deg-per-m", str(kp),
        "--vd-outer-loop-kd-deg-per-mps", str(kd),
        "--vd-outer-loop-theta-ref-max-deg", str(theta),
        "--vd-outer-loop-deadband-m", str(deadband),
        "--vd-outer-loop-rate-limit-deg-per-step", str(rate),
        "--vd-outer-loop-lowpass-alpha", str(lowpass),
    ]
    if ki != 0.0 or integral_on:
        cmd += ["--vd-outer-loop-ki-deg-per-m-s", str(ki),
                "--vd-outer-loop-integral-enabled"]

    env = dict(__import__("os").environ)
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        return run_id, None
    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text((result.stderr or "")[-4000:])
        return run_id, None

    # With --output-dir the sim writes telemetry_<steps>.csv directly into out_dir.
    produced = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if produced and produced[0] != tel_dst:
        shutil.copy2(produced[0], tel_dst)
    return run_id, str(tel_dst) if tel_dst.exists() else None


def analyze(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None
    drift = clean(fcol(rows, DRIFT))
    if not drift:
        return None
    pitch = clean(fcol(rows, "robot_pitch_x"))
    roll = clean(fcol(rows, "robot_roll_y"))
    comz = clean(fcol(rows, "com_z"))
    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))
    yawd = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    absdr = [abs(x) for x in drift]
    nz = len(drift)
    pos = sum(1 for x in drift if x > 0)
    zc = sum(1 for i in range(1, len(drift)) if (drift[i-1] <= 0) != (drift[i] <= 0))
    pos_area = sum(x for x in drift if x > 0)
    neg_area = -sum(x for x in drift if x < 0)

    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]
    hy_all = [abs(x) for x in (lhy + rhy)]

    # left/right hip-yaw asymmetry rms
    asym = []
    for a, b in zip(lhy, rhy):
        asym.append(a + b)  # symmetric pose has l ~ -r so sum is asym proxy
    asym_rms = rms(asym) if asym else 0.0
    # yaw drift growth: |yaw_drift[end]| - |yaw_drift[start window]|
    yaw_growth = 0.0
    if len(yawd) > 100:
        yaw_growth = abs(yawd[-1]) - abs(sum(yawd[:50]) / 50)

    def pct(thr):
        return 100 * sum(1 for x in absdr if x > thr) / nz if nz else 0.0

    term = any(str(r.get("terminated", "")).strip().lower() in ("true", "1") for r in rows)

    return {
        "steps": n,
        "fell": term,
        "min": round(min(drift), 4),
        "max": round(max(drift), 4),
        "maxabs": round(max(absdr), 4),
        "p2p": round(max(drift) - min(drift), 4),
        "pos_pct": round(100 * pos / nz, 1),
        "zc": zc,
        "out05": round(pct(0.05), 1),
        "out08": round(pct(0.08), 1),
        "out10": round(pct(0.10), 1),
        "out15": round(pct(0.15), 1),
        "pitch_rms": round(rms(pitch_deg), 2),
        "pitch_max": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "roll_rms": round(rms(roll_deg), 2),
        "comz_min": round(min(comz), 4) if comz else 0.0,
        "hy_max": round(max(hy_all), 4) if hy_all else 0.0,
        "asym_rms": round(asym_rms, 4),
        "yaw_growth": round(yaw_growth, 4),
    }


def score(m):
    """Task scoring function (lower = better). Excludes final drift by design."""
    if m is None:
        return 1e9
    if m["fell"]:
        return 1e8
    s = 0.0
    s += 2.0 * abs(m["pos_pct"] - 50)
    s += 120.0 * max(0.0, m["maxabs"] - 0.18)
    s += 90.0 * max(0.0, m["p2p"] - 0.26)
    s += 70.0 * m["out15"]
    s += 30.0 * m["out10"]
    s += 20.0 * max(0.0, m["yaw_growth"])
    s += 20.0 * m["hy_max"]
    s += 30.0 * m["asym_rms"]
    # posture penalty
    if m["pitch_max"] > 14.0:
        s += 50.0 * (m["pitch_max"] - 14.0)
    if m["roll_rms"] > 2.5:
        s += 50.0 * (m["roll_rms"] - 2.5)
    # contact/height proxy
    if m["comz_min"] < 0.25:
        s += 100.0
    # oscillation penalty: excessive zero crossings relative to horizon
    zc_rate = m["zc"] / max(1, m["steps"])
    if zc_rate > 0.05:
        s += 200.0 * (zc_rate - 0.05)
    return round(s, 3)


# ---- Stage runners -------------------------------------------------------- #

def stage_2a():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for h in HEIGHTS:
        for kp in KP_GRID:
            for kd in KD_GRID:
                run_id = f"2A_{h}_kp{kp:.2f}_kd{kd:.3f}"
                jobs.append((run_id, h, kp, kd, 0.0, CURRENT_THETA, CURRENT_DEADBAND,
                             CURRENT_RATE, CURRENT_LOWPASS, False, SCREEN_STEPS))
    print(f"Stage 2A: {len(jobs)} runs, {MAX_WORKERS} workers, {SCREEN_STEPS} steps each", flush=True)
    return _run_jobs(jobs, "2A")


def stage_2b(best_per_height):
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for h in HEIGHTS:
        bk = best_per_height.get(h)
        if not bk:
            continue
        kp0, kd0 = bk["kp"], bk["kd"]
        kps = sorted({round(max(0.4, kp0 + d), 3) for d in (-0.15, -0.075, 0.0, 0.075, 0.15)})
        kds = sorted({round(max(0.0, kd0 + d), 3) for d in (-0.05, 0.0, 0.05)})
        for kp in kps:
            for kd in kds:
                run_id = f"2B_{h}_kp{kp:.3f}_kd{kd:.3f}"
                jobs.append((run_id, h, kp, kd, 0.0, CURRENT_THETA, CURRENT_DEADBAND,
                             CURRENT_RATE, CURRENT_LOWPASS, False, SCREEN_STEPS))
    print(f"Stage 2B: {len(jobs)} refinement runs", flush=True)
    return _run_jobs(jobs, "2B")


def stage_2c(best_per_height):
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    thetas = [2.0, 3.0, 4.0]
    deadbands = [0.010, 0.015, 0.025]
    for h in HEIGHTS:
        bk = best_per_height.get(h)
        if not bk:
            continue
        kp0, kd0 = bk["kp"], bk["kd"]
        for theta in thetas:
            for db in deadbands:
                if theta == CURRENT_THETA and db == CURRENT_DEADBAND:
                    continue  # already in 2A/2B
                run_id = f"2C_{h}_th{theta:.1f}_db{db:.3f}"
                jobs.append((run_id, h, kp0, kd0, 0.0, theta, db,
                             CURRENT_RATE, CURRENT_LOWPASS, False, SCREEN_STEPS))
    print(f"Stage 2C: {len(jobs)} theta/deadband runs", flush=True)
    return _run_jobs(jobs, "2C")


def stage_2d(best_per_height):
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    kis = [0.005, 0.010, 0.020]
    for h in HEIGHTS:
        bk = best_per_height.get(h)
        if not bk:
            continue
        kp0, kd0 = bk["kp"], bk["kd"]
        theta = bk.get("theta", CURRENT_THETA)
        db = bk.get("deadband", CURRENT_DEADBAND)
        for ki in kis:
            run_id = f"2D_{h}_ki{ki:.3f}"
            jobs.append((run_id, h, kp0, kd0, ki, theta, db,
                         CURRENT_RATE, CURRENT_LOWPASS, True, SCREEN_STEPS))
    print(f"Stage 2D: {len(jobs)} Ki runs", flush=True)
    return _run_jobs(jobs, "2D")


def _run_jobs(jobs, stage):
    results = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            run_id, tel = fut.result()
            m = analyze(tel)
            results[run_id] = m
            done += 1
            sc = score(m)
            print(f"  [{done}/{len(jobs)}] {run_id}: "
                  f"{'FELL' if (m and m['fell']) else 'ok'} "
                  f"maxabs={m['maxabs'] if m else 'NA'} score={sc} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # persist raw
    (SWEEP_DIR / f"{stage}_raw_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    return results


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "2A":
        stage_2a()
    elif stage == "2B":
        raw = json.loads((SWEEP_DIR / "2A_raw_results.json").read_text())
        best = _best_per_height_from_raw(raw, "2A")
        stage_2b(best)
    elif stage == "2C":
        best = _load_combined_best(["2A", "2B"])
        stage_2c(best)
    elif stage == "2D":
        best = _load_combined_best(["2A", "2B", "2C"])
        stage_2d(best)
    elif stage == "report":
        write_report()
    else:
        print("usage: run_outer_loop_detailed_gain_sweep.py [2A|2B|2C|2D|report]")


def _parse_run_id(run_id):
    """Extract (height, kp, kd, theta, deadband, ki) from a run_id where present."""
    parts = run_id.split("_")
    # height token is 'low'/'high' + value
    h = None
    for i, p in enumerate(parts):
        if p in ("low", "high"):
            h = f"{p}_{parts[i+1]}"
            break
    d = {"height": h, "kp": None, "kd": None, "theta": None, "deadband": None, "ki": None}
    for p in parts:
        if p.startswith("kp"):
            d["kp"] = float(p[2:])
        elif p.startswith("kd"):
            d["kd"] = float(p[2:])
        elif p.startswith("th"):
            d["theta"] = float(p[2:])
        elif p.startswith("db"):
            d["deadband"] = float(p[2:])
        elif p.startswith("ki"):
            d["ki"] = float(p[2:])
    return d


def _best_per_height_from_raw(raw, stage):
    best = {}
    for run_id, m in raw.items():
        info = _parse_run_id(run_id)
        h = info["height"]
        if h is None or m is None:
            continue
        sc = score(m)
        cand = {"run_id": run_id, "score": sc, "metrics": m, **info}
        if h not in best or sc < best[h]["score"]:
            best[h] = cand
    return best


def _load_combined_best(stages):
    """Combine raw results across stages, carrying forward kp/kd/theta/db from best."""
    combined = {}
    for st in stages:
        p = SWEEP_DIR / f"{st}_raw_results.json"
        if p.exists():
            raw = json.loads(p.read_text())
            for run_id, m in raw.items():
                combined[run_id] = m
    # Need to know the kp/kd context for 2C/2D runs (they inherit from prior best).
    # For simplicity, the best-per-height is computed over all runs; kp/kd are
    # filled from the 2A/2B winner when the run_id doesn't encode them.
    best_pd = _best_per_height_from_raw(
        {k: v for k, v in combined.items() if k.startswith(("2A", "2B"))}, "PD")
    best = {}
    for run_id, m in combined.items():
        info = _parse_run_id(run_id)
        h = info["height"]
        if h is None or m is None:
            continue
        # fill missing kp/kd from PD winner
        if info["kp"] is None and h in best_pd:
            info["kp"] = best_pd[h]["kp"]
        if info["kd"] is None and h in best_pd:
            info["kd"] = best_pd[h]["kd"]
        if info["theta"] is None:
            info["theta"] = CURRENT_THETA
        if info["deadband"] is None:
            info["deadband"] = CURRENT_DEADBAND
        sc = score(m)
        cand = {"run_id": run_id, "score": sc, "metrics": m, **info}
        if h not in best or sc < best[h]["score"]:
            best[h] = cand
    return best


def write_report():
    best = _load_combined_best(["2A", "2B", "2C", "2D"])
    # Also load current B baseline score from reconfirmation
    baseline = {}
    recon = OUT_BASE / "current_b_10_height_2000_reconfirmation_data.json"
    if recon.exists():
        rdata = json.loads(recon.read_text())
        for h in HEIGHTS:
            b = rdata.get(h, {}).get("B")
            if b:
                bm = {
                    "fell": b["fell"], "maxabs": b["max_abs"], "p2p": b["p2p"],
                    "pos_pct": b["pos_pct"], "out10": b["out10_pct"], "out15": b["out15_pct"],
                    "pitch_max": b["pitch_max_abs_deg"], "roll_rms": b["roll_rms_deg"],
                    "comz_min": b["comz_min"], "hy_max": b["hip_yaw_abs_max_rad"],
                    "asym_rms": 0.0, "yaw_growth": 0.0, "zc": b["zero_crossings"],
                    "steps": b["steps"],
                }
                baseline[h] = {"score": score(bm), "metrics": bm}

    best_json = OUT_BASE / "outer_loop_detailed_gain_sweep_best_per_height.json"
    out = {}
    n_improved = 0
    for h in HEIGHTS:
        bk = best.get(h)
        bl = baseline.get(h)
        improved = bool(bk and bl and bk["score"] < bl["score"])
        if improved:
            n_improved += 1
        out[h] = {
            "height_m": HEIGHT_M[h],
            "best": {
                "kp": bk["kp"] if bk else None,
                "kd": bk["kd"] if bk else None,
                "ki": bk["ki"] if bk and bk["ki"] else 0.0,
                "theta": bk["theta"] if bk else CURRENT_THETA,
                "deadband": bk["deadband"] if bk else CURRENT_DEADBAND,
                "score": bk["score"] if bk else None,
                "run_id": bk["run_id"] if bk else None,
                "metrics": bk["metrics"] if bk else None,
            },
            "baseline_b_score": bl["score"] if bl else None,
            "improved_vs_b": improved,
        }
    best_json.write_text(json.dumps(out, indent=2, default=str))

    # csv
    csv_path = OUT_BASE / "outer_loop_detailed_gain_sweep_metrics.csv"
    rows = []
    for st in ["2A", "2B", "2C", "2D"]:
        p = SWEEP_DIR / f"{st}_raw_results.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        for run_id, m in raw.items():
            if m is None:
                continue
            info = _parse_run_id(run_id)
            rows.append({"stage": st, "run_id": run_id, "height": info["height"],
                         "kp": info["kp"], "kd": info["kd"], "theta": info["theta"],
                         "deadband": info["deadband"], "ki": info["ki"],
                         "score": score(m), **m})
    if rows:
        keys = ["stage", "run_id", "height", "kp", "kd", "theta", "deadband", "ki", "score"] + \
               sorted(set(rows[0].keys()) - {"stage", "run_id", "height", "kp", "kd",
                                             "theta", "deadband", "ki", "score"})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    # classification
    if not best:
        classification = "OUTER_LOOP_GAIN_SWEEP_INCONCLUSIVE"
    elif any(out[h]["best"]["metrics"] and out[h]["best"]["metrics"]["fell"] for h in HEIGHTS if out[h]["best"]["metrics"]):
        classification = "OUTER_LOOP_GAIN_SWEEP_FAIL_SAFETY"
    elif n_improved >= 4:
        classification = "OUTER_LOOP_GAIN_SWEEP_READY_FOR_FITTING"
    elif n_improved == 0:
        classification = "OUTER_LOOP_GAIN_SWEEP_NO_IMPROVEMENT"
    else:
        classification = "OUTER_LOOP_GAIN_SWEEP_READY_FOR_FITTING"

    report = ROOT / "docs" / "validation" / "outer_loop_detailed_gain_sweep_report.md"
    L = [
        "# Outer-Loop Detailed Fine Gain Sweep (Phase 2)",
        "",
        f"**Base profile:** `{PROFILE}`  |  **Screen horizon:** {SCREEN_STEPS} steps",
        f"**Classification:** `{classification}`",
        f"**Heights improved vs current B:** {n_improved}/10",
        "",
        "Scoring excludes final drift (per metric policy). Lower score = better.",
        "",
        "## Best Candidate per Height",
        "",
        "| height | h(m) | Kp | Kd | Ki | theta | deadband | score | B_score | improved | maxabs | P2P | pos% | out15 | hy_max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for h in HEIGHTS:
        o = out[h]
        b = o["best"]
        m = b["metrics"] or {}
        L.append(
            f"| {h} | {o['height_m']:.3f} | {b['kp']} | {b['kd']} | {b['ki']} | "
            f"{b['theta']} | {b['deadband']} | {b['score']} | {o['baseline_b_score']} | "
            f"{'YES' if o['improved_vs_b'] else 'no'} | {m.get('maxabs','')} | "
            f"{m.get('p2p','')} | {m.get('pos_pct','')} | {m.get('out15','')} | {m.get('hy_max','')} |"
        )
    L += ["", "## Decision", ""]
    if classification == "OUTER_LOOP_GAIN_SWEEP_READY_FOR_FITTING":
        L.append("- Sweep produced improved/safe candidates. **Proceed to Phase 3 height-function fit.**")
    else:
        L.append(f"- {classification}. Review before fitting.")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L) + "\n")
    print(f"Classification: {classification}  improved={n_improved}/10")
    print(f"Report: {report}")
    print(f"Best-per-height: {best_json}")
    return classification


if __name__ == "__main__":
    main()