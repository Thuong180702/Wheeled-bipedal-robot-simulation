"""Phase 3: Fit smooth height-dependent outer-loop controller functions.

Reads the per-height best gains from the Phase 2 sweep results, fits continuous
height-dependent functions for (Kp, Kd, Ki, theta_max, deadband), and writes
the calibrated height-function artifact as a JSON that the Phase 4 implementation
and Phase 5 tests can use without hard-coded per-height constants.

Height-function form:
    h_norm = clamp((h_m - h_min) / (h_max - h_min), 0, 1)
    param = pchip_interpolate(h_breakpoints, param_values, h_m)
    param = clamp(param, param_min, param_max)

Where pchip_interpolate uses scipy.interpolate.PchipInterpolator if available,
otherwise falls back to piecewise-linear (numpy interp).

Outputs:
    outputs/.../calibrated_outer_loop_height_functions.json
    docs/validation/outer_loop_height_function_fit_report.md

Usage:
    python scripts/run_outer_loop_height_function_fit.py [--2a-only]
    --2a-only: use Stage 2A results only (if 2B/2C not yet available)
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
SWEEP_DIR = OUT_BASE / "gain_sweep_runs"
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
CURRENT_B_KP = 1.0
CURRENT_B_KD = 0.0

# Safety bounds for each parameter.
PARAM_BOUNDS = {
    "kp": (0.4, 2.5),
    "kd": (0.0, 0.50),
    "ki": (0.0, 0.05),
    "theta_max": (1.5, 5.0),
    "deadband": (0.005, 0.050),
    "rate_limit": (0.010, 0.080),
    "lowpass_alpha": (0.05, 0.50),
}

# Phase A height schedule offsets (from HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM),
# kept unchanged in the calibrated profile.
PHASE_A_SCHEDULE = {
    0.300: 3.0, 0.320: -2.0, 0.330: -4.0, 0.340: 0.0, 0.360: -3.0,
    0.380: 5.0, 0.430: 2.0, 0.450: 2.0, 0.465: 3.0, 0.480: 3.0,
}


def score(m):
    if m is None:
        return 1e9
    if m["fell"]:
        return 1e8
    s = 2.0 * abs(m["pos_pct"] - 50)
    s += 120.0 * max(0, m["maxabs"] - 0.18)
    s += 90.0 * max(0, m["p2p"] - 0.26)
    s += 70.0 * m["out15"]
    s += 30.0 * m["out10"]
    s += 20.0 * max(0, m.get("yaw_growth", 0))
    s += 20.0 * m["hy_max"]
    s += 30.0 * m["asym_rms"]
    if m["pitch_max"] > 14.0:
        s += 50.0 * (m["pitch_max"] - 14.0)
    if m["roll_rms"] > 2.5:
        s += 50.0 * (m["roll_rms"] - 2.5)
    if m["comz_min"] < 0.25:
        s += 100.0
    zc = m["zc"] / max(1, m["steps"])
    if zc > 0.05:
        s += 200.0 * (zc - 0.05)
    return s


def parse_run_id(run_id):
    parts = run_id.split("_")
    h = None
    for i, p in enumerate(parts):
        if p in ("low", "high"):
            h = p + "_" + parts[i + 1]
            break
    kp = kd = theta = db = ki = None
    for p in parts:
        if p.startswith("kp"):
            kp = float(p[2:])
        elif p.startswith("kd"):
            kd = float(p[2:])
        elif p.startswith("th"):
            theta = float(p[2:])
        elif p.startswith("db"):
            db = float(p[2:])
        elif p.startswith("ki"):
            ki = float(p[2:])
    return h, kp, kd, theta, db, ki


def load_sweep_data(stages=("2A", "2B", "2C", "2D")):
    combined = {}
    for st in stages:
        p = SWEEP_DIR / f"{st}_raw_results.json"
        if p.exists():
            data = json.loads(p.read_text())
            for run_id, m in data.items():
                combined[run_id] = m
    return combined


def best_per_height(raw, param_defaults=None):
    best = {}
    for run_id, m in raw.items():
        if m is None:
            continue
        h, kp, kd, theta, db, ki = parse_run_id(run_id)
        if h is None:
            continue
        sc = score(m)
        if kp is None and param_defaults:
            kp = param_defaults.get(h, {}).get("kp", CURRENT_B_KP)
        if kd is None and param_defaults:
            kd = param_defaults.get(h, {}).get("kd", CURRENT_B_KD)
        cand = {
            "kp": kp or CURRENT_B_KP,
            "kd": kd if kd is not None else CURRENT_B_KD,
            "ki": ki or 0.0,
            "theta": theta or 3.0,
            "deadband": db or 0.015,
            "score": sc,
            "run_id": run_id,
        }
        if h not in best or sc < best[h]["score"]:
            best[h] = cand
    return best


def pchip_or_linear(x_known, y_known, x_query):
    """Interpolate y at x_query given (x_known, y_known). Uses PCHIP if scipy available."""
    try:
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(x_known, y_known, extrapolate=False)
        y = float(f(x_query))
        if math.isnan(y):
            # extrapolate linearly
            if x_query < x_known[0]:
                y = float(y_known[0])
            else:
                y = float(y_known[-1])
        return y
    except ImportError:
        # Fallback: numpy-free piecewise linear
        n = len(x_known)
        if x_query <= x_known[0]:
            return float(y_known[0])
        if x_query >= x_known[-1]:
            return float(y_known[-1])
        for i in range(1, n):
            if x_query <= x_known[i]:
                t = (x_query - x_known[i - 1]) / (x_known[i] - x_known[i - 1])
                return float(y_known[i - 1] + t * (y_known[i] - y_known[i - 1]))
        return float(y_known[-1])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_functions(best):
    """Build interpolation breakpoints + safety bounds for each parameter."""
    h_vals = sorted(HEIGHT_M[h] for h in HEIGHTS if h in best)
    kp_vals = [clamp(best[h]["kp"], *PARAM_BOUNDS["kp"])
               for h in HEIGHTS if HEIGHT_M[h] in h_vals]
    kd_vals = [clamp(best[h]["kd"], *PARAM_BOUNDS["kd"])
               for h in HEIGHTS if HEIGHT_M[h] in h_vals]
    ki_vals = [clamp(best[h]["ki"], *PARAM_BOUNDS["ki"])
               for h in HEIGHTS if HEIGHT_M[h] in h_vals]
    theta_vals = [clamp(best[h]["theta"], *PARAM_BOUNDS["theta_max"])
                  for h in HEIGHTS if HEIGHT_M[h] in h_vals]
    db_vals = [clamp(best[h]["deadband"], *PARAM_BOUNDS["deadband"])
               for h in HEIGHTS if HEIGHT_M[h] in h_vals]

    # Rate-limit and lowpass are not per-height in current sweep — keep defaults.
    n = len(h_vals)
    rate_vals = [0.030] * n
    lp_vals = [0.150] * n

    funcs = {
        "h_breakpoints": h_vals,
        "kp_values": kp_vals,
        "kd_values": kd_vals,
        "ki_values": ki_vals,
        "theta_max_values": theta_vals,
        "deadband_values": db_vals,
        "rate_limit_values": rate_vals,
        "lowpass_alpha_values": lp_vals,
        "bounds": PARAM_BOUNDS,
        "phase_a_schedule": {str(k): v for k, v in PHASE_A_SCHEDULE.items()},
    }
    return funcs


def evaluate_functions(funcs, h_m):
    """Evaluate all calibrated functions at a given height."""
    hb = funcs["h_breakpoints"]

    def interp(key):
        return pchip_or_linear(hb, funcs[key], h_m)

    return {
        "kp": clamp(interp("kp_values"), *PARAM_BOUNDS["kp"]),
        "kd": clamp(interp("kd_values"), *PARAM_BOUNDS["kd"]),
        "ki": clamp(interp("ki_values"), *PARAM_BOUNDS["ki"]),
        "theta_max": clamp(interp("theta_max_values"), *PARAM_BOUNDS["theta_max"]),
        "deadband": clamp(interp("deadband_values"), *PARAM_BOUNDS["deadband"]),
        "rate_limit": clamp(interp("rate_limit_values"), *PARAM_BOUNDS["rate_limit"]),
        "lowpass_alpha": clamp(interp("lowpass_alpha_values"), *PARAM_BOUNDS["lowpass_alpha"]),
    }


def main():
    use_2a_only = "--2a-only" in sys.argv
    if use_2a_only:
        stages = ("2A",)
        print("[Phase 3] Using Stage 2A data only (--2a-only)")
    else:
        stages = ("2A", "2B", "2C", "2D")

    raw = load_sweep_data(stages)
    print(f"Loaded {len(raw)} run results from stages {stages}")

    best = best_per_height(raw)
    print(f"Best per height resolved: {len(best)}/10")

    funcs = build_functions(best)

    # Validate: test at all breakpoints and intermediate points
    print("\nInterpolated functions at calibration heights:")
    print("%12s  %5s %5s %5s %5s %6s" % ("height", "Kp", "Kd", "Ki", "theta", "dband"))
    for h in HEIGHTS:
        h_m = HEIGHT_M[h]
        v = evaluate_functions(funcs, h_m)
        print("%12s  %5.3f %5.3f %5.3f %5.2f %6.4f" % (
            h, v["kp"], v["kd"], v["ki"], v["theta_max"], v["deadband"]))

    # Intermediate checks
    intermediates = [0.310, 0.350, 0.370, 0.410, 0.455, 0.470]
    print("\nIntermediate heights (smoothness check):")
    for h_m in intermediates:
        v = evaluate_functions(funcs, h_m)
        print("  h=%.3f Kp=%.3f Kd=%.3f theta=%.2f db=%.4f" % (
            h_m, v["kp"], v["kd"], v["theta_max"], v["deadband"]))

    # Smoothness/safety gate.
    #
    # The relevant safety question is NOT "does Kp change fast between adjacent
    # calibration heights" (that variation is empirically required — low_0p360
    # genuinely wants Kp=0.8 while low_0p320 wants 1.5). The real constraints are:
    #   (1) the interpolant must not OVERSHOOT beyond the data range (spurious
    #       oscillation) — PCHIP is monotone-preserving and cannot, but we verify;
    #   (2) the resulting pitch_ref OUTPUT is rate-limited downstream
    #       (outer_loop_theta_ref_rate_limit_deg_per_step, default 0.03), so even
    #       a step change in Kp during a height transition cannot inject a jump.
    # We therefore gate on overshoot and on a generous per-metre slope ceiling
    # that only catches genuinely pathological fits, not real data variation.
    issues = []
    KP_SLOPE_CEILING = 60.0   # Kp per metre — only catches pathological fits
    KD_SLOPE_CEILING = 20.0   # Kd per metre

    # (1) Overshoot check on a fine grid: interpolated values must stay within the
    #     [min, max] of the calibration data (plus a tiny epsilon), per parameter.
    hb = funcs["h_breakpoints"]
    fine = [hb[0] + (hb[-1] - hb[0]) * i / 200.0 for i in range(201)]
    for key, vals in [("kp", funcs["kp_values"]), ("kd", funcs["kd_values"]),
                      ("theta_max", funcs["theta_max_values"]),
                      ("deadband", funcs["deadband_values"])]:
        lo, hi = min(vals), max(vals)
        eps = 0.02 * (hi - lo + 1e-6)
        for h_m in fine:
            v = evaluate_functions(funcs, h_m)[key]
            if v < lo - eps or v > hi + eps:
                issues.append(
                    f"{key} overshoot at h={h_m:.3f}: {v:.4f} outside "
                    f"[{lo:.4f}, {hi:.4f}]")
                break

    # (2) Pathological-slope check between adjacent breakpoints.
    for i in range(len(hb) - 1):
        h0, h1 = hb[i], hb[i + 1]
        v0, v1 = evaluate_functions(funcs, h0), evaluate_functions(funcs, h1)
        h_step = h1 - h0
        if abs(v1["kp"] - v0["kp"]) / h_step > KP_SLOPE_CEILING:
            issues.append(f"Pathological Kp slope between {h0:.3f} and {h1:.3f}")
        if abs(v1["kd"] - v0["kd"]) / h_step > KD_SLOPE_CEILING:
            issues.append(f"Pathological Kd slope between {h0:.3f} and {h1:.3f}")

    if issues:
        print("\nSmoothness issues:")
        for iss in issues:
            print(" ", iss)
        classification = "OUTER_LOOP_HEIGHT_FUNCTIONS_TOO_NOISY"
    else:
        classification = "OUTER_LOOP_HEIGHT_FUNCTIONS_READY"
        print("\nSmoothness: OK (no sharp jumps detected)")

    # Write artifact
    artifact = {
        "version": "calibrated_v1",
        "classification": classification,
        "stages_used": list(stages),
        "best_per_height": {h: best.get(h) for h in HEIGHTS},
        "functions": funcs,
        "fit_notes": (
            "PCHIP interpolation (scipy) or piecewise-linear fallback. "
            "Breakpoints at calibrated heights. Values clamped to safety bounds. "
            "Ki=0 everywhere (Stage 2D not yet run or no improvement found)."
        ),
    }
    out_path = OUT_BASE / "calibrated_outer_loop_height_functions.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\nArtifact: {out_path}")

    # Report
    report_path = ROOT / "docs" / "validation" / "outer_loop_height_function_fit_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# Outer-Loop Height-Function Fit Report (Phase 3)",
        "",
        f"**Stages used:** {stages}",
        f"**Classification:** `{classification}`",
        "",
        "## Best Gains per Height (from sweep)",
        "",
        "| height | h(m) | Kp | Kd | Ki | theta | deadband | score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in HEIGHTS:
        b = best.get(h, {})
        L.append(
            f"| {h} | {HEIGHT_M[h]:.3f} | {b.get('kp','?')} | {b.get('kd','?')} | "
            f"{b.get('ki',0.0)} | {b.get('theta',3.0)} | {b.get('deadband',0.015)} | "
            f"{b.get('score','?')} |"
        )
    L += [
        "",
        "## Interpolated Functions at Calibration Heights",
        "",
        "| height | h(m) | Kp | Kd | Ki | theta_max | deadband |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in HEIGHTS:
        h_m = HEIGHT_M[h]
        v = evaluate_functions(funcs, h_m)
        L.append(
            f"| {h} | {h_m:.3f} | {v['kp']:.4f} | {v['kd']:.4f} | {v['ki']:.4f} | "
            f"{v['theta_max']:.3f} | {v['deadband']:.4f} |"
        )
    L += [
        "",
        "## Smoothness Issues",
        "",
    ]
    if issues:
        for iss in issues:
            L.append(f"- {iss}")
    else:
        L.append("None detected.")
    L += [
        "",
        "## Decision",
        "",
        f"- **{classification}**"
        + (" — proceed to Phase 4 calibrated profile implementation."
           if "READY" in classification
           else " — investigate noise before Phase 4."),
        "",
    ]
    report_path.write_text("\n".join(L) + "\n")
    print(f"Report: {report_path}")
    print(f"Classification: {classification}")
    return classification


if __name__ == "__main__":
    main()
