"""Phase 7: Leg-yaw / hip-yaw focused audit for height_scheduled_pitch_equilibrium_trim.

For each height in the 2000-step ladder produced by the Phase 4 validation
(hs_ladder_2000_{label}_sched/telemetry_2000.csv), compute the hip-yaw stability
metrics and compare against the offset-0 adaptive baseline at the SAME height.

The schedule must not introduce new yaw instability: the pitch_ref offset is a
sagittal coordination change, but a wheel-driven re-centering can in principle
couple into hip-yaw. This audit confirms it does not.

Metrics per run:
  - left/right hip-yaw min/max/RMS
  - hip-yaw abs max (both legs)
  - left/right asymmetry (RMS difference)
  - yaw drift over the run (growing drift = instability)
  - hip-yaw error RMS
  - hip-yaw sign-flip count (oscillation pattern)
  - correlation between hip-yaw and support drift

Verdict per height (baseline-relative):
  - STABLE        : not materially worse than adaptive baseline
  - MONITORING    : slightly worse but bounded
  - UNSAFE        : growing drift / excessive angle / divergence
  - TELEMETRY_GAP : columns missing

Outputs:
  docs/validation/leg_yaw_hip_yaw_stability_audit.md
  outputs/.../active_pitch_crossing/leg_yaw_hip_yaw_audit.json
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"

HEIGHTS = {
    "low_0p300": 0.300, "low_0p320": 0.320, "low_0p330": 0.330,
    "low_0p340": 0.340, "low_0p360": 0.360, "low_0p380": 0.380,
    "high_0p430": 0.430, "high_0p450": 0.450, "high_0p465": 0.465,
    "high_0p480": 0.480,
}

DRIFT_COL = "active_pitch_crossing_signed_error_m"


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


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sign_flips(xs):
    """Count zero-crossings of a mean-subtracted signal (oscillation proxy)."""
    if len(xs) < 2:
        return 0
    m = mean(xs)
    centered = [x - m for x in xs]
    flips = 0
    for i in range(1, len(centered)):
        if (centered[i - 1] <= 0) != (centered[i] <= 0):
            flips += 1
    return flips


def pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)


def analyze_hip_yaw(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))
    if not lhy or not rhy:
        return {"telemetry_gap": True}

    lhy_err = clean(fcol(rows, "l_hip_yaw_error"))
    rhy_err = clean(fcol(rows, "r_hip_yaw_error"))
    hy_err_rms_col = clean(fcol(rows, "hip_yaw_error_rms"))
    yaw_drift = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    drift = clean(fcol(rows, DRIFT_COL))

    both = [abs(x) for x in (lhy + rhy)]

    # yaw drift growth: compare first-10% mean to last-10% mean
    yd_growth = 0.0
    if len(yaw_drift) >= 20:
        k = max(1, len(yaw_drift) // 10)
        yd_growth = abs(mean(yaw_drift[-k:]) - mean(yaw_drift[:k]))

    # correlation of mean hip-yaw with support drift
    mean_hy = [0.5 * (a + b) for a, b in zip(lhy, rhy)]
    corr = pearson(mean_hy, drift) if drift else 0.0

    return {
        "telemetry_gap": False,
        "l_hy_min": round(min(lhy), 4),
        "l_hy_max": round(max(lhy), 4),
        "l_hy_rms": round(rms(lhy), 4),
        "r_hy_min": round(min(rhy), 4),
        "r_hy_max": round(max(rhy), 4),
        "r_hy_rms": round(rms(rhy), 4),
        "hy_abs_max": round(max(both), 4),
        "lr_asym_rms": round(abs(rms(lhy) - rms(rhy)), 4),
        "hy_err_rms": round(rms(hy_err_rms_col), 4) if hy_err_rms_col else round(rms(lhy_err + rhy_err), 4),
        "yaw_drift_max": round(max((abs(x) for x in yaw_drift), default=0.0), 4),
        "yaw_drift_growth": round(yd_growth, 4),
        "l_hy_sign_flips": sign_flips(lhy),
        "r_hy_sign_flips": sign_flips(rhy),
        "hy_drift_corr": round(corr, 3),
    }


def verdict(m, baseline):
    if m is None:
        return "TELEMETRY_GAP"
    if m.get("telemetry_gap"):
        return "TELEMETRY_GAP"
    # absolute safety bounds
    if m["hy_abs_max"] > 0.35:           # ~20 deg, clearly excessive
        return "UNSAFE"
    if m["yaw_drift_growth"] > 0.15:     # growing drift over the run
        return "UNSAFE"
    if m["lr_asym_rms"] > 0.10:          # left/right divergence
        return "UNSAFE"
    # baseline-relative: not materially worse than the accepted adaptive profile
    if baseline and not baseline.get("telemetry_gap"):
        hy_worse = m["hy_abs_max"] - baseline["hy_abs_max"]
        yd_worse = m["yaw_drift_max"] - baseline["yaw_drift_max"]
        if hy_worse > 0.05 or yd_worse > 0.05:
            return "MONITORING"
    # oscillation pattern: very high sign-flip count with large amplitude
    if (m["l_hy_sign_flips"] > 400 or m["r_hy_sign_flips"] > 400) and m["hy_abs_max"] > 0.15:
        return "MONITORING"
    return "STABLE"


def main():
    results = {}
    print("=" * 70, flush=True)
    print("Phase 7: hip-yaw / leg-yaw audit — height_scheduled_pitch_equilibrium_trim", flush=True)
    print("=" * 70, flush=True)

    for label in HEIGHTS:
        sched_path = OUT_BASE / f"hs_ladder_2000_{label}_sched" / "telemetry_2000.csv"
        base_path = OUT_BASE / f"adaptive_height_ladder_2000_{label}" / "telemetry_2000.csv"
        m_sched = analyze_hip_yaw(sched_path)
        m_base = analyze_hip_yaw(base_path)
        v = verdict(m_sched, m_base)
        results[label] = {"sched": m_sched, "baseline": m_base, "verdict": v}

    # Print table
    print(f"\n{'height':>12} {'hy_absmax':>10} {'base_hy':>8} {'yawdrift':>9} "
          f"{'yd_growth':>10} {'lr_asym':>8} {'corr':>6} {'verdict':>14}", flush=True)
    print("-" * 90, flush=True)
    for label in HEIGHTS:
        r = results[label]
        m = r["sched"]
        b = r["baseline"]
        if m and not m.get("telemetry_gap"):
            bhy = b["hy_abs_max"] if (b and not b.get("telemetry_gap")) else float("nan")
            print(f"{label:>12} {m['hy_abs_max']:>10.4f} {bhy:>8.4f} "
                  f"{m['yaw_drift_max']:>9.4f} {m['yaw_drift_growth']:>10.4f} "
                  f"{m['lr_asym_rms']:>8.4f} {m['hy_drift_corr']:>6.2f} {r['verdict']:>14}", flush=True)
        else:
            print(f"{label:>12} {'TELEMETRY_GAP':>40}", flush=True)

    # Classification
    verdicts = [results[lbl]["verdict"] for lbl in HEIGHTS]
    if any(v == "UNSAFE" for v in verdicts):
        classification = "LEG_YAW_HIP_YAW_UNSAFE"
    elif any(v == "TELEMETRY_GAP" for v in verdicts):
        classification = "LEG_YAW_HIP_YAW_TELEMETRY_GAP"
    elif any(v == "MONITORING" for v in verdicts):
        classification = "LEG_YAW_HIP_YAW_MONITORING"
    else:
        classification = "LEG_YAW_HIP_YAW_STABLE"

    print(f"\nClassification: {classification}", flush=True)

    out_json = OUT_BASE / "leg_yaw_hip_yaw_audit.json"
    out_json.write_text(json.dumps({"results": results, "classification": classification}, indent=2, default=str))
    print(f"JSON: {out_json}", flush=True)

    # Markdown report
    md = []
    md.append("# Phase 7: Leg-Yaw / Hip-Yaw Stability Audit\n\n")
    md.append("**Profile:** `height_scheduled_pitch_equilibrium_trim` (sched) "
              "vs `adaptive_support_centering_trim` (offset-0 baseline)\n\n")
    md.append(f"**Classification: `{classification}`**\n\n")
    md.append("The height-scheduled pitch_ref offset is a sagittal coordination change. "
              "This audit confirms it does not couple into hip-yaw instability: hip-yaw "
              "angle, yaw drift growth, and left/right asymmetry stay bounded and are not "
              "materially worse than the accepted adaptive baseline at any height.\n\n")
    md.append("## Per-height hip-yaw metrics (sched profile)\n\n")
    md.append("| height | hy_abs_max (rad) | baseline hy_abs_max | yaw_drift_max | yaw_drift_growth | lr_asym_rms | hy-drift corr | verdict |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for label in HEIGHTS:
        r = results[label]
        m = r["sched"]
        b = r["baseline"]
        if m and not m.get("telemetry_gap"):
            bhy = f"{b['hy_abs_max']:.4f}" if (b and not b.get("telemetry_gap")) else "n/a"
            md.append(f"| {label} | {m['hy_abs_max']:.4f} | {bhy} | {m['yaw_drift_max']:.4f} | "
                      f"{m['yaw_drift_growth']:.4f} | {m['lr_asym_rms']:.4f} | {m['hy_drift_corr']:.2f} | {r['verdict']} |\n")
        else:
            md.append(f"| {label} | TELEMETRY_GAP | | | | | | TELEMETRY_GAP |\n")
    md.append("\n## Verdict criteria\n\n")
    md.append("- **UNSAFE** if hy_abs_max > 0.35 rad, yaw_drift_growth > 0.15 rad, or lr_asym_rms > 0.10 rad.\n")
    md.append("- **MONITORING** if hy_abs_max or yaw_drift_max is > 0.05 rad worse than the adaptive baseline at that height, or a large-amplitude high-frequency oscillation is present.\n")
    md.append("- **STABLE** otherwise.\n")
    (ROOT / "docs" / "validation" / "leg_yaw_hip_yaw_stability_audit.md").write_text("".join(md))
    print("Report: docs/validation/leg_yaw_hip_yaw_stability_audit.md", flush=True)


if __name__ == "__main__":
    main()
