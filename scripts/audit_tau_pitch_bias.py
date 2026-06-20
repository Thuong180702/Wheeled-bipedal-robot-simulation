"""Phase 1: tau_pitch bias audit across four profiles.

Reads high_0p480 5000-step telemetry for:
- adaptive_support_centering_trim
- zero_crossing_support_recenter
- early_zero_crossing_recenter
- early_zero_crossing_recenter_v2

Computes:
- tau_pitch mean/median/min/max/RMS overall and conditional
- Correlations vs pitch, pitch_rate, support_error, drift direction
- Comparison against pitch reference, pitch error, tau_position, damping
"""
from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

ROOT = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation")
EVAL_DIR = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"

PROFILES = OrderedDict([
    ("adaptive_support_centering_trim", "adaptive_5000_high_0p480/telemetry_5000.csv"),
    ("zero_crossing_support_recenter", "zc_5000_high_0p480/telemetry_5000.csv"),
    ("early_zero_crossing_recenter", "ezc_5000_high_0p480/telemetry_5000.csv"),
    ("early_zero_crossing_recenter_v2", "ezc_v2_5000_high_0p480/telemetry_5000.csv"),
])

# Columns we need
COLS_NEEDED = [
    "tau_pitch", "tau_pitch_raw", "tau_pitch_clipped",
    "tau_position", "tau_support_velocity",
    "tau_pitch_rate",
    "pitch_x", "pitch_x_rad", "pitch_rate_x_rad_s",
    "pitch_x_ref_rad", "pitch_x_error_rad",
    "active_pitch_crossing_signed_error_m",
    "support_position_error_m",
    "sagittal_position_error_m",
    "tau_total_clipped_l_wheel", "tau_total_clipped_r_wheel",
    "final_wheel_tau_with_apc", "final_wheel_tau_without_apc",
    "tau_static_feedforward",
    "control_pitch_x", "control_pitch_rate_x",
    "tau_wheel_velocity_left", "tau_wheel_velocity_right",
    "ezc_active",
    "adaptive_bias_tau_nm",
]


def parse_float(v: str) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_telemetry(path: Path):
    """Load telemetry CSV, returning dict of column -> list[float]."""
    data: dict[str, list[float]] = {c: [] for c in COLS_NEEDED}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in COLS_NEEDED:
                data[c].append(parse_float(row.get(c, "")))
    return data


def stat_summary(vals: list[float]) -> dict:
    """Compute mean/median/min/max/RMS for a list of floats, ignoring NaN."""
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"), "rms": float("nan"),
                "std": float("nan")}
    n = len(clean)
    mean = sum(clean) / n
    sq = sum(v * v for v in clean) / n
    rms = math.sqrt(sq)
    var = sum((v - mean) ** 2 for v in clean) / n
    std = math.sqrt(var)
    sorted_vals = sorted(clean)
    median = sorted_vals[n // 2] if n % 2 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    return {"n": n, "mean": mean, "median": median, "min": min(clean),
            "max": max(clean), "rms": rms, "std": std}


def conditional(vals: list[float], cond: list[bool]) -> list[float]:
    return [v for v, c in zip(vals, cond) if c]


def pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    pairs = [(a, b) for a, b in zip(x[:n], y[:n]) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    den = math.sqrt(sxx * syy)
    return sxy / den if den > 0 else float("nan")


def percent_window(cond_flags: list[bool]) -> float:
    n = len(cond_flags)
    if n == 0:
        return 0.0
    return 100.0 * sum(cond_flags) / n


def audit_profile(name: str, csv_path: Path) -> dict:
    print(f"\n{'='*70}\n  Profile: {name}\n  File:    {csv_path.name}\n{'='*70}")
    data = load_telemetry(csv_path)

    n_steps = len(data["tau_pitch"])
    print(f"  Total steps: {n_steps}")

    # Choose drift error column (active pitch crossing signed error)
    drift = data["active_pitch_crossing_signed_error_m"]

    # Pitch column - pitch_x is in rad already? check
    pitch_rad = data["pitch_x"]  # already rad based on column name pitch_x_rad
    if all(math.isnan(v) for v in pitch_rad):
        pitch_rad = data["pitch_x_rad"]
    pitch_deg = [math.degrees(v) if not math.isnan(v) else float("nan") for v in pitch_rad]

    # Conditions
    near_zero_pitch = [(not math.isnan(p)) and abs(p) < 1.0 for p in pitch_deg]
    near_zero_drift = [(not math.isnan(d)) and abs(d) < 0.03 for d in drift]
    pos_drift = [(not math.isnan(d)) and d > 0.0 for d in drift]
    neg_drift = [(not math.isnan(d)) and d < 0.0 for d in drift]

    summary = {
        "profile": name,
        "csv": str(csv_path),
        "n_steps": n_steps,
        "near_zero_pitch_pct": percent_window(near_zero_pitch),
        "near_zero_drift_pct": percent_window(near_zero_drift),
        "pos_drift_pct": percent_window(pos_drift),
        "neg_drift_pct": percent_window(neg_drift),
    }

    # tau_pitch overall
    summary["tau_pitch_all"] = stat_summary(data["tau_pitch"])
    summary["tau_pitch_raw_all"] = stat_summary(data["tau_pitch_raw"])
    summary["tau_pitch_clipped_all"] = stat_summary(data["tau_pitch_clipped"])

    # Conditional tau_pitch
    summary["tau_pitch_near_zero_pitch"] = stat_summary(
        conditional(data["tau_pitch"], near_zero_pitch))
    summary["tau_pitch_near_zero_drift"] = stat_summary(
        conditional(data["tau_pitch"], near_zero_drift))
    summary["tau_pitch_pos_drift"] = stat_summary(
        conditional(data["tau_pitch"], pos_drift))
    summary["tau_pitch_neg_drift"] = stat_summary(
        conditional(data["tau_pitch"], neg_drift))

    # Conjunction: near zero pitch AND near zero drift = quiescent posture
    quiescent = [a and b for a, b in zip(near_zero_pitch, near_zero_drift)]
    summary["quiescent_pct"] = percent_window(quiescent)
    summary["tau_pitch_quiescent"] = stat_summary(
        conditional(data["tau_pitch"], quiescent))

    # Pitch summary
    summary["pitch_deg_all"] = stat_summary(pitch_deg)
    summary["pitch_x_ref_rad"] = stat_summary(data["pitch_x_ref_rad"])
    summary["pitch_x_error_rad"] = stat_summary(data["pitch_x_error_rad"])
    summary["pitch_rate_rad_s"] = stat_summary(data["pitch_rate_x_rad_s"])
    summary["control_pitch_x"] = stat_summary(data["control_pitch_x"])

    # Other torques
    summary["tau_position"] = stat_summary(data["tau_position"])
    summary["tau_support_velocity"] = stat_summary(data["tau_support_velocity"])
    summary["tau_pitch_rate"] = stat_summary(data["tau_pitch_rate"])
    summary["final_wheel_tau_with_apc"] = stat_summary(data["final_wheel_tau_with_apc"])

    # Wheel velocity damping
    wheel_damp_avg = []
    for l, r in zip(data["tau_wheel_velocity_left"], data["tau_wheel_velocity_right"]):
        if math.isnan(l) and math.isnan(r):
            wheel_damp_avg.append(float("nan"))
        else:
            wheel_damp_avg.append(0.5 * ((l if not math.isnan(l) else 0) +
                                         (r if not math.isnan(r) else 0)))
    summary["tau_wheel_velocity_avg"] = stat_summary(wheel_damp_avg)

    # Correlations
    summary["corr_taup_vs_drift"] = pearson(data["tau_pitch"], drift)
    summary["corr_taup_vs_pitch_deg"] = pearson(data["tau_pitch"], pitch_deg)
    summary["corr_taup_vs_pitch_rate"] = pearson(
        data["tau_pitch"], data["pitch_rate_x_rad_s"])
    summary["corr_taup_vs_pitch_error"] = pearson(
        data["tau_pitch"], data["pitch_x_error_rad"])

    # Print headline numbers
    tp = summary["tau_pitch_all"]
    tp_q = summary["tau_pitch_quiescent"]
    tp_nzp = summary["tau_pitch_near_zero_pitch"]
    tp_nzd = summary["tau_pitch_near_zero_drift"]
    tp_pos = summary["tau_pitch_pos_drift"]
    tp_neg = summary["tau_pitch_neg_drift"]
    p_deg = summary["pitch_deg_all"]
    p_ref = summary["pitch_x_ref_rad"]
    p_err = summary["pitch_x_error_rad"]
    print(f"  tau_pitch all       : mean={tp['mean']:+.3f} median={tp['median']:+.3f} min={tp['min']:+.3f} max={tp['max']:+.3f} RMS={tp['rms']:.3f}")
    print(f"  tau_pitch |p|<1deg  : mean={tp_nzp['mean']:+.3f} median={tp_nzp['median']:+.3f} n={tp_nzp['n']} ({summary['near_zero_pitch_pct']:.1f}%)")
    print(f"  tau_pitch |e|<0.03  : mean={tp_nzd['mean']:+.3f} median={tp_nzd['median']:+.3f} n={tp_nzd['n']} ({summary['near_zero_drift_pct']:.1f}%)")
    print(f"  tau_pitch quiescent : mean={tp_q['mean']:+.3f} median={tp_q['median']:+.3f} n={tp_q['n']} ({summary['quiescent_pct']:.1f}%)")
    print(f"  tau_pitch +drift    : mean={tp_pos['mean']:+.3f} n={tp_pos['n']} ({summary['pos_drift_pct']:.1f}%)")
    print(f"  tau_pitch -drift    : mean={tp_neg['mean']:+.3f} n={tp_neg['n']} ({summary['neg_drift_pct']:.1f}%)")
    print(f"  pitch (deg)         : mean={p_deg['mean']:+.3f} median={p_deg['median']:+.3f} min={p_deg['min']:+.3f} max={p_deg['max']:+.3f} RMS={p_deg['rms']:.3f}")
    print(f"  pitch_ref (rad)     : mean={p_ref['mean']:+.5f} median={p_ref['median']:+.5f} min={p_ref['min']:+.5f} max={p_ref['max']:+.5f}")
    print(f"  pitch_err (rad)     : mean={p_err['mean']:+.5f} median={p_err['median']:+.5f} min={p_err['min']:+.5f} max={p_err['max']:+.5f}")
    print(f"  tau_position        : mean={summary['tau_position']['mean']:+.3f}")
    print(f"  tau_support_velocity: mean={summary['tau_support_velocity']['mean']:+.3f}")
    print(f"  tau_wheel_vel avg   : mean={summary['tau_wheel_velocity_avg']['mean']:+.3f}")
    print(f"  final tau (apc)     : mean={summary['final_wheel_tau_with_apc']['mean']:+.3f}")
    print(f"  corr tau_pitch vs drift     : {summary['corr_taup_vs_drift']:+.3f}")
    print(f"  corr tau_pitch vs pitch_deg : {summary['corr_taup_vs_pitch_deg']:+.3f}")
    print(f"  corr tau_pitch vs pitch_rate: {summary['corr_taup_vs_pitch_rate']:+.3f}")
    print(f"  corr tau_pitch vs pitch_err : {summary['corr_taup_vs_pitch_error']:+.3f}")

    return summary


def main():
    out_dir = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "tau_pitch_bias_audit.json"

    results = OrderedDict()
    for name, rel_path in PROFILES.items():
        csv_path = EVAL_DIR / rel_path
        if not csv_path.exists():
            print(f"MISSING: {csv_path}")
            continue
        results[name] = audit_profile(name, csv_path)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved JSON: {out_json}")


if __name__ == "__main__":
    main()
