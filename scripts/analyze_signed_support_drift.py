"""
Signed Support Drift and Phase Behavior Audit Script

Analyzes D2, E2, E2b telemetry to:
1. Compute signed support drift metrics
2. Detect phase reversal / missed recenter opportunities
3. Generate audit reports
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/signed_support_drift_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_FILES = {
    "D2": "outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv",
    "E2": "outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500/e2_low_0p300_500_telemetry.csv",
    "E2b": "outputs/step_e_extreme_support_fix_eval/e2b_low_0p300_500/e2b_low_0p300_500_telemetry.csv",
}


def load_telemetry(name: str, path: str) -> pd.DataFrame:
    """Load telemetry CSV with proper types."""
    df = pd.read_csv(path)
    print(f"[{name}] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def find_column(df: pd.DataFrame, patterns: list) -> str:
    """Find first matching column from patterns list."""
    cols = df.columns.tolist()
    for p in patterns:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return None


def compute_signed_drift_metrics(series: pd.Series, name: str) -> dict:
    """Compute comprehensive signed drift metrics."""
    valid = series.dropna()

    if len(valid) == 0:
        return {"error": "No valid data"}

    mean = valid.mean()
    median = valid.median()
    std = valid.std()
    rms = np.sqrt(np.mean(valid**2))
    final = valid.iloc[-1] if len(valid) > 0 else np.nan
    min_val = valid.min()
    max_val = valid.max()

    # Percent time positive/negative
    n_positive = (valid > 0).sum()
    n_negative = (valid < 0).sum()
    pct_positive = 100.0 * n_positive / len(valid)
    pct_negative = 100.0 * n_negative / len(valid)

    # Zero crossings
    signs = np.sign(valid.values)
    crossings = np.sum(np.diff(signs) != 0)
    pct_zero_crossings = 100.0 * crossings / max(1, len(valid) - 1)

    # Longest same-sign interval
    max_same_sign_run = 0
    current_run = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            current_run += 1
        else:
            max_same_sign_run = max(max_same_sign_run, current_run)
            current_run = 1
    max_same_sign_run = max(max_same_sign_run, current_run)

    # Peak drift
    peak_positive = valid.max()
    peak_negative = valid.min()

    # Bias ratio: how much time spent on one side vs the other
    bias_ratio = pct_positive / max(0.1, pct_negative)

    # Classify behavior
    classification = classify_drift(mean, median, final, std, pct_positive, pct_negative, bias_ratio, max_same_sign_run, len(valid))

    return {
        "variable": name,
        "n_samples": len(valid),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "rms": float(rms),
        "final": float(final),
        "min": float(min_val),
        "max": float(max_val),
        "peak_positive": float(peak_positive),
        "peak_negative": float(peak_negative),
        "pct_positive": float(pct_positive),
        "pct_negative": float(pct_negative),
        "zero_crossings": int(crossings),
        "pct_zero_crossings": float(pct_zero_crossings),
        "max_same_sign_run": int(max_same_sign_run),
        "bias_ratio": float(bias_ratio),
        "classification": classification,
    }


def classify_drift(mean, median, final, std, pct_pos, pct_neg, bias_ratio, max_run, n_samples) -> str:
    """Classify the signed drift behavior."""
    # Check for conclusive bias
    if abs(bias_ratio - 1.0) < 0.3 and pct_pos > 40 and pct_pos < 60:
        # Oscillates around zero
        if std < 0.05:
            return "OSCILLATES_AROUND_ZERO_TIGHT"
        elif std < 0.10:
            return "OSCILLATES_AROUND_ZERO"
        else:
            return "OSCILLATES_AROUND_ZERO_LARGE_AMPLITUDE"

    # Strong one-sided bias
    if pct_pos > 70:
        if bias_ratio > 3.0:
            return "POSITIVE_BIASED_DRIFT_STRONG"
        elif bias_ratio > 2.0:
            return "POSITIVE_BIASED_DRIFT_MODERATE"
        else:
            return "POSITIVE_BIASED_DRIFT_WEAK"
    if pct_neg > 70:
        if bias_ratio < 0.33:
            return "NEGATIVE_BIASED_DRIFT_STRONG"
        elif bias_ratio < 0.5:
            return "NEGATIVE_BIASED_DRIFT_MODERATE"
        else:
            return "NEGATIVE_BIASED_DRIFT_WEAK"

    # Ratcheting: final far from zero, one-sided
    if abs(final) > 0.10 and pct_pos > 65:
        return "RATCHETING_DRIFT_POSITIVE"
    if abs(final) > 0.10 and pct_neg > 65:
        return "RATCHETING_DRIFT_NEGATIVE"

    # Long same-sign runs indicate ratcheting tendency
    if max_run > 0.3 * n_samples and std > 0.05:
        if mean > 0.02:
            return "RATCHETING_DRIFT_POSITIVE"
        elif mean < -0.02:
            return "RATCHETING_DRIFT_NEGATIVE"
        else:
            return "RATCHETING_DRIFT_UNCLEAR"

    return "SIGNED_DRIFT_INCONCLUSIVE"


def detect_reversal_windows(df: pd.DataFrame, config: dict) -> list:
    """Detect phase reversal windows where support should recenter."""
    windows = []

    # Find columns
    support_col = find_column(df, ["support_center_x", "com_error_x", "cp_error_x"])
    pitch_col = find_column(df, ["pitch_x", "pitch_x_rad"])
    pitch_rate_col = find_column(df, ["pitch_rate_x", "pitch_rate_x_rad_s"])
    wheel_vel_col = find_column(df, ["wheel_vel_mean", "stage2c_wheel_vel_mean"])
    tau_pos_col = find_column(df, ["tau_position"])
    hip_yaw_col = find_column(df, ["hip_yaw_abs_max"])

    if not all([support_col, pitch_col, pitch_rate_col, wheel_vel_col, tau_pos_col]):
        print(f"  Warning: Missing columns. Found: support={support_col}, pitch={pitch_col}, pitch_rate={pitch_rate_col}, wheel_vel={wheel_vel_col}, tau_pos={tau_pos_col}")
        return []

    # Get data
    support = df[support_col].values
    pitch = df[pitch_col].values
    pitch_rate = df[pitch_rate_col].values
    wheel_vel = df[wheel_vel_col].values
    tau_pos = df[tau_pos_col].values
    hip_yaw = df[hip_yaw_col].values if hip_yaw_col else np.zeros_like(support)

    # Thresholds
    support_deadband = config.get("support_deadband", 0.05)  # 5cm deadband
    pitch_safe_threshold = config.get("pitch_safe_threshold", 0.1)  # 0.1 rad (~6 deg)
    reversal_min_steps = config.get("reversal_min_steps", 20)

    n = len(support)
    i = 0

    while i < n - reversal_min_steps:
        # Look for support far from zero
        if abs(support[i]) < support_deadband:
            i += 1
            continue

        # Track this window
        start_idx = i
        sign_start = np.sign(support[i])
        support_start = support[i]

        # Find when pitch reverses or support returns toward zero
        found_reversal = False
        for j in range(i + 1, min(i + 200, n)):
            # Check if pitch rate has reversed sign (indicates body moving opposite direction)
            if j > i + 5:
                pitch_rate_sign_start = np.sign(pitch_rate[i])
                pitch_rate_sign_now = np.sign(pitch_rate[j])

                # Check if support velocity has reversed
                if j > i + 3:
                    support_vel_start = (support[min(j-3, n-1)] - support[i]) / 3
                    support_vel_now = (support[min(j, n-1)] - support[min(j-3, n-1)]) / 3

                    # Support moving toward zero
                    toward_zero = (sign_start > 0 and support_vel_now < 0) or (sign_start < 0 and support_vel_now > 0)

                    # Pitch reversed
                    pitch_reversed = pitch_rate_sign_start * pitch_rate_sign_now < 0

                    # Detect reversal window
                    if toward_zero and pitch_reversed:
                        end_idx = j
                        support_end = support[j]
                        wheel_vel_in_window = wheel_vel[i:j]
                        tau_pos_in_window = tau_pos[i:j]

                        # Check if wheel reversed too aggressively
                        wheel_vel_max = np.max(np.abs(wheel_vel_in_window))
                        wheel_vel_mean = np.mean(wheel_vel_in_window)

                        # Determine if recentering worked
                        recenter_amount = abs(support_start) - abs(support_end)
                        support_moved_toward_zero = recenter_amount > 0.01

                        # Check hip yaw
                        hip_yaw_max_in_window = np.max(hip_yaw[i:j]) if hip_yaw_col else 0

                        # Classify
                        if support_moved_toward_zero:
                            if abs(wheel_vel_mean) > 5.0:
                                behavior = "RECENTERING_WORKS_BUT_AGGRESSIVE"
                            else:
                                behavior = "RECENTERING_WORKS"
                        else:
                            if abs(wheel_vel_mean) > 3.0:
                                behavior = "RECENTERING_PREMATURELY_REVERSED"
                            elif hip_yaw_max_in_window > 0.10:
                                behavior = "POSITION_TERM_TOO_AGGRESSIVE_CAUSES_HIP_YAW"
                            else:
                                behavior = "RECENTERING_TOO_WEAK"

                        windows.append({
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "duration_steps": int(end_idx - start_idx),
                            "support_start": float(support_start),
                            "support_end": float(support_end),
                            "recenter_amount": float(recenter_amount),
                            "support_moved_toward_zero": bool(support_moved_toward_zero),
                            "wheel_vel_max": float(wheel_vel_max),
                            "wheel_vel_mean": float(wheel_vel_mean),
                            "tau_position_mean": float(np.mean(np.abs(tau_pos_in_window))),
                            "hip_yaw_max_in_window": float(hip_yaw_max_in_window),
                            "behavior": behavior,
                        })
                        found_reversal = True
                        i = j
                        break

        if not found_reversal:
            i += 1

    return windows


def compute_summary_stats(windows: list) -> dict:
    """Compute summary statistics from reversal windows."""
    if not windows:
        return {
            "total_reversal_windows": 0,
            "behavior_summary": {},
            "summary": "NO_REVERSAL_WINDOWS_DETECTED",
        }

    behavior_counts = {}
    total_recenter_amount = 0
    total_windows = len(windows)

    for w in windows:
        behavior = w["behavior"]
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        total_recenter_amount += w["recenter_amount"]

    avg_recenter = total_recenter_amount / total_windows if total_windows > 0 else 0

    return {
        "total_reversal_windows": total_windows,
        "behavior_summary": {k: {"count": v, "pct": 100.0 * v / total_windows} for k, v in behavior_counts.items()},
        "avg_recenter_amount": float(avg_recenter),
        "total_recenter_amount": float(total_recenter_amount),
    }


def main():
    print("=" * 60)
    print("SIGNED SUPPORT DRIFT AND PHASE BEHAVIOR AUDIT")
    print("=" * 60)

    all_results = {}
    all_windows = {}
    all_summaries = {}

    config = {
        "support_deadband": 0.05,  # 5cm
        "pitch_safe_threshold": 0.1,  # 0.1 rad
        "reversal_min_steps": 20,
    }

    for name, path in TELEMETRY_FILES.items():
        if not os.path.exists(path):
            print(f"[{name}] WARNING: File not found: {path}")
            continue

        print(f"\n[{name}] Processing telemetry...")

        # Load telemetry
        df = load_telemetry(name, path)

        # Find support column
        support_col = find_column(df, ["support_center_x", "com_error_x", "cp_error_x"])
        if not support_col:
            print(f"[{name}] ERROR: Could not find support column")
            continue

        print(f"[{name}] Using support column: {support_col}")

        # Compute signed drift metrics
        support_series = df[support_col]
        drift_metrics = compute_signed_drift_metrics(support_series, support_col)

        # Also check com_error_x if available
        com_error_col = find_column(df, ["com_error_x"])
        if com_error_col and com_error_col != support_col:
            com_metrics = compute_signed_drift_metrics(df[com_error_col], com_error_col)
        else:
            com_metrics = None

        # Check hip_yaw
        hip_yaw_col = find_column(df, ["hip_yaw_abs_max"])
        if hip_yaw_col:
            hip_yaw_metrics = compute_signed_drift_metrics(df[hip_yaw_col], hip_yaw_col)
        else:
            hip_yaw_metrics = None

        # Detect reversal windows
        windows = detect_reversal_windows(df, config)
        window_summary = compute_summary_stats(windows)

        all_results[name] = {
            "support_drift": drift_metrics,
            "com_error_drift": com_metrics,
            "hip_yaw_metrics": hip_yaw_metrics,
        }
        all_windows[name] = windows
        all_summaries[name] = window_summary

        # Print summary
        print(f"\n[{name}] SIGNED SUPPORT DRIFT METRICS:")
        print(f"  Mean: {drift_metrics['mean']:.4f} m")
        print(f"  Median: {drift_metrics['median']:.4f} m")
        print(f"  Final: {drift_metrics['final']:.4f} m")
        print(f"  RMS: {drift_metrics['rms']:.4f} m")
        print(f"  % Positive: {drift_metrics['pct_positive']:.1f}%")
        print(f"  % Negative: {drift_metrics['pct_negative']:.1f}%")
        print(f"  Zero crossings: {drift_metrics['zero_crossings']}")
        print(f"  Max same-sign run: {drift_metrics['max_same_sign_run']} steps")
        print(f"  Bias ratio: {drift_metrics['bias_ratio']:.2f}")
        print(f"  Classification: {drift_metrics['classification']}")

        if hip_yaw_metrics:
            print(f"\n[{name}] HIP_YAW METRICS:")
            print(f"  Mean: {hip_yaw_metrics['mean']:.4f} rad")
            print(f"  Max: {hip_yaw_metrics['max']:.4f} rad")
            print(f"  RMS: {hip_yaw_metrics['rms']:.4f} rad")

        print(f"\n[{name}] REVERSAL WINDOW SUMMARY:")
        print(f"  Total windows: {window_summary['total_reversal_windows']}")
        for behavior, stats in window_summary.get('behavior_summary', {}).items():
            print(f"  {behavior}: {stats['count']} ({stats['pct']:.1f}%)")

    # Save results
    output_json = OUTPUT_DIR / "signed_drift_metrics.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics to: {output_json}")

    # Save CSV summary
    csv_rows = []
    for name, results in all_results.items():
        dm = results.get("support_drift", {})
        hm = results.get("hip_yaw_metrics", {})
        ws = all_summaries.get(name, {})
        csv_rows.append({
            "variant": name,
            "support_mean_m": dm.get("mean"),
            "support_median_m": dm.get("median"),
            "support_final_m": dm.get("final"),
            "support_rms_m": dm.get("rms"),
            "support_pct_positive": dm.get("pct_positive"),
            "support_pct_negative": dm.get("pct_negative"),
            "support_zero_crossings": dm.get("zero_crossings"),
            "support_max_same_sign_run": dm.get("max_same_sign_run"),
            "support_bias_ratio": dm.get("bias_ratio"),
            "support_classification": dm.get("classification"),
            "hip_yaw_mean_rad": hm.get("mean"),
            "hip_yaw_max_rad": hm.get("max"),
            "hip_yaw_rms_rad": hm.get("rms"),
            "reversal_windows_total": ws.get("total_reversal_windows"),
        })

    csv_df = pd.DataFrame(csv_rows)
    output_csv = OUTPUT_DIR / "signed_drift_metrics.csv"
    csv_df.to_csv(output_csv, index=False)
    print(f"Saved CSV to: {output_csv}")

    # Save reversal windows
    all_windows_flat = []
    for name, windows in all_windows.items():
        for w in windows:
            w_with_name = {"variant": name, **w}
            all_windows_flat.append(w_with_name)

    if all_windows_flat:
        windows_df = pd.DataFrame(all_windows_flat)
        windows_csv = OUTPUT_DIR / "reversal_windows.csv"
        windows_df.to_csv(windows_csv, index=False)
        print(f"Saved reversal windows to: {windows_csv}")

    # Save phase behavior summary
    phase_summary = {}
    for name, windows in all_windows.items():
        ws = all_summaries.get(name, {})
        phase_summary[name] = ws

    phase_json = OUTPUT_DIR / "phase_behavior_summary.json"
    with open(phase_json, 'w') as f:
        json.dump(phase_summary, f, indent=2)
    print(f"Saved phase behavior summary to: {phase_json}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return all_results, all_windows, all_summaries


if __name__ == "__main__":
    main()
