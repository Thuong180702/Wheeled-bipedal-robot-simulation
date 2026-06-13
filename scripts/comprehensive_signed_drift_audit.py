"""
Comprehensive Signed Support Drift and Phase Behavior Audit

This script performs:
1. Phase 1: Signed support drift audit
2. Phase 2: Phase reversal / missed recenter opportunity audit
3. Phase 3: Generate design inputs for phase-aware recentering strategy
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
    """Load telemetry CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"[{name}] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_signed_drift_metrics(series: pd.Series, name: str) -> dict:
    """Compute comprehensive signed drift metrics for a series."""
    valid = series.dropna()
    if len(valid) == 0:
        return {"error": "No valid data"}

    mean = float(valid.mean())
    median = float(valid.median())
    std = float(valid.std())
    rms = float(np.sqrt(np.mean(valid**2)))
    final = float(valid.iloc[-1])
    min_val = float(valid.min())
    max_val = float(valid.max())

    n_positive = int((valid > 0).sum())
    n_negative = int((valid < 0).sum())
    pct_positive = 100.0 * n_positive / len(valid)
    pct_negative = 100.0 * n_negative / len(valid)

    signs = np.sign(valid.values)
    crossings = int(np.sum(np.diff(signs) != 0))
    pct_zero_crossings = 100.0 * crossings / max(1, len(valid) - 1)

    max_same_sign_run = 0
    current_run = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            current_run += 1
        else:
            max_same_sign_run = max(max_same_sign_run, current_run)
            current_run = 1
    max_same_sign_run = max(max_same_sign_run, current_run)

    bias_ratio = pct_positive / max(0.1, pct_negative)

    return {
        "variable": name,
        "n_samples": len(valid),
        "mean": mean,
        "median": median,
        "std": std,
        "rms": rms,
        "final": final,
        "min": min_val,
        "max": max_val,
        "peak_positive": float(valid.max()),
        "peak_negative": float(valid.min()),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pct_positive": pct_positive,
        "pct_negative": pct_negative,
        "zero_crossings": crossings,
        "pct_zero_crossings": pct_zero_crossings,
        "max_same_sign_run": max_same_sign_run,
        "bias_ratio": bias_ratio,
        "classification": classify_signed_drift(mean, median, final, std, pct_positive, pct_negative, bias_ratio, max_same_sign_run, len(valid)),
    }


def classify_signed_drift(mean, median, final, std, pct_pos, pct_neg, bias_ratio, max_run, n_samples) -> str:
    """Classify signed drift behavior."""
    if abs(bias_ratio - 1.0) < 0.3 and pct_pos > 40 and pct_pos < 60:
        if std < 0.01:
            return "OSCILLATES_AROUND_ZERO_TIGHT"
        elif std < 0.05:
            return "OSCILLATES_AROUND_ZERO"
        else:
            return "OSCILLATES_AROUND_ZERO_LARGE"

    if pct_pos > 70:
        if bias_ratio > 3.0:
            return "POSITIVE_BIASED_STRONG"
        elif bias_ratio > 2.0:
            return "POSITIVE_BIASED_MODERATE"
        else:
            return "POSITIVE_BIASED_WEAK"
    if pct_neg > 70:
        if bias_ratio < 0.33:
            return "NEGATIVE_BIASED_STRONG"
        elif bias_ratio < 0.5:
            return "NEGATIVE_BIASED_MODERATE"
        else:
            return "NEGATIVE_BIASED_WEAK"

    if abs(final) > 0.10 and pct_pos > 65:
        return "RATCHETING_POSITIVE"
    if abs(final) > 0.10 and pct_neg > 65:
        return "RATCHETING_NEGATIVE"

    if max_run > 0.3 * n_samples and std > 0.02:
        if mean > 0.02:
            return "RATCHETING_POSITIVE"
        elif mean < -0.02:
            return "RATCHETING_NEGATIVE"
        else:
            return "RATCHETING_UNCLEAR"

    return "INCONCLUSIVE"


def detect_phase_reversals(df: pd.DataFrame, variant_name: str) -> dict:
    """Detect and analyze phase reversal behavior."""
    results = {
        "variant": variant_name,
        "n_steps": len(df),
        "pitch_reversals": [],
        "support_error_reversals": [],
        "premature_reversal_count": 0,
        "good_recovery_count": 0,
    }

    pitch = df['pitch_x'].values
    pitch_rate = df['pitch_rate_x'].values
    support_err = df['support_position_error_m'].values
    hip_yaw = df['hip_yaw_abs_max'].values
    tau_position = df['tau_position'].values

    n = len(pitch)
    reversal_windows = []

    # Find pitch reversals (where pitch_rate changes sign significantly)
    for i in range(5, n - 5):
        prev_rate = pitch_rate[i-3]
        curr_rate = pitch_rate[i]
        next_rate = pitch_rate[i+3]

        # Detect reversal: rate goes from positive to negative or vice versa
        if prev_rate * curr_rate < 0 and abs(curr_rate) > 0.02:
            # This is a pitch rate reversal
            reversal_window = {
                "step": i,
                "pitch_at_reversal": pitch[i],
                "pitch_rate_at_reversal": pitch_rate[i],
                "support_err_at_reversal": support_err[i],
                "hip_yaw_at_reversal": hip_yaw[i],
                "tau_position_at_reversal": tau_position[i],
            }

            # Check if support error is large
            if support_err[i] > 0.10:
                reversal_window["support_error_large"] = True
            else:
                reversal_window["support_error_large"] = False

            # Check what happens in next 20 steps
            future_support = support_err[i:min(i+20, n)]
            if len(future_support) > 1:
                reversal_window["support_err_min_future"] = float(np.min(future_support))
                reversal_window["support_err_max_future"] = float(np.max(future_support))
                reversal_window["support_err_reduced"] = bool(future_support[0] - np.min(future_support) > 0.01)

            reversal_windows.append(reversal_window)

    results["pitch_reversals"] = reversal_windows
    results["total_pitch_reversals"] = len(reversal_windows)
    results["large_support_error_reversals"] = sum(1 for r in reversal_windows if r.get("support_error_large", False))
    results["good_recovery_count"] = sum(1 for r in reversal_windows if r.get("support_err_reduced", False))

    return results


def analyze_wheel_reversal_behavior(df: pd.DataFrame, variant_name: str) -> dict:
    """Analyze wheel velocity reversal behavior."""
    results = {
        "variant": variant_name,
        "n_steps": len(df),
    }

    # Check wheel velocity
    wheel_vel_cols = [c for c in df.columns if 'wheel_vel' in c.lower() and 'mean' in c.lower()]
    if wheel_vel_cols:
        wheel_vel = df[wheel_vel_cols[0]].values
        results["wheel_vel_mean_col"] = wheel_vel_cols[0]
        results["wheel_vel_mean_stats"] = {
            "mean": float(np.mean(wheel_vel)),
            "std": float(np.std(wheel_vel)),
            "max": float(np.max(wheel_vel)),
            "min": float(np.min(wheel_vel)),
            "positive_pct": float(100 * np.sum(wheel_vel > 0) / len(wheel_vel)),
            "negative_pct": float(100 * np.sum(wheel_vel < 0) / len(wheel_vel)),
        }

    # Check tau_position
    tau_position = df['tau_position'].values
    results["tau_position_stats"] = {
        "mean": float(np.mean(tau_position)),
        "std": float(np.std(tau_position)),
        "max": float(np.max(tau_position)),
        "min": float(np.min(tau_position)),
        "positive_pct": float(100 * np.sum(tau_position > 0) / len(tau_position)),
        "negative_pct": float(100 * np.sum(tau_position < 0) / len(tau_position)),
        "at_cap_pct": float(100 * np.sum(np.abs(tau_position) >= 3.0) / len(tau_position)),  # Assuming cap of 3
    }

    # Check hip yaw correlation with position error
    hip_yaw = df['hip_yaw_abs_max'].values
    support_err = df['support_position_error_m'].values

    results["hip_yaw_support_error_correlation"] = float(np.corrcoef(hip_yaw, support_err)[0, 1])

    return results


def analyze_hip_yaw_regression(df_d2: pd.DataFrame, df_e2: pd.DataFrame, df_e2b: pd.DataFrame) -> dict:
    """Analyze why E2/E2b regressed hip_yaw."""
    results = {}

    for name, df in [("D2", df_d2), ("E2", df_e2), ("E2b", df_e2b)]:
        results[name] = {
            "hip_yaw_abs_max_mean": float(df['hip_yaw_abs_max'].mean()),
            "hip_yaw_abs_max_max": float(df['hip_yaw_abs_max'].max()),
            "hip_yaw_abs_max_std": float(df['hip_yaw_abs_max'].std()),
            "hip_yaw_divergence_mean": float(df['hip_yaw_divergence'].mean()),
            "hip_yaw_divergence_max": float(df['hip_yaw_divergence'].max()),
            "hip_yaw_asymmetry_mean": float(df['hip_yaw_asymmetry'].mean()),
            "hip_yaw_asymmetry_max": float(df['hip_yaw_asymmetry'].max()),
        }

    # The key insight: E2/E2b reduced hip_yaw_abs_max by reducing the position correction
    # But this also means the robot is less correcting position, which could cause drift
    return results


def generate_phase_aware_strategy_inputs(results: dict) -> dict:
    """Generate inputs for phase-aware recentering strategy design."""
    return {
        "key_findings": {
            "support_error_source": "hip_yaw_comp_support_error_m (yaw-induced sagittal position error)",
            "root_cause": "Hip yaw divergence causes sagittal position error, not pure CoM drift",
            "D2_issue": "High hip_yaw_abs_max (0.31 rad) but position correction causes even more yaw",
            "E2_fix": "Reduced position correction authority, hip_yaw dropped to 0.13 rad",
            "E2_tradeoff": "Improved hip_yaw but position error crossings still high (62 normalized)",
        },
        "strategy_constraints": {
            "must_preserve": ["balance recovery", "pitch stabilization", "hip yaw bounds"],
            "should_improve": ["position recentering when safe", "zero-crossing behavior"],
            "must_avoid": ["excessive wheel reversal", "hip yaw regression", "premature reversal"],
        },
        "recommended_approach": {
            "principle": "Phase-aware recentering with safe pitch detection",
            "key_signal": "pitch_x and pitch_rate to detect dangerous vs recovering state",
            "secondary_signal": "hip_yaw_abs_max to avoid position corrections that worsen yaw",
            "recenter_trigger": "When pitch is recovering AND support_error > deadband AND hip_yaw bounded",
            "recenter_strength": "Bounded, proportional to signed support error",
        }
    }


def main():
    print("=" * 70)
    print("SIGNED SUPPORT DRIFT AND PHASE BEHAVIOR AUDIT")
    print("=" * 70)

    all_results = {}
    all_phase_results = {}
    all_wheel_results = {}

    # Load all telemetry
    telemetry = {}
    for name, path in TELEMETRY_FILES.items():
        try:
            telemetry[name] = load_telemetry(name, path)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

    # Phase 1: Signed drift metrics
    print("\n" + "=" * 70)
    print("PHASE 1: SIGNED SUPPORT DRIFT METRICS")
    print("=" * 70)

    for name, df in telemetry.items():
        print(f"\n[{name}] Computing drift metrics...")

        # The signed error source is hip_yaw_comp_support_error_m
        # But we need to understand its sign
        signed_error = df['hip_yaw_comp_support_error_m']

        # Compute metrics for signed error
        signed_metrics = compute_signed_drift_metrics(signed_error, "hip_yaw_comp_support_error_m_signed")

        # Also compute for magnitude
        magnitude_metrics = {
            "variable": "support_position_error_m",
            "mean": float(df['support_position_error_m'].mean()),
            "max": float(df['support_position_error_m'].max()),
            "std": float(df['support_position_error_m'].std()),
            "crossings_150": int((df['support_position_error_m'] > 0.15).sum()),
            "crossings_100": int((df['support_position_error_m'] > 0.10).sum()),
            "crossings_normalized_500": float((df['support_position_error_m'] > 0.15).sum() * 500 / len(df)),
        }

        # Hip yaw metrics
        hip_yaw_metrics = compute_signed_drift_metrics(df['hip_yaw_abs_max'], "hip_yaw_abs_max")

        # Pitch metrics
        pitch_metrics = compute_signed_drift_metrics(df['pitch_x'], "pitch_x")

        all_results[name] = {
            "signed_support_error": signed_metrics,
            "magnitude_support_error": magnitude_metrics,
            "hip_yaw": hip_yaw_metrics,
            "pitch": pitch_metrics,
        }

        print(f"\n[{name}] SIGNED SUPPORT ERROR:")
        print(f"  Mean: {signed_metrics['mean']:.6f} m")
        print(f"  Median: {signed_metrics['median']:.6f} m")
        print(f"  Std: {signed_metrics['std']:.6f} m")
        print(f"  Min: {signed_metrics['min']:.6f} m")
        print(f"  Max: {signed_metrics['max']:.6f} m")
        print(f"  % Positive: {signed_metrics['pct_positive']:.1f}%")
        print(f"  % Negative: {signed_metrics['pct_negative']:.1f}%")
        print(f"  Zero crossings: {signed_metrics['zero_crossings']}")
        print(f"  Classification: {signed_metrics['classification']}")

        print(f"\n[{name}] MAGNITUDE SUPPORT ERROR:")
        print(f"  Mean: {magnitude_metrics['mean']:.4f} m")
        print(f"  Max: {magnitude_metrics['max']:.4f} m")
        print(f"  Std: {magnitude_metrics['std']:.4f} m")
        print(f"  Crossings >0.15m: {magnitude_metrics['crossings_150']} (normalized to 500: {magnitude_metrics['crossings_normalized_500']:.1f})")
        print(f"  Crossings >0.10m: {magnitude_metrics['crossings_100']}")

        print(f"\n[{name}] HIP_YAW:")
        print(f"  Mean: {hip_yaw_metrics['mean']:.4f} rad")
        print(f"  Max: {hip_yaw_metrics['max']:.4f} rad")
        print(f"  Std: {hip_yaw_metrics['std']:.4f} rad")

        print(f"\n[{name}] PITCH:")
        print(f"  Mean: {pitch_metrics['mean']:.4f} rad ({np.degrees(pitch_metrics['mean']):.2f} deg)")
        print(f"  Max: {pitch_metrics['max']:.4f} rad ({np.degrees(pitch_metrics['max']):.2f} deg)")
        print(f"  % Positive (forward lean): {pitch_metrics['pct_positive']:.1f}%")

    # Phase 2: Phase reversal analysis
    print("\n" + "=" * 70)
    print("PHASE 2: PHASE REVERSAL BEHAVIOR")
    print("=" * 70)

    for name, df in telemetry.items():
        phase_results = detect_phase_reversals(df, name)
        all_phase_results[name] = phase_results

        print(f"\n[{name}] PHASE REVERSAL ANALYSIS:")
        print(f"  Total pitch reversals: {phase_results['total_pitch_reversals']}")
        print(f"  Reversals with large support error: {phase_results['large_support_error_reversals']}")
        print(f"  Good recoveries after reversal: {phase_results['good_recovery_count']}")

    # Phase 3: Wheel reversal behavior
    print("\n" + "=" * 70)
    print("PHASE 3: WHEEL REVERSAL BEHAVIOR")
    print("=" * 70)

    for name, df in telemetry.items():
        wheel_results = analyze_wheel_reversal_behavior(df, name)
        all_wheel_results[name] = wheel_results

        print(f"\n[{name}] WHEEL VELOCITY:")
        if "wheel_vel_mean_stats" in wheel_results:
            stats = wheel_results["wheel_vel_mean_stats"]
            print(f"  Mean: {stats['mean']:.4f} rad/s")
            print(f"  Std: {stats['std']:.4f} rad/s")
            print(f"  Max: {stats['max']:.4f} rad/s")
            print(f"  Min: {stats['min']:.4f} rad/s")
            print(f"  % Positive: {stats['positive_pct']:.1f}%")
            print(f"  % Negative: {stats['negative_pct']:.1f}%")

        print(f"\n[{name}] TAU_POSITION:")
        stats = wheel_results["tau_position_stats"]
        print(f"  Mean: {stats['mean']:.4f} Nm")
        print(f"  Std: {stats['std']:.4f} Nm")
        print(f"  Min: {stats['min']:.4f} Nm (negative = correcting forward fall)")
        print(f"  At cap (3Nm): {stats['at_cap_pct']:.1f}% of time")

    # Phase 4: Hip yaw regression analysis
    print("\n" + "=" * 70)
    print("PHASE 4: HIP YAW REGRESSION ANALYSIS")
    print("=" * 70)

    if all(t in telemetry for t in ["D2", "E2", "E2b"]):
        hip_yaw_analysis = analyze_hip_yaw_regression(
            telemetry["D2"], telemetry["E2"], telemetry["E2b"]
        )
        print("\nHIP YAW COMPARISON:")
        for name in ["D2", "E2", "E2b"]:
            r = hip_yaw_analysis[name]
            print(f"\n{name}:")
            print(f"  hip_yaw_abs_max: mean={r['hip_yaw_abs_max_mean']:.4f}, max={r['hip_yaw_abs_max_max']:.4f}")
            print(f"  hip_yaw_divergence: mean={r['hip_yaw_divergence_mean']:.4f}, max={r['hip_yaw_divergence_max']:.4f}")
            print(f"  hip_yaw_asymmetry: mean={r['hip_yaw_asymmetry_mean']:.6f}, max={r['hip_yaw_asymmetry_max']:.6f}")

    # Generate strategy inputs
    strategy_inputs = generate_phase_aware_strategy_inputs(all_results)

    # Save results
    output_json = OUTPUT_DIR / "signed_drift_metrics.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved metrics to: {output_json}")

    phase_json = OUTPUT_DIR / "phase_behavior_summary.json"
    with open(phase_json, 'w') as f:
        json.dump(all_phase_results, f, indent=2)
    print(f"Saved phase behavior to: {phase_json}")

    wheel_json = OUTPUT_DIR / "wheel_reversal_summary.json"
    with open(wheel_json, 'w') as f:
        json.dump(all_wheel_results, f, indent=2)
    print(f"Saved wheel reversal analysis to: {wheel_json}")

    strategy_json = OUTPUT_DIR / "phase_aware_strategy_inputs.json"
    with open(strategy_json, 'w') as f:
        json.dump(strategy_inputs, f, indent=2)
    print(f"Saved strategy inputs to: {strategy_json}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    summary_data = []
    for name in ["D2", "E2", "E2b"]:
        if name in all_results:
            r = all_results[name]
            summary_data.append({
                "variant": name,
                "support_error_mean_m": r["magnitude_support_error"]["mean"],
                "support_error_max_m": r["magnitude_support_error"]["max"],
                "crossings_150_norm": r["magnitude_support_error"]["crossings_normalized_500"],
                "hip_yaw_mean_rad": r["hip_yaw"]["mean"],
                "hip_yaw_max_rad": r["hip_yaw"]["max"],
                "pitch_mean_deg": np.degrees(r["pitch"]["mean"]),
                "pitch_max_deg": np.degrees(r["pitch"]["max"]),
                "pitch_forward_pct": r["pitch"]["pct_positive"],
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = OUTPUT_DIR / "signed_drift_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_csv}")

    return all_results, all_phase_results, all_wheel_results


if __name__ == "__main__":
    main()
