"""Analyze T6B 5000-step validation results.

Compares T6B vs T5 at full 5000 steps, focusing on windows 2-7 (steps 500-3500).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

T6B_5000 = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_best_high_0p480_5000/telemetry_1781244201.csv"
T5_5000 = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"

DRIFT_COLUMN = "active_pitch_crossing_signed_error_m"

def compute_metrics(df, column=DRIFT_COLUMN):
    """Compute full-run drift metrics."""
    e = df[column].values
    abs_e = np.abs(e)

    return {
        "survived_steps": len(df),
        "outside_0p08_pct": 100.0 * np.sum(abs_e > 0.08) / len(df),
        "outside_0p10_pct": 100.0 * np.sum(abs_e > 0.10) / len(df),
        "outside_0p15_pct": 100.0 * np.sum(abs_e > 0.15) / len(df),
        "max_abs_e_m": float(np.max(abs_e)),
        "mean_abs_e_m": float(np.mean(abs_e)),
        "final_e_m": float(e[-1]),
    }

def compute_window_metrics(df, column=DRIFT_COLUMN):
    """Compute 500-step window metrics."""
    windows = []
    for i in range(10):
        start = i * 500
        end = (i + 1) * 500
        if end > len(df):
            break
        window_df = df.iloc[start:end]
        e_window = window_df[column].values
        abs_e_window = np.abs(e_window)

        windows.append({
            "window_id": i,
            "start_step": start,
            "end_step": end,
            "outside_0p08_pct": 100.0 * np.sum(abs_e_window > 0.08) / len(e_window),
            "outside_0p10_pct": 100.0 * np.sum(abs_e_window > 0.10) / len(e_window),
            "outside_0p15_pct": 100.0 * np.sum(abs_e_window > 0.15) / len(e_window),
            "max_abs_e_m": float(np.max(abs_e_window)),
            "mean_abs_e_m": float(np.mean(abs_e_window)),
        })
    return windows

def compute_drift_accumulation(df, column=DRIFT_COLUMN):
    """Compute drift accumulation ratio."""
    e = df[column].values
    abs_e = np.abs(e)

    first_1000_mean = np.mean(abs_e[:1000])
    last_1000_mean = np.mean(abs_e[-1000:])

    ratio = last_1000_mean / first_1000_mean if first_1000_mean > 0 else 1.0

    return {
        "first_1000_mean": first_1000_mean,
        "last_1000_mean": last_1000_mean,
        "ratio": ratio,
    }

def check_gates(metrics, windows, drift_acc):
    """Check if T6B passes Step E gates."""
    gates = {
        "survived_4900": metrics["survived_steps"] >= 4900,
        "outside_0p08_lte_30": metrics["outside_0p08_pct"] <= 30.0,
        "outside_0p10_lte_10": metrics["outside_0p10_pct"] <= 10.0,
        "outside_0p15_lte_5": metrics["outside_0p15_pct"] <= 5.0,
        "max_e_lte_0p20": metrics["max_abs_e_m"] <= 0.20,
        "drift_acc_lt_1p5": drift_acc["ratio"] < 1.5,
    }

    all_pass = all(gates.values())

    return gates, all_pass

def classify_result(gates, gates_pass, metrics):
    """Classify T6B result."""
    if not gates_pass:
        if not gates["outside_0p08_lte_30"]:
            return "T6_BEST_HIGH_0P480_5000_FAIL_BAND_TARGET"
        elif not gates["drift_acc_lt_1p5"]:
            return "T6_BEST_HIGH_0P480_5000_FAIL_DRIFT_ACCUMULATION"
        else:
            return "T6_BEST_HIGH_0P480_5000_FAIL_STABILITY"
    else:
        # Passed all gates
        if metrics["outside_0p08_pct"] <= 25.0:
            return "T6_BEST_HIGH_0P480_5000_PASS_PROCEED_LOW_SANITY"
        else:
            return "T6_BEST_HIGH_0P480_5000_PASS_WITH_MONITORING"

def main():
    print("="*80)
    print("T6B 5000-Step Validation Analysis")
    print("="*80)

    # Load T6B
    print("\nLoading T6B 5000-step run...")
    t6b_df = pd.read_csv(T6B_5000)
    t6b_metrics = compute_metrics(t6b_df)
    t6b_windows = compute_window_metrics(t6b_df)
    t6b_drift_acc = compute_drift_accumulation(t6b_df)

    print(f"T6B: {t6b_metrics['outside_0p08_pct']:.1f}% outside ±0.08 m")
    print(f"T6B: {t6b_metrics['survived_steps']} steps survived")

    # Load T5
    print("\nLoading T5 5000-step run...")
    t5_df = pd.read_csv(T5_5000)
    t5_metrics = compute_metrics(t5_df)
    t5_windows = compute_window_metrics(t5_df)
    t5_drift_acc = compute_drift_accumulation(t5_df)

    print(f"T5: {t5_metrics['outside_0p08_pct']:.1f}% outside ±0.08 m")
    print(f"T5: {t5_metrics['survived_steps']} steps survived")

    # Check gates
    print("\n" + "="*80)
    print("Checking T6B Step E gates...")
    print("="*80)

    gates, gates_pass = check_gates(t6b_metrics, t6b_windows, t6b_drift_acc)

    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate_name}: {status}")

    classification = classify_result(gates, gates_pass, t6b_metrics)

    print(f"\nClassification: {classification}")

    # Compare windows 2-7
    print("\n" + "="*80)
    print("Windows 2-7 Comparison (steps 500-3500)")
    print("="*80)

    print("\n| Window | T5 ±0.08% | T6B ±0.08% | Improvement |")
    print("|--------|-----------|------------|-------------|")

    for i in range(2, 8):
        if i < len(t5_windows) and i < len(t6b_windows):
            t5_w = t5_windows[i]["outside_0p08_pct"]
            t6b_w = t6b_windows[i]["outside_0p08_pct"]
            improvement = t5_w - t6b_w
            print(f"| {i} | {t5_w:.1f}% | {t6b_w:.1f}% | {improvement:+.1f}% |")

    # Write summary
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "t6b_metrics": t6b_metrics,
        "t6b_windows": t6b_windows,
        "t6b_drift_accumulation": t6b_drift_acc,
        "t5_metrics": t5_metrics,
        "t5_windows": t5_windows,
        "t5_drift_accumulation": t5_drift_acc,
        "gates": {k: bool(v) for k, v in gates.items()},
        "gates_pass": bool(gates_pass),
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_best_high_0p480_5000.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_path}")

    # Write window metrics CSV
    window_rows = []
    for i in range(min(len(t5_windows), len(t6b_windows))):
        window_rows.append({
            "window_id": i,
            "start_step": i * 500,
            "end_step": (i + 1) * 500,
            "t5_outside_0p08_pct": t5_windows[i]["outside_0p08_pct"],
            "t6b_outside_0p08_pct": t6b_windows[i]["outside_0p08_pct"],
            "improvement_pct": t5_windows[i]["outside_0p08_pct"] - t6b_windows[i]["outside_0p08_pct"],
        })

    csv_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_best_high_0p480_5000_window_metrics.csv")
    pd.DataFrame(window_rows).to_csv(csv_path, index=False)
    print(f"Window metrics written to: {csv_path}")

    print("\n" + "="*80)
    print("Phase 7 complete!")
    print("="*80)

if __name__ == "__main__":
    main()
