#!/usr/bin/env python3
"""Compare D2 vs F1b 500-step telemetry for signed support drift analysis."""

import csv
import json
import argparse
from pathlib import Path


def load_telemetry(csv_path):
    """Load telemetry from CSV file."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def compute_metrics(rows, name):
    """Compute metrics from telemetry rows."""
    if not rows:
        return {}

    metrics = {"name": name, "n_rows": len(rows)}

    # Extract key columns
    # Use hip_yaw_comp_support_error_m as primary signed support metric
    # or fall back to support_position_error_m

    if "hip_yaw_comp_support_error_m" in rows[0]:
        signed_support = [float(r["hip_yaw_comp_support_error_m"]) for r in rows]
        metrics["signed_support_source"] = "hip_yaw_comp_support_error_m"
    elif "support_position_error_m" in rows[0]:
        signed_support = [float(r["support_position_error_m"]) for r in rows]
        metrics["signed_support_source"] = "support_position_error_m"
    else:
        signed_support = []
        metrics["signed_support_source"] = "NOT FOUND"

    if signed_support:
        metrics["signed_support_mean"] = sum(signed_support) / len(signed_support)
        metrics["signed_support_min"] = min(signed_support)
        metrics["signed_support_max"] = max(signed_support)
        metrics["signed_support_final"] = signed_support[-1]
        metrics["signed_support_positive_count"] = sum(1 for x in signed_support if x > 0)
        metrics["signed_support_positive_pct"] = 100 * metrics["signed_support_positive_count"] / len(signed_support)
        metrics["signed_support_negative_count"] = sum(1 for x in signed_support if x < 0)
        metrics["signed_support_negative_pct"] = 100 * metrics["signed_support_negative_count"] / len(signed_support)

        # Zero crossings
        crossings = 0
        for i in range(1, len(signed_support)):
            if (signed_support[i-1] >= 0) != (signed_support[i] >= 0):
                crossings += 1
        metrics["signed_support_zero_crossings"] = crossings

        # Time outside ±0.15
        metrics["signed_support_outside_0.15_positive"] = sum(1 for x in signed_support if x > 0.15)
        metrics["signed_support_outside_0.15_negative"] = sum(1 for x in signed_support if x < -0.15)
        metrics["signed_support_outside_0.15_total"] = metrics["signed_support_outside_0.15_positive"] + metrics["signed_support_outside_0.15_negative"]

        # RMS
        metrics["signed_support_rms"] = (sum(x*x for x in signed_support) / len(signed_support)) ** 0.5

        # Mean absolute error
        metrics["signed_support_mae"] = sum(abs(x) for x in signed_support) / len(signed_support)

        # Longest same-sign interval
        longest_positive = 0
        longest_negative = 0
        current_positive = 0
        current_negative = 0
        for x in signed_support:
            if x > 0:
                current_positive += 1
                current_negative = 0
                longest_positive = max(longest_positive, current_positive)
            elif x < 0:
                current_negative += 1
                current_positive = 0
                longest_negative = max(longest_negative, current_negative)
            else:
                current_positive = 0
                current_negative = 0
        metrics["signed_support_longest_positive_interval"] = longest_positive
        metrics["signed_support_longest_negative_interval"] = longest_negative
        metrics["signed_support_longest_same_sign_interval"] = max(longest_positive, longest_negative)

    # Support position error (magnitude-like)
    if "support_position_error_m" in rows[0]:
        support_mag = [abs(float(r["support_position_error_m"])) for r in rows]
        metrics["support_position_error_abs_max"] = max(support_mag)
        metrics["support_position_error_abs_mean"] = sum(support_mag) / len(support_mag)
        metrics["support_position_error_abs_final"] = support_mag[-1]
        metrics["support_position_error_abs_crossings_0.15"] = sum(1 for x in support_mag if x > 0.15)
    else:
        metrics["support_position_error_source"] = "NOT FOUND"

    # Phase recenter telemetry (F1b specific)
    if "phase_recenter_active" in rows[0]:
        recenter_active = [r["phase_recenter_active"].lower() == "true" for r in rows]
        metrics["phase_recenter_active_count"] = sum(1 for x in recenter_active if x)
        metrics["phase_recenter_active_pct"] = 100 * metrics["phase_recenter_active_count"] / len(recenter_active)

        if "phase_recenter_tau" in rows[0]:
            recenter_tau = [float(r["phase_recenter_tau"]) for r in rows]
            metrics["phase_recenter_tau_max"] = max(abs(x) for x in recenter_tau)
            metrics["phase_recenter_tau_mean"] = sum(abs(x) for x in recenter_tau) / len(recenter_tau)
            metrics["phase_recenter_tau_final"] = recenter_tau[-1]

        if "phase_recenter_signed_error_m" in rows[0]:
            recenter_signed_error = [float(r["phase_recenter_signed_error_m"]) for r in rows]
            metrics["phase_recenter_signed_error_mean"] = sum(recenter_signed_error) / len(recenter_signed_error)
    else:
        metrics["phase_recenter_source"] = "NOT FOUND"

    # Stability metrics
    if "pitch_x" in rows[0]:
        pitch_x = [float(r["pitch_x"]) * 57.3 for r in rows]  # Convert to degrees
        metrics["pitch_x_max_deg"] = max(abs(x) for x in pitch_x)
        metrics["pitch_x_rms_deg"] = (sum(x*x for x in pitch_x) / len(pitch_x)) ** 0.5
        metrics["pitch_x_final_deg"] = pitch_x[-1]

    if "roll_y" in rows[0]:
        roll_y = [float(r["roll_y"]) * 57.3 for r in rows]
        metrics["roll_y_max_deg"] = max(abs(x) for x in roll_y)
        metrics["roll_y_rms_deg"] = (sum(x*x for x in roll_y) / len(roll_y)) ** 0.5
        metrics["roll_y_final_deg"] = roll_y[-1]

    # Hip yaw metrics
    if "hip_yaw_abs_max" in rows[0]:
        hip_yaw = [float(r["hip_yaw_abs_max"]) for r in rows]
        metrics["hip_yaw_abs_max"] = max(hip_yaw)
        metrics["hip_yaw_abs_final"] = hip_yaw[-1]
    elif "hip_yaw_abs_max_rad" in rows[0]:
        hip_yaw = [float(r["hip_yaw_abs_max_rad"]) for r in rows]
        metrics["hip_yaw_abs_max"] = max(hip_yaw)
        metrics["hip_yaw_abs_final"] = hip_yaw[-1]

    # Wheel velocity
    if "wheel_vel_mean_rad_s" in rows[0]:
        wheel_vel = [abs(float(r["wheel_vel_mean_rad_s"])) for r in rows]
        metrics["wheel_vel_abs_max"] = max(wheel_vel)
        metrics["wheel_vel_abs_mean"] = sum(wheel_vel) / len(wheel_vel)

    # Contact state
    if "contact_supervisor_state" in rows[0]:
        contact_states = [r["contact_supervisor_state"] for r in rows]
        state_counts = {}
        for state in contact_states:
            state_counts[state] = state_counts.get(state, 0) + 1
        metrics["contact_state_counts"] = state_counts
        metrics["contact_state_most_common"] = max(state_counts.items(), key=lambda x: x[1])[0]

    # WBC gate
    if "wbc_gate_passed" in rows[0]:
        wbc_passed = [r["wbc_gate_passed"].lower() == "true" for r in rows]
        metrics["wbc_gate_passed_pct"] = 100 * sum(wbc_passed) / len(wbc_passed)

    # Hidden torque and ownership violations
    if "hidden_torque_norm" in rows[0]:
        hidden_torque = [float(r["hidden_torque_norm"]) for r in rows]
        metrics["hidden_torque_norm_max"] = max(hidden_torque)

    if "ownership_violation_count" in rows[0]:
        ownership_violations = [int(r["ownership_violation_count"]) for r in rows]
        metrics["ownership_violation_count_max"] = max(ownership_violations)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare D2 vs F1b telemetry")
    parser.add_argument("--d2", required=True, help="D2 telemetry CSV path")
    parser.add_argument("--f1b", required=True, help="F1b telemetry CSV path")
    parser.add_argument("--output-json", help="Output JSON path")
    parser.add_argument("--output-csv", help="Output CSV comparison path")
    args = parser.parse_args()

    print("Loading D2 telemetry...")
    d2_rows = load_telemetry(args.d2)
    print(f"  Loaded {len(d2_rows)} rows")

    print("Loading F1b telemetry...")
    f1b_rows = load_telemetry(args.f1b)
    print(f"  Loaded {len(f1b_rows)} rows")

    print("\nComputing D2 metrics...")
    d2_metrics = compute_metrics(d2_rows, "D2")

    print("Computing F1b metrics...")
    f1b_metrics = compute_metrics(f1b_rows, "F1b")

    # Print comparison
    print("\n" + "=" * 80)
    print("D2 vs F1b Comparison - Signed Support Drift Analysis")
    print("=" * 80)

    print("\n--- Survival & Structure ---")
    print(f"D2 rows: {d2_metrics.get('n_rows', 0)}")
    print(f"F1b rows: {f1b_metrics.get('n_rows', 0)}")
    print(f"D2 contact state: {d2_metrics.get('contact_state_most_common', 'N/A')}")
    print(f"F1b contact state: {f1b_metrics.get('contact_state_most_common', 'N/A')}")

    print("\n--- Signed Support Drift (Primary Metric) ---")
    print(f"Source: {d2_metrics.get('signed_support_source', 'N/A')}")
    print(f"")
    print(f"                         D2              F1b")
    print(f"Mean:            {d2_metrics.get('signed_support_mean', 0):>10.6f}    {f1b_metrics.get('signed_support_mean', 0):>10.6f}")
    print(f"Min:             {d2_metrics.get('signed_support_min', 0):>10.6f}    {f1b_metrics.get('signed_support_min', 0):>10.6f}")
    print(f"Max:             {d2_metrics.get('signed_support_max', 0):>10.6f}    {f1b_metrics.get('signed_support_max', 0):>10.6f}")
    print(f"Final:           {d2_metrics.get('signed_support_final', 0):>10.6f}    {f1b_metrics.get('signed_support_final', 0):>10.6f}")
    print(f"RMS:             {d2_metrics.get('signed_support_rms', 0):>10.6f}    {f1b_metrics.get('signed_support_rms', 0):>10.6f}")
    print(f"MAE:             {d2_metrics.get('signed_support_mae', 0):>10.6f}    {f1b_metrics.get('signed_support_mae', 0):>10.6f}")
    print(f"Positive %:       {d2_metrics.get('signed_support_positive_pct', 0):>10.1f}    {f1b_metrics.get('signed_support_positive_pct', 0):>10.1f}")
    print(f"Negative %:      {d2_metrics.get('signed_support_negative_pct', 0):>10.1f}    {f1b_metrics.get('signed_support_negative_pct', 0):>10.1f}")
    print(f"Zero crossings:   {d2_metrics.get('signed_support_zero_crossings', 0):>10d}    {f1b_metrics.get('signed_support_zero_crossings', 0):>10d}")
    print(f"Outside +0.15:   {d2_metrics.get('signed_support_outside_0.15_positive', 0):>10d}    {f1b_metrics.get('signed_support_outside_0.15_positive', 0):>10d}")
    print(f"Outside -0.15:   {d2_metrics.get('signed_support_outside_0.15_negative', 0):>10d}    {f1b_metrics.get('signed_support_outside_0.15_negative', 0):>10d}")
    print(f"Outside total:   {d2_metrics.get('signed_support_outside_0.15_total', 0):>10d}    {f1b_metrics.get('signed_support_outside_0.15_total', 0):>10d}")
    print(f"Longest pos int: {d2_metrics.get('signed_support_longest_positive_interval', 0):>10d}    {f1b_metrics.get('signed_support_longest_positive_interval', 0):>10d}")
    print(f"Longest neg int: {d2_metrics.get('signed_support_longest_negative_interval', 0):>10d}    {f1b_metrics.get('signed_support_longest_negative_interval', 0):>10d}")

    print("\n--- Phase Recenter (F1b Specific) ---")
    print(f"Source: {f1b_metrics.get('phase_recenter_source', 'N/A')}")
    print(f"Recenter active %: {f1b_metrics.get('phase_recenter_active_pct', 0):.1f}%")
    if f1b_metrics.get('phase_recenter_tau_max') is not None:
        print(f"Recenter tau max: {f1b_metrics.get('phase_recenter_tau_max', 0):.4f}")
        print(f"Recenter tau mean: {f1b_metrics.get('phase_recenter_tau_mean', 0):.4f}")
        print(f"Recenter tau final: {f1b_metrics.get('phase_recenter_tau_final', 0):.4f}")
        print(f"Recenter signed error mean: {f1b_metrics.get('phase_recenter_signed_error_mean', 0):.6f}")

    print("\n--- Support Position Error (Magnitude) ---")
    print(f"                         D2              F1b")
    print(f"Abs max:         {d2_metrics.get('support_position_error_abs_max', 0):>10.6f}    {f1b_metrics.get('support_position_error_abs_max', 0):>10.6f}")
    print(f"Abs mean:       {d2_metrics.get('support_position_error_abs_mean', 0):>10.6f}    {f1b_metrics.get('support_position_error_abs_mean', 0):>10.6f}")
    print(f"Crossings >0.15: {d2_metrics.get('support_position_error_abs_crossings_0.15', 0):>10d}    {f1b_metrics.get('support_position_error_abs_crossings_0.15', 0):>10d}")

    print("\n--- Stability ---")
    print(f"                         D2              F1b")
    print(f"Pitch max deg:   {d2_metrics.get('pitch_x_max_deg', 0):>10.2f}    {f1b_metrics.get('pitch_x_max_deg', 0):>10.2f}")
    print(f"Pitch RMS deg:   {d2_metrics.get('pitch_x_rms_deg', 0):>10.2f}    {f1b_metrics.get('pitch_x_rms_deg', 0):>10.2f}")
    print(f"Pitch final deg: {d2_metrics.get('pitch_x_final_deg', 0):>10.2f}    {f1b_metrics.get('pitch_x_final_deg', 0):>10.2f}")
    print(f"Roll max deg:    {d2_metrics.get('roll_y_max_deg', 0):>10.2f}    {f1b_metrics.get('roll_y_max_deg', 0):>10.2f}")
    print(f"Roll RMS deg:    {d2_metrics.get('roll_y_rms_deg', 0):>10.2f}    {f1b_metrics.get('roll_y_rms_deg', 0):>10.2f}")

    print("\n--- Hip Yaw ---")
    print(f"                         D2              F1b")
    print(f"Abs max:         {d2_metrics.get('hip_yaw_abs_max', 0):>10.4f}    {f1b_metrics.get('hip_yaw_abs_max', 0):>10.4f}")
    print(f"Abs final:       {d2_metrics.get('hip_yaw_abs_final', 0):>10.4f}    {f1b_metrics.get('hip_yaw_abs_final', 0):>10.4f}")

    print("\n--- Wheel Velocity (Monitor) ---")
    print(f"                         D2              F1b")
    print(f"Abs max:         {d2_metrics.get('wheel_vel_abs_max', 0):>10.4f}    {f1b_metrics.get('wheel_vel_abs_max', 0):>10.4f}")
    print(f"Abs mean:        {d2_metrics.get('wheel_vel_abs_mean', 0):>10.4f}    {f1b_metrics.get('wheel_vel_abs_mean', 0):>10.4f}")

    print("\n--- Controller Integrity ---")
    print(f"                         D2              F1b")
    print(f"Hidden torque:   {d2_metrics.get('hidden_torque_norm_max', 0):>10.4f}    {f1b_metrics.get('hidden_torque_norm_max', 0):>10.4f}")
    print(f"Ownership viol: {d2_metrics.get('ownership_violation_count_max', 0):>10d}    {f1b_metrics.get('ownership_violation_count_max', 0):>10d}")

    # Write JSON output
    if args.output_json:
        output = {
            "d2": d2_metrics,
            "f1b": f1b_metrics,
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON output written to: {args.output_json}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
