#!/usr/bin/env python3
"""Hip-yaw root cause audit for Step E controller behavior.

Diagnostic-first systematic analysis of hip-yaw posture failure across
low_0p300, nominal, and high_0p480 height variants.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def load_telemetry(telemetry_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load telemetry CSV files for all three height variants."""
    cases = {
        "low_0p300": telemetry_dir / "low_0p300_5000_telemetry.csv",
        "nominal": telemetry_dir / "nominal_5000_telemetry.csv",
        "high_0p480": telemetry_dir / "high_0p480_5000_telemetry.csv",
    }

    data = {}
    for case_name, path in cases.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing telemetry: {path}")
        df = pd.read_csv(path)
        data[case_name] = df
        print(f"[LOADED] {case_name}: {len(df)} steps, {len(df.columns)} columns")

    return data


def find_first_threshold_crossing(series: pd.Series, threshold: float) -> int:
    """Find first timestep where series exceeds threshold (absolute value)."""
    mask = series.abs() > threshold
    crossings = np.where(mask)[0]
    return int(crossings[0]) if len(crossings) > 0 else -1


def audit_event_order(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute event-order audit for all heights.

    For each height, determine when various failure indicators first appear.
    Classify event order without assuming causality.
    """
    events = []

    for case_name, df in data.items():
        event = {"case": case_name}

        # Hip-yaw thresholds
        event["hip_yaw_003"] = find_first_threshold_crossing(df["hip_yaw_abs_max"], 0.03)
        event["hip_yaw_007"] = find_first_threshold_crossing(df["hip_yaw_abs_max"], 0.07)
        event["hip_yaw_010"] = find_first_threshold_crossing(df["hip_yaw_abs_max"], 0.10)

        # Support position error thresholds
        event["support_005"] = find_first_threshold_crossing(df["support_position_error_m"], 0.05)
        event["support_010"] = find_first_threshold_crossing(df["support_position_error_m"], 0.10)
        event["support_015"] = find_first_threshold_crossing(df["support_position_error_m"], 0.15)

        # Height error
        event["height_002"] = find_first_threshold_crossing(df["height_error_m"], 0.02)

        # Roll
        event["roll_005"] = find_first_threshold_crossing(df["roll_y"], 0.05)

        # Pitch diagnostic levels
        event["pitch_010"] = find_first_threshold_crossing(df["pitch_x"], 0.10)
        event["pitch_015"] = find_first_threshold_crossing(df["pitch_x"], 0.15)
        event["pitch_020"] = find_first_threshold_crossing(df["pitch_x"], 0.20)

        # Contact validity
        if "contact_valid_pct" in df.columns:
            # Contact valid is a percentage, failure is < 100
            invalid_mask = df["contact_valid_pct"] < 99.0
            invalid_steps = np.where(invalid_mask)[0]
            event["contact_invalid"] = int(invalid_steps[0]) if len(invalid_steps) > 0 else -1
        else:
            event["contact_invalid"] = -1

        # Non-wheel contact
        if "non_wheel_floor_contacts" in df.columns:
            non_wheel_mask = df["non_wheel_floor_contacts"] > 0
            non_wheel_steps = np.where(non_wheel_mask)[0]
            event["non_wheel_contact"] = int(non_wheel_steps[0]) if len(non_wheel_steps) > 0 else -1
        else:
            event["non_wheel_contact"] = -1

        events.append(event)

    df_events = pd.DataFrame(events)

    # Classify event order
    for idx, row in df_events.iterrows():
        case = row["case"]

        # Get first significant hip-yaw and support events
        hip_yaw_first = min([v for v in [row["hip_yaw_003"], row["hip_yaw_007"]] if v >= 0], default=-1)
        support_first = min([v for v in [row["support_005"], row["support_010"]] if v >= 0], default=-1)

        # Classify
        if hip_yaw_first < 0 and support_first < 0:
            classification = "no_significant_events"
        elif hip_yaw_first >= 0 and support_first < 0:
            classification = "hip_yaw_only"
        elif support_first >= 0 and hip_yaw_first < 0:
            classification = "support_only"
        elif abs(hip_yaw_first - support_first) <= 50:  # Within 0.5s
            classification = "simultaneous"
        elif hip_yaw_first < support_first:
            classification = "hip_yaw_first"
        else:
            classification = "support_first"

        df_events.loc[idx, "classification"] = classification

    return df_events


def audit_hip_yaw_reference_tracking(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit hip-yaw reference and tracking behavior for all heights."""
    results = []

    for case_name, df in data.items():
        result = {"case": case_name}

        # Check if required columns exist
        required_cols = ["l_hip_yaw_pos", "r_hip_yaw_pos", "hip_yaw_abs_max"]
        if not all(col in df.columns for col in required_cols):
            result["status"] = "missing_columns"
            results.append(result)
            continue

        # Basic position metrics
        result["l_hip_yaw_pos_max"] = df["l_hip_yaw_pos"].abs().max()
        result["r_hip_yaw_pos_max"] = df["r_hip_yaw_pos"].abs().max()
        result["l_hip_yaw_pos_final"] = df["l_hip_yaw_pos"].iloc[-1]
        result["r_hip_yaw_pos_final"] = df["r_hip_yaw_pos"].iloc[-1]
        result["hip_yaw_abs_max"] = df["hip_yaw_abs_max"].max()
        result["hip_yaw_abs_final"] = df["hip_yaw_abs_max"].iloc[-1]

        # Reference and error metrics if available
        if "l_hip_yaw_ref" in df.columns and "r_hip_yaw_ref" in df.columns:
            result["l_hip_yaw_ref_initial"] = df["l_hip_yaw_ref"].iloc[0]
            result["r_hip_yaw_ref_initial"] = df["r_hip_yaw_ref"].iloc[0]
            result["l_hip_yaw_ref_final"] = df["l_hip_yaw_ref"].iloc[-1]
            result["r_hip_yaw_ref_final"] = df["r_hip_yaw_ref"].iloc[-1]
            result["l_hip_yaw_ref_changed"] = abs(result["l_hip_yaw_ref_final"] - result["l_hip_yaw_ref_initial"]) > 0.01
            result["r_hip_yaw_ref_changed"] = abs(result["r_hip_yaw_ref_final"] - result["r_hip_yaw_ref_initial"]) > 0.01

        if "l_hip_yaw_error" in df.columns and "r_hip_yaw_error" in df.columns:
            result["l_hip_yaw_error_max"] = df["l_hip_yaw_error"].abs().max()
            result["r_hip_yaw_error_max"] = df["r_hip_yaw_error"].abs().max()
            result["l_hip_yaw_error_final"] = df["l_hip_yaw_error"].iloc[-1]
            result["r_hip_yaw_error_final"] = df["r_hip_yaw_error"].iloc[-1]
            result["l_hip_yaw_error_rms"] = np.sqrt((df["l_hip_yaw_error"] ** 2).mean())
            result["r_hip_yaw_error_rms"] = np.sqrt((df["r_hip_yaw_error"] ** 2).mean())

            # Divergence and common mode analysis
            hip_yaw_divergence = (df["l_hip_yaw_error"] - df["r_hip_yaw_error"]).abs()
            hip_yaw_common_mode = (df["l_hip_yaw_error"] + df["r_hip_yaw_error"]).abs()

            result["hip_yaw_divergence_max"] = hip_yaw_divergence.max()
            result["hip_yaw_divergence_rms"] = np.sqrt((hip_yaw_divergence ** 2).mean())
            result["hip_yaw_common_mode_max"] = hip_yaw_common_mode.max()
            result["hip_yaw_common_mode_rms"] = np.sqrt((hip_yaw_common_mode ** 2).mean())

            # Classify error pattern
            if result["hip_yaw_divergence_rms"] > result["hip_yaw_common_mode_rms"]:
                result["error_pattern"] = "divergence_dominant"
            else:
                result["error_pattern"] = "common_mode_dominant"

            # Check for monotonic drift
            left_slope = np.polyfit(np.arange(len(df)), df["l_hip_yaw_error"], 1)[0]
            right_slope = np.polyfit(np.arange(len(df)), df["r_hip_yaw_error"], 1)[0]
            result["l_hip_yaw_error_slope"] = left_slope
            result["r_hip_yaw_error_slope"] = right_slope
            result["error_drift_monotonic"] = (abs(left_slope) > 1e-5) or (abs(right_slope) > 1e-5)
        else:
            result["error_pattern"] = "no_error_columns"

        result["status"] = "complete"
        results.append(result)

    return pd.DataFrame(results)


def audit_hip_yaw_torque_authority(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit hip-yaw torque authority and ownership for all heights."""
    results = []

    for case_name, df in data.items():
        result = {"case": case_name}

        # Check torque columns
        torque_cols = ["l_hip_yaw_tau_shape_raw", "r_hip_yaw_tau_shape_raw",
                      "l_hip_yaw_tau_shape_final", "r_hip_yaw_tau_shape_final"]

        if all(col in df.columns for col in torque_cols):
            result["l_hip_yaw_tau_raw_max"] = df["l_hip_yaw_tau_shape_raw"].abs().max()
            result["r_hip_yaw_tau_raw_max"] = df["r_hip_yaw_tau_shape_raw"].abs().max()
            result["l_hip_yaw_tau_final_max"] = df["l_hip_yaw_tau_shape_final"].abs().max()
            result["r_hip_yaw_tau_final_max"] = df["r_hip_yaw_tau_shape_final"].abs().max()
            result["l_hip_yaw_tau_raw_rms"] = np.sqrt((df["l_hip_yaw_tau_shape_raw"] ** 2).mean())
            result["r_hip_yaw_tau_raw_rms"] = np.sqrt((df["r_hip_yaw_tau_shape_raw"] ** 2).mean())

            # Check if torque is applied (raw vs final)
            result["torque_applied_left"] = result["l_hip_yaw_tau_final_max"] > 0.01
            result["torque_applied_right"] = result["r_hip_yaw_tau_final_max"] > 0.01

            # Check torque sign correctness if error available
            if "l_hip_yaw_error" in df.columns:
                # Torque should oppose error (negative error needs positive torque)
                l_sign_correct = (df["l_hip_yaw_tau_shape_final"] * df["l_hip_yaw_error"]) < 0
                r_sign_correct = (df["r_hip_yaw_tau_shape_final"] * df["r_hip_yaw_error"]) < 0
                result["l_hip_yaw_sign_correct_pct"] = (l_sign_correct.sum() / len(df)) * 100
                result["r_hip_yaw_sign_correct_pct"] = (r_sign_correct.sum() / len(df)) * 100
        else:
            result["status"] = "missing_torque_columns"

        # Check saturation flags if available
        if "hip_yaw_torque_saturation_flag_left" in df.columns:
            result["hip_yaw_sat_left_count"] = df["hip_yaw_torque_saturation_flag_left"].sum()
            result["hip_yaw_sat_right_count"] = df["hip_yaw_torque_saturation_flag_right"].sum()

        # Check torque margin
        if "hip_yaw_torque_margin_left" in df.columns:
            result["hip_yaw_margin_left_min"] = df["hip_yaw_torque_margin_left"].min()
            result["hip_yaw_margin_right_min"] = df["hip_yaw_torque_margin_right"].min()

        # Check ownership violations
        if "ownership_violation_count" in df.columns:
            result["ownership_violations_total"] = df["ownership_violation_count"].sum()
            result["ownership_violations_max"] = df["ownership_violation_count"].max()

        # Check hidden torque
        if "hidden_torque_norm" in df.columns:
            result["hidden_torque_max"] = df["hidden_torque_norm"].max()
            result["hidden_torque_mean"] = df["hidden_torque_norm"].mean()

        results.append(result)

    return pd.DataFrame(results)


def compute_lag_correlation(signal_a: pd.Series, signal_b: pd.Series, max_lag: int = 100) -> Tuple[float, int]:
    """Compute best correlation and lag between two signals."""
    best_corr = 0.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = signal_a.iloc[:lag].corr(signal_b.iloc[-lag:])
        elif lag > 0:
            corr = signal_a.iloc[lag:].corr(signal_b.iloc[:-lag])
        else:
            corr = signal_a.corr(signal_b)

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_corr, best_lag


def audit_coupling_correlation(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit correlations between hip-yaw drift and other signals."""
    results = []

    for case_name, df in data.items():
        result = {"case": case_name}

        if "hip_yaw_abs_max" not in df.columns:
            result["status"] = "missing_hip_yaw"
            results.append(result)
            continue

        hip_yaw_signal = df["hip_yaw_abs_max"]

        # Correlate with support position error
        if "support_position_error_m" in df.columns:
            corr, lag = compute_lag_correlation(hip_yaw_signal, df["support_position_error_m"])
            result["support_corr"] = corr
            result["support_lag"] = lag

        # Correlate with wheel velocity
        if "wheel_vel_mean_rad_s" in df.columns:
            corr, lag = compute_lag_correlation(hip_yaw_signal, df["wheel_vel_mean_rad_s"])
            result["wheel_vel_corr"] = corr
            result["wheel_vel_lag"] = lag

        # Correlate with pitch
        if "pitch_x" in df.columns:
            corr, lag = compute_lag_correlation(hip_yaw_signal, df["pitch_x"])
            result["pitch_corr"] = corr
            result["pitch_lag"] = lag

        # Correlate with height error
        if "height_error_m" in df.columns:
            corr, lag = compute_lag_correlation(hip_yaw_signal, df["height_error_m"])
            result["height_corr"] = corr
            result["height_lag"] = lag

        results.append(result)

    return pd.DataFrame(results)


def audit_wbc_structural_invariant(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Determine if WBC is diagnostic-only or actually applied to final torque."""
    wbc_status = {}

    for case_name, df in data.items():
        status = {"case": case_name}

        # Check if WBC norm exists
        if "tau_wbc_norm" in df.columns:
            wbc_norm_max = df["tau_wbc_norm"].max()
            wbc_norm_mean = df["tau_wbc_norm"].mean()
            status["wbc_norm_max"] = wbc_norm_max
            status["wbc_norm_mean"] = wbc_norm_mean

            # Check if WBC appears in ownership
            if "active_torque_owner_per_joint" in df.columns:
                # This is a string column, check if it contains "wbc"
                wbc_ownership = df["active_torque_owner_per_joint"].str.contains("wbc", case=False, na=False)
                status["wbc_ownership_count"] = wbc_ownership.sum()

            # Check hidden torque
            if "hidden_torque_norm" in df.columns:
                status["hidden_torque_max"] = df["hidden_torque_norm"].max()

            # Classify
            if wbc_norm_max > 0.1:
                if status.get("wbc_ownership_count", 0) > 0:
                    status["classification"] = "WBC_ACTUALLY_APPLIED"
                else:
                    # WBC computed but not in ownership - likely diagnostic only
                    status["classification"] = "WBC_DIAGNOSTIC_ONLY"
            else:
                status["classification"] = "WBC_DISABLED_OR_ZERO"
        else:
            status["classification"] = "WBC_COLUMNS_MISSING"

        wbc_status[case_name] = status

    return wbc_status


def classify_hip_yaw_failure_mechanism(
    case_name: str,
    df: pd.DataFrame,
    event_row: pd.Series,
    ref_tracking: pd.Series,
    torque_audit: pd.Series,
    coupling: pd.Series,
) -> Dict[str, Any]:
    """Classify hip-yaw failure mechanism for a single height based on evidence."""
    classification = {
        "case": case_name,
        "mechanism": "unclear_requires_analysis",
        "evidence": [],
        "ruled_out": [],
    }

    # Check if hip-yaw failure even occurred
    if df["hip_yaw_abs_max"].max() < 0.05:
        classification["mechanism"] = "no_hip_yaw_failure"
        classification["evidence"].append("hip_yaw_abs_max stays below 0.05 rad")
        return classification

    # Check reference tracking
    if ref_tracking.get("status") == "complete":
        if ref_tracking.get("l_hip_yaw_ref_changed") or ref_tracking.get("r_hip_yaw_ref_changed"):
            classification["mechanism"] = "hip_yaw_reference_drift"
            classification["evidence"].append("reference changed during run")

        error_pattern = ref_tracking.get("error_pattern", "")
        if error_pattern == "divergence_dominant":
            classification["mechanism"] = "hip_yaw_divergence_mode_uncontrolled"
            classification["evidence"].append(f"divergence RMS ({ref_tracking.get('hip_yaw_divergence_rms', 0):.4f}) > common mode RMS")
        elif error_pattern == "common_mode_dominant":
            classification["mechanism"] = "hip_yaw_common_mode_uncontrolled"
            classification["evidence"].append(f"common mode RMS ({ref_tracking.get('hip_yaw_common_mode_rms', 0):.4f}) > divergence RMS")

    # Check torque authority
    if not torque_audit.get("torque_applied_left", False) or not torque_audit.get("torque_applied_right", False):
        classification["mechanism"] = "hip_yaw_torque_not_applied"
        classification["evidence"].append("final torque magnitude near zero")
        return classification

    # Check torque sign correctness
    if torque_audit.get("l_hip_yaw_sign_correct_pct", 100) < 80 or torque_audit.get("r_hip_yaw_sign_correct_pct", 100) < 80:
        classification["mechanism"] = "hip_yaw_torque_sign_error"
        classification["evidence"].append("torque does not consistently oppose error")

    # Check saturation
    if torque_audit.get("hip_yaw_sat_left_count", 0) > 100 or torque_audit.get("hip_yaw_sat_right_count", 0) > 100:
        classification["mechanism"] = "hip_yaw_torque_saturation"
        classification["evidence"].append("torque saturation flags present")

    # Check coupling
    if coupling.get("support_corr", 0) > 0.7:
        classification["mechanism"] = "hip_yaw_coupled_with_support_drift"
        classification["evidence"].append(f"high correlation with support error (r={coupling.get('support_corr', 0):.3f})")

    # Event order
    event_class = event_row.get("classification", "")
    if event_class:
        classification["evidence"].append(f"event order: {event_class}")

    return classification


def main():
    parser = argparse.ArgumentParser(description="Audit Step E hip-yaw root cause")
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path("outputs/step_e_best_current_profile_5000_eval"),
        help="Directory containing telemetry CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hip_yaw_root_cause_audit"),
        help="Output directory for audit artifacts",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Hip-Yaw Root Cause Audit - Step E Controller Behavior")
    print("=" * 80)
    print()

    # Phase 1: Load telemetry
    print("[PHASE 1] Loading telemetry data...")
    try:
        data = load_telemetry(args.telemetry_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    print()

    # Phase 2: Event-order audit
    print("[PHASE 2] Running event-order audit...")
    df_events = audit_event_order(data)
    print(df_events)
    df_events.to_csv(args.output_dir / "hip_yaw_event_order_comparison.csv", index=False)
    print(f"[SAVED] {args.output_dir / 'hip_yaw_event_order_comparison.csv'}")
    print()

    # Phase 3: Hip-yaw reference and tracking audit
    print("[PHASE 3] Running hip-yaw reference and tracking audit...")
    df_ref_tracking = audit_hip_yaw_reference_tracking(data)
    print(df_ref_tracking.T)
    df_ref_tracking.to_csv(args.output_dir / "hip_yaw_reference_command_audit.csv", index=False)
    print(f"[SAVED] {args.output_dir / 'hip_yaw_reference_command_audit.csv'}")
    print()

    # Phase 4: Hip-yaw torque and ownership audit
    print("[PHASE 4] Running hip-yaw torque and ownership audit...")
    df_torque = audit_hip_yaw_torque_authority(data)
    print(df_torque.T)
    df_torque.to_csv(args.output_dir / "hip_yaw_torque_authority_audit.csv", index=False)
    print(f"[SAVED] {args.output_dir / 'hip_yaw_torque_authority_audit.csv'}")
    print()

    # Phase 5: Coupling and correlation audit
    print("[PHASE 5] Running coupling and correlation audit...")
    df_coupling = audit_coupling_correlation(data)
    print(df_coupling.T)
    df_coupling.to_csv(args.output_dir / "hip_yaw_correlation_lag_audit.csv", index=False)
    print(f"[SAVED] {args.output_dir / 'hip_yaw_correlation_lag_audit.csv'}")
    print()

    # Phase 6: WBC structural invariant audit
    print("[PHASE 6] Running WBC structural invariant audit...")
    wbc_status = audit_wbc_structural_invariant(data)
    for case_name, status in wbc_status.items():
        print(f"  {case_name}: {status.get('classification', 'UNKNOWN')}")
    print()

    # Phase 7: Height-by-height classification
    print("[PHASE 7] Classifying hip-yaw failure mechanism per height...")
    classifications = []
    for case_name, df in data.items():
        event_row = df_events[df_events["case"] == case_name].iloc[0]
        ref_row = df_ref_tracking[df_ref_tracking["case"] == case_name].iloc[0]
        torque_row = df_torque[df_torque["case"] == case_name].iloc[0]
        coupling_row = df_coupling[df_coupling["case"] == case_name].iloc[0]

        classification = classify_hip_yaw_failure_mechanism(
            case_name, df, event_row, ref_row, torque_row, coupling_row
        )
        classifications.append(classification)
        print(f"\n  {case_name}:")
        print(f"    Mechanism: {classification['mechanism']}")
        print(f"    Evidence: {classification['evidence']}")

    print()

    # Extract peak windows for each height
    print("[PHASE 8] Extracting peak windows...")
    for case_name, df in data.items():
        peak_idx = df["hip_yaw_abs_max"].idxmax()
        window_start = max(0, peak_idx - 100)
        window_end = min(len(df), peak_idx + 100)
        window = df.iloc[window_start:window_end]

        window_path = args.output_dir / f"{case_name}_hip_yaw_peak_window.csv"
        window.to_csv(window_path, index=False)
        print(f"[SAVED] {window_path}")
    print()

    # Generate summary JSON
    print("[PHASE 9] Generating summary JSON...")
    summary = {
        "telemetry_source": str(args.telemetry_dir),
        "telemetry_files_used": [
            "low_0p300_5000_telemetry.csv",
            "nominal_5000_telemetry.csv",
            "high_0p480_5000_telemetry.csv",
        ],
        "simulations_rerun": False,
        "event_order": df_events.to_dict(orient="records"),
        "reference_tracking": df_ref_tracking.to_dict(orient="records"),
        "torque_authority": df_torque.to_dict(orient="records"),
        "coupling_correlation": df_coupling.to_dict(orient="records"),
        "wbc_structural_status": wbc_status,
        "classifications": classifications,
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    summary = convert_numpy_types(summary)

    summary_path = args.output_dir / "hip_yaw_root_cause_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {summary_path}")
    print()

    # Generate markdown report
    print("[PHASE 10] Generating markdown report...")
    generate_markdown_report(summary, args.output_dir, data)
    print()

    print("=" * 80)
    print("Hip-Yaw Root Cause Audit Complete")
    print("=" * 80)
    print(f"\nArtifacts saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the markdown report at:")
    print(f"   docs/validation/step_e_hip_yaw_root_cause_audit.md")
    print("2. Examine peak windows and failure windows for detailed time-series analysis")
    print("3. Based on classification, implement targeted fixes")

    return 0


def generate_markdown_report(summary: Dict[str, Any], output_dir: Path, data: Dict[str, pd.DataFrame]):
    """Generate comprehensive markdown report."""
    report_path = Path("docs/validation/step_e_hip_yaw_root_cause_audit.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step E Hip-Yaw Root Cause Audit\n\n")
        f.write("**Date:** 2026-06-05\n\n")
        f.write("**Objective:** Diagnostic-first systematic analysis of hip-yaw posture failure\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This audit investigates the root cause of hip-yaw posture failure across\n")
        f.write("three height variants (low_0p300, nominal, high_0p480) in the Step E controller.\n\n")

        f.write("### Priority Order\n\n")
        f.write("1. Robot must survive and keep contact/height\n")
        f.write("2. Robot must keep posture, especially hip-yaw (legs must not twist)\n")
        f.write("3. Support-position drift should be improved\n")
        f.write("4. Pitch handled later by task-aware pitch control (not blocking this audit)\n\n")

        f.write("## Telemetry Source\n\n")
        f.write(f"- **Directory:** `{summary['telemetry_source']}`\n")
        f.write(f"- **Simulations rerun:** {summary['simulations_rerun']}\n")
        f.write("- **Files used:**\n")
        for file in summary['telemetry_files_used']:
            f.write(f"  - {file}\n")
        f.write("\n")

        f.write("## Metrics Summary\n\n")
        f.write("| Case | Survived | Support Max | Hip-Yaw Max | Pitch Max | Roll Max | Height Error |\n")
        f.write("|------|----------|-------------|-------------|-----------|----------|---------------|\n")
        for case_name, df in data.items():
            survived = "✓" if len(df) >= 5000 else "✗"
            support_max = df["support_position_error_m"].abs().max()
            hip_yaw_max = df["hip_yaw_abs_max"].max()
            pitch_max = df["pitch_x"].abs().max()
            roll_max = df["roll_y"].abs().max()
            height_err = df["height_error_m"].abs().iloc[-1]
            f.write(f"| {case_name} | {survived} | {support_max:.4f} m | {hip_yaw_max:.4f} rad | {pitch_max:.4f} rad | {roll_max:.4f} rad | {height_err:.4f} m |\n")
        f.write("\n")

        f.write("## Event Order Analysis\n\n")
        f.write("| Case | Classification | Hip-Yaw 0.03 | Support 0.05 | Hip-Yaw 0.10 | Support 0.15 |\n")
        f.write("|------|----------------|--------------|--------------|--------------|---------------|\n")
        for event in summary['event_order']:
            f.write(f"| {event['case']} | {event.get('classification', 'N/A')} | ")
            f.write(f"{event.get('hip_yaw_003', -1)} | {event.get('support_005', -1)} | ")
            f.write(f"{event.get('hip_yaw_010', -1)} | {event.get('support_015', -1)} |\n")
        f.write("\n")

        f.write("## WBC Structural Invariant Status\n\n")
        for case_name, status in summary['wbc_structural_status'].items():
            f.write(f"### {case_name}\n\n")
            f.write(f"- **Classification:** `{status.get('classification', 'UNKNOWN')}`\n")
            if "wbc_norm_max" in status:
                f.write(f"- **WBC norm max:** {status['wbc_norm_max']:.4f} Nm\n")
            if "hidden_torque_max" in status:
                f.write(f"- **Hidden torque max:** {status['hidden_torque_max']:.4f} Nm\n")
            f.write("\n")

        f.write("## Hip-Yaw Failure Mechanism Classification\n\n")
        for classification in summary['classifications']:
            f.write(f"### {classification['case']}\n\n")
            f.write(f"**Mechanism:** `{classification['mechanism']}`\n\n")
            f.write("**Evidence:**\n\n")
            for evidence in classification['evidence']:
                f.write(f"- {evidence}\n")
            f.write("\n")
            if classification.get('ruled_out'):
                f.write("**Ruled out:**\n\n")
                for ruled_out in classification['ruled_out']:
                    f.write(f"- {ruled_out}\n")
                f.write("\n")

        f.write("## Pitch Policy Statement\n\n")
        f.write("Pitch is tracked and reported in this audit but is **not the primary objective**.\n\n")
        f.write("Pitch will later be converted to task-aware pitch control:\n")
        f.write("- Static pitch reference for standing\n")
        f.write("- Dynamic pitch reference for future forward/backward motion\n")
        f.write("- Absolute safety bound\n\n")
        f.write("For this audit:\n")
        f.write("- Pitch metrics are recorded\n")
        f.write("- Pitch must not cause fall/contact loss/height failure\n")
        f.write("- Hip-yaw posture failure is prioritized\n\n")

        f.write("## Final Decision\n\n")

        # Determine final decision based on classifications
        all_mechanisms = [c['mechanism'] for c in summary['classifications']]
        if all(m == "no_hip_yaw_failure" for m in all_mechanisms):
            decision = "NO_HIP_YAW_FAILURE_AT_CURRENT_HEIGHTS"
        elif any("unclear" in m or "requires" in m for m in all_mechanisms):
            decision = "HIP_YAW_ROOT_CAUSE_REQUIRES_MORE_TELEMETRY"
        elif any("WBC_ACTUALLY_APPLIED" in str(summary['wbc_structural_status'][c['case']].get('classification', '')) for c in summary['classifications']):
            decision = "STRUCTURAL_INVARIANT_BLOCKS_HIP_YAW_DIAGNOSIS"
        else:
            decision = "HIP_YAW_ROOT_CAUSE_IDENTIFIED"

        f.write(f"**Decision:** `{decision}`\n\n")

        f.write("## Restrictions Followed\n\n")
        f.write("- ✓ Did NOT add WBC\n")
        f.write("- ✓ Did NOT enable legacy WBC paths\n")
        f.write("- ✓ Did NOT modify hip-roll\n")
        f.write("- ✓ Did NOT proceed to Step C or Step D\n")
        f.write("- ✓ Did NOT commit\n\n")

        f.write("## Artifacts Generated\n\n")
        f.write("All artifacts saved to: `outputs/hip_yaw_root_cause_audit/`\n\n")
        f.write("- `hip_yaw_root_cause_summary.json`\n")
        f.write("- `hip_yaw_event_order_comparison.csv`\n")
        f.write("- `hip_yaw_reference_command_audit.csv`\n")
        f.write("- `hip_yaw_torque_authority_audit.csv`\n")
        f.write("- `hip_yaw_correlation_lag_audit.csv`\n")
        f.write("- `low_0p300_hip_yaw_peak_window.csv`\n")
        f.write("- `nominal_hip_yaw_peak_window.csv`\n")
        f.write("- `high_0p480_hip_yaw_peak_window.csv`\n\n")

    print(f"[SAVED] {report_path}")


if __name__ == "__main__":
    main()
