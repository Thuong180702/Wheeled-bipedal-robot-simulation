"""Analyze boundary Step E and Step C logs and telemetry.

This script reads telemetry CSV files and stdout/stderr logs from boundary
height validation runs (low_0p300, high_0p480) for both Step E hold and Step C
recovery, then generates comprehensive analysis including:

- Event order (first threshold crossings)
- Failure windows and time series
- Metric comparisons
- Mechanism classification
- Hypothesis confirmation/falsification

Usage:
    python scripts/analyze_boundary_step_e_step_c_logs.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STEP_E_DIR = Path("outputs/boundary_step_e_step_c_log_verification/step_e")
STEP_C_DIR = Path("outputs/boundary_step_e_step_c_log_verification/step_c")
ANALYSIS_DIR = Path("outputs/boundary_step_e_step_c_log_verification/analysis")

BOUNDARY_CASES = ["low_0p300", "high_0p480"]
STAGES = ["step_e", "step_c"]


def load_telemetry(case_name: str, stage: str) -> pd.DataFrame | None:
    """Load telemetry CSV for given case and stage."""
    stage_dir = STEP_E_DIR if stage == "step_e" else STEP_C_DIR
    telemetry_path = stage_dir / f"{case_name}_{stage}_telemetry.csv"

    if not telemetry_path.exists():
        print(f"[WARNING] Telemetry not found: {telemetry_path}")
        return None

    try:
        df = pd.read_csv(telemetry_path)
        print(f"[OK] Loaded telemetry: {case_name}_{stage} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {telemetry_path}: {e}")
        return None


def load_stdout_log(case_name: str, stage: str) -> list[str]:
    """Load stdout log lines for given case and stage."""
    stage_dir = STEP_E_DIR if stage == "step_e" else STEP_C_DIR
    log_path = stage_dir / f"{case_name}_{stage}_stdout.log"

    if not log_path.exists():
        print(f"[WARNING] Stdout log not found: {log_path}")
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        print(f"[OK] Loaded stdout log: {case_name}_{stage} ({len(lines)} lines)")
        return lines
    except Exception as e:
        print(f"[ERROR] Failed to load {log_path}: {e}")
        return []


def load_stderr_log(case_name: str, stage: str) -> list[str]:
    """Load stderr log lines for given case and stage."""
    stage_dir = STEP_E_DIR if stage == "step_e" else STEP_C_DIR
    log_path = stage_dir / f"{case_name}_{stage}_stderr.log"

    if not log_path.exists():
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if lines:
            print(f"[OK] Loaded stderr log: {case_name}_{stage} ({len(lines)} lines)")
        return lines
    except Exception as e:
        print(f"[ERROR] Failed to load {log_path}: {e}")
        return []


def compute_core_metrics(df: pd.DataFrame, case_name: str, stage: str) -> dict[str, Any]:
    """Compute core metrics from telemetry."""
    if df is None or df.empty:
        return {"error": "empty_or_missing_telemetry"}

    metrics = {
        "case_name": case_name,
        "stage": stage,
        "row_count": len(df),
        "final_source_step_index": int(df["source_step_index"].iloc[-1]) if "source_step_index" in df.columns else None,
    }

    # Height metrics
    if "current_com_z" in df.columns:
        metrics["current_com_z_m_min"] = float(df["current_com_z"].min())
        metrics["current_com_z_m_max"] = float(df["current_com_z"].max())
        metrics["current_com_z_m_final"] = float(df["current_com_z"].iloc[-1])
        metrics["current_com_z_m_mean"] = float(df["current_com_z"].mean())
        metrics["current_com_z_m_std"] = float(df["current_com_z"].std())

    # Hip yaw metrics
    if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
        l_hip_yaw = df["l_hip_yaw_pos"].abs()
        r_hip_yaw = df["r_hip_yaw_pos"].abs()
        hip_yaw_abs_max = pd.concat([l_hip_yaw, r_hip_yaw], axis=1).max(axis=1)

        metrics["hip_yaw_abs_max_max"] = float(hip_yaw_abs_max.max())
        metrics["hip_yaw_abs_max_final"] = float(hip_yaw_abs_max.iloc[-1])
        metrics["hip_yaw_abs_max_mean"] = float(hip_yaw_abs_max.mean())

        metrics["l_hip_yaw_error_max"] = float(df["l_hip_yaw_error"].abs().max()) if "l_hip_yaw_error" in df.columns else None
        metrics["r_hip_yaw_error_max"] = float(df["r_hip_yaw_error"].abs().max()) if "r_hip_yaw_error" in df.columns else None
        metrics["l_hip_yaw_error_final"] = float(df["l_hip_yaw_error"].iloc[-1]) if "l_hip_yaw_error" in df.columns else None
        metrics["r_hip_yaw_error_final"] = float(df["r_hip_yaw_error"].iloc[-1]) if "r_hip_yaw_error" in df.columns else None

    # Posture metrics
    if "control_pitch_x" in df.columns:
        metrics["pitch_x_rad_max_abs"] = float(df["control_pitch_x"].abs().max())
        metrics["pitch_x_rad_final"] = float(df["control_pitch_x"].iloc[-1])
        metrics["pitch_x_rad_rms"] = float(np.sqrt((df["control_pitch_x"] ** 2).mean()))

    if "control_roll_y" in df.columns:
        metrics["roll_y_rad_max_abs"] = float(df["control_roll_y"].abs().max())
        metrics["roll_y_rad_final"] = float(df["control_roll_y"].iloc[-1])
        metrics["roll_y_rad_rms"] = float(np.sqrt((df["control_roll_y"] ** 2).mean()))

    if "yaw_z_rad" in df.columns:
        metrics["yaw_z_rad_max_abs"] = float(df["yaw_z_rad"].abs().max())
        metrics["yaw_z_rad_final"] = float(df["yaw_z_rad"].iloc[-1])

    # Support position metrics
    if "sagittal_position_error_m" in df.columns:
        metrics["support_position_error_m_max_abs"] = float(df["sagittal_position_error_m"].abs().max())
        metrics["support_position_error_m_final"] = float(df["sagittal_position_error_m"].iloc[-1])
        metrics["support_position_error_m_rms"] = float(np.sqrt((df["sagittal_position_error_m"] ** 2).mean()))

    # Wheel velocity metrics
    if "l_wheel_vel" in df.columns and "r_wheel_vel" in df.columns:
        wheel_vel_mean = (df["l_wheel_vel"] + df["r_wheel_vel"]) / 2.0
        metrics["wheel_vel_mean_rad_s_max_abs"] = float(wheel_vel_mean.abs().max())
        metrics["wheel_vel_mean_rad_s_rms"] = float(np.sqrt((wheel_vel_mean ** 2).mean()))

    # Contact metrics
    if "contact_valid" in df.columns:
        metrics["contact_valid_percent"] = float((df["contact_valid"] == 1.0).sum() / len(df) * 100)
        metrics["invalid_contact_rows"] = int((df["contact_valid"] != 1.0).sum())

    if "non_wheel_floor_contacts" in df.columns:
        metrics["non_wheel_floor_contacts_max"] = int(df["non_wheel_floor_contacts"].max())

    # Torque metrics
    if "tau_position_raw" in df.columns:
        metrics["tau_position_raw_max_abs"] = float(df["tau_position_raw"].abs().max())

    if "tau_position" in df.columns:
        metrics["tau_position_clipped_max_abs"] = float(df["tau_position"].abs().max())

    if "tau_pitch" in df.columns:
        metrics["tau_pitch_max_abs"] = float(df["tau_pitch"].abs().max())

    # WBC status
    if "applied_wbc_contribution_norm" in df.columns:
        metrics["wbc_applied"] = bool((df["applied_wbc_contribution_norm"] > 0.0).any())
        metrics["wbc_contribution_max"] = float(df["applied_wbc_contribution_norm"].max())
    else:
        metrics["wbc_applied"] = False
        metrics["wbc_contribution_max"] = 0.0

    # Ownership metrics
    if "ownership_violation_count" in df.columns:
        metrics["ownership_violation_count_max"] = int(df["ownership_violation_count"].max())

    if "hidden_torque_norm" in df.columns:
        metrics["hidden_torque_norm_max"] = float(df["hidden_torque_norm"].max())

    return metrics


def find_first_threshold_crossing(df: pd.DataFrame, column: str, threshold: float, direction: str = "abs") -> int | None:
    """Find first row index where column crosses threshold.

    Args:
        df: Telemetry dataframe
        column: Column name
        threshold: Threshold value
        direction: "abs" (absolute), "positive", or "negative"

    Returns:
        Row index of first crossing, or None if never crossed
    """
    if column not in df.columns:
        return None

    if direction == "abs":
        mask = df[column].abs() > threshold
    elif direction == "positive":
        mask = df[column] > threshold
    elif direction == "negative":
        mask = df[column] < -threshold
    else:
        raise ValueError(f"Unknown direction: {direction}")

    crossing_indices = df[mask].index.tolist()
    return int(crossing_indices[0]) if crossing_indices else None


def compute_event_order(df: pd.DataFrame, case_name: str, stage: str) -> dict[str, Any]:
    """Compute event order (first threshold crossings)."""
    if df is None or df.empty:
        return {"case_name": case_name, "stage": stage, "error": "empty_telemetry"}

    events = {
        "case_name": case_name,
        "stage": stage,
    }

    # Hip yaw thresholds
    if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
        hip_yaw_abs_max = pd.concat([df["l_hip_yaw_pos"].abs(), df["r_hip_yaw_pos"].abs()], axis=1).max(axis=1)
        df_temp = df.copy()
        df_temp["hip_yaw_abs_max"] = hip_yaw_abs_max

        events["hip_yaw_0p03_cross"] = find_first_threshold_crossing(df_temp, "hip_yaw_abs_max", 0.03, "abs")
        events["hip_yaw_0p07_cross"] = find_first_threshold_crossing(df_temp, "hip_yaw_abs_max", 0.07, "abs")
        events["hip_yaw_0p10_cross"] = find_first_threshold_crossing(df_temp, "hip_yaw_abs_max", 0.10, "abs")

    # Support position thresholds
    if "sagittal_position_error_m" in df.columns:
        events["support_0p05_cross"] = find_first_threshold_crossing(df, "sagittal_position_error_m", 0.05, "abs")
        events["support_0p10_cross"] = find_first_threshold_crossing(df, "sagittal_position_error_m", 0.10, "abs")
        events["support_0p15_cross"] = find_first_threshold_crossing(df, "sagittal_position_error_m", 0.15, "abs")

    # Pitch thresholds
    if "control_pitch_x" in df.columns:
        events["pitch_0p05_cross"] = find_first_threshold_crossing(df, "control_pitch_x", np.deg2rad(2.865), "abs")  # ~0.05 rad
        events["pitch_0p10_cross"] = find_first_threshold_crossing(df, "control_pitch_x", np.deg2rad(5.730), "abs")  # ~0.10 rad

    # Classify event order
    event_times = {k: v for k, v in events.items() if v is not None and k != "case_name" and k != "stage"}
    if not event_times:
        events["classification"] = "no_thresholds_crossed"
        events["first_event"] = None
        events["second_event"] = None
        events["third_event"] = None
    else:
        sorted_events = sorted(event_times.items(), key=lambda x: x[1])
        events["first_event"] = sorted_events[0][0] if len(sorted_events) > 0 else None
        events["second_event"] = sorted_events[1][0] if len(sorted_events) > 1 else None
        events["third_event"] = sorted_events[2][0] if len(sorted_events) > 2 else None

        first = events["first_event"]
        if first and "hip_yaw" in first:
            events["classification"] = "hip_yaw_led"
        elif first and "support" in first:
            events["classification"] = "support_position_led"
        elif first and "pitch" in first:
            events["classification"] = "pitch_led"
        else:
            events["classification"] = "unclear_coupled"

    return events


def extract_failure_windows(df: pd.DataFrame, case_name: str, stage: str) -> pd.DataFrame:
    """Extract failure windows (rows where thresholds are violated)."""
    if df is None or df.empty:
        return pd.DataFrame()

    failure_rows = []

    for idx, row in df.iterrows():
        is_failure = False
        failure_reasons = []

        # Hip yaw check
        if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
            hip_yaw_max = max(abs(row["l_hip_yaw_pos"]), abs(row["r_hip_yaw_pos"]))
            if hip_yaw_max > 0.07:
                is_failure = True
                failure_reasons.append(f"hip_yaw={hip_yaw_max:.4f}")

        # Support position check
        if "sagittal_position_error_m" in df.columns:
            support_error = abs(row["sagittal_position_error_m"])
            if support_error > 0.15:
                is_failure = True
                failure_reasons.append(f"support={support_error:.4f}")

        # Pitch check
        if "control_pitch_x" in df.columns:
            pitch = abs(row["control_pitch_x"])
            if pitch > 0.10:
                is_failure = True
                failure_reasons.append(f"pitch={pitch:.4f}")

        if is_failure:
            failure_row = {
                "case_name": case_name,
                "stage": stage,
                "row_index": int(idx),
                "source_step_index": int(row["source_step_index"]) if "source_step_index" in df.columns else None,
                "time_s": float(row["source_step_index"] * 0.01) if "source_step_index" in df.columns else None,
                "failure_reasons": "; ".join(failure_reasons),
            }

            # Add key telemetry columns
            for col in ["l_hip_yaw_pos", "r_hip_yaw_pos", "sagittal_position_error_m", "control_pitch_x", "control_roll_y"]:
                if col in df.columns:
                    failure_row[col] = float(row[col])

            failure_rows.append(failure_row)

    return pd.DataFrame(failure_rows)


def scan_logs_for_patterns(stdout_lines: list[str], stderr_lines: list[str], case_name: str, stage: str) -> dict[str, Any]:
    """Scan stdout/stderr logs for warnings, errors, and patterns."""
    patterns = {
        "case_name": case_name,
        "stage": stage,
        "warnings": [],
        "errors": [],
        "termination_messages": [],
        "wbc_messages": [],
        "contact_invalid_messages": [],
        "height_floor_messages": [],
        "nan_messages": [],
        "saturation_messages": [],
        "ownership_messages": [],
    }

    # Scan stdout
    for line in stdout_lines:
        line_lower = line.lower()
        if "warning" in line_lower or "[warning]" in line_lower:
            patterns["warnings"].append(line.strip())
        if "error" in line_lower and "error:" in line_lower:
            patterns["errors"].append(line.strip())
        if "termination" in line_lower or "terminated" in line_lower:
            patterns["termination_messages"].append(line.strip())
        if "wbc" in line_lower:
            patterns["wbc_messages"].append(line.strip())
        if "contact" in line_lower and ("invalid" in line_lower or "non_wheel" in line_lower):
            patterns["contact_invalid_messages"].append(line.strip())
        if "height" in line_lower and ("floor" in line_lower or "too low" in line_lower):
            patterns["height_floor_messages"].append(line.strip())
        if "nan" in line_lower or "inf" in line_lower:
            patterns["nan_messages"].append(line.strip())
        if "saturation" in line_lower or "saturated" in line_lower or "clipped" in line_lower:
            patterns["saturation_messages"].append(line.strip())
        if "ownership" in line_lower or "hidden_torque" in line_lower:
            patterns["ownership_messages"].append(line.strip())

    # Scan stderr
    for line in stderr_lines:
        line_lower = line.lower()
        if "error" in line_lower or "exception" in line_lower or "traceback" in line_lower:
            patterns["errors"].append(line.strip())

    # Truncate long lists
    for key in patterns:
        if isinstance(patterns[key], list) and len(patterns[key]) > 20:
            patterns[key] = patterns[key][:10] + ["... (truncated)"] + patterns[key][-10:]

    return patterns


def main():
    """Main analysis pipeline."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Boundary Step E & Step C Log Verification Analysis")
    print("=" * 80)

    all_metrics = []
    all_event_orders = []
    all_log_patterns = []

    for case_name in BOUNDARY_CASES:
        for stage in STAGES:
            print(f"\n--- Analyzing: {case_name} / {stage} ---")

            # Load data
            df = load_telemetry(case_name, stage)
            stdout_lines = load_stdout_log(case_name, stage)
            stderr_lines = load_stderr_log(case_name, stage)

            # Compute metrics
            metrics = compute_core_metrics(df, case_name, stage)
            all_metrics.append(metrics)

            # Compute event order
            event_order = compute_event_order(df, case_name, stage)
            all_event_orders.append(event_order)

            # Extract failure windows
            failure_windows = extract_failure_windows(df, case_name, stage)
            if not failure_windows.empty:
                failure_path = ANALYSIS_DIR / f"{case_name}_{stage}_failure_windows.csv"
                failure_windows.to_csv(failure_path, index=False)
                print(f"[OK] Saved failure windows: {failure_path} ({len(failure_windows)} rows)")

            # Scan logs
            log_patterns = scan_logs_for_patterns(stdout_lines, stderr_lines, case_name, stage)
            all_log_patterns.append(log_patterns)

    # Save summary JSON
    summary = {
        "metrics": all_metrics,
        "event_orders": all_event_orders,
        "log_patterns": all_log_patterns,
    }

    summary_path = ANALYSIS_DIR / "boundary_step_e_step_c_log_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved summary: {summary_path}")

    # Save metrics comparison CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = ANALYSIS_DIR / "boundary_step_e_step_c_metric_comparison.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[OK] Saved metrics comparison: {metrics_csv_path}")

    # Save event order JSON
    for event_order in all_event_orders:
        case_name = event_order["case_name"]
        stage = event_order["stage"]
        event_path = ANALYSIS_DIR / f"{case_name}_{stage}_event_order.json"
        with open(event_path, "w") as f:
            json.dump(event_order, f, indent=2)
        print(f"[OK] Saved event order: {event_path}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
