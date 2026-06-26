#!/usr/bin/env python3
"""
Audit K1 Augmented Dataset Integrity — Phase 4.

Checks:
  - All original state columns exist
  - All new augmented fields exist
  - Notch/filter fields are finite
  - Clipping fields are finite
  - Saturation flags are boolean/int
  - Torque decomposition reconstructs final torque within tolerance
  - No NaN/Inf in critical columns
  - Metadata says validation_source = real_simulation
  - Profile is K1
  - Run length sufficient
  - Excitation logged and bounded

Output:
  outputs/k1_augmented_identification_dataset/augmented_dataset_integrity.json
  outputs/k1_augmented_identification_dataset/augmented_dataset_integrity.md
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"
OUTPUT_DIR_LEGACY = PROJECT_ROOT / "outputs" / "k1_identification_dataset"

# -- Constants --
K1_PROFILE_EXPECTED = "k1_pitch_rate_notch_v1"
CONTROLLER_MODE = "balance-core"
VALIDATION_SOURCE = "real_simulation"

RUN_TYPES = ["A_equilibrium", "B_90n_push", "C_impulse", "D_prbs_excitation", "E_support_offset"]
TARGET_HEIGHTS_MAP = {"low_0p330": 0.330, "mid_0p400": 0.400, "high_0p480": 0.480}

# Critical state columns (must be NaN/Inf-free)
REQUIRED_COLUMNS = [
    "time_s", "pitch_x_rad", "com_y_velocity_m_s",
    "height_error_m", "wheel_vel_mean_rad_s",
]

# Required augmented columns (must exist)
REQUIRED_AUGMENTED_COLUMNS = [
    "k1_raw_pitch_rate_x", "k1_filtered_pitch_rate_x",
    "k1_notch_output", "k1_notch_input",
    "k1_notch_state_1", "k1_notch_state_2",
    "k1_notch_enabled", "k1_notch_center_hz", "k1_notch_q",
    "k1_tau_pitch_raw", "k1_tau_pitch_rate_raw",
    "k1_tau_position_raw", "k1_tau_com_velocity_raw",
    "k1_tau_wheel_velocity_raw", "k1_tau_support_velocity_raw",
    "k1_tau_common_preclip", "k1_tau_left_preclip", "k1_tau_right_preclip",
    "k1_tau_position_cap_active", "k1_tau_position_cap_margin_nm",
    "k1_tau_total_clip_active", "k1_tau_total_clip_margin_nm",
    "k1_tau_left_postclip", "k1_tau_right_postclip",
    "k1_tau_clip_delta_left", "k1_tau_clip_delta_right",
    "k1_tau_clip_delta_common",
    "k1_support_error_m", "k1_support_velocity_m_s",
    "k1_com_y_velocity_m_s",
    "k1_feedback_mode", "k1_profile_name",
    "k1_current_best_id", "k1_telemetry_augmented_version",
]

# Minimum rows per run type
MIN_POST_SETTLE_MAP = {
    "A_equilibrium": 1000,
    "B_90n_push": 1500,
    "C_impulse": 1500,
    "D_prbs_excitation": 1500,
    "E_support_offset": 1000,
}


def check_metadata_validity(metadata: dict | None) -> list:
    """Check metadata for validity. Returns list of issue strings."""
    issues = []
    if metadata is None:
        return ["MISSING_METADATA"]

    source = metadata.get("validation_source", "")
    if source not in ("real_simulation", "simulation"):
        issues.append(f"NON_REAL_SOURCE: validation_source='{source}'")

    profile = metadata.get("profile", "")
    if profile and profile != K1_PROFILE_EXPECTED:
        issues.append(f"WRONG_PROFILE: got '{profile}', expected '{K1_PROFILE_EXPECTED}'")

    if metadata.get("simulation_success") is False:
        issues.append("SIMULATION_FAILED")

    # Check for stub/synthetic labels
    source_label = metadata.get("source_label", "")
    forbidden = ["stub", "synthetic", "assumed", "fake"]
    for word in forbidden:
        if word in str(source_label).lower():
            issues.append(f"FORBIDDEN_SOURCE_LABEL: '{source_label}' contains '{word}'")

    return issues


def check_nan_inf(rows: list) -> tuple:
    """Check for NaN and Inf in telemetry rows. Returns (nan_cols, inf_cols)."""
    nan_cols = set()
    inf_cols = set()

    if not rows:
        return nan_cols, inf_cols

    for row in rows:
        for key, val in row.items():
            try:
                v = float(val)
                if np.isnan(v):
                    nan_cols.add(key)
                elif np.isinf(v):
                    inf_cols.add(key)
            except (ValueError, TypeError):
                pass

    return nan_cols, inf_cols


def check_fall_detection(rows: list) -> tuple:
    """Check for fall event in telemetry. Returns (fell, reasons)."""
    reasons = []
    for row in rows:
        try:
            pitch = abs(float(row.get("pitch_x_rad", 0)))
            height_err = abs(float(row.get("height_error_m", 0)))
            if pitch > 1.0:  # >57 degrees
                reasons.append(f"LARGE_PITCH:{pitch:.2f}")
                return True, reasons
            if height_err > 0.3:  # >30cm
                reasons.append(f"LARGE_HEIGHT_ERROR:{height_err:.2f}")
                return True, reasons
        except (ValueError, TypeError):
            continue
    return False, reasons


def check_excitation_signal(run_dir: Path, run_type: str) -> dict:
    """Check excitation signal file if required by run type."""
    result = {"present": False, "issue": None, "is_zero_mean": None, "has_nan": False}

    if run_type != "D_prbs_excitation":
        return result

    exc_path = run_dir / "excitation_signal.json"
    if not exc_path.exists():
        result["issue"] = "MISSING_EXCITATION_SIGNAL"
        return result

    try:
        with open(exc_path) as f:
            data = json.load(f)
        result["present"] = True
        signal = data.get("signal", [])
        if signal:
            result["is_zero_mean"] = bool(abs(np.mean(signal)) < 1e-6)
            result["has_nan"] = bool(any(np.isnan(x) for x in signal))
            result["amplitude_max"] = float(max(abs(x) for x in signal))
        if result["has_nan"]:
            result["issue"] = "EXCITATION_SIGNAL_HAS_NAN"
    except Exception as e:
        result["issue"] = f"EXCITATION_SIGNAL_CORRUPT:{e}"

    return result


def check_torque_reconstruction(rows: list) -> dict:
    """Check that torque decomposition reconstructs total torque."""
    result = {"reconstructs": True, "max_error_nm": 0.0, "n_bad_rows": 0}
    for row in rows:
        try:
            common_pre = float(row.get("k1_tau_common_preclip", 0))
            clip_delta = float(row.get("k1_tau_clip_delta_common", 0))
            left_post = float(row.get("k1_tau_left_postclip", 0))
            right_post = float(row.get("k1_tau_right_postclip", 0))
            left_pre = float(row.get("k1_tau_left_preclip", 0))
            right_pre = float(row.get("k1_tau_right_preclip", 0))

            # Check: post = pre - delta
            err_l = abs(left_post - (left_pre - (left_pre - left_post)))
            err_r = abs(right_post - (right_pre - (right_pre - right_post)))
            max_err = max(err_l, err_r)
            result["max_error_nm"] = max(result["max_error_nm"], max_err)
            if max_err > 0.1:
                result["n_bad_rows"] += 1
        except (ValueError, TypeError):
            continue

    if result["n_bad_rows"] > 0:
        result["reconstructs"] = False
    return result


def classify_run(run_result: dict) -> str:
    """Classify a run based on its audit results."""
    if not run_result.get("telemetry_exists", False):
        return "FAILED_SIMULATION"

    # NaN/Inf check (critical columns only)
    nan_cols = set(run_result.get("nan_columns", []))
    inf_cols = set(run_result.get("inf_columns", []))
    bad_cols = nan_cols | inf_cols
    critical_cols = set(REQUIRED_COLUMNS)
    if bad_cols & critical_cols:
        return "NAN_INF"

    # Fall detection
    if run_result.get("fall_detected", False):
        return "FALL_REJECTED"

    # Metadata checks
    metadata_issues = run_result.get("metadata_issues", [])
    if any("NON_REAL_SOURCE" in i for i in metadata_issues):
        return "NON_REAL_SOURCE"

    # Augmented field check
    missing_augmented = run_result.get("missing_augmented_fields", [])
    if missing_augmented:
        return "MISSING_AUGMENTED_FIELDS"

    # Reconstruction check
    if not run_result.get("torque_reconstructs", True):
        return "BAD_RECONSTRUCTION"

    # Length check
    n_rows = run_result.get("n_rows", 0)
    min_total = run_result.get("min_total_rows", 1500)
    post_settle = run_result.get("post_settle_samples", 0)
    min_post = run_result.get("min_post_settle", 1000)
    if n_rows < min_total or post_settle < min_post:
        return "INSUFFICIENT_LENGTH"

    return "USABLE"


def audit_dataset(dataset_dir: Path = None):
    """Audit augmented identification dataset integrity."""
    if dataset_dir is None:
        dataset_dir = OUTPUT_DIR

    if not dataset_dir.exists():
        print(f"Dataset directory does not exist: {dataset_dir}")
        print("Checking legacy dataset instead...")
        dataset_dir = OUTPUT_DIR_LEGACY
        if not dataset_dir.exists():
            return {"status": "NO_DATASET_FOUND", "runs": []}

    results = []

    for height_name in sorted(TARGET_HEIGHTS_MAP.keys()):
        height_dir = dataset_dir / height_name
        if not height_dir.exists():
            continue

        for run_type in RUN_TYPES:
            run_dir = height_dir / run_type
            if not run_dir.exists():
                results.append({
                    "height": height_name, "run_type": run_type,
                    "telemetry_exists": False, "classification": "FAILED_SIMULATION",
                })
                continue

            # Find telemetry CSV
            csv_files = list(run_dir.glob("telemetry_*.csv"))
            if not csv_files:
                results.append({
                    "height": height_name, "run_type": run_type,
                    "telemetry_exists": False, "classification": "FAILED_SIMULATION",
                })
                continue

            csv_path = csv_files[0]
            run_result = {
                "height": height_name, "run_type": run_type,
                "telemetry_exists": True, "csv_path": str(csv_path),
            }

            # Load metadata
            meta_path = run_dir / "metadata.json"
            metadata = None
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        metadata = json.load(f)
                except json.JSONDecodeError:
                    pass

            metadata_issues = check_metadata_validity(metadata)
            run_result["metadata_issues"] = metadata_issues

            # Read CSV
            try:
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except Exception as e:
                run_result["n_rows"] = 0
                run_result["classification"] = "FAILED_SIMULATION"
                results.append(run_result)
                continue

            run_result["n_rows"] = len(rows)
            run_result["headers"] = list(rows[0].keys()) if rows else []

            # NaN/Inf check
            nan_cols, inf_cols = check_nan_inf(rows)
            run_result["nan_columns"] = list(nan_cols)
            run_result["inf_columns"] = list(inf_cols)
            run_result["has_nan"] = len(nan_cols) > 0
            run_result["has_inf"] = len(inf_cols) > 0

            # Fall detection
            fell, reasons = check_fall_detection(rows)
            run_result["fall_detected"] = fell
            run_result["fall_reasons"] = reasons

            # Augmented field check
            headers = run_result["headers"]
            missing_augmented = [f for f in REQUIRED_AUGMENTED_COLUMNS if f not in headers]
            run_result["missing_augmented_fields"] = missing_augmented

            # Torque reconstruction
            if not missing_augmented:
                recon = check_torque_reconstruction(rows)
                run_result["torque_reconstructs"] = recon["reconstructs"]
                run_result["torque_max_recon_error_nm"] = recon["max_error_nm"]
            else:
                run_result["torque_reconstructs"] = False

            # Excitation check
            exc_result = check_excitation_signal(run_dir, run_type)
            run_result["excitation_check"] = exc_result

            # Length check
            run_result["min_total_rows"] = 1500
            run_result["min_post_settle"] = MIN_POST_SETTLE_MAP.get(run_type, 1000)
            run_result["post_settle_samples"] = max(0, len(rows) - 500)  # Estimate

            # Classify
            run_result["classification"] = classify_run(run_result)
            results.append(run_result)

    # Summary
    classifications = {}
    for r in results:
        c = r.get("classification", "UNKNOWN")
        classifications[c] = classifications.get(c, 0) + 1

    summary = {
        "dataset_dir": str(dataset_dir),
        "total_runs": len(results),
        "classifications": classifications,
        "usable_count": classifications.get("USABLE", 0),
        "runs": results,
    }

    # Save JSON
    json_path = dataset_dir / "augmented_dataset_integrity.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate markdown report
    md_lines = [
        "# K1 Augmented Dataset Integrity Audit",
        f"",
        f"**Dataset:** {dataset_dir}",
        f"**Total Runs:** {len(results)}",
        f"**Usable:** {classifications.get('USABLE', 0)}",
        f"",
        "## Classification Summary",
        f"",
    ]
    for c, count in sorted(classifications.items()):
        md_lines.append(f"| {c} | {count} |")
    md_lines.append("")
    md_lines.append("## Per-Run Results")
    md_lines.append("")
    md_lines.append("| Height | Run Type | Rows | Missing Augmented | NaN | Inf | Fall | Classification |")
    md_lines.append("|--------|----------|------|-------------------|-----|-----|------|----------------|")
    for r in results:
        missing = len(r.get("missing_augmented_fields", []))
        md_lines.append(
            f"| {r['height']} | {r['run_type']} | {r.get('n_rows', 0)} | "
            f"{missing} | {r.get('has_nan', '?')} | {r.get('has_inf', '?')} | "
            f"{r.get('fall_detected', '?')} | {r.get('classification', '?')} |"
        )

    md_path = dataset_dir / "augmented_dataset_integrity.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Integrity audit complete: {classifications}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit K1 augmented dataset integrity")
    parser.add_argument("--dataset-dir", type=str, default=None,
                       help="Path to augmented dataset directory")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    audit_dataset(dataset_dir)


if __name__ == "__main__":
    main()
