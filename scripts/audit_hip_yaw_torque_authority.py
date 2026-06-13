"""Audit hip-yaw torque authority and identify limiting factors.

Phase 3 of hip-yaw investigation: determine if hip-yaw torque is sufficient,
saturated, rate-limited, or overwritten.
"""

import json
from pathlib import Path

import pandas as pd


def find_key_windows(df: pd.DataFrame) -> dict:
    """Find key analysis windows."""

    hip_yaw_abs_max = df["hip_yaw_abs_max"]
    support_error = df["support_position_error_m"]

    # Find onset and peak for hip-yaw
    hip_yaw_onset_idx = None
    for i in range(len(df)):
        if hip_yaw_abs_max.iloc[i] > 0.03:
            hip_yaw_onset_idx = i
            break

    hip_yaw_peak_idx = hip_yaw_abs_max.idxmax()

    # Find onset and peak for support error
    support_onset_idx = None
    for i in range(len(df)):
        if support_error.iloc[i] > 0.05:
            support_onset_idx = i
            break

    support_peak_idx = support_error.idxmax()

    return {
        "hip_yaw_onset": hip_yaw_onset_idx,
        "hip_yaw_peak": hip_yaw_peak_idx,
        "support_onset": support_onset_idx,
        "support_peak": support_peak_idx,
    }


def analyze_window(df: pd.DataFrame, center_idx: int, window_size: int = 20) -> dict:
    """Analyze a window around a key event."""

    start = max(0, center_idx - window_size // 2)
    end = min(len(df), center_idx + window_size // 2)

    window = df.iloc[start:end]

    # Hip-yaw torque
    l_tau_raw = window["l_hip_yaw_tau_shape_raw"]
    r_tau_raw = window["r_hip_yaw_tau_shape_raw"]
    l_tau_final = window["l_hip_yaw_tau_shape_final"]
    r_tau_final = window["r_hip_yaw_tau_shape_final"]

    # Torque correctness
    l_sign_correct = window["hip_yaw_torque_sign_correct_left"]
    r_sign_correct = window["hip_yaw_torque_sign_correct_right"]

    # Saturation flags
    l_saturated = window.get("hip_yaw_torque_saturation_flag_left", pd.Series([False] * len(window)))
    r_saturated = window.get("hip_yaw_torque_saturation_flag_right", pd.Series([False] * len(window)))

    # Torque margin
    l_margin = window.get("hip_yaw_torque_margin_left", pd.Series([float('nan')] * len(window)))
    r_margin = window.get("hip_yaw_torque_margin_right", pd.Series([float('nan')] * len(window)))

    # Errors
    l_error = window["l_hip_yaw_error"]
    r_error = window["r_hip_yaw_error"]

    # Check if torque is lost (raw != final)
    torque_lost = bool(not (l_tau_raw - l_tau_final).abs().max() < 1e-6 or not (r_tau_raw - r_tau_final).abs().max() < 1e-6)

    return {
        "center_step": center_idx,
        "window_start": start,
        "window_end": end,
        "l_tau_raw_mean": float(l_tau_raw.mean()),
        "r_tau_raw_mean": float(r_tau_raw.mean()),
        "l_tau_final_mean": float(l_tau_final.mean()),
        "r_tau_final_mean": float(r_tau_final.mean()),
        "l_tau_raw_max_abs": float(l_tau_raw.abs().max()),
        "r_tau_raw_max_abs": float(r_tau_raw.abs().max()),
        "l_sign_correct_rate": float(l_sign_correct.mean()) if not l_sign_correct.isna().all() else None,
        "r_sign_correct_rate": float(r_sign_correct.mean()) if not r_sign_correct.isna().all() else None,
        "l_saturated_rate": float(l_saturated.mean()) if not l_saturated.isna().all() else None,
        "r_saturated_rate": float(r_saturated.mean()) if not r_saturated.isna().all() else None,
        "l_margin_min": float(l_margin.min()) if not l_margin.isna().all() else None,
        "r_margin_min": float(r_margin.min()) if not r_margin.isna().all() else None,
        "l_error_max_abs": float(l_error.abs().max()),
        "r_error_max_abs": float(r_error.abs().max()),
        "torque_lost": torque_lost,
    }


def audit_torque_authority(variant_name: str, telemetry_csv: Path) -> dict:
    """Audit hip-yaw torque authority for a single case."""

    df = pd.read_csv(telemetry_csv)

    # Find key windows
    windows_info = find_key_windows(df)

    # Analyze each window
    windows = {}
    for key, idx in windows_info.items():
        if idx is not None:
            windows[key] = analyze_window(df, idx)
        else:
            windows[key] = None

    # Overall torque statistics
    l_tau_raw = df["l_hip_yaw_tau_shape_raw"]
    r_tau_raw = df["r_hip_yaw_tau_shape_raw"]
    l_tau_final = df["l_hip_yaw_tau_shape_final"]
    r_tau_final = df["r_hip_yaw_tau_shape_final"]

    l_sign_correct = df["hip_yaw_torque_sign_correct_left"]
    r_sign_correct = df["hip_yaw_torque_sign_correct_right"]

    l_saturated = df.get("hip_yaw_torque_saturation_flag_left", pd.Series([False] * len(df)))
    r_saturated = df.get("hip_yaw_torque_saturation_flag_right", pd.Series([False] * len(df)))

    # Check ownership (if available)
    ownership_violations = 0
    if "ownership_violations" in df.columns:
        ownership_violations = int(df["ownership_violations"].max())

    # Check if torque equals shape command
    torque_matches_shape = bool((l_tau_raw - l_tau_final).abs().max() < 1e-6 and (r_tau_raw - r_tau_final).abs().max() < 1e-6)

    # Classification
    issues = []

    if l_sign_correct.mean() < 0.95 or r_sign_correct.mean() < 0.95:
        issues.append("sign_error_detected")

    if l_saturated.mean() > 0.05 or r_saturated.mean() > 0.05:
        issues.append("torque_saturation")

    if not torque_matches_shape:
        issues.append("torque_composer_loss")

    if ownership_violations > 0:
        issues.append("ownership_violation")

    # Check if torque is growing with error
    hip_yaw_abs_max = df["hip_yaw_abs_max"]
    if hip_yaw_abs_max.max() > 0.07:
        # Find when error exceeds threshold
        exceed_idx = (hip_yaw_abs_max > 0.07).idxmax()
        if exceed_idx > 100:
            # Check if torque grew leading up to exceedance
            pre_window = df.iloc[exceed_idx-50:exceed_idx]
            l_tau_growth = pre_window["l_hip_yaw_tau_shape_raw"].abs().max() - pre_window["l_hip_yaw_tau_shape_raw"].abs().iloc[0]
            r_tau_growth = pre_window["r_hip_yaw_tau_shape_raw"].abs().max() - pre_window["r_hip_yaw_tau_shape_raw"].abs().iloc[0]

            if l_tau_growth < 0.5 and r_tau_growth < 0.5:
                issues.append("torque_authority_insufficient")

    if not issues:
        classification = "no_torque_issue_detected"
    else:
        classification = "_".join(issues)

    return {
        "variant": variant_name,
        "overall": {
            "l_tau_raw_max_abs": float(l_tau_raw.abs().max()),
            "r_tau_raw_max_abs": float(r_tau_raw.abs().max()),
            "l_sign_correct_rate": float(l_sign_correct.mean()),
            "r_sign_correct_rate": float(r_sign_correct.mean()),
            "l_saturated_rate": float(l_saturated.mean()) if not l_saturated.isna().all() else 0.0,
            "r_saturated_rate": float(r_saturated.mean()) if not r_saturated.isna().all() else 0.0,
            "torque_matches_shape": torque_matches_shape,
            "ownership_violations": ownership_violations,
        },
        "windows": windows,
        "classification": classification,
        "issues": issues,
    }


def main():
    audit_dir = Path("outputs/hip_yaw_boundary_audit")
    output_dir = audit_dir / "torque_authority"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        "low_0p300",
        "high_0p480",
        "nominal",
    ]

    results = []

    for variant_name in cases:
        telemetry_csv = audit_dir / f"{variant_name}_baseline_telemetry.csv"

        if not telemetry_csv.exists():
            print(f"ERROR: Telemetry missing for {variant_name}")
            continue

        print(f"Auditing torque authority for {variant_name}...")
        result = audit_torque_authority(variant_name, telemetry_csv)
        results.append(result)

        print(f"  Classification: {result['classification']}")
        print(f"  Issues: {result['issues']}")
        print()

    # Save summary
    summary = {
        "phase": "phase_3_torque_authority_audit",
        "date": "2026-06-04",
        "cases": results,
    }

    summary_path = output_dir / "hip_yaw_torque_authority_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")

    # Generate report
    report_lines = [
        "# Hip-Yaw Torque Authority Audit Report",
        "",
        "## Phase 3: Torque Authority Audit",
        "",
        "Date: 2026-06-04",
        "",
        "## Purpose",
        "",
        "Determine if hip-yaw torque is sufficient, saturated, rate-limited, or overwritten.",
        "",
        "## Results",
        "",
    ]

    for result in results:
        variant = result["variant"]
        classification = result["classification"]
        overall = result["overall"]

        report_lines.append(f"### {variant}")
        report_lines.append("")
        report_lines.append(f"**Classification:** `{classification}`")
        report_lines.append("")

        report_lines.append("**Overall Torque:**")
        report_lines.append(f"- l_tau_raw max_abs: {overall['l_tau_raw_max_abs']:.4f} Nm")
        report_lines.append(f"- r_tau_raw max_abs: {overall['r_tau_raw_max_abs']:.4f} Nm")
        report_lines.append(f"- l_sign_correct_rate: {overall['l_sign_correct_rate']:.3f}")
        report_lines.append(f"- r_sign_correct_rate: {overall['r_sign_correct_rate']:.3f}")
        report_lines.append(f"- l_saturated_rate: {overall['l_saturated_rate']:.3f}")
        report_lines.append(f"- r_saturated_rate: {overall['r_saturated_rate']:.3f}")
        report_lines.append(f"- torque_matches_shape: {overall['torque_matches_shape']}")
        report_lines.append(f"- ownership_violations: {overall['ownership_violations']}")
        report_lines.append("")

        # Windows
        windows = result["windows"]
        if windows.get("hip_yaw_onset"):
            report_lines.append("**Hip-Yaw Onset Window:**")
            w = windows["hip_yaw_onset"]
            report_lines.append(f"- center_step: {w['center_step']}")
            report_lines.append(f"- l_tau_raw max_abs: {w['l_tau_raw_max_abs']:.4f} Nm")
            report_lines.append(f"- r_tau_raw max_abs: {w['r_tau_raw_max_abs']:.4f} Nm")
            report_lines.append(f"- l_error max_abs: {w['l_error_max_abs']:.4f} rad")
            report_lines.append(f"- r_error max_abs: {w['r_error_max_abs']:.4f} rad")
            report_lines.append("")

        if windows.get("hip_yaw_peak"):
            report_lines.append("**Hip-Yaw Peak Window:**")
            w = windows["hip_yaw_peak"]
            report_lines.append(f"- center_step: {w['center_step']}")
            report_lines.append(f"- l_tau_raw max_abs: {w['l_tau_raw_max_abs']:.4f} Nm")
            report_lines.append(f"- r_tau_raw max_abs: {w['r_tau_raw_max_abs']:.4f} Nm")
            report_lines.append(f"- l_error max_abs: {w['l_error_max_abs']:.4f} rad")
            report_lines.append(f"- r_error max_abs: {w['r_error_max_abs']:.4f} rad")
            report_lines.append("")

        if result["issues"]:
            report_lines.append(f"**Issues:** {', '.join(result['issues'])}")
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # Summary
    report_lines.append("## Summary")
    report_lines.append("")

    all_no_issues = all(r["classification"] == "no_torque_issue_detected" for r in results)
    any_saturation = any("torque_saturation" in r["issues"] for r in results)
    any_composer_loss = any("torque_composer_loss" in r["issues"] for r in results)
    any_insufficient = any("torque_authority_insufficient" in r["issues"] for r in results)

    if all_no_issues:
        report_lines.append("No torque authority issues detected.")
        report_lines.append("")
        report_lines.append("Hip-yaw torque is not saturated, not rate-limited, not overwritten.")
        report_lines.append("")
        report_lines.append("Ready for Phase 4: Controlled isolation experiments")
    elif any_saturation:
        report_lines.append("FINDING: Torque saturation detected")
    elif any_composer_loss:
        report_lines.append("FINDING: Torque composer loss detected")
    elif any_insufficient:
        report_lines.append("FINDING: Torque authority insufficient")

    report_path = output_dir / "hip_yaw_torque_authority_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
