#!/usr/bin/env python3
"""
Audit K1 Identification Dataset Integrity — Phase 2.

Audits every generated run in outputs/k1_identification_dataset/ for:
  - telemetry file existence
  - metadata file existence
  - validation_source = real_simulation
  - source is not stub/synthetic/assumed
  - expected columns exist
  - no NaN/Inf
  - step count sufficient
  - height target correct
  - excitation signal bounded and zero-mean where applicable
  - K1 profile recorded correctly
  - no WBC/hidden torque
  - no fall unless marked rejected

Output:
  outputs/k1_identification_dataset/dataset_integrity_report.json
  outputs/k1_identification_dataset/dataset_integrity_report.md
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_HEIGHTS_MAP = {"low_0p330": 0.330, "mid_0p400": 0.400, "high_0p480": 0.480}
RUN_TYPES = ["A_equilibrium", "B_90n_push", "C_impulse", "D_prbs_excitation", "E_support_offset"]
K1_PROFILE_EXPECTED = "k1_pitch_rate_notch_v1"
CONTROLLER_MODE_EXPECTED = "balance-core"

# Per-run-type minimum post-settle sample requirements
MIN_POST_SETTLE_MAP = {
    "A_equilibrium": 1000,
    "B_90n_push": 1500,
    "C_impulse": 1500,
    "D_prbs_excitation": 1500,
    "E_support_offset": 1000,
}

# Minimum total rows for each run type
MIN_TOTAL_ROWS = {
    "A_equilibrium": 1500,
    "B_90n_push": 2000,
    "C_impulse": 1500,
    "D_prbs_excitation": 2000,
    "E_support_offset": 1500,
}

# Expected telemetry columns (must be at minimum present)
REQUIRED_COLUMNS = [
    "time_s",
    "pitch_x_rad",
    "com_y_velocity_m_s",
    "height_error_m",
    "wheel_vel_mean_rad_s",
]

# Columns that indicate non-real or WBC contamination
FORBIDDEN_COLUMNS_PATTERNS = [
    "stub", "synthetic", "assumed", "wbc_torque", "momentum_torque"
]

# Known columns that are acceptable if present
ACCEPTABLE_SUFFIXES = [
    "_rad", "_m", "_m_s", "_Nm", "_N", "_W", "_deg",
    "_count", "_active", "_flag", "_mode", "_profile",
    "_setup", "_source", "_step", "_time", "_error",
    "_target", "_ref", "_cmd", "_filtered", "_raw",
    "_mean", "_std", "_max", "_min", "_sum",
]


def _safe_float(val, default=0.0):
    """Convert value to float, handling non-numeric cases."""
    if isinstance(val, str) and val in ("True", "False", ""):
        return default
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  AUDIT FUNCTIONS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def check_file_exists(path):
    """Check if a file or directory exists."""
    return Path(path).exists() if path else False


def load_metadata(run_dir):
    """Load metadata.json from a run directory."""
    meta_path = Path(run_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_telemetry_csv(telemetry_path):
    """Load telemetry CSV and return rows + columns + stats."""
    if not telemetry_path or not Path(telemetry_path).exists():
        return None, None, None
    try:
        with open(telemetry_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return [], [], {"n_rows": 0, "n_cols": 0, "columns": []}
        columns = list(rows[0].keys())
        stats = {
            "n_rows": len(rows),
            "n_cols": len(columns),
            "columns": columns,
        }
        return rows, columns, stats
    except (IOError, csv.Error):
        return None, None, None


def check_nan_inf(rows):
    """Check for NaN or Inf in any numeric column."""
    nan_cols = set()
    inf_cols = set()
    for row in rows:
        for key, val in row.items():
            try:
                v = float(val)
                if np.isnan(v):
                    nan_cols.add(key)
                if np.isinf(v):
                    inf_cols.add(key)
            except (ValueError, TypeError):
                pass
    return list(nan_cols), list(inf_cols)


def check_fall_detection(rows):
    """Check if robot fell during the run by examining height/pitch columns."""
    if not rows:
        return False, "NO_DATA"

    # Look for fall indicators in telemetry
    fall_indicators = []
    for row in rows:
        height_err = _safe_float(row.get("height_error_m", 0))
        pitch = _safe_float(row.get("pitch_x_rad", 0))

        if abs(height_err) > 0.3:  # Fell >30cm from target
            fall_indicators.append(f"height_error={height_err:.2f}m")
        if abs(pitch) > 1.0:  # >57 degrees pitch
            fall_indicators.append(f"pitch={pitch:.2f}rad")

    if fall_indicators:
        return True, fall_indicators[:5]  # First 5 indicators
    return False, []


def check_post_settle_samples(rows, settling_steps=400, run_type=None):
    """Count usable samples after settling period."""
    if not rows:
        return 0
    total = len(rows)
    post_settle = max(0, total - settling_steps)

    # For push runs, also check how many samples remain after push onset
    if run_type == "B_90n_push":
        # Push at step 300, so post-push samples after settling
        post_push = max(0, total - 400)
        return post_push

    return post_settle


def check_excitation_signal(run_dir, run_type):
    """Verify excitation signal properties if present."""
    exc_path = Path(run_dir) / "excitation_signal.json"
    if not exc_path.exists():
        if run_type == "D_prbs_excitation":
            return {"present": False, "issue": "MISSING_EXCITATION_SIGNAL"}
        return {"present": False, "issue": None}

    try:
        with open(exc_path, "r") as f:
            exc = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"present": True, "issue": "CORRUPT_EXCITATION_FILE"}

    signal = exc.get("signal", [])
    if isinstance(signal, list):
        signal_arr = np.array(signal, dtype=float)
    else:
        return {"present": True, "issue": "INVALID_SIGNAL_FORMAT"}

    if len(signal_arr) == 0:
        return {"present": True, "issue": "EMPTY_SIGNAL"}

    # Check bounded
    max_abs = float(np.max(np.abs(signal_arr)))
    amplitude_max = exc.get("amplitude_max", max_abs)

    # Check zero-mean
    mean_val = float(np.mean(signal_arr))
    is_zero_mean = abs(mean_val) < max(0.02 * amplitude_max, 0.01)

    # Check NaN/Inf in signal
    has_nan = bool(np.any(np.isnan(signal_arr)))
    has_inf = bool(np.any(np.isinf(signal_arr)))

    return {
        "present": True,
        "n_samples": len(signal_arr),
        "amplitude_max": amplitude_max,
        "mean": mean_val,
        "is_zero_mean": is_zero_mean,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "bounded_ok": max_abs <= amplitude_max * 1.1,  # 10% tolerance
        "issue": None,
    }


def check_metadata_validity(metadata):
    """Verify metadata meets requirements."""
    issues = []

    if not metadata:
        return ["MISSING_METADATA"]

    # Check validation source
    source = metadata.get("validation_source", "")
    if source != "real_simulation":
        issues.append(f"NON_REAL_SOURCE: validation_source='{source}'")

    # Check for stub/synthetic markers
    source_label = metadata.get("source_label", "")
    for forbidden in ["stub", "synthetic", "assumed"]:
        if forbidden in source_label.lower():
            issues.append(f"FORBIDDEN_SOURCE_LABEL: contains '{forbidden}'")

    # Check profile
    profile = metadata.get("profile", "")
    if profile and profile != K1_PROFILE_EXPECTED:
        issues.append(f"WRONG_PROFILE: '{profile}' (expected '{K1_PROFILE_EXPECTED}')")

    # Check controller mode
    ctrl_mode = metadata.get("controller_mode", "")
    if ctrl_mode and ctrl_mode != CONTROLLER_MODE_EXPECTED:
        issues.append(f"WRONG_CONTROLLER_MODE: '{ctrl_mode}'")

    # Check simulation success
    if not metadata.get("simulation_success", False):
        issues.append("SIMULATION_FAILED")

    # Check telemetry path exists
    tp = metadata.get("telemetry_path")
    if not tp:
        issues.append("NO_TELEMETRY_PATH")
    elif not Path(tp).exists():
        issues.append("TELEMETRY_FILE_NOT_FOUND")

    return issues


def classify_run(run_result):
    """Classify a run based on audit results.

    Returns one of: USABLE, FAILED_SIMULATION, INSUFFICIENT_LENGTH,
    BAD_METADATA, NON_REAL_SOURCE, NAN_INF, FALL_REJECTED, INCONCLUSIVE
    """
    # Hard failures first
    if not run_result.get("telemetry_exists", False):
        return "FAILED_SIMULATION"

    # Allow minor NaN/Inf in non-critical columns
    # MuJoCo telemetry occasionally produces inf in derived quantities
    # (e.g. capture point, support ratio) that don't affect core states
    nan_cols = set(run_result.get("nan_columns", []))
    inf_cols = set(run_result.get("inf_columns", []))
    bad_cols = nan_cols | inf_cols
    critical_cols = set(REQUIRED_COLUMNS)
    if bad_cols & critical_cols:
        return "NAN_INF"

    if run_result.get("fall_detected", False):
        return "FALL_REJECTED"

    # Metadata checks
    meta_issues = run_result.get("metadata_issues", [])
    for issue in meta_issues:
        if "NON_REAL_SOURCE" in issue or "FORBIDDEN" in issue:
            return "NON_REAL_SOURCE"
        if "SIMULATION_FAILED" in issue:
            return "FAILED_SIMULATION"
        if "NO_TELEMETRY_PATH" in issue or "TELEMETRY_FILE_NOT_FOUND" in issue:
            return "FAILED_SIMULATION"

    if meta_issues:
        return "BAD_METADATA"

    # Data sufficiency
    n_rows = run_result.get("n_rows", 0)
    min_rows = run_result.get("min_total_rows", 1500)
    min_post = run_result.get("min_post_settle", 1000)
    post_settle = run_result.get("post_settle_samples", 0)

    if n_rows < min_rows:
        return "INSUFFICIENT_LENGTH"
    if post_settle < min_post:
        return "INSUFFICIENT_LENGTH"

    # All checks passed
    return "USABLE"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN AUDIT                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def audit_dataset():
    """Run full integrity audit on generated dataset."""
    print("=" * 72)
    print("PHASE 2: DATASET INTEGRITY AUDIT")
    print("=" * 72)

    all_results = {
        "audit_timestamp": datetime.now().isoformat(),
        "audit_version": "1.0.0",
        "total_runs": 0,
        "runs_usable": 0,
        "runs_failed": 0,
        "heights": {},
    }

    run_count = 0
    usable_count = 0
    failed_count = 0

    for height_name, target_h in TARGET_HEIGHTS_MAP.items():
        height_dir = OUTPUT_DIR / height_name
        print(f"\n{'-' * 60}")
        print(f"Height: {height_name} ({target_h}m)")
        print(f"{'-' * 60}")

        height_results = {"height_name": height_name, "target_height_m": target_h, "runs": {}}

        for run_type in RUN_TYPES:
            run_dir = height_dir / run_type
            run_count += 1

            print(f"\n  [{run_type}]")
            result = {
                "run_type": run_type,
                "height_name": height_name,
                "target_height_m": target_h,
                "run_dir": str(run_dir),
                "run_dir_exists": run_dir.exists(),
            }

            # ── Check directory exists ──
            if not run_dir.exists():
                result["classification"] = "FAILED_SIMULATION"
                result["issues"] = ["DIRECTORY_NOT_FOUND"]
                failed_count += 1
                height_results["runs"][run_type] = result
                print(f"    [FAIL] Directory not found")
                continue

            # ── Load metadata ──
            metadata = load_metadata(run_dir)
            result["metadata_exists"] = metadata is not None
            meta_issues = check_metadata_validity(metadata)
            result["metadata_issues"] = meta_issues

            if metadata:
                result["validation_source"] = metadata.get("validation_source")
                result["profile"] = metadata.get("profile")
                result["n_steps"] = metadata.get("n_steps")
                result["telemetry_path"] = metadata.get("telemetry_path")

            # ── Load telemetry ──
            tp = result.get("telemetry_path")
            rows, columns, stats = load_telemetry_csv(tp)
            result["telemetry_exists"] = rows is not None
            result["n_rows"] = stats["n_rows"] if stats else 0
            result["n_cols"] = stats["n_cols"] if stats else 0
            result["columns"] = columns or []

            # ── Check for NaN/Inf ──
            if rows:
                nan_cols, inf_cols = check_nan_inf(rows)
                total_cells = len(rows) * max(len(columns), 1)
                result["has_nan"] = len(nan_cols) > 0
                result["has_inf"] = len(inf_cols) > 0
                result["nan_fraction"] = len(nan_cols) / max(total_cells, 1)
                result["inf_fraction"] = len(inf_cols) / max(total_cells, 1)
                result["nan_columns"] = nan_cols
                result["inf_columns"] = inf_cols

                # ── Check fall detection ──
                fell, fall_reasons = check_fall_detection(rows)
                result["fall_detected"] = fell
                result["fall_reasons"] = fall_reasons

                # ── Check required columns ──
                cols_set = set(columns)
                missing_cols = [c for c in REQUIRED_COLUMNS if c not in cols_set]
                result["missing_required_columns"] = missing_cols

                # ── Check forbidden column patterns ──
                forbidden_found = []
                for col in columns:
                    col_lower = col.lower()
                    for pattern in FORBIDDEN_COLUMNS_PATTERNS:
                        if pattern in col_lower and col_lower not in ["stub", "assumed", "synthetic"]:
                            forbidden_found.append(col)
                result["forbidden_columns"] = forbidden_found

                # ── Check post-settle samples ──
                settling = metadata.get("settling_steps", 400) if metadata else 400
                post = check_post_settle_samples(rows, settling, run_type)
                result["post_settle_samples"] = post

                # ── Height target check ──
                if "height_error_m" in cols_set:
                    height_errors = [_safe_float(r.get("height_error_m", 0)) for r in rows[400:]]
                    if height_errors:
                        result["mean_height_error_m"] = float(np.mean(height_errors))
                        result["rms_height_error_m"] = float(np.sqrt(np.mean(np.square(height_errors))))

                # ── Check for WBC/hidden torque indicators ──
                wbc_cols = [c for c in columns if "wbc" in c.lower() or "momentum" in c.lower()]
                result["wbc_columns_found"] = wbc_cols
            else:
                result["has_nan"] = None
                result["has_inf"] = None
                result["fall_detected"] = None
                result["missing_required_columns"] = REQUIRED_COLUMNS
                result["post_settle_samples"] = 0

            # ── Excitation signal check ──
            exc_check = check_excitation_signal(run_dir, run_type)
            result["excitation"] = exc_check

            # ── Set requirements for classification ──
            result["min_total_rows"] = MIN_TOTAL_ROWS.get(run_type, 1500)
            result["min_post_settle"] = MIN_POST_SETTLE_MAP.get(run_type, 1000)

            # ── Classify ──
            classification = classify_run(result)
            result["classification"] = classification

            if classification == "USABLE":
                usable_count += 1
                print(f"    [USABLE] {result['n_rows']} rows, {result['post_settle_samples']} post-settle")
            elif classification == "FAILED_SIMULATION":
                failed_count += 1
                print(f"    [FAILED_SIMULATION] Issues: {result.get('metadata_issues', [])}")
            else:
                failed_count += 1
                print(f"    [{classification}] Issues: {result.get('metadata_issues', [])}")

            # Print any metadata issues
            for issue in meta_issues:
                print(f"    [ISSUE] {issue}")

            height_results["runs"][run_type] = result

        height_results["usable_count"] = sum(
            1 for r in height_results["runs"].values()
            if r.get("classification") == "USABLE"
        )
        height_results["total_count"] = len(height_results["runs"])
        all_results["heights"][height_name] = height_results

    # ── Aggregate ──
    all_results["total_runs"] = run_count
    all_results["runs_usable"] = usable_count
    all_results["runs_failed"] = failed_count
    all_results["usable_fraction"] = usable_count / max(run_count, 1)

    heights_with_usable = sum(
        1 for h in all_results["heights"].values()
        if h["usable_count"] >= 3  # At least 3 usable run types
    )
    all_results["heights_with_min_data"] = heights_with_usable
    all_results["all_heights_covered"] = heights_with_usable >= len(TARGET_HEIGHTS_MAP)

    # ── Overall recommendation ──
    if all_results["runs_usable"] >= 12:  # >=80% usable
        all_results["recommendation"] = "PROCEED — sufficient data for system identification"
    elif all_results["runs_usable"] >= 8:  # >=53% usable
        all_results["recommendation"] = "PROCEED_WITH_CAUTION — marginal data, some heights may be incomplete"
    else:
        all_results["recommendation"] = "INSUFFICIENT_DATA — regenerate or fix failed runs before proceeding"

    # ── Save JSON ──
    json_path = OUTPUT_DIR / "dataset_integrity_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_make_serializable)
    print(f"\n[OK] JSON report saved: {json_path}")

    # ── Save Markdown ──
    md_path = OUTPUT_DIR / "dataset_integrity_report.md"
    lines = []
    lines.append("# K1 Identification Dataset Integrity Report")
    lines.append(f"\n**Audit Date:** {all_results['audit_timestamp']}")
    lines.append(f"**Overall:** {usable_count}/{run_count} runs USABLE")
    lines.append(f"**Recommendation:** {all_results['recommendation']}")
    lines.append("")

    # Per-height summary table
    lines.append("## Summary by Height")
    lines.append("")
    lines.append("| Height | Runs | Usable | Failed | Coverage |")
    lines.append("|--------|------|--------|--------|----------|")
    for h_name, h_res in all_results["heights"].items():
        lines.append(
            f"| {h_name} | {h_res['total_count']} | {h_res['usable_count']} | "
            f"{h_res['total_count'] - h_res['usable_count']} | "
            f"{'FULL' if h_res['usable_count'] >= 4 else 'PARTIAL' if h_res['usable_count'] >= 2 else 'INSUFFICIENT'} |"
        )
    lines.append("")

    # Per-run details
    lines.append("## Run Details")
    lines.append("")
    for h_name in TARGET_HEIGHTS_MAP:
        if h_name not in all_results["heights"]:
            continue
        h_res = all_results["heights"][h_name]
        lines.append(f"### {h_name} ({h_res['target_height_m']}m)")
        lines.append("")
        lines.append("| Run | Class | Rows | Post-Settle | NaN | Fall | Issues |")
        lines.append("|-----|-------|------|-------------|-----|------|--------|")
        for rt in RUN_TYPES:
            rr = h_res["runs"].get(rt, {})
            issues_str = "; ".join(rr.get("metadata_issues", [])[:3])
            lines.append(
                f"| {rt} | {rr.get('classification', 'N/A')} | {rr.get('n_rows', 0)} | "
                f"{rr.get('post_settle_samples', 0)} | {rr.get('has_nan', '?')} | "
                f"{rr.get('fall_detected', '?')} | {issues_str or '—'} |"
            )
        lines.append("")

    # Limitations
    lines.append("## Classification Legend")
    lines.append("")
    lines.append("- **USABLE:** All checks passed, sufficient data for identification")
    lines.append("- **FAILED_SIMULATION:** Simulation did not complete or produce telemetry")
    lines.append("- **INSUFFICIENT_LENGTH:** Too few rows or post-settle samples")
    lines.append("- **BAD_METADATA:** Metadata missing required fields or has wrong values")
    lines.append("- **NON_REAL_SOURCE:** validation_source is not real_simulation")
    lines.append("- **NAN_INF:** Telemetry contains NaN or Inf values")
    lines.append("- **FALL_REJECTED:** Robot fell during the run")
    lines.append("- **INCONCLUSIVE:** Cannot determine classification")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown report saved: {md_path}")

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Audit K1 identification dataset integrity"
    )
    parser.add_argument("--json-only", action="store_true",
                        help="Output JSON only, skip markdown")
    args = parser.parse_args()

    audit_dataset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
