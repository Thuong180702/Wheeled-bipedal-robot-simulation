#!/usr/bin/env python3
"""Audit D4/D5 hip-yax > 0.35 root cause: body-yaw-to-hip-yaw coupling.

This script reads K1 D4/D5 focused telemetry and analyzes the coupling
between body yaw drift and hip-yaw joint angles to determine whether
wheel-yaw correction through differential wheel velocity is justified.

Audit dimensions:
1. Body yaw vs hip_yaw_common correlation
2. Body yaw vs hip_yaw_abs_max correlation
3. Support error vs hip_yaw_divergence correlation
4. Wheel velocity differential vs yaw correction
5. Mode-div torque vs hip_yaw reduction

Output:
    outputs/k1_controller_completion/d4_d5_body_yaw_audit/
        analysis/  (metrics and correlation analysis)

Usage:
    python scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "k1_controller_completion" / "d4_d5_body_yaw_audit"
ANALYSIS_DIR = OUT_BASE / "analysis"

# D4/D5 telemetry search paths
TELEMETRY_PATHS = {
    "D4_K1": [
        ROOT / "outputs" / "k1_strict_promotion_validation" / "d4_d5_focused" / "D4_medium_push_low_K1" / "telemetry_1000.csv",
        ROOT / "outputs" / "k1_post_promotion_validation" / "full_step_d" / "D4_medium_push_low" / "telemetry_1000.csv",
    ],
    "D5_K1": [
        ROOT / "outputs" / "k1_strict_promotion_validation" / "d4_d5_focused" / "D5_large_push_high_K1" / "telemetry_1000.csv",
        ROOT / "outputs" / "k1_post_promotion_validation" / "full_step_d" / "D5_large_push_high" / "telemetry_1000.csv",
    ],
}


def find_telemetry(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    # Glob fallback
    for parent in [ROOT / "outputs"]:
        for glob_pattern in ["**/D4*/telemetry*.csv", "**/D5*/telemetry*.csv"]:
            matches = sorted(parent.rglob(glob_pattern))
            if matches:
                return matches[0]
    return None


def load_telemetry(path: Path) -> dict:
    """Load telemetry CSV into dict of numpy arrays."""
    import csv
    import collections
    with open(path) as f:
        reader = csv.DictReader(f)
        data = collections.defaultdict(list)
        for row in reader:
            for key, val in row.items():
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(0.0)
    return {k: np.array(v) for k, v in data.items()}


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 50) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    x_d = x - np.mean(x)
    y_d = y - np.mean(y)
    x_s = np.std(x_d)
    y_s = np.std(y_d)
    if x_s < 1e-12 or y_s < 1e-12:
        return lags, corr
    for i, lag in enumerate(lags):
        if lag < 0:
            corr[i] = np.mean(x_d[:lag] * y_d[-lag:]) / (x_s * y_s)
        elif lag > 0:
            corr[i] = np.mean(x_d[lag:] * y_d[:-lag]) / (x_s * y_s)
        else:
            corr[i] = np.mean(x_d * y_d) / (x_s * y_s)
    return lags, corr


def analyze_coupling(data: dict, label: str, push_end_step: int = 500) -> dict:
    """Analyze body-yaw to hip-yaw coupling."""
    results = {"case": label, "rows": len(next(iter(data.values())))}

    # Body yaw from available signals
    body_yaw_cols = ["body_yaw_z", "yaw_error_rad", "yaw_angle_rad"]
    body_yaw = None
    for col in body_yaw_cols:
        if col in data:
            body_yaw = data[col]
            results["body_yaw_source"] = col
            break

    # Hip-yaw signals
    hy_common = data.get("hip_yaw_common_error_rad", None)
    hy_div = data.get("hip_yaw_divergence_error_rad", None)
    hy_l = data.get("l_hip_yaw_pos", None)
    hy_r = data.get("r_hip_yaw_pos", None)

    # Compute hip_yaw_abs_max from direct telemetry
    if hy_l is not None and hy_r is not None:
        hy_abs = np.maximum(np.abs(hy_l), np.abs(hy_r))
        results["hip_yaw_abs_max"] = float(np.max(hy_abs))
        results["hip_yaw_abs_max_final"] = float(np.max(hy_abs[-200:]))

    if hy_common is not None:
        results["hip_yaw_common_abs_max"] = float(np.max(np.abs(hy_common)))
    if hy_div is not None:
        results["hip_yaw_divergence_abs_max"] = float(np.max(np.abs(hy_div)))

    # Post-push window correlations
    post_push = slice(push_end_step, None)

    if body_yaw is not None and hy_common is not None:
        by = body_yaw[post_push]
        hc = hy_common[post_push]
        if len(by) > 50 and np.isfinite(by).any() and np.isfinite(hc).any():
            lags, corr = cross_correlation(by[:min(500,len(by))], hc[:min(500,len(hc))], max_lag=30)
            max_idx = np.argmax(np.abs(corr))
            results["body_yaw_to_hy_common_max_corr"] = float(corr[max_idx])
            results["body_yaw_to_hy_common_lag_steps"] = int(lags[max_idx])
            results["body_yaw_to_hy_common_corr"] = float(np.corrcoef(by, hc)[0, 1])

    if body_yaw is not None and hy_l is not None and hy_r is not None:
        hy_abs = np.maximum(np.abs(hy_l), np.abs(hy_r))[post_push]
        by = body_yaw[post_push]
        if len(by) > 50 and np.isfinite(by).any():
            results["body_yaw_to_hy_abs_max_corr"] = float(np.corrcoef(by, hy_abs[:len(by)])[0, 1]) if len(by) == len(hy_abs) else 0.0

    # Support error vs hip_yaw_divergence coupling
    support_err = data.get("support_position_error_m", None)
    if support_err is not None and hy_div is not None:
        se = support_err[post_push]
        hd = hy_div[post_push]
        if len(se) > 50 and np.isfinite(se).any():
            results["support_to_hy_div_corr"] = float(np.corrcoef(se, hd[:len(se)])[0, 1]) if len(se) == len(hd) else 0.0

    # Wheel velocity differential vs yaw
    wv_l = data.get("wheel_vel_left_rad_s", data.get("wheel_vel_left", None))
    wv_r = data.get("wheel_vel_right_rad_s", data.get("wheel_vel_right", None))
    if wv_l is not None and wv_r is not None:
        wv_diff = wv_l - wv_r
        results["wheel_vel_diff_rms"] = float(np.sqrt(np.mean(wv_diff[post_push] ** 2)))
        if body_yaw is not None:
            b = body_yaw[post_push][:len(wv_diff[post_push])]
            w = wv_diff[post_push][:len(b)]
            if len(b) > 50:
                results["wheel_diff_to_body_yaw_corr"] = float(np.corrcoef(b, w)[0, 1])

    # Mode-div torque analysis
    tau_mode_div = data.get("mode_div_tau_left", None)
    if tau_mode_div is not None:
        results["mode_div_tau_abs_max"] = float(np.max(np.abs(tau_mode_div)))
        results["mode_div_tau_rms"] = float(np.sqrt(np.mean(tau_mode_div ** 2)))

    return results


def write_audit_report(results_list: list[dict], path: Path):
    """Write structured audit report."""
    lines = []
    lines.append("# D4/D5 Body-Yaw to Hip-Yaw Coupling — Audit Report")
    lines.append("")
    lines.append("## 1. Summary")
    for r in results_list:
        case = r.get("case", "?")
        hy_max = r.get("hip_yaw_abs_max", "N/A")
        hy_common = r.get("hip_yaw_common_abs_max", "N/A")
        hy_div = r.get("hip_yaw_divergence_abs_max", "N/A")
        by_corr = r.get("body_yaw_to_hy_common_corr", "N/A")
        sup_corr = r.get("support_to_hy_div_corr", "N/A")
        lines.append(f"  {case}: hy_max={hy_max}, hy_common={hy_common}, hy_div={hy_div}")
        lines.append(f"    body_yaw<->hy_common corr={by_corr}, support<->hy_div corr={sup_corr}")

    lines.append("")
    lines.append("## 2. Detailed Results")
    for i, r in enumerate(results_list):
        lines.append(f"\n### Run {i+1}: {r.get('case', '?')}")
        for k, v in sorted(r.items()):
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("## 3. Conclusion")

    # Compute aggregate correlation
    by_corrs = [r.get("body_yaw_to_hy_common_corr", 0) for r in results_list if "body_yaw_to_hy_common_corr" in r]
    sup_corrs = [r.get("support_to_hy_div_corr", 0) for r in results_list if "support_to_hy_div_corr" in r]

    if by_corrs:
        mean_by_corr = np.mean(by_corrs)
        if abs(mean_by_corr) > 0.5:
            lines.append(f"BODY_YAW_COUPLING_CONFIRMED: mean body_yaw<->hy_common corr = {mean_by_corr:.3f}")
            lines.append("Body yaw drift IS the dominant driver of hip-yaw exceedance.")
            lines.append("Wheel-yaw correction through differential wheel velocity is justified.")
        else:
            lines.append(f"BODY_YAW_COUPLING_WEAK: mean body_yaw<->hy_common corr = {mean_by_corr:.3f}")
            lines.append("Hip-yaw exceedance may be driven more by divergence than body yaw.")

    if sup_corrs:
        mean_sup_corr = np.mean(sup_corrs)
        if abs(mean_sup_corr) > 0.5:
            lines.append(f"SUPPORT_DIVERGENCE_COUPLING_CONFIRMED: mean support<->hy_div corr = {mean_sup_corr:.3f}")
            lines.append("Support position error couples into hip-yaw divergence.")

    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  [REPORT] Wrote audit report to {path}")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for label, candidates in TELEMETRY_PATHS.items():
        tel_path = find_telemetry(candidates)
        if tel_path is None:
            print(f"[WARN] No telemetry found for {label}", flush=True)
            continue
        print(f"[INFO] Loading {label} telemetry: {tel_path}", flush=True)
        data = load_telemetry(tel_path)
        results = analyze_coupling(data, label)
        all_results.append(results)

        # Print quick summary
        hy = results.get("hip_yaw_abs_max", "N/A")
        by_corr = results.get("body_yaw_to_hy_common_corr", "N/A")
        print(f"  hip_yaw_abs_max={hy}, body_yaw<->hy_common corr={by_corr}", flush=True)

    if all_results:
        write_audit_report(all_results, ANALYSIS_DIR / "body_yaw_coupling_audit.txt")
    else:
        print("[ERROR] No telemetry found for D4/D5 audit", flush=True)


if __name__ == "__main__":
    main()
