#!/usr/bin/env python3
"""
D4/D5 Hip-Yaw Universal Limit Audit Script.

Audit the remaining D4/D5 hip-yaw universal limit across all profiles A/B/C/D.
Computes windowed metrics, reference-vs-error decomposition, torque budget analysis,
and mode decomposition from raw telemetry.

Usage:
    python scripts/audit_d4_d5_hip_yaw_universal_limit.py

Output:
    outputs/d4_d5_hip_yaw_universal_limit_audit/
"""

import csv
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_BASE = os.path.join(
    REPO_ROOT, "outputs", "mode_hip_yaw_div_full_real_validation", "d4_d5_focused_1000"
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "d4_d5_hip_yaw_universal_limit_audit")

# Cases to audit
CASES = {
    "D4_medium_push_low": {"height": "low_0p330", "push_N": 60, "duration": 5, "interval": 150},
    "D5_large_push_high": {"height": "high_0p480", "push_N": 90, "duration": 5, "interval": 200},
}
PROFILES = ["A", "B", "C", "D"]
PROFILE_DIR_NAMES = {p: f"_{p}" for p in PROFILES}


# ---------------------------------------------------------------------------
# Telemetry loading helpers
# ---------------------------------------------------------------------------
def _float_safe(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def load_telemetry(case_id: str, profile: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    dir_name = f"{case_id}_{profile}"
    telemetry_path = os.path.join(TELEMETRY_BASE, dir_name, "telemetry_1000.csv")
    if not os.path.isfile(telemetry_path):
        return [], telemetry_path
    with open(telemetry_path, "r", newline="") as f:
        reader = list(csv.DictReader(f))
    return reader, telemetry_path


def get_float_col(rows: List[Dict], col: str, default: float = 0.0) -> List[float]:
    if not rows or col not in rows[0]:
        return [default] * len(rows)
    return [_float_safe(r.get(col, str(default))) for r in rows]


# ---------------------------------------------------------------------------
# Window definitions
# ---------------------------------------------------------------------------
def compute_windows(
    n_steps: int, push_interval: int, push_duration: int
) -> Dict[str, Tuple[int, int]]:
    push_start = push_interval
    push_end = push_interval + push_duration
    windows = OrderedDict()
    windows["startup"] = (0, min(50, n_steps))
    windows["pre_push"] = (max(0, push_start - 50), push_start)
    windows["push_active"] = (push_start, min(push_end, n_steps))
    windows["post_push"] = (push_end, min(push_end + 50, n_steps))
    windows["peak_window"] = (max(0, push_start), min(push_start + 300, n_steps))
    windows["recovery"] = (min(push_end + 50, n_steps), min(push_end + 400, n_steps))
    windows["final_steady"] = (max(0, n_steps - 100), n_steps)
    return windows


def slice_rows(rows: List[Dict], start: int, end: int) -> List[Dict]:
    return rows[start:end]


# ---------------------------------------------------------------------------
# Window metrics computation
# ---------------------------------------------------------------------------
def compute_window_metrics(
    rows_window: List[Dict],
    col_map: Dict[str, str],
) -> Dict[str, float]:
    """Compute per-window metrics."""
    m: Dict[str, float] = {}

    def g(canonical: str) -> List[float]:
        col = col_map.get(canonical, canonical)
        return get_float_col(rows_window, col)

    n = len(rows_window)
    if n == 0:
        return {}

    # Hip-yaw positions
    l_pos = g("l_hip_yaw_pos")
    r_pos = g("r_hip_yaw_pos")
    l_ref = g("l_hip_yaw_ref")
    r_ref = g("r_hip_yaw_ref")
    l_err = g("l_hip_yaw_error")
    r_err = g("r_hip_yaw_error")
    l_vel = g("l_hip_yaw_vel")
    r_vel = g("r_hip_yaw_vel")

    # Modes
    common = [(l_pos[i] + r_pos[i]) / 2 for i in range(n)]
    divergence = [(l_pos[i] - r_pos[i]) / 2 for i in range(n)]
    common_ref = [(l_ref[i] + r_ref[i]) / 2 for i in range(n)]
    divergence_ref = [(l_ref[i] - r_ref[i]) / 2 for i in range(n)]
    common_err = [(l_err[i] + r_err[i]) / 2 for i in range(n)]
    divergence_err = [(l_err[i] - r_err[i]) / 2 for i in range(n)]

    # Per-joint abs
    hy_abs = [max(abs(l_pos[i]), abs(r_pos[i])) for i in range(n)]

    m["hip_yaw_abs_max"] = max(hy_abs) if hy_abs else 0.0
    m["hip_yaw_abs_mean"] = sum(hy_abs) / n if n else 0.0
    m["common_abs_max"] = max(abs(c) for c in common) if common else 0.0
    m["divergence_abs_max"] = max(abs(d) for d in divergence) if divergence else 0.0
    m["divergence_error_abs_max"] = max(abs(de) for de in divergence_err) if divergence_err else 0.0
    m["divergence_err_rms"] = (
        (sum(de * de for de in divergence_err) / n) ** 0.5 if n else 0.0
    )
    m["common_error_abs_max"] = max(abs(ce) for ce in common_err) if common_err else 0.0

    # Torques
    tau_sl = g("l_hip_yaw_tau_shape_raw")
    tau_sr = g("r_hip_yaw_tau_shape_raw")
    tau_fl = g("l_hip_yaw_tau_shape_final")
    tau_fr = g("r_hip_yaw_tau_shape_final")
    tau_mode_l = g("mode_hip_yaw_div_tau_left")
    tau_mode_r = g("mode_hip_yaw_div_tau_right")
    tau_div_l = g("hip_yaw_div_left")
    tau_div_r = g("hip_yaw_div_right")

    # Mode decomposition of final applied torque
    tau_common_final = [(tau_fl[i] + tau_fr[i]) / 2 for i in range(n)]
    tau_div_final = [(tau_fl[i] - tau_fr[i]) / 2 for i in range(n)]

    m["tau_shape_l_abs_max"] = max(abs(t) for t in tau_sl) if tau_sl else 0.0
    m["tau_shape_r_abs_max"] = max(abs(t) for t in tau_sr) if tau_sr else 0.0
    m["tau_common_final_abs_max"] = max(abs(t) for t in tau_common_final) if tau_common_final else 0.0
    m["tau_div_final_abs_max"] = max(abs(t) for t in tau_div_final) if tau_div_final else 0.0
    m["tau_shape_l_at_peak"] = tau_sl[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 else 0.0
    m["tau_shape_r_at_peak"] = tau_sr[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 else 0.0

    if tau_mode_l and tau_mode_r:
        m["mode_div_tau_l_abs_max"] = max(abs(t) for t in tau_mode_l)
        m["mode_div_tau_r_abs_max"] = max(abs(t) for t in tau_mode_r)
        m["mode_div_tau_l_at_peak"] = tau_mode_l[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 else 0.0
        m["mode_div_tau_r_at_peak"] = tau_mode_r[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 else 0.0

    # Torque sign correctness
    # tau should oppose error: positive error needs negative torque
    if hy_abs and m["hip_yaw_abs_max"] > 0:
        peak_idx = hy_abs.index(m["hip_yaw_abs_max"])
        m["tau_sign_correct_left"] = 1.0 if (l_err[peak_idx] * tau_sl[peak_idx] <= 0) else 0.0
        m["tau_sign_correct_right"] = 1.0 if (r_err[peak_idx] * tau_sr[peak_idx] <= 0) else 0.0

    # Support / yaw coupling
    support_err = g("support_position_error_m")
    yaw_drift = g("yaw_drift_from_initial_rad")
    euler_yaw = g("euler_yaw_z")
    m["support_err_abs_max"] = max(abs(s) for s in support_err) if support_err else 0.0
    m["support_err_at_peak"] = (
        support_err[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 and len(support_err) > hy_abs.index(m["hip_yaw_abs_max"]) else 0.0
    )
    m["yaw_drift_at_peak"] = (
        yaw_drift[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 and len(yaw_drift) > hy_abs.index(m["hip_yaw_abs_max"]) else 0.0
    )
    m["yaw_drift_abs_max"] = max(abs(y) for y in yaw_drift) if yaw_drift else 0.0
    m["euler_yaw_at_peak"] = (
        euler_yaw[hy_abs.index(m["hip_yaw_abs_max"])] if hy_abs and m["hip_yaw_abs_max"] > 0 and len(euler_yaw) > hy_abs.index(m["hip_yaw_abs_max"]) else 0.0
    )

    # Reference-vs-error at peak
    if hy_abs and m["hip_yaw_abs_max"] > 0:
        peak_idx = hy_abs.index(m["hip_yaw_abs_max"])
        ref_mag = abs(l_ref[peak_idx]) + abs(r_ref[peak_idx])
        err_mag = abs(l_err[peak_idx]) + abs(r_err[peak_idx])
        total = ref_mag + err_mag
        m["ref_contrib_pct"] = 100.0 * ref_mag / total if total > 0 else 0.0
        m["err_contrib_pct"] = 100.0 * err_mag / total if total > 0 else 0.0
    else:
        m["ref_contrib_pct"] = 0.0
        m["err_contrib_pct"] = 0.0

    # Timing
    m["peak_step"] = float(hy_abs.index(m["hip_yaw_abs_max"])) if hy_abs and m["hip_yaw_abs_max"] > 0 else 0.0

    return m


# ---------------------------------------------------------------------------
# Reference-vs-error classification
# ---------------------------------------------------------------------------
def classify_ref_vs_error(m: Dict[str, float]) -> str:
    """Classify whether hip-yaw peak is dominated by reference or tracking error."""
    err_pct = m.get("err_contrib_pct", 0.0)
    if err_pct >= 70.0:
        return "TRACKING_ERROR_DOMINANT"
    elif err_pct <= 30.0:
        return "REFERENCE_TOO_LARGE"
    else:
        return "MIXED"


# ---------------------------------------------------------------------------
# Torque budget classification
# ---------------------------------------------------------------------------
def classify_torque_budget(m: Dict[str, float], profile: str, case: str) -> str:
    """Classify whether torque budget is saturated, margin, or unknown."""
    # Mode-div torque max is 2.0 Nm (config)
    mode_div_max = 2.0
    mode_div_used = max(
        m.get("mode_div_tau_l_abs_max", 0.0),
        m.get("mode_div_tau_r_abs_max", 0.0),
    )
    if mode_div_used > 0:
        mode_div_pct = 100.0 * mode_div_used / mode_div_max
        if mode_div_pct >= 95.0:
            return "MODE_DIV_SATURATED"
        elif mode_div_pct >= 80.0:
            return "MODE_DIV_NEAR_SATURATED"

    # Total shape torque margin
    tau_l_max = m.get("tau_shape_l_abs_max", 0.0)
    tau_r_max = m.get("tau_shape_r_abs_max", 0.0)
    total_shape = max(tau_l_max, tau_r_max)
    # Estimated actuator limit ~10 Nm for hip-yaw
    if total_shape >= 9.0:
        return "SHAPE_TORQUE_SATURATED_OR_LIMIT"
    elif total_shape >= 6.0:
        return "SHAPE_TORQUE_HIGH"
    return "TORQUE_WITHIN_MARGIN"


# ---------------------------------------------------------------------------
# Mode decomposition
# ---------------------------------------------------------------------------
def compute_mode_decomposition(
    rows_window: List[Dict], col_map: Dict[str, str]
) -> Dict[str, float]:
    """Decompose hip-yaw into common/divergence modes and their relationships."""
    m: Dict[str, float] = {}
    n = len(rows_window)
    if n == 0:
        return m

    def g(canonical: str) -> List[float]:
        col = col_map.get(canonical, canonical)
        return get_float_col(rows_window, col)

    l_pos = g("l_hip_yaw_pos")
    r_pos = g("r_hip_yaw_pos")
    l_ref = g("l_hip_yaw_ref")
    r_ref = g("r_hip_yaw_ref")
    l_err = g("l_hip_yaw_error")
    r_err = g("r_hip_yaw_error")
    tau_fl = g("l_hip_yaw_tau_shape_final")
    tau_fr = g("r_hip_yaw_tau_shape_final")

    common = [(l_pos[i] + r_pos[i]) / 2 for i in range(n)]
    divergence = [(l_pos[i] - r_pos[i]) / 2 for i in range(n)]
    divergence_ref = [(l_ref[i] - r_ref[i]) / 2 for i in range(n)]
    divergence_err = [(l_err[i] - r_err[i]) / 2 for i in range(n)]
    tau_common = [(tau_fl[i] + tau_fr[i]) / 2 for i in range(n)]
    tau_div = [(tau_fl[i] - tau_fr[i]) / 2 for i in range(n)]

    m["common_mean"] = sum(common) / n
    m["common_abs_max"] = max(abs(c) for c in common)
    m["divergence_mean"] = sum(divergence) / n
    m["divergence_abs_max"] = max(abs(d) for d in divergence)
    m["divergence_ref_abs_max"] = max(abs(dr) for dr in divergence_ref)
    m["divergence_err_abs_max"] = max(abs(de) for de in divergence_err)

    # Ratios
    m["div_common_ratio_max"] = (
        max(
            abs(divergence[i]) / (abs(common[i]) + 1e-8)
            for i in range(n)
        )
        if any(abs(c) > 1e-6 for c in common)
        else float("inf")
    )

    # Torque mode decomposition
    m["tau_common_abs_max"] = max(abs(tc) for tc in tau_common)
    m["tau_div_abs_max"] = max(abs(td) for td in tau_div)
    m["tau_div_common_ratio_max"] = (
        max(
            abs(tau_div[i]) / (abs(tau_common[i]) + 1e-8)
            for i in range(n)
        )
        if any(abs(tc) > 1e-6 for tc in tau_common)
        else float("inf")
    )

    return m


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_audit() -> Dict[str, Any]:
    """Run the full audit and return results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Column name mapping (canonical -> actual column if different)
    COL_MAP: Dict[str, str] = {}

    results: Dict[str, Any] = {
        "metadata": {
            "task": "d4_d5_hip_yaw_universal_limit_audit",
            "output_dir": OUTPUT_DIR,
            "cases": CASES,
            "profiles": PROFILES,
        },
        "windowed_metrics": [],
        "peak_events": [],
        "mode_decomposition": [],
        "torque_budget": [],
        "ref_vs_error": [],
        "classification": {},
    }

    for case_id, case_info in CASES.items():
        height = case_info["height"]
        push_N = case_info["push_N"]
        push_duration = case_info["duration"]
        push_interval = case_info["interval"]

        for profile in PROFILES:
            rows, telemetry_path = load_telemetry(case_id, profile)
            if not rows:
                print(f"  [MISSING] {case_id} / Profile {profile} — no telemetry at {telemetry_path}")
                continue

            n_steps = len(rows)
            print(f"  [OK] {case_id} / Profile {profile}: {n_steps} rows")

            windows = compute_windows(n_steps, push_interval, push_duration)

            # Per-window metrics
            for win_name, (w_start, w_end) in windows.items():
                if w_start >= w_end:
                    continue
                w_rows = slice_rows(rows, w_start, w_end)
                w_metrics = compute_window_metrics(w_rows, COL_MAP)
                w_metrics["case"] = case_id
                w_metrics["profile"] = profile
                w_metrics["height"] = height
                w_metrics["push_N"] = push_N
                w_metrics["window"] = win_name
                w_metrics["window_start"] = w_start
                w_metrics["window_end"] = w_end
                w_metrics["window_rows"] = len(w_rows)
                results["windowed_metrics"].append(w_metrics)

            # Peak event (hip_yaw_abs_max across full run)
            full_metrics = compute_window_metrics(rows, COL_MAP)
            full_metrics["case"] = case_id
            full_metrics["profile"] = profile
            full_metrics["height"] = height
            full_metrics["push_N"] = push_N
            full_metrics["window"] = "full_run"
            full_metrics["window_rows"] = n_steps
            results["windowed_metrics"].append(full_metrics)

            # Peak event record
            peak = {
                "case": case_id,
                "profile": profile,
                "height": height,
                "push_N": push_N,
                "hip_yaw_abs_max": full_metrics.get("hip_yaw_abs_max", 0.0),
                "peak_step": full_metrics.get("peak_step", 0.0),
                "common_at_peak": None,
                "divergence_at_peak": None,
                "l_pos_at_peak": None,
                "r_pos_at_peak": None,
                "l_ref_at_peak": None,
                "r_ref_at_peak": None,
                "tau_shape_l_at_peak": full_metrics.get("tau_shape_l_at_peak", 0.0),
                "tau_shape_r_at_peak": full_metrics.get("tau_shape_r_at_peak", 0.0),
                "support_err_at_peak": full_metrics.get("support_err_at_peak", 0.0),
                "yaw_drift_at_peak": full_metrics.get("yaw_drift_at_peak", 0.0),
                "ref_contrib_pct": full_metrics.get("ref_contrib_pct", 0.0),
                "err_contrib_pct": full_metrics.get("err_contrib_pct", 0.0),
            }

            # Fill in peak values from raw data
            peak_step = int(full_metrics.get("peak_step", 0))
            if peak_step < n_steps:
                r = rows[peak_step]
                peak["l_pos_at_peak"] = _float_safe(r.get("l_hip_yaw_pos", "0"))
                peak["r_pos_at_peak"] = _float_safe(r.get("r_hip_yaw_pos", "0"))
                peak["l_ref_at_peak"] = _float_safe(r.get("l_hip_yaw_ref", "0"))
                peak["r_ref_at_peak"] = _float_safe(r.get("r_hip_yaw_ref", "0"))
                l_p = peak["l_pos_at_peak"]
                r_p = peak["r_pos_at_peak"]
                peak["common_at_peak"] = (l_p + r_p) / 2
                peak["divergence_at_peak"] = (l_p - r_p) / 2
            results["peak_events"].append(peak)

            # Mode decomposition (full run)
            mode_dec = compute_mode_decomposition(rows, COL_MAP)
            mode_dec["case"] = case_id
            mode_dec["profile"] = profile
            results["mode_decomposition"].append(mode_dec)

            # Reference-vs-error
            ref_err = {
                "case": case_id,
                "profile": profile,
                "ref_contrib_pct": full_metrics.get("ref_contrib_pct", 0.0),
                "err_contrib_pct": full_metrics.get("err_contrib_pct", 0.0),
                "classification": classify_ref_vs_error(full_metrics),
            }
            results["ref_vs_error"].append(ref_err)

            # Torque budget
            tb = {
                "case": case_id,
                "profile": profile,
                "tau_shape_l_abs_max": full_metrics.get("tau_shape_l_abs_max", 0.0),
                "tau_shape_r_abs_max": full_metrics.get("tau_shape_r_abs_max", 0.0),
                "mode_div_tau_l_abs_max": full_metrics.get("mode_div_tau_l_abs_max", 0.0),
                "mode_div_tau_r_abs_max": full_metrics.get("mode_div_tau_r_abs_max", 0.0),
                "classification": classify_torque_budget(full_metrics, profile, case_id),
            }
            results["torque_budget"].append(tb)

    return results


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_csv(filename: str, records: List[Dict[str, Any]]):
    if not records:
        return
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"  Wrote {len(records)} rows to {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(results: Dict[str, Any]):
    print("\n" + "=" * 72)
    print("D4/D5 HIP-YAW UNIVERSAL LIMIT AUDIT — SUMMARY")
    print("=" * 72)

    # Reference-vs-error classification
    print("\n--- Reference vs Error Classification ---")
    ref_counts: Dict[str, int] = {}
    for re in results["ref_vs_error"]:
        cl = re["classification"]
        ref_counts[cl] = ref_counts.get(cl, 0) + 1
        print(f"  {re['case']} {re['profile']}: {cl} (ref={re['ref_contrib_pct']:.1f}%, err={re['err_contrib_pct']:.1f}%)")
    print(f"  Total: {ref_counts}")

    # Torque budget
    print("\n--- Torque Budget Classification ---")
    tb_counts: Dict[str, int] = {}
    for tb in results["torque_budget"]:
        cl = tb["classification"]
        tb_counts[cl] = tb_counts.get(cl, 0) + 1
        md = max(tb.get("mode_div_tau_l_abs_max", 0), tb.get("mode_div_tau_r_abs_max", 0))
        sh = max(tb.get("tau_shape_l_abs_max", 0), tb.get("tau_shape_r_abs_max", 0))
        print(f"  {tb['case']} {tb['profile']}: {cl} (shape={sh:.2f}, mode_div={md:.2f})")
    print(f"  Total: {tb_counts}")

    # Peak hip-yaw table
    print("\n--- Peak Hip-Yaw Table ---")
    print(f"  {'Case':<25} {'Profile':<8} {'hip_yaw_abs_max':<18} {'common':<10} {'divergence':<12} {'yaw_drift':<12}")
    print(f"  {'-'*25} {'-'*8} {'-'*18} {'-'*10} {'-'*12} {'-'*12}")
    for pe in results["peak_events"]:
        c = pe.get("common_at_peak") or 0
        d = pe.get("divergence_at_peak") or 0
        print(f"  {pe['case']:<25} {pe['profile']:<8} {pe['hip_yaw_abs_max']:<18.4f} {c:<10.4f} {d:<12.4f} {pe['yaw_drift_at_peak']:<12.4f}")

    # Divergence-dominant check
    print("\n--- Divergence Dominance Check ---")
    for md in results["mode_decomposition"]:
        dc_ratio = md.get("div_common_ratio_max", 0)
        if dc_ratio == float("inf"):
            dom = "DIVERGENCE_ONLY"
        elif dc_ratio > 5:
            dom = "DIVERGENCE_DOMINANT"
        elif dc_ratio > 1:
            dom = "DIVERGENCE_BIASED"
        else:
            dom = "COMMON_DOMINANT"
        print(f"  {md['case']} {md['profile']}: div_common_ratio={dc_ratio:.1f} -> {dom}")

    # Overall universal finding
    print("\n--- Universal Finding ---")
    for case_id in CASES:
        peak_vals = [
            pe["hip_yaw_abs_max"]
            for pe in results["peak_events"]
            if pe["case"] == case_id
        ]
        if peak_vals:
            print(f"  {case_id}: range=[{min(peak_vals):.4f}, {max(peak_vals):.4f}], all above 0.35={all(v > 0.35 for v in peak_vals)}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("D4/D5 Hip-Yaw Universal Limit Audit")
    print(f"Telemetry base: {TELEMETRY_BASE}")
    print(f"Output: {OUTPUT_DIR}\n")

    results = run_audit()

    # Write CSVs
    print("\nWriting output files...")
    write_csv("d4_d5_windowed_metrics.csv", results["windowed_metrics"])
    write_csv("d4_d5_peak_event_table.csv", results["peak_events"])
    write_csv("d4_d5_mode_decomposition_timeseries_summary.csv", results["mode_decomposition"])
    write_csv("d4_d5_torque_budget_summary.csv", results["torque_budget"])
    write_csv("d4_d5_reference_vs_error_summary.csv", results["ref_vs_error"])

    # Write JSON summary
    summary_path = os.path.join(OUTPUT_DIR, "audit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Wrote JSON summary to {summary_path}")

    print_summary(results)
    print(f"\nAudit complete. All files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
