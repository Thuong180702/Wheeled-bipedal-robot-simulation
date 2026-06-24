"""Analyze G1_sg080 single-push recovery diagnostic.

Reads telemetry CSV from a G1_sg080 single-push run and computes:

- Basic completion verification
- Push timing verification
- Peak response metrics
- Recovery metrics (post-push)
- Final 500-step stability metrics
- Stability classification verdict

Usage:
    python scripts/analyze_g1_sg080_single_push_recovery.py
        [--telemetry path/to/telemetry.csv]
        [--output-dir outputs/g1_sg080_single_90n_10step_push_recovery_2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIP_YAW_GATE_RAD = 0.35
SUPPORT_RECOVERY_THRESHOLD_M = 0.05
SUPPORT_HIGH_THRESHOLD_M = 0.10
PITCH_RECOVERY_DEG = 5.0
ROLL_RECOVERY_DEG = 5.0
HIP_YAW_RECOVERY_RAD = 0.35
STABILITY_CONSECUTIVE_STEPS = 200
FINAL_WINDOW_STEPS = 500
DEG = 180.0 / math.pi


def _float_safe(v: str) -> float:
    """Parse a telemetry field safely, returning 0.0 for empty/missing."""
    try:
        return float(v) if v.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Classification enum
# ---------------------------------------------------------------------------
CLASSIFICATION = {
    "SINGLE_PUSH_RECOVERY_PASS": "SINGLE_PUSH_RECOVERY_PASS",
    "SINGLE_PUSH_RECOVERY_PASS_WITH_HIP_YAW_LIMIT": "SINGLE_PUSH_RECOVERY_PASS_WITH_HIP_YAW_LIMIT",
    "SINGLE_PUSH_RECOVERY_FAIL_HIP_YAW": "SINGLE_PUSH_RECOVERY_FAIL_HIP_YAW",
    "SINGLE_PUSH_RECOVERY_FAIL_SUPPORT": "SINGLE_PUSH_RECOVERY_FAIL_SUPPORT",
    "SINGLE_PUSH_RECOVERY_FAIL_FALL": "SINGLE_PUSH_RECOVERY_FAIL_FALL",
    "SINGLE_PUSH_RECOVERY_FAIL_UNSTABLE_FINAL_WINDOW": "SINGLE_PUSH_RECOVERY_FAIL_UNSTABLE_FINAL_WINDOW",
    "SINGLE_PUSH_RECOVERY_INCONCLUSIVE": "SINGLE_PUSH_RECOVERY_INCONCLUSIVE",
}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze(telemetry_path: Path) -> dict:
    """Full analysis pipeline for a G1_sg080 single-push run."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    sidecar_path = telemetry_path.with_suffix(".summary.json")
    sidecar = None
    if sidecar_path.exists():
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    else:
        candidate_sidecars = sorted(
            telemetry_path.parent.glob("telemetry_*.summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidate_sidecars:
            with open(candidate_sidecars[0], encoding="utf-8") as f:
                sidecar = json.load(f)

    if not rows:
        return {"error": "Empty telemetry", "classification": CLASSIFICATION["SINGLE_PUSH_RECOVERY_INCONCLUSIVE"]}

    # -----------------------------------------------------------------------
    # 1. Basic completion
    # -----------------------------------------------------------------------
    n = len(rows)
    steps = [_float_safe(r.get("step", 0)) for r in rows]
    requested_steps = int(sidecar.get("requested_steps", 0)) if sidecar else int(max(steps) + 1 if steps else 0)
    actual_steps = int(sidecar.get("actual_steps", n)) if sidecar else n
    terminated_vals = [r.get("terminated", "False") for r in rows]
    terminated = bool(sidecar.get("terminated", False)) if sidecar else any(v.lower() == "true" for v in terminated_vals)
    term_reason = sidecar.get("termination_reason", rows[-1].get("termination_reason", "")) if sidecar else rows[-1].get("termination_reason", "")

    has_nan = False
    # Only check critical numeric scalar columns (skip string/multivalue columns).
    numeric_scalar_cols = [
        "hip_yaw_abs_max", "support_position_error_m", "robot_pitch_x",
        "robot_roll_y", "robot_yaw_z", "com_z", "com_x", "com_y",
        "pitch_rate_rad_s", "roll_rate_rad_s", "yaw_rate_rad_s",
        "mode_hip_yaw_div_tau_left_raw", "mode_hip_yaw_div_tau_right_raw",
        "mode_hip_yaw_div_error", "mode_hip_yaw_div_tau_left",
        "mode_hip_yaw_div_tau_right",
    ]
    for col_name in numeric_scalar_cols:
        for r in rows:
            v = r.get(col_name, "").strip()
            if v and v.lower() in ("nan", "inf", "-inf"):
                # Accept a single row of inf (initialization artifact) per column
                count_bad = sum(1 for rr in rows if rr.get(col_name, "").strip().lower() in ("nan", "inf", "-inf"))
                if count_bad > 1:
                    has_nan = True
                    break
        if has_nan:
            break

    completed_full = (not terminated) and (actual_steps >= requested_steps)

    # -----------------------------------------------------------------------
    # 2. Push timing
    # -----------------------------------------------------------------------
    push_active_col = [r.get("push_active", "False") for r in rows]
    push_on_steps = [i for i, v in enumerate(push_active_col) if v == "True"]
    push_on_step_numbers = [int(steps[i]) for i in push_on_steps]

    n_push_active = len(push_on_steps)
    push_start_step = min(push_on_step_numbers) if push_on_step_numbers else -1
    push_end_step = (max(push_on_step_numbers) + 1) if push_on_step_numbers else -1
    push_duration_actual = n_push_active

    # Count push windows (consecutive active sequences)
    push_windows = 0
    if push_on_step_numbers:
        push_windows = 1
        for i in range(1, len(push_on_step_numbers)):
            if push_on_step_numbers[i] != push_on_step_numbers[i - 1] + 1:
                push_windows += 1

    # -----------------------------------------------------------------------
    # 3. Extract numeric arrays for key fields
    # -----------------------------------------------------------------------
    def col(name: str, default=0.0) -> list[float]:
        return [_float_safe(r.get(name, default)) for r in rows]

    hip_yaw_abs = col("hip_yaw_abs_max")
    sup_err = col("support_position_error_m")
    sup_err_abs = [abs(v) for v in sup_err]
    # Use controller/body-frame pitch_x and roll_y for stability metrics.
    pitch_deg = [v * DEG for v in col("robot_pitch_x")]
    roll_deg = [v * DEG for v in col("robot_roll_y")]
    yaw_rad = col("robot_yaw_z")
    com_z = col("com_z")

    # Mode-div fields
    md_enabled = [r.get("mode_hip_yaw_div_enabled", "False") for r in rows]
    md_tau_left = col("mode_hip_yaw_div_tau_left_raw")
    md_tau_right = col("mode_hip_yaw_div_tau_right_raw")
    md_tau_left_sat = [r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True" for r in rows]
    md_tau_right_sat = [r.get("mode_hip_yaw_div_tau_right_sat", "False") == "True" for r in rows]
    md_tau_left_clipped = col("mode_hip_yaw_div_tau_left")
    md_tau_right_clipped = col("mode_hip_yaw_div_tau_right")
    md_div_error = col("mode_hip_yaw_div_error")
    md_div_rate = col("mode_hip_yaw_div_rate")

    # Pitch / roll rate
    pitch_rate = col("pitch_rate_rad_s")
    roll_rate = col("roll_rate_rad_s")
    yaw_rate = col("yaw_rate_rad_s")

    # Contact
    contact_valid = [r.get("contact_force_valid", "False") for r in rows]

    # -----------------------------------------------------------------------
    # 4. Peak response
    # -----------------------------------------------------------------------
    max_hip_yaw_abs = max(hip_yaw_abs) if hip_yaw_abs else 0.0
    max_sup_err_abs = max(sup_err_abs) if sup_err_abs else 0.0
    max_pitch_abs_deg = max(abs(v) for v in pitch_deg) if pitch_deg else 0.0
    max_roll_abs_deg = max(abs(v) for v in roll_deg) if roll_deg else 0.0
    max_yaw_abs = max(abs(v) for v in yaw_rad) if yaw_rad else 0.0

    # Peak during push window
    push_indices = push_on_steps
    hy_during_push = max(hip_yaw_abs[i] for i in push_indices) if push_indices else 0.0
    sup_during_push = max(sup_err_abs[i] for i in push_indices) if push_indices else 0.0

    # Peak after push (from push_end to end)
    after_push_start = min(n - 1, (push_end_step if push_end_step > 0 else 0))
    after_push_indices = list(range(after_push_start, n))
    hy_after_push = max(hip_yaw_abs[i] for i in after_push_indices) if after_push_indices else 0.0
    sup_after_push = max(sup_err_abs[i] for i in after_push_indices) if after_push_indices else 0.0

    # Support P2P
    support_p2p = max(sup_err_abs) - min(sup_err_abs) if sup_err_abs else 0.0

    # Mode-div torque peak
    md_tau_left_max = max(md_tau_left) if md_tau_left else 0.0
    md_tau_right_max = max(md_tau_right) if md_tau_right else 0.0
    md_tau_left_clipped_max = max(md_tau_left_clipped) if md_tau_left_clipped else 0.0
    md_tau_right_clipped_max = max(md_tau_right_clipped) if md_tau_right_clipped else 0.0
    md_sat_rows = sum(1 for i in range(n) if md_tau_left_sat[i] or md_tau_right_sat[i])

    # -----------------------------------------------------------------------
    # 5. Recovery metrics
    # -----------------------------------------------------------------------
    recovery = {}
    if push_end_step > 0 and push_end_step < n:
        recovery_window = list(range(push_end_step, n))

        # Time to support < 0.05 m
        sup_recovery_05 = None
        for i in recovery_window:
            if sup_err_abs[i] < SUPPORT_RECOVERY_THRESHOLD_M and all(
                sup_err_abs[j] < SUPPORT_RECOVERY_THRESHOLD_M
                for j in range(i, min(i + STABILITY_CONSECUTIVE_STEPS, n))
            ):
                sup_recovery_05 = i - push_end_step
                break

        # Time to support < 0.10 m
        sup_recovery_10 = None
        for i in recovery_window:
            if sup_err_abs[i] < SUPPORT_HIGH_THRESHOLD_M and all(
                sup_err_abs[j] < SUPPORT_HIGH_THRESHOLD_M
                for j in range(i, min(i + STABILITY_CONSECUTIVE_STEPS, n))
            ):
                sup_recovery_10 = i - push_end_step
                break

        # Time to pitch < 5 deg
        pitch_recovery = None
        for i in recovery_window:
            if abs(pitch_deg[i]) < PITCH_RECOVERY_DEG and all(
                abs(pitch_deg[j]) < PITCH_RECOVERY_DEG
                for j in range(i, min(i + STABILITY_CONSECUTIVE_STEPS, n))
            ):
                pitch_recovery = i - push_end_step
                break

        # Time to roll < 5 deg
        roll_recovery = None
        for i in recovery_window:
            if abs(roll_deg[i]) < ROLL_RECOVERY_DEG and all(
                abs(roll_deg[j]) < ROLL_RECOVERY_DEG
                for j in range(i, min(i + STABILITY_CONSECUTIVE_STEPS, n))
            ):
                roll_recovery = i - push_end_step
                break

        # Time to hip_yaw < 0.35 rad (if it exceeded)
        hy_recovery = None
        if max_hip_yaw_abs >= HIP_YAW_RECOVERY_RAD:
            for i in recovery_window:
                if hip_yaw_abs[i] < HIP_YAW_RECOVERY_RAD and all(
                    hip_yaw_abs[j] < HIP_YAW_RECOVERY_RAD
                    for j in range(i, min(i + STABILITY_CONSECUTIVE_STEPS, n))
                ):
                    hy_recovery = i - push_end_step
                    break

        recovery = {
            "push_end_step": push_end_step,
            "sup_recovery_to_005m_steps": sup_recovery_05,
            "sup_recovery_to_010m_steps": sup_recovery_10,
            "pitch_recovery_to_5deg_steps": pitch_recovery,
            "roll_recovery_to_5deg_steps": roll_recovery,
            "hip_yaw_recovery_to_035rad_steps": hy_recovery,
            "hip_yaw_exceeded_gate": max_hip_yaw_abs >= HIP_YAW_RECOVERY_RAD,
        }
    else:
        recovery = {"error": "Cannot determine push_end_step"}

    # -----------------------------------------------------------------------
    # 6. Final 500-step stability
    # -----------------------------------------------------------------------
    final_start = max(0, n - FINAL_WINDOW_STEPS)
    final_indices = list(range(final_start, n))

    f_hy_max = max(hip_yaw_abs[i] for i in final_indices) if final_indices else 0.0
    f_hy_mean = sum(hip_yaw_abs[i] for i in final_indices) / len(final_indices) if final_indices else 0.0
    f_sup_max = max(sup_err_abs[i] for i in final_indices) if final_indices else 0.0
    f_sup_mean = sum(sup_err_abs[i] for i in final_indices) / len(final_indices) if final_indices else 0.0
    f_sup_rms = math.sqrt(sum(v * v for v in [sup_err_abs[i] for i in final_indices]) / len(final_indices)) if final_indices else 0.0
    f_pitch_max = max(abs(pitch_deg[i]) for i in final_indices) if final_indices else 0.0
    f_pitch_mean = sum(abs(pitch_deg[i]) for i in final_indices) / len(final_indices) if final_indices else 0.0
    f_pitch_rms = math.sqrt(sum(pitch_deg[i] ** 2 for i in final_indices) / len(final_indices)) if final_indices else 0.0
    f_roll_max = max(abs(roll_deg[i]) for i in final_indices) if final_indices else 0.0
    f_roll_mean = sum(abs(roll_deg[i]) for i in final_indices) / len(final_indices) if final_indices else 0.0
    f_roll_rms = math.sqrt(sum(roll_deg[i] ** 2 for i in final_indices) / len(final_indices)) if final_indices else 0.0
    f_yaw_drift = abs(yaw_rad[-1] - yaw_rad[final_start]) if final_start < len(yaw_rad) and len(yaw_rad) > final_start else 0.0
    f_yaw_range = max(yaw_rad[final_start:]) - min(yaw_rad[final_start:]) if final_start < len(yaw_rad) else 0.0

    # Divergence error in final window
    f_hy_div_max = max(abs(md_div_error[i]) for i in final_indices) if final_indices and md_div_error else 0.0

    # Final COM drift
    f_com_z_drift = abs(com_z[-1] - com_z[final_start]) if final_start < len(com_z) and len(com_z) > final_start else 0.0

    final_stability = {
        "final_window_start_step": int(steps[final_start]) if final_start < len(steps) else 0,
        "final_window_n_steps": len(final_indices),
        "hip_yaw_abs_max": round(f_hy_max, 6),
        "hip_yaw_abs_mean": round(f_hy_mean, 6),
        "sup_err_abs_max": round(f_sup_max, 6),
        "sup_err_abs_mean": round(f_sup_mean, 6),
        "sup_err_abs_rms": round(f_sup_rms, 6),
        "pitch_abs_max_deg": round(f_pitch_max, 4),
        "pitch_abs_mean_deg": round(f_pitch_mean, 4),
        "pitch_rms_deg": round(f_pitch_rms, 4),
        "roll_abs_max_deg": round(f_roll_max, 4),
        "roll_abs_mean_deg": round(f_roll_mean, 4),
        "roll_rms_deg": round(f_roll_rms, 4),
        "yaw_drift_rad": round(f_yaw_drift, 6),
        "yaw_range_rad": round(f_yaw_range, 6),
        "hip_yaw_divergence_error_abs_max": round(f_hy_div_max, 6),
        "com_z_drift_m": round(f_com_z_drift, 6),
    }

    # -----------------------------------------------------------------------
    # 7. Stability verdict
    # -----------------------------------------------------------------------
    classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_PASS"]

    if terminated:
        classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_FAIL_FALL"]
    elif has_nan:
        classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_INCONCLUSIVE"]
    elif push_windows != 1 or n_push_active != 10:
        classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_INCONCLUSIVE"]
    elif f_sup_max > SUPPORT_HIGH_THRESHOLD_M:
        classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_FAIL_SUPPORT"]
    elif max_hip_yaw_abs >= HIP_YAW_GATE_RAD and f_hy_max >= HIP_YAW_GATE_RAD:
        classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_FAIL_HIP_YAW"]
    elif max_hip_yaw_abs >= HIP_YAW_GATE_RAD > f_hy_max:
        classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_PASS_WITH_HIP_YAW_LIMIT"]

    # Check for instability in final window
    if classification == CLASSIFICATION["SINGLE_PUSH_RECOVERY_PASS"]:
        # Growing trend check: last 200 vs second-to-last 200
        if n >= 600:
            last200 = sup_err_abs[-200:]
            prev200 = sup_err_abs[-400:-200]
            if last200 and prev200:
                last_mean = sum(last200) / len(last200)
                prev_mean = sum(prev200) / len(prev200)
                if last_mean > prev_mean * 1.5 and last_mean > 0.03:
                    classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_FAIL_UNSTABLE_FINAL_WINDOW"]

        # Pitch oscillation check
        if n >= 600:
            last200_p = [abs(pitch_deg[i]) for i in range(n - 200, n)]
            prev200_p = [abs(pitch_deg[i]) for i in range(n - 400, n - 200)]
            if last200_p and prev200_p:
                last_p_mean = sum(last200_p) / len(last200_p)
                prev_p_mean = sum(prev200_p) / len(prev200_p)
                if last_p_mean > prev_p_mean * 1.5 and last_p_mean > 3.0:
                    classification = CLASSIFICATION["SINGLE_PUSH_RECOVERY_FAIL_UNSTABLE_FINAL_WINDOW"]

    # -----------------------------------------------------------------------
    # 8. Assemble full result
    # -----------------------------------------------------------------------
    result = {
        "case_id": "G1_sg080_single_90n_10step_push_high_2000",
        "controller_profile": "G1_sg080",
        "validation_source": "real_simulation",
        "classification": classification,
        "basic_completion": {
            "requested_steps": requested_steps,
            "actual_steps": actual_steps,
            "actual_rows": n,
            "completed_full_duration": completed_full,
            "fall": bool(terminated),
            "early_termination": bool(terminated),
            "termination_reason": term_reason,
            "has_nan_inf": has_nan,
        },
        "push_timing": {
            "requested_start_step": 500,
            "actual_start_step": push_start_step,
            "actual_end_step": push_end_step,
            "requested_duration_steps": 10,
            "actual_active_frames": n_push_active,
            "push_windows": push_windows,
            "push_count_verified": push_windows == 1,
            "push_duration_verified": n_push_active == 10,
        },
        "peak_response": {
            "hip_yaw_abs_max_full_run": round(max_hip_yaw_abs, 6),
            "hip_yaw_abs_max_during_push": round(hy_during_push, 6),
            "hip_yaw_abs_max_after_push": round(hy_after_push, 6),
            "support_error_abs_max": round(max_sup_err_abs, 6),
            "support_error_during_push": round(sup_during_push, 6),
            "support_error_after_push": round(sup_after_push, 6),
            "support_p2p": round(support_p2p, 6),
            "pitch_abs_max_deg": round(max_pitch_abs_deg, 4),
            "roll_abs_max_deg": round(max_roll_abs_deg, 4),
            "yaw_abs_max_rad": round(max_yaw_abs, 6),
        },
        "mode_div_torque": {
            "tau_left_raw_max": round(md_tau_left_max, 6),
            "tau_right_raw_max": round(md_tau_right_max, 6),
            "tau_left_clipped_max": round(md_tau_left_clipped_max, 6),
            "tau_right_clipped_max": round(md_tau_right_clipped_max, 6),
            "saturation_rows": md_sat_rows,
        },
        "recovery_metrics": recovery,
        "final_500_stability": final_stability,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze G1_sg080 single-push recovery diagnostic.")
    parser.add_argument(
        "--telemetry",
        type=str,
        default=None,
        help="Path to telemetry CSV. Default: auto-detect in output dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print classification and key metrics.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "g1_sg080_single_90n_10step_push_recovery_2000"
    )

    if args.telemetry:
        tele_path = Path(args.telemetry)
    else:
        csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            print(f"ERROR: No telemetry CSV found in {out_dir}")
            print("Run scripts/run_g1_sg080_single_90n_10step_push_recovery.py first.")
            sys.exit(1)
        tele_path = csvs[0]

    print(f"Analyzing: {tele_path}")
    result = analyze(tele_path)

    # Write analysis result JSON
    result_path = out_dir / "analysis_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Analysis written: {result_path}")

    # Print summary
    print()
    print("=" * 70)
    print("G1_sg080 SINGLE-PUSH RECOVERY ANALYSIS")
    print("=" * 70)

    b = result["basic_completion"]
    print(f"Completed {b['requested_steps']} steps: {'YES' if b['completed_full_duration'] else 'NO'}")
    print(f"Fall: {'YES' if b['fall'] else 'NO'}")
    print(f"NaN/Inf: {'YES' if b['has_nan_inf'] else 'NO'}")

    p = result["push_timing"]
    print(f"Push windows: {p['push_windows']} (verified count=1: {p['push_count_verified']})")
    print(f"Push active frames: {p['actual_active_frames']} (verified duration=10: {p['push_duration_verified']})")
    print(f"Push start/end: step {p['actual_start_step']} / {p['actual_end_step']}")

    peak = result["peak_response"]
    print(f"Peak hip_yaw_abs: {peak['hip_yaw_abs_max_full_run']:.4f} rad (during push: {peak['hip_yaw_abs_max_during_push']:.4f})")
    print(f"Peak support_err_abs: {peak['support_error_abs_max']:.4f} m")
    print(f"Peak pitch: {peak['pitch_abs_max_deg']:.2f} deg, roll: {peak['roll_abs_max_deg']:.2f} deg")

    fin = result["final_500_stability"]
    print(f"Final500 hip_yaw_abs_max: {fin['hip_yaw_abs_max']:.4f} rad")
    print(f"Final500 sup_err_abs_max: {fin['sup_err_abs_max']:.4f} m")
    print(f"Final500 pitch_abs_max: {fin['pitch_abs_max_deg']:.2f} deg")
    print(f"Final500 roll_abs_max: {fin['roll_abs_max_deg']:.2f} deg")

    md = result["mode_div_torque"]
    print(f"Mode-div tau L/R max raw: {md['tau_left_raw_max']:.3f} / {md['tau_right_raw_max']:.3f} Nm")
    print(f"Mode-div sat rows: {md['saturation_rows']}")

    rec = result["recovery_metrics"]
    if "sup_recovery_to_005m_steps" in rec:
        sup_05 = rec["sup_recovery_to_005m_steps"]
        print(f"Recovery: support<0.05m in {sup_05} steps" if sup_05 is not None else "Recovery: support NEVER below 0.05m")
    if "hip_yaw_recovery_to_035rad_steps" in rec:
        print(f"Hip_yaw recovery: {'YES' if rec['hip_yaw_recovery_to_035rad_steps'] is not None else 'NOT RECOVERED'}")

    print()
    print(f"CLASSIFICATION: {result['classification']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
