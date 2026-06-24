#!/usr/bin/env python3
"""Root-cause audit for support-reference reacquisition and pitch-support limit cycle.

Reads telemetry CSV from the G1_sg080 baseline run and determines why the
pitch-support limit cycle persists after a single 90 N / 10-step push.

Usage:
    python scripts/audit_support_reference_reacquisition_root_cause.py
        [--telemetry path/to/telemetry.csv]
        [--output-dir outputs/support_reference_reacquisition_and_pitch_support_limit_cycle_fix/baseline_audit]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

DEG = 180.0 / math.pi

# Windows
FINAL_WINDOW_START = 2500
FINAL_WINDOW_END = 2999
GATE_REOPEN_EXPECTED_AFTER = 500  # steps after push end


def _float_safe(v: str) -> float:
    try:
        return float(v) if v.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _abs_max(values: list[float]) -> float:
    return max(abs(v) for v in values) if values else 0.0


def _gate_bool(v: str) -> float:
    """Parse gate_pass column: 'True' -> 1.0, 'False' -> 0.0."""
    return 1.0 if v.strip().lower() == "true" else 0.0


def audit(telemetry_path: Path, output_dir: Path) -> dict:
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    steps = [_float_safe(r.get("step", 0)) for r in rows]

    def col(name: str, default=0.0) -> list[float]:
        return [_float_safe(r.get(name, default)) for r in rows]

    def indices(start_step: int, end_step: int) -> list[int]:
        return [i for i, s in enumerate(steps) if start_step <= s <= end_step]

    # -----------------------------------------------------------------------
    # 1. Gate analysis: why does the gate close and when does it reopen?
    # -----------------------------------------------------------------------
    gate_pass = [_gate_bool(r.get("outer_loop_gate_pass", "False")) for r in rows]
    gate_reason = [r.get("outer_loop_block_reason", "") for r in rows]

    # Find gate transition timeline
    prev_gate = None
    gate_timeline = []
    for i, (g, s) in enumerate(zip(gate_pass, steps)):
        step = int(s)
        if g != prev_gate:
            gate_timeline.append({
                "step": step, "row": i, "gate_pass": bool(g),
                "reason": gate_reason[i],
            })
            prev_gate = g

    # Compute gate pass fraction per window
    pre_idx = indices(0, 299)
    early_idx = indices(310, 799)
    med_idx = indices(800, 1299)
    late_idx = indices(1300, 1999)
    final_idx = indices(2500, 2999)

    gate_frac = {
        "pre_push": _mean([gate_pass[i] for i in pre_idx]) if pre_idx else 0.0,
        "early_recovery": _mean([gate_pass[i] for i in early_idx]) if early_idx else 0.0,
        "medium_recovery": _mean([gate_pass[i] for i in med_idx]) if med_idx else 0.0,
        "late_recovery": _mean([gate_pass[i] for i in late_idx]) if late_idx else 0.0,
        "final_window": _mean([gate_pass[i] for i in final_idx]) if final_idx else 0.0,
    }

    # Count block reasons in each window
    def reason_counter(idx_list):
        return Counter(gate_reason[i] for i in idx_list)

    block_reasons = {
        "pre_push": dict(reason_counter(pre_idx)) if pre_idx else {},
        "early_recovery": dict(reason_counter(early_idx)) if early_idx else {},
        "medium_recovery": dict(reason_counter(med_idx)) if med_idx else {},
        "late_recovery": dict(reason_counter(late_idx)) if late_idx else {},
        "final_window": dict(reason_counter(final_idx)) if final_idx else {},
    }

    # -----------------------------------------------------------------------
    # 2. Support reference analysis
    # -----------------------------------------------------------------------
    sup_err = col("support_position_error_m")
    sup_err_abs = [abs(v) for v in sup_err]
    outer_loop_sup_err = col("outer_loop_support_error_m")
    outer_loop_dynamic_deg = col("outer_loop_pitch_ref_dynamic_deg")
    support_outer_kp_eff = col("support_outer_loop_kp_effective")
    support_outer_height_scale = col("support_outer_loop_height_scale")
    pitch_ref_total = col("outer_loop_pitch_ref_total_deg")
    pitch_ref_offset_scheduled = col("pitch_ref_offset_scheduled_deg")
    # Note: telemetry column "calibrated_kp_deg_per_m" actually stores the FINAL
    # outer_loop_kp AFTER low-band support override — NOT the raw calibrated value.
    # Use support_outer_loop_kp_effective for the effective Kp instead.
    calibrated_outer_loop_active = [r.get("calibrated_outer_loop_active", "False") for r in rows]
    cal_active_final = any(
        calibrated_outer_loop_active[i].strip().lower() == "true"
        for i in (final_idx or [])
    )

    # Final-window support stats
    f_sup_mean = _mean([outer_loop_sup_err[i] for i in final_idx]) if final_idx else 0.0
    f_sup_rms = _rms([outer_loop_sup_err[i] for i in final_idx]) if final_idx else 0.0
    f_sup_abs_max = _abs_max([outer_loop_sup_err[i] for i in final_idx]) if final_idx else 0.0
    f_dynamic_mean = _mean([outer_loop_dynamic_deg[i] for i in final_idx]) if final_idx else 0.0
    f_dynamic_abs_max = _abs_max([outer_loop_dynamic_deg[i] for i in final_idx]) if final_idx else 0.0
    f_sup_kp_mean = _mean([support_outer_kp_eff[i] for i in final_idx]) if final_idx else 0.0
    f_height_scale_mean = _mean([support_outer_height_scale[i] for i in final_idx]) if final_idx else 0.0
    f_kp_eff_nonzero = sum(1 for i in final_idx if support_outer_kp_eff[i] > 0.001) if final_idx else 0

    # Check: is there any time when dynamic correction > 0.01 deg?
    any_correction_applied = any(abs(outer_loop_dynamic_deg[i]) > 0.01 for i in range(n_rows))

    support_reference_assessment = {
        "final_window_support_error_mean_m": round(f_sup_mean, 6),
        "final_window_support_error_rms_m": round(f_sup_rms, 6),
        "final_window_support_error_abs_max_m": round(f_sup_abs_max, 6),
        "final_window_dynamic_correction_mean_deg": round(f_dynamic_mean, 6),
        "final_window_dynamic_correction_abs_max_deg": round(f_dynamic_abs_max, 6),
        "final_window_support_outer_kp_effective_mean": round(f_sup_kp_mean, 6),
        "final_window_support_outer_height_scale_mean": round(f_height_scale_mean, 6),
        "final_window_calibrated_outer_loop_active": cal_active_final,
        "final_window_kp_effective_nonzero_count": f_kp_eff_nonzero,
        "any_correction_applied_over_full_run": any_correction_applied,
        "support_reference_fixed": abs(f_sup_mean) > 0.01 and f_dynamic_abs_max < 0.001,
        "support_reference_tracks_error": any_correction_applied,
    }

    # -----------------------------------------------------------------------
    # 3. Pitch reference analysis
    # -----------------------------------------------------------------------
    pitch_rad = col("robot_pitch_x")
    pitch_deg = [v * DEG for v in pitch_rad]
    pitch_x_ref = col("pitch_x_ref_rad")

    f_pitch_mean = _mean([pitch_deg[i] for i in final_idx]) if final_idx else 0.0
    f_pitch_rms = _rms([pitch_deg[i] for i in final_idx]) if final_idx else 0.0
    f_pitch_ref_total_mean = _mean([pitch_ref_total[i] for i in final_idx]) if final_idx else 0.0
    f_pitch_scheduled_mean = _mean([pitch_ref_offset_scheduled[i] for i in final_idx]) if final_idx else 0.0
    f_pitch_x_ref_mean = _mean([pitch_x_ref[i] * DEG for i in final_idx]) if final_idx else 0.0

    pitch_reference_assessment = {
        "final_window_pitch_mean_deg": round(f_pitch_mean, 4),
        "final_window_pitch_rms_deg": round(f_pitch_rms, 4),
        "final_window_pitch_ref_total_mean_deg": round(f_pitch_ref_total_mean, 4),
        "final_window_pitch_scheduled_offset_mean_deg": round(f_pitch_scheduled_mean, 4),
        "final_window_pitch_x_ref_mean_deg": round(f_pitch_x_ref_mean, 4),
        "pitch_ref_dominated_by_scheduled_offset": abs(f_pitch_ref_total_mean - f_pitch_scheduled_mean) < 0.01,
        "dynamic_correction_absent": f_dynamic_abs_max < 0.001,
    }

    # -----------------------------------------------------------------------
    # 4. Limit cycle analysis
    # -----------------------------------------------------------------------
    def estimate_freq(signal, window_indices):
        vals = [signal[i] for i in window_indices if i < len(signal)]
        if len(vals) < 50:
            return None
        sign_changes = 0
        for i in range(1, len(vals)):
            if vals[i] * vals[i - 1] < 0:
                sign_changes += 1
        cycles = sign_changes / 2.0
        duration_s = len(vals) * 0.002
        return cycles / duration_s if duration_s > 0 else None

    pitch_freq = estimate_freq(pitch_deg, final_idx) if final_idx else None
    sup_freq = estimate_freq(sup_err, final_idx) if final_idx else None

    limit_cycle_assessment = {
        "pitch_oscillation_freq_hz": round(pitch_freq, 4) if pitch_freq else None,
        "support_oscillation_freq_hz": round(sup_freq, 4) if sup_freq else None,
        "pitch_rms_trend": {
            "early_recovery": round(_rms([pitch_deg[i] for i in early_idx]) if early_idx else 0, 4),
            "medium_recovery": round(_rms([pitch_deg[i] for i in med_idx]) if med_idx else 0, 4),
            "late_recovery": round(_rms([pitch_deg[i] for i in late_idx]) if late_idx else 0, 4),
            "final_window": round(f_pitch_rms, 4),
        },
        "support_rms_trend": {
            "early_recovery": round(_rms([sup_err_abs[i] for i in early_idx]) if early_idx else 0, 4),
            "medium_recovery": round(_rms([sup_err_abs[i] for i in med_idx]) if med_idx else 0, 4),
            "late_recovery": round(_rms([sup_err_abs[i] for i in late_idx]) if late_idx else 0, 4),
            "final_window": round(f_sup_rms, 4),
        },
    }

    # -----------------------------------------------------------------------
    # 5. Root cause determination
    # -----------------------------------------------------------------------
    root_cause = None

    # RC1: support_outer_loop_kp_effective is zero at high height because
    # the low-band support height scale is 0. This is the PRIMARY root cause.
    if f_sup_kp_mean < 0.001 and f_height_scale_mean < 0.001 and cal_active_final:
        root_cause = "SUPPORT_OUTER_LOOP_KP_ZEROED_BY_LOW_BAND_SCALE"
        root_cause_detail = (
            f"The low-band support outer loop height_scale is {f_height_scale_mean:.3f} at 0.480 m height "
            f"(Gaussian centered at 0.320 m with sigma=0.004 m). This zeros the "
            f"effective Kp ({f_sup_kp_mean:.3f} deg/m) even though the calibrated "
            f"outer loop is active (calibrated_outer_loop_active=True). "
            f"The support correction is never applied. "
            f"Root cause: kp = scale * peak_kp (no blend with base Kp) in "
            f"support_outer_loop_low_band.py."
        )

    # RC2: Support reference fixed
    if root_cause is None and support_reference_assessment.get("support_reference_fixed"):
        root_cause = "SUPPORT_REFERENCE_FIXED_NOT_REACQUIRED"
        root_cause_detail = (
            "The support reference does not re-center to the robot's actual position "
            "after the push. The support error persists with mean "
            f"{f_sup_mean:.3f} m."
        )

    # RC3: Gate incorrectly closed
    if root_cause is None and gate_frac.get("final_window", 0) < 0.5:
        primary_reasons = [k for k, v in block_reasons.get("final_window", {}).items() if k != "active"]
        root_cause = "SUPPORT_GATE_FALSE_NEGATIVE_AFTER_PUSH"
        root_cause_detail = (
            f"The support outer loop gate is closed for "
            f"{100 * (1 - gate_frac.get('final_window', 0)):.0f}% of the final window. "
            f"Primary block reasons: {primary_reasons}."
        )

    # RC4: No root cause identified
    if root_cause is None:
        root_cause = "MISSING_TELEMETRY"
        root_cause_detail = "Root cause could not be determined from available telemetry."

    # -----------------------------------------------------------------------
    # 6. Recommendations
    # -----------------------------------------------------------------------
    recommendations = []

    if "SUPPORT_OUTER_LOOP_KP_ZEROED_BY_LOW_BAND_SCALE" in root_cause:
        recommendations.append(
            "Fix the low-band support outer loop blend: change "
            "'kp = scale * peak_kp' to 'kp = (1-scale) * base_kp + scale * peak_kp' "
            "in support_outer_loop_low_band.py. This preserves the low-band boost "
            "near 0.320 m while restoring the calibrated outer loop Kp at tall heights."
        )
        recommendations.append(
            "The I1 candidate (i_support_reference_reacquisition_v1) implements this fix "
            "with low_band_support_blend_with_base=True. Run the focused single-push "
            "diagnostic with the I1 sagittal profile to evaluate."
        )

    recommendations.append(
        "After support correction is restored, evaluate whether Kp or Ki needs tuning "
        "to fully damp the 2.5 Hz pitch-support limit cycle."
    )

    # -----------------------------------------------------------------------
    # Result
    # -----------------------------------------------------------------------
    result = {
        "audit_source": "scripts/audit_support_reference_reacquisition_root_cause.py",
        "telemetry_path": str(telemetry_path),
        "analysis_windows": {
            "pre_push": "0-299",
            "early_recovery": "310-799",
            "medium_recovery": "800-1299",
            "late_recovery": "1300-1999",
            "final_window": "2500-2999",
        },
        "gate_analysis": {
            "gate_pass_fraction_by_window": gate_frac,
            "block_reason_counts_by_window": block_reasons,
            "gate_transition_count": len(gate_timeline),
            "gate_first_closes_due_to_error_too_large": any(
                t["reason"] == "error_too_large" for t in gate_timeline[:5]
            ),
            "gate_active_in_final_window": gate_frac.get("final_window", 0) > 0.99,
        },
        "support_reference_assessment": support_reference_assessment,
        "pitch_reference_assessment": pitch_reference_assessment,
        "limit_cycle_assessment": limit_cycle_assessment,
        "root_cause": root_cause,
        "root_cause_detail": root_cause_detail,
        "recommendations": recommendations,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Root-cause audit for support reference reacquisition."
    )
    parser.add_argument("--telemetry", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    default_dir = (
        root / "outputs" / "g1_sg080_single_90n_10step_push_step300_3000"
    )

    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "support_reference_reacquisition_and_pitch_support_limit_cycle_fix" / "baseline_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.telemetry:
        tele_path = Path(args.telemetry)
    else:
        csvs = sorted(default_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            print(f"ERROR: No telemetry CSV found in {default_dir}")
            sys.exit(1)
        tele_path = csvs[0]

    print(f"Telemetry: {tele_path}")
    result = audit(tele_path, out_dir)

    out_path = out_dir / "root_cause_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Audit written: {out_path}")

    print("\n" + "=" * 70)
    print("SUPPORT REFERENCE REACQUISITION — ROOT-CAUSE AUDIT")
    print("=" * 70)

    ga = result["gate_analysis"]
    print(f"Gate active in final window: {ga['gate_active_in_final_window']}")
    print(f"Gate pass fraction by window: {ga['gate_pass_fraction_by_window']}")

    sa = result["support_reference_assessment"]
    print(f"\nSupport outer Kp effective mean (final): {sa['final_window_support_outer_kp_effective_mean']}")
    print(f"Support outer height scale mean (final): {sa['final_window_support_outer_height_scale_mean']}")
    print(f"Calibrated outer loop active: {sa['final_window_calibrated_outer_loop_active']}")
    print(f"Any correction applied: {sa['any_correction_applied_over_full_run']}")
    print(f"Dynamic correction mean (final): {sa['final_window_dynamic_correction_mean_deg']} deg")

    pa = result["pitch_reference_assessment"]
    print(f"\nPitch ref dominated by scheduled offset: {pa['pitch_ref_dominated_by_scheduled_offset']}")

    lc = result["limit_cycle_assessment"]
    print(f"\nPitch freq: {lc['pitch_oscillation_freq_hz']} Hz")
    print(f"Support freq: {lc['support_oscillation_freq_hz']} Hz")

    print(f"\nROOT CAUSE: {result['root_cause']}")
    print(f"Detail: {result['root_cause_detail']}")

    print(f"\nRecommendations:")
    for r in result["recommendations"]:
        print(f"  - {r}")

    print("=" * 70)


if __name__ == "__main__":
    main()
