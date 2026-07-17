#!/usr/bin/env python3
"""Posture recovery failure audit for G1_sg080 single-push diagnostic.

Reads telemetry CSV and posture_recovery_analysis.json, then performs root-cause
audit to determine why posture did not recover and what failure mechanisms are at play.

Usage:
    python scripts/audit_g1_sg080_posture_recovery_failure.py
        [--telemetry path/to/telemetry.csv]
        [--analysis path/to/posture_recovery_analysis.json]
        [--output-dir outputs/g1_sg080_single_90n_10step_push_step300_3000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

DEG = 180.0 / math.pi


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


def audit(telemetry_path: Path, analysis_path: Path | None, output_dir: Path) -> dict:
    """Run root-cause audit."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    steps = [_float_safe(r.get("step", 0)) for r in rows]

    # Load analysis if available
    analysis = None
    if analysis_path and analysis_path.exists():
        with open(analysis_path, encoding="utf-8") as f:
            analysis = json.load(f)

    def col(name: str, default=0.0) -> list[float]:
        return [_float_safe(r.get(name, default)) for r in rows]

    # Extract signals
    pitch_rad = col("robot_pitch_x")
    pitch_deg = [v * DEG for v in pitch_rad]
    roll_rad = col("robot_roll_y")
    roll_deg = [v * DEG for v in roll_rad]
    yaw_rad = col("robot_yaw_z")
    sup_err_raw = col("support_position_error_m")
    sup_err_abs = [abs(v) for v in sup_err_raw]
    com_z = col("com_z")
    target_com_z = col("target_com_z_m")
    height_error = col("height_error_m")
    hip_yaw_abs = col("hip_yaw_abs_max")
    md_tau_left_raw = col("mode_hip_yaw_div_tau_left_raw")
    md_tau_right_raw = col("mode_hip_yaw_div_tau_right_raw")
    md_div_error = col("mode_hip_yaw_div_error")
    hy_common_error = col("hip_yaw_common_error_rad")
    hy_divergence_error = col("hip_yaw_divergence_error_rad")
    outer_loop_support_error = col("outer_loop_support_error_m")
    outer_loop_support_rate = col("outer_loop_support_error_rate_mps")
    outer_loop_pitch_ref_total_deg = col("outer_loop_pitch_ref_total_deg")
    physics_eq_pitch_ref_deg = col("physics_equivalent_pitch_ref_deg")
    pitch_error_rad = col("pitch_error_x_rad") if "pitch_error_x_rad" in rows[0] else col("pitch_error")
    pitch_x_ref_rad = col("pitch_x_ref_rad")
    outer_loop_gate_pass = col("outer_loop_gate_pass")
    support_outer_loop_kp_effective = col("support_outer_loop_kp_effective")
    support_outer_loop_kd_effective = col("support_outer_loop_kd_effective")

    # Phase analysis windows (by index)
    def indices(start_step, end_step):
        return [i for i, s in enumerate(steps) if start_step <= s <= end_step]

    pre_idx = indices(0, 299)
    early_idx = indices(310, 799)
    med_idx = indices(800, 1299)
    late_idx = indices(1300, 1999)
    final_idx = indices(2500, 2999)

    # -----------------------------------------------------------------------
    # 1. True limit cycle or slow decay?
    # -----------------------------------------------------------------------
    pitch_rms_by_window = {
        "pre_push": _rms([pitch_deg[i] for i in pre_idx]) if pre_idx else 0.0,
        "early_recovery": _rms([pitch_deg[i] for i in early_idx]) if early_idx else 0.0,
        "medium_recovery": _rms([pitch_deg[i] for i in med_idx]) if med_idx else 0.0,
        "late_recovery": _rms([pitch_deg[i] for i in late_idx]) if late_idx else 0.0,
        "final_window": _rms([pitch_deg[i] for i in final_idx]) if final_idx else 0.0,
    }
    sup_rms_by_window = {
        "pre_push": _rms([sup_err_abs[i] for i in pre_idx]) if pre_idx else 0.0,
        "early_recovery": _rms([sup_err_abs[i] for i in early_idx]) if early_idx else 0.0,
        "medium_recovery": _rms([sup_err_abs[i] for i in med_idx]) if med_idx else 0.0,
        "late_recovery": _rms([sup_err_abs[i] for i in late_idx]) if late_idx else 0.0,
        "final_window": _rms([sup_err_abs[i] for i in final_idx]) if final_idx else 0.0,
    }

    # Estimate oscillation frequency from zero-crossings (last 1000 steps)
    def estimate_freq(signal, window_indices):
        vals = [signal[i] for i in window_indices if i < len(signal)]
        if len(vals) < 50:
            return None
        sign_changes = 0
        for i in range(1, len(vals)):
            if vals[i] * vals[i - 1] < 0:
                sign_changes += 1
        # Each sign change is half a cycle
        cycles = sign_changes / 2.0
        duration_s = len(vals) * 0.002  # 500 Hz
        if duration_s > 0:
            return cycles / duration_s
        return None

    pitch_freq = estimate_freq(pitch_deg, final_idx) if final_idx else None
    sup_freq = estimate_freq(sup_err_raw, final_idx) if final_idx else None

    # Cross-correlation: pitch vs support error in final window
    def cross_correlation(a, b, max_lag=100):
        """Simple cross-correlation with lag analysis."""
        a_vals = [a[i] for i in final_idx if i < len(a)] if final_idx else []
        b_vals = [b[i] for i in final_idx if i < len(b)] if final_idx else []
        n = min(len(a_vals), len(b_vals))
        if n < 50:
            return {"error": "too few samples"}
        a_vals = a_vals[:n]
        b_vals = b_vals[:n]
        a_mean = _mean(a_vals)
        b_mean = _mean(b_vals)
        a_std = math.sqrt(_rms([v - a_mean for v in a_vals]))
        b_std = math.sqrt(_rms([v - b_mean for v in b_vals]))
        if a_std < 1e-9 or b_std < 1e-9:
            return {"correlation": 0.0, "lag_est": 0}
        # Compute correlations at several lags to find best alignment
        best_corr = 0.0
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x = a_vals[-lag:]
                y = b_vals[:lag]
            elif lag > 0:
                x = a_vals[:-lag]
                y = b_vals[lag:]
            else:
                x = a_vals
                y = b_vals
            if len(x) < 10:
                continue
            corr = sum((xi - a_mean) * (yi - b_mean) for xi, yi in zip(x, y)) / (len(x) * a_std * b_std)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        return {"correlation": best_corr, "lag_steps": best_lag}

    pitch_sup_xcorr = cross_correlation(pitch_deg, sup_err_raw)

    # Envelope decay rate
    def envelope_decay_rate(signal, start_idx, end_idx, window=200):
        """Estimate decay rate constant from rolling max envelope."""
        vals = [signal[i] for i in range(start_idx, min(end_idx, len(signal)))]
        if len(vals) < 2 * window:
            return None
        peaks = []
        for i in range(window, len(vals) - window, window // 2):
            seg = vals[max(0, i - window // 2):min(len(vals), i + window // 2)]
            peaks.append(max(abs(v) for v in seg))
        if len(peaks) < 3:
            return None
        # Fit exponential decay: ln(peak) vs time
        import math
        t_vals = [i * (window // 2) * 0.002 for i in range(len(peaks))]
        # Only fit positive peaks
        valid = [(t, math.log(max(p, 1e-9))) for t, p in zip(t_vals, peaks) if p > 1e-6]
        if len(valid) < 3:
            return None
        n = len(valid)
        sum_t = sum(t for t, _ in valid)
        sum_ln = sum(ln for _, ln in valid)
        sum_t2 = sum(t * t for t, _ in valid)
        sum_t_ln = sum(t * ln for t, ln in valid)
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-9:
            return None
        slope = (n * sum_t_ln - sum_t * sum_ln) / denom
        return {"decay_rate_constant": slope, "half_life_s": -math.log(2) / slope if slope < 0 else None}

    pitch_decay_rate = envelope_decay_rate(pitch_deg, 310, n_rows)
    sup_decay_rate = envelope_decay_rate(sup_err_abs, 310, n_rows)

    # -----------------------------------------------------------------------
    # 2. Pitch equilibrium feedforward offset?
    # -----------------------------------------------------------------------
    # Compute mean pitch error and mean PFF contribution in final window
    f_pitch_error_mean = _mean([pitch_error_rad[i] * DEG for i in final_idx]) if final_idx else 0.0
    f_pitch_mean = _mean([pitch_deg[i] for i in final_idx]) if final_idx else 0.0
    f_pitch_ref_mean = _mean([pitch_x_ref_rad[i] * DEG for i in final_idx]) if final_idx else 0.0
    f_pff_mean = _mean([physics_eq_pitch_ref_deg[i] for i in final_idx]) if (final_idx and physics_eq_pitch_ref_deg) else 0.0
    # Check if actual pitch follows the ref (which may include PFF offset)
    f_pitch_error_vs_ref = _mean([(pitch_rad[i] - pitch_x_ref_rad[i]) * DEG for i in final_idx]) if final_idx else 0.0

    pitch_eq_offset_assessment = {
        "final_mean_pitch_deg": round(f_pitch_mean, 4),
        "final_mean_pitch_ref_deg": round(f_pitch_ref_mean, 4),
        "final_mean_pitch_error_deg": round(f_pitch_error_mean, 4),
        "final_mean_pff_deg": round(f_pff_mean, 4),
        "final_mean_pitch_error_vs_ref_deg": round(f_pitch_error_vs_ref, 4),
        "pff_maybe_offset": abs(f_pitch_error_vs_ref) > 2.0,
    }

    # -----------------------------------------------------------------------
    # 3. Support outer loop underdamped?
    # -----------------------------------------------------------------------
    f_sup_err_rms = _rms([sup_err_abs[i] for i in final_idx]) if final_idx else 0.0
    f_sup_err_mean = _mean([outer_loop_support_error[i] for i in final_idx]) if final_idx else 0.0
    f_outer_loop_pitch_ref_rms = _rms([outer_loop_pitch_ref_total_deg[i] for i in final_idx]) if final_idx else 0.0
    f_sup_kp_mean = _mean([support_outer_loop_kp_effective[i] for i in final_idx]) if (final_idx and support_outer_loop_kp_effective) else 0.0
    f_sup_kd_mean = _mean([support_outer_loop_kd_effective[i] for i in final_idx]) if (final_idx and support_outer_loop_kd_effective) else 0.0
    f_gate_pass_mean = _mean([outer_loop_gate_pass[i] for i in final_idx]) if (final_idx and outer_loop_gate_pass) else 1.0

    # Check for sustained oscillations: support sign changes
    sup_sign_changes_final = 0
    sup_final_vals = [sup_err_raw[i] for i in final_idx if i < len(sup_err_raw)]
    for i in range(1, len(sup_final_vals)):
        if sup_final_vals[i] * sup_final_vals[i - 1] < 0:
            sup_sign_changes_final += 1

    support_loop_assessment = {
        "final_window_sup_err_rms_m": round(f_sup_err_rms, 6),
        "final_window_sup_err_mean_m": round(f_sup_err_mean, 6),
        "final_window_outer_loop_pitch_ref_rms_deg": round(f_outer_loop_pitch_ref_rms, 4),
        "final_window_sup_kp_mean": round(f_sup_kp_mean, 4),
        "final_window_sup_kd_mean": round(f_sup_kd_mean, 4),
        "final_window_gate_pass_mean": round(f_gate_pass_mean, 4),
        "final_window_sup_sign_changes": sup_sign_changes_final,
        "oscillating_support": sup_sign_changes_final > 5 and f_sup_err_rms > 0.03,
        "suggests_underdamped": sup_sign_changes_final > 10 and f_sup_err_rms > 0.05,
    }

    # -----------------------------------------------------------------------
    # 4. Mode-div authority coupling into sagittal recovery?
    # -----------------------------------------------------------------------
    # Correlate mode-div torque with support/pitch oscillation
    f_md_tau_left_abs_max = _abs_max([md_tau_left_raw[i] for i in final_idx]) if final_idx else 0.0
    f_md_tau_right_abs_max = _abs_max([md_tau_right_raw[i] for i in final_idx]) if final_idx else 0.0
    f_md_div_error_abs_max = _abs_max([md_div_error[i] for i in final_idx]) if final_idx else 0.0

    # Check if mode-div torque varies with support error (cross-correlation)
    md_sup_xcorr = cross_correlation(md_tau_left_raw, sup_err_raw, max_lag=50)

    # Phase relationship between md_tau and sup_err
    md_sup_phase = md_sup_xcorr.get("lag_steps", 0)

    mode_div_coupling_assessment = {
        "final_md_tau_left_abs_max": round(f_md_tau_left_abs_max, 6),
        "final_md_tau_right_abs_max": round(f_md_tau_right_abs_max, 6),
        "final_md_div_error_abs_max": round(f_md_div_error_abs_max, 6),
        "md_tau_vs_sup_correlation": round(md_sup_xcorr.get("correlation", 0), 4),
        "md_tau_vs_sup_lag_steps": md_sup_xcorr.get("lag_steps", 0),
        "md_tau_vs_sup_error": md_sup_xcorr.get("error"),
        "suggests_coupling": abs(md_sup_xcorr.get("correlation", 0)) > 0.3 and f_md_tau_left_abs_max > 1.0,
    }

    # -----------------------------------------------------------------------
    # 5. Yaw/Roll coupling?
    # -----------------------------------------------------------------------
    f_yaw_drift = abs(yaw_rad[final_idx[-1]] - yaw_rad[final_idx[0]]) if (final_idx and len(yaw_rad) > max(final_idx) >= min(final_idx)) else 0.0
    f_roll_abs_max = _abs_max([roll_deg[i] for i in final_idx]) if final_idx else 0.0
    f_yaw_rate_rms = _rms([col("yaw_rate_rad_s")[i] for i in final_idx]) if final_idx else 0.0
    f_hy_common_error_abs_max = _abs_max([hy_common_error[i] for i in final_idx]) if (final_idx and hy_common_error) else 0.0
    f_hy_divergence_error_abs_max = _abs_max([hy_divergence_error[i] for i in final_idx]) if (final_idx and hy_divergence_error) else 0.0

    yaw_roll_assessment = {
        "final_yaw_drift_deg": round(f_yaw_drift * DEG, 4),
        "final_roll_abs_max_deg": round(f_roll_abs_max, 4),
        "final_yaw_rate_rms_deg_s": round(f_yaw_rate_rms * DEG, 4),
        "final_hy_common_error_abs_max_rad": round(f_hy_common_error_abs_max, 6),
        "final_hy_divergence_error_abs_max_rad": round(f_hy_divergence_error_abs_max, 6),
        "yaw_roll_coupling_suspected": f_yaw_drift > 0.05 or f_roll_abs_max > 1.0,
    }

    # -----------------------------------------------------------------------
    # 6. COM height recovery?
    # -----------------------------------------------------------------------
    f_com_z_mean = _mean([com_z[i] for i in final_idx]) if final_idx else 0.0
    f_target_com_z_mean = _mean([target_com_z[i] for i in final_idx]) if (final_idx and target_com_z) else 0.0
    f_height_error_mean = _mean([height_error[i] for i in final_idx]) if (final_idx and height_error) else 0.0
    f_com_z_rms = _rms([com_z[i] for i in final_idx]) if final_idx else 0.0

    # COM height during push vs recovery
    com_z_during_push = _mean([com_z[i] for i in indices(300, 309)]) if indices(300, 309) else 0.0

    com_height_assessment = {
        "final_mean_com_z_m": round(f_com_z_mean, 6),
        "final_target_com_z_mean_m": round(f_target_com_z_mean, 6),
        "final_height_error_mean_m": round(f_height_error_mean, 6),
        "final_com_z_rms_m": round(f_com_z_rms, 6),
        "com_z_during_push_mean_m": round(com_z_during_push, 6),
        "height_recovered": abs(f_height_error_mean) < 0.01,
    }

    # -----------------------------------------------------------------------
    # 7. Is push too severe?
    # -----------------------------------------------------------------------
    # Note: this is a qualitative assessment based on known D baseline results
    push_severity_assessment = {
        "push_magnitude_N": 90.0,
        "push_duration_steps": 10,
        "push_at_step": 300,
        "height_variant": "high_0p480",
        "d_baseline_result_at_step500": "FALL at step 856",
        "g1_sg080_at_step500_result": "SURVIVED but no settle (2000 steps)",
        "previous_assessment": "90N/10-step is severe for tall height; D falls, G1 survives but oscillates",
        "push_severity_verdict": "severe_but_survivable",
        "note": "The push is at the boundary of what the controller can survive. Posture does not fully recover because recovery dynamics are underdamped, not because the push is fatal.",
    }

    # -----------------------------------------------------------------------
    # 8. Failure class determination
    # -----------------------------------------------------------------------
    failure_classes = []

    # Check support outer loop underdamped
    if support_loop_assessment.get("suggests_underdamped"):
        failure_classes.append("LOW_BAND_SUPPORT_OUTER_LOOP_UNDERDAMPED")

    # Check pitch equilibrium feedforward offset
    if pitch_eq_offset_assessment.get("pff_maybe_offset"):
        failure_classes.append("PITCH_EQUILIBRIUM_FEEDFORWARD_OFFSET")

    # Check support position target not reacquired
    if f_sup_err_mean > 0.05 and f_sup_err_rms < 0.03:
        failure_classes.append("SUPPORT_POSITION_TARGET_NOT_REACQUIRED")
    elif f_sup_err_rms > 0.03:
        failure_classes.append("SUPPORT_POSITION_TARGET_NOT_REACQUIRED")

    # Check mode-div sagittal coupling
    if mode_div_coupling_assessment.get("suggests_coupling"):
        failure_classes.append("MODE_DIV_AUTHORITY_SAGITTAL_COUPLING")

    # Check yaw/hip-yaw conflict
    if yaw_roll_assessment.get("yaw_roll_coupling_suspected") and f_hy_common_error_abs_max > 0.05:
        failure_classes.append("YAW_HIP_YAW_CONFLICT")

    # Check COM height recovery limit
    if not com_height_assessment.get("height_recovered") and abs(com_height_assessment["final_height_error_mean_m"]) > 0.01:
        failure_classes.append("COM_HEIGHT_RECOVERY_LIMIT")

    # Add physical push severity as context
    failure_classes.append("PHYSICAL_PUSH_TOO_SEVERE_FOR_CURRENT_CONTROLLER")

    # If nothing specific found
    if len(failure_classes) <= 1:
        failure_classes.append("MISSING_TELEMETRY")

    # -----------------------------------------------------------------------
    # 9. Recommended fix direction
    # -----------------------------------------------------------------------
    recommendations = []

    if "LOW_BAND_SUPPORT_OUTER_LOOP_UNDERDAMPED" in failure_classes:
        recommendations.append(
            "Audit high-height support outer-loop damping. The support error oscillates through "
            "the final window, and the outer-loop pitch reference likely oscillates in phase. "
            "Consider increasing support_outer_loop_kd or reducing support_outer_loop_kp at tall height."
        )
    if "PITCH_EQUILIBRIUM_FEEDFORWARD_OFFSET" in failure_classes:
        recommendations.append(
            "Audit pitch equilibrium feedforward recovery. The PFF term may not return to nominal "
            "after the push, leaving a persistent pitch offset that drives support error. "
            "Check if the physics equilibrium pitch ref resets after disturbance."
        )
    if "SUPPORT_POSITION_TARGET_NOT_REACQUIRED" in failure_classes:
        recommendations.append(
            "Audit support target re-centering/recovery policy. The support reference may not "
            "re-center to the robot's actual position after large displacement. "
            "Consider support recovery mode after large transient."
        )
    if "MODE_DIV_AUTHORITY_SAGITTAL_COUPLING" in failure_classes:
        recommendations.append(
            "Audit G1_sg080 soft_gain/kd trade-off. sg=0.80 provides strong hip-yaw divergence "
            "suppression, but the resulting torque may inject coupling into sagittal dynamics. "
            "Consider intermediate soft_gain values between 0.25 and 0.80 or kd increase."
        )
    if "YAW_HIP_YAW_CONFLICT" in failure_classes:
        recommendations.append(
            "Audit yaw controller / hip-yaw conflict. Yaw drift and hip-yaw common error suggest "
            "the yaw and hip-yaw divergence controllers may be pulling in opposite directions."
        )
    if "COM_HEIGHT_RECOVERY_LIMIT" in failure_classes:
        recommendations.append(
            "Audit COM height recovery. Height error may prevent pitch equilibrium from returning "
            "to nominal, as the pitch-height coupling shifts the equilibrium point."
        )
    recommendations.append(
        "After controller baseline is characterized, consider PPO residual recovery training "
        "to learn corrective actions for the persistent oscillation mode."
    )

    # -----------------------------------------------------------------------
    # Assemble result
    # -----------------------------------------------------------------------
    result = {
        "audit_source": "scripts/audit_g1_sg080_posture_recovery_failure.py",
        "telemetry_path": str(telemetry_path),
        "analysis_used": str(analysis_path) if analysis_path else None,
        "classification_from_analysis": analysis.get("classification") if analysis else None,
        "limit_cycle_or_slow_decay": {
            "pitch_rms_by_window": {k: round(v, 6) for k, v in pitch_rms_by_window.items()},
            "support_rms_by_window": {k: round(v, 6) for k, v in sup_rms_by_window.items()},
            "pitch_oscillation_freq_hz": round(pitch_freq, 4) if pitch_freq else None,
            "support_oscillation_freq_hz": round(sup_freq, 4) if sup_freq else None,
            "pitch_support_cross_correlation": pitch_sup_xcorr,
            "pitch_envelope_decay": pitch_decay_rate,
            "support_envelope_decay": sup_decay_rate,
        },
        "pitch_equilibrium_feedforward_offset": pitch_eq_offset_assessment,
        "support_outer_loop_assessment": support_loop_assessment,
        "mode_div_coupling_assessment": mode_div_coupling_assessment,
        "yaw_roll_assessment": yaw_roll_assessment,
        "com_height_assessment": com_height_assessment,
        "push_severity_assessment": push_severity_assessment,
        "failure_classes": failure_classes,
        "recommendations": recommendations,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Posture recovery failure audit for G1_sg080.")
    parser.add_argument("--telemetry", type=str, default=None)
    parser.add_argument("--analysis", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "g1_sg080_single_90n_10step_push_step300_3000"
    )
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Find telemetry
    if args.telemetry:
        tele_path = Path(args.telemetry)
    else:
        csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            print(f"ERROR: No telemetry CSV found in {out_dir}")
            sys.exit(1)
        tele_path = csvs[0]

    # Find analysis result
    analysis_path = None
    if args.analysis:
        analysis_path = Path(args.analysis)
    else:
        candidate = out_dir / "posture_recovery_analysis.json"
        if candidate.exists():
            analysis_path = candidate

    print(f"Telemetry: {tele_path}")
    print(f"Analysis:  {analysis_path}")
    result = audit(tele_path, analysis_path, out_dir)

    # Write audit result
    audit_path = audit_dir / "audit_result.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Audit written: {audit_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("G1_sg080 POSTURE RECOVERY AUDIT")
    print("=" * 70)

    lim = result["limit_cycle_or_slow_decay"]
    print(f"\nPitch RMS by window: {lim['pitch_rms_by_window']}")
    print(f"Support RMS by window: {lim['support_rms_by_window']}")
    print(f"Pitch freq: {lim['pitch_oscillation_freq_hz']} Hz, Support freq: {lim['support_oscillation_freq_hz']} Hz")
    print(f"Pitch-support cross-correlation: {lim['pitch_support_cross_correlation']}")
    if lim.get("pitch_envelope_decay"):
        print(f"Pitch envelope decay rate: {lim['pitch_envelope_decay'].get('decay_rate_constant')}, half-life: {lim['pitch_envelope_decay'].get('half_life_s')}s")
    if lim.get("support_envelope_decay"):
        print(f"Support envelope decay rate: {lim['support_envelope_decay'].get('decay_rate_constant')}, half-life: {lim['support_envelope_decay'].get('half_life_s')}s")

    pe = result["pitch_equilibrium_feedforward_offset"]
    print(f"\nPitch equilibrium: mean_pitch={pe['final_mean_pitch_deg']:.2f} deg, ref={pe['final_mean_pitch_ref_deg']:.2f} deg, pff={pe['final_mean_pff_deg']:.2f} deg")
    print(f"PFF offset suspected: {pe['pff_maybe_offset']}")

    so = result["support_outer_loop_assessment"]
    print(f"\nSupport loop: rms={so['final_window_sup_err_rms_m']:.4f} m, sign_changes={so['final_window_sup_sign_changes']}, underdamped={so.get('suggests_underdamped')}")

    md = result["mode_div_coupling_assessment"]
    print(f"\nMode-div coupling: corr={md['md_tau_vs_sup_correlation']}, lag={md['md_tau_vs_sup_lag_steps']} steps, coupling={md.get('suggests_coupling')}")

    print(f"\nFailure classes: {result['failure_classes']}")
    print(f"\nRecommendations:")
    for r in result["recommendations"]:
        print(f"  - {r[:100]}...")
    print("=" * 70)


if __name__ == "__main__":
    main()
