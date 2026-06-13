"""Audit hip-yaw divergence after sign convention fix.

This script performs comprehensive diagnostic analysis to understand
why divergence increased after the hip-yaw sign convention fix.
"""

import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict


def parse_telemetry(telemetry_path: Path) -> tuple[list, list]:
    """Parse telemetry CSV into rows and column names."""
    with open(telemetry_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames if reader.fieldnames else []
    return rows, columns


def make_serializable(obj):
    """Convert numpy/jax types to native Python for JSON serialization."""
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    try:
        float(obj)
        return float(obj)
    except (TypeError, ValueError):
        return obj


def analyze_telemetry(rows: list, height_name: str, columns: list) -> dict:
    """Comprehensive telemetry analysis."""
    metrics = {
        "height": height_name,
        "steps": len(rows),
        "survived": len(rows) >= 5000,
    }

    if not rows:
        return {"error": "No telemetry rows", **metrics}

    # Initialize accumulators
    l_errors, r_errors = [], []
    l_vels, r_vels = [], []
    l_tau_raw, r_tau_raw = [], []
    l_tau_final, r_tau_final = [], []
    divergences, common_modes = [], []
    support_errors = []
    height_errors = []
    roll_vals, pitch_vals = [], []
    wheel_vels = []
    sign_correct_l, sign_correct_r = [], []
    l_torque_raw_vals, r_torque_raw_vals = [], []
    divergences_vel = []

    for row in rows:
        # Sign correctness
        try:
            s_l = row.get("hip_yaw_torque_sign_correct_left", "False") in ("True", "1", "true", "1.0")
            s_r = row.get("hip_yaw_torque_sign_correct_right", "False") in ("True", "1", "true", "1.0")
            sign_correct_l.append(s_l)
            sign_correct_r.append(s_r)
        except:
            pass

        # Hip-yaw errors
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            l_errors.append(l_e)
            r_errors.append(r_e)

            # Divergence: l - r (positive = left ahead)
            div = l_e - r_e
            divergences.append(div)

            # Common mode: (l + r) / 2
            common_modes.append((l_e + r_e) / 2)
        except (ValueError, TypeError):
            pass

        # Hip-yaw velocities
        try:
            l_v = float(row.get("l_hip_yaw_vel", 0))
            r_v = float(row.get("r_hip_yaw_vel", 0))
            l_vels.append(l_v)
            r_vels.append(r_v)
            divergences_vel.append(l_v - r_v)
        except (ValueError, TypeError):
            pass

        # Torques
        try:
            l_t_raw = float(row.get("l_hip_yaw_tau_shape_raw", 0))
            r_t_raw = float(row.get("r_hip_yaw_tau_shape_raw", 0))
            l_t_final = float(row.get("l_hip_yaw_tau_shape_final", 0))
            r_t_final = float(row.get("r_hip_yaw_tau_shape_final", 0))
            l_tau_raw.append(l_t_raw)
            r_tau_raw.append(r_t_raw)
            l_tau_final.append(l_t_final)
            r_tau_final.append(r_t_final)
            l_torque_raw_vals.append(l_t_raw)
            r_torque_raw_vals.append(r_t_raw)
        except (ValueError, TypeError):
            pass

        # Support position error
        try:
            supp_err = float(row.get("support_position_error_m", 0))
            support_errors.append(abs(supp_err))
        except (ValueError, TypeError):
            pass

        # Height error
        try:
            h_err = float(row.get("height_error", 0))
            height_errors.append(abs(h_err))
        except (ValueError, TypeError):
            pass

        # Roll and pitch
        try:
            roll = float(row.get("euler_roll_x", row.get("robot_roll_y", 0)))
            pitch = float(row.get("euler_pitch_y", row.get("robot_pitch_x", 0)))
            roll_vals.append(abs(roll))
            pitch_vals.append(abs(pitch))
        except (ValueError, TypeError):
            pass

        # Wheel velocity
        try:
            wv = float(row.get("l_wheel_velocity", 0))
            wheel_vels.append(abs(wv))
        except (ValueError, TypeError):
            pass

    n = len(rows)

    # Sign correctness
    if sign_correct_l:
        metrics["sign_correct_left_pct"] = sum(sign_correct_l) / len(sign_correct_l) * 100
    if sign_correct_r:
        metrics["sign_correct_right_pct"] = sum(sign_correct_r) / len(sign_correct_r) * 100

    # Error metrics
    if l_errors:
        metrics["l_error_max"] = max(abs(e) for e in l_errors)
        metrics["l_error_final"] = l_errors[-1]
        metrics["l_error_rms"] = (sum(e**2 for e in l_errors) / len(l_errors)) ** 0.5
    if r_errors:
        metrics["r_error_max"] = max(abs(e) for e in r_errors)
        metrics["r_error_final"] = r_errors[-1]
        metrics["r_error_rms"] = (sum(e**2 for e in r_errors) / len(r_errors)) ** 0.5
    if l_errors and r_errors:
        metrics["hip_yaw_abs_max"] = max(max(abs(e) for e in l_errors), max(abs(e) for e in r_errors))

    # Divergence metrics
    if divergences:
        metrics["divergence_max"] = max(abs(d) for d in divergences)
        metrics["divergence_final"] = divergences[-1]
        metrics["divergence_rms"] = (sum(d**2 for d in divergences) / len(divergences)) ** 0.5

    # Common mode metrics
    if common_modes:
        metrics["common_mode_max"] = max(abs(c) for c in common_modes)
        metrics["common_mode_final"] = common_modes[-1]
        metrics["common_mode_rms"] = (sum(c**2 for c in common_modes) / len(common_modes)) ** 0.5

    # Velocity metrics
    if divergences_vel:
        metrics["divergence_vel_max"] = max(abs(v) for v in divergences_vel)
        metrics["divergence_vel_rms"] = (sum(v**2 for v in divergences_vel) / len(divergences_vel)) ** 0.5

    # Torque metrics
    if l_tau_raw:
        metrics["l_tau_raw_max"] = max(abs(t) for t in l_tau_raw)
        metrics["l_tau_raw_rms"] = (sum(t**2 for t in l_tau_raw) / len(l_tau_raw)) ** 0.5
    if r_tau_raw:
        metrics["r_tau_raw_max"] = max(abs(t) for t in r_tau_raw)
        metrics["r_tau_raw_rms"] = (sum(t**2 for t in r_tau_raw) / len(r_tau_raw)) ** 0.5
    if l_tau_final:
        metrics["l_tau_final_max"] = max(abs(t) for t in l_tau_final)
        metrics["l_tau_final_rms"] = (sum(t**2 for t in l_tau_final) / len(l_tau_final)) ** 0.5
    if r_tau_final:
        metrics["r_tau_final_max"] = max(abs(t) for t in r_tau_final)
        metrics["r_tau_final_rms"] = (sum(t**2 for t in r_tau_final) / len(r_tau_final)) ** 0.5

    # Torque decomposition: common and divergence modes
    if l_tau_raw and r_tau_raw:
        common_torques = [(l + r) / 2 for l, r in zip(l_tau_raw, r_tau_raw)]
        div_torques = [(l - r) / 2 for l, r in zip(l_tau_raw, r_tau_raw)]
        metrics["common_torque_max"] = max(abs(t) for t in common_torques)
        metrics["div_torque_max"] = max(abs(t) for t in div_torques)
        metrics["common_torque_rms"] = (sum(t**2 for t in common_torques) / len(common_torques)) ** 0.5
        metrics["div_torque_rms"] = (sum(t**2 for t in div_torques) / len(div_torques)) ** 0.5

    # Support, height, roll, pitch
    if support_errors:
        metrics["support_error_max"] = max(support_errors)
        metrics["support_error_rms"] = (sum(e**2 for e in support_errors) / len(support_errors)) ** 0.5
    if height_errors:
        metrics["height_error_max"] = max(height_errors)
        metrics["height_error_rms"] = (sum(e**2 for e in height_errors) / len(height_errors)) ** 0.5
    if roll_vals:
        metrics["roll_max"] = max(roll_vals)
        metrics["roll_rms"] = (sum(v**2 for v in roll_vals) / len(roll_vals)) ** 0.5
    if pitch_vals:
        metrics["pitch_max"] = max(pitch_vals)
        metrics["pitch_rms"] = (sum(v**2 for v in pitch_vals) / len(pitch_vals)) ** 0.5
    if wheel_vels:
        metrics["wheel_vel_max"] = max(wheel_vels)
        metrics["wheel_vel_rms"] = (sum(v**2 for v in wheel_vels) / len(wheel_vels)) ** 0.5

    return metrics


def compute_event_order(rows: list) -> dict:
    """Compute first occurrence of key events."""
    events = {
        "hip_yaw_abs_gt_0.03": None,
        "hip_yaw_abs_gt_0.07": None,
        "hip_yaw_abs_gt_0.10": None,
        "divergence_gt_0.03": None,
        "divergence_gt_0.07": None,
        "divergence_gt_0.10": None,
        "common_gt_0.03": None,
        "support_gt_0.05": None,
        "height_gt_0.02": None,
        "roll_gt_0.05": None,
        "pitch_gt_0.15": None,
    }

    for i, row in enumerate(rows):
        # Hip-yaw abs
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            abs_max = max(abs(l_e), abs(r_e))
            if events["hip_yaw_abs_gt_0.03"] is None and abs_max > 0.03:
                events["hip_yaw_abs_gt_0.03"] = i
            if events["hip_yaw_abs_gt_0.07"] is None and abs_max > 0.07:
                events["hip_yaw_abs_gt_0.07"] = i
            if events["hip_yaw_abs_gt_0.10"] is None and abs_max > 0.10:
                events["hip_yaw_abs_gt_0.10"] = i
        except:
            pass

        # Divergence
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            div = abs(l_e - r_e)
            if events["divergence_gt_0.03"] is None and div > 0.03:
                events["divergence_gt_0.03"] = i
            if events["divergence_gt_0.07"] is None and div > 0.07:
                events["divergence_gt_0.07"] = i
            if events["divergence_gt_0.10"] is None and div > 0.10:
                events["divergence_gt_0.10"] = i
        except:
            pass

        # Common mode
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            common = abs((l_e + r_e) / 2)
            if events["common_gt_0.03"] is None and common > 0.03:
                events["common_gt_0.03"] = i
        except:
            pass

        # Support
        try:
            supp = abs(float(row.get("support_position_error_m", 0)))
            if events["support_gt_0.05"] is None and supp > 0.05:
                events["support_gt_0.05"] = i
        except:
            pass

        # Height
        try:
            h = abs(float(row.get("height_error", 0)))
            if events["height_gt_0.02"] is None and h > 0.02:
                events["height_gt_0.02"] = i
        except:
            pass

        # Roll
        try:
            roll = abs(float(row.get("euler_roll_x", row.get("robot_roll_y", 0))))
            if events["roll_gt_0.05"] is None and roll > 0.05:
                events["roll_gt_0.05"] = i
        except:
            pass

        # Pitch
        try:
            pitch = abs(float(row.get("euler_pitch_y", row.get("robot_pitch_x", 0))))
            if events["pitch_gt_0.15"] is None and pitch > 0.15:
                events["pitch_gt_0.15"] = i
        except:
            pass

    return events


def compute_torque_sign_analysis(rows: list) -> dict:
    """Analyze whether torques oppose errors."""
    l_correct, r_correct = 0, 0
    l_opposes, r_opposes = 0, 0
    l_total, r_total = 0, 0

    for row in rows:
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            l_t = float(row.get("l_hip_yaw_tau_shape_raw", 0))
            r_t = float(row.get("r_hip_yaw_tau_shape_raw", 0))

            # Check sign: torque should oppose error
            # Positive error -> positive torque (increases position)
            if abs(l_e) > 1e-6:
                l_total += 1
                if l_e * l_t > 0:  # Same sign = correct (opposes velocity tendency)
                    l_correct += 1
                else:
                    l_opposes += 1

            if abs(r_e) > 1e-6:
                r_total += 1
                if r_e * r_t > 0:
                    r_correct += 1
                else:
                    r_opposes += 1
        except:
            pass

    return {
        "l_correct_count": l_correct,
        "l_opposes_count": l_opposes,
        "l_total_nonzero": l_total,
        "r_correct_count": r_correct,
        "r_opposes_count": r_opposes,
        "r_total_nonzero": r_total,
        "l_correct_pct": l_correct / max(l_total, 1) * 100,
        "r_correct_pct": r_correct / max(r_total, 1) * 100,
    }


def compute_mode_decomposition(rows: list) -> dict:
    """Compute mode decomposition metrics."""
    common_errors, div_errors = [], []
    common_vels, div_vels = [], []
    common_torques, div_torques = [], []

    for row in rows:
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            l_v = float(row.get("l_hip_yaw_vel", 0))
            r_v = float(row.get("r_hip_yaw_vel", 0))
            l_t = float(row.get("l_hip_yaw_tau_shape_raw", 0))
            r_t = float(row.get("r_hip_yaw_tau_shape_raw", 0))

            common_errors.append((l_e + r_e) / 2)
            div_errors.append(l_e - r_e)
            common_vels.append((l_v + r_v) / 2)
            div_vels.append(l_v - r_v)
            common_torques.append((l_t + r_t) / 2)
            div_torques.append(l_t - r_t)
        except:
            pass

    result = {}

    if common_errors:
        result["common_error_max"] = max(abs(e) for e in common_errors)
        result["common_error_final"] = common_errors[-1]
        result["common_error_rms"] = (sum(e**2 for e in common_errors) / len(common_errors)) ** 0.5

    if div_errors:
        result["div_error_max"] = max(abs(e) for e in div_errors)
        result["div_error_final"] = div_errors[-1]
        result["div_error_rms"] = (sum(e**2 for e in div_errors) / len(div_errors)) ** 0.5

    if common_vels:
        result["common_vel_max"] = max(abs(v) for v in common_vels)
        result["common_vel_rms"] = (sum(v**2 for v in common_vels) / len(common_vels)) ** 0.5

    if div_vels:
        result["div_vel_max"] = max(abs(v) for v in div_vels)
        result["div_vel_rms"] = (sum(v**2 for v in div_vels) / len(div_vels)) ** 0.5

    if common_torques:
        result["common_torque_max"] = max(abs(t) for t in common_torques)
        result["common_torque_rms"] = (sum(t**2 for t in common_torques) / len(common_torques)) ** 0.5

    if div_torques:
        result["div_torque_max"] = max(abs(t) for t in div_torques)
        result["div_torque_rms"] = (sum(t**2 for t in div_torques) / len(div_torques)) ** 0.5

    # Check if torques oppose errors
    # Common: torque should oppose common_error
    # Divergence: antisymmetric torque should oppose divergence_error
    if common_errors and common_torques:
        n = len(common_errors)
        common_opposes = sum(1 for e, t in zip(common_errors, common_torques) if e * t > 0)
        result["common_torque_opposes_error_pct"] = common_opposes / n * 100

    if div_errors and div_torques:
        n = len(div_errors)
        # For divergence: torque = +k*(l-r) so left gets +k*div, right gets -k*div
        # If div is positive (left > right), left needs positive torque
        div_opposes = sum(1 for e, t in zip(div_errors, div_torques) if e * t > 0)
        result["div_torque_opposes_error_pct"] = div_opposes / n * 100

    return result


def compute_correlations(rows: list) -> dict:
    """Compute correlations between divergence and other signals."""
    divergences = []
    support_errors = []
    height_errors = []
    roll_vals = []
    pitch_vals = []
    wheel_vels = []
    l_vels = []
    r_vels = []

    for row in rows:
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            div = l_e - r_e
            divergences.append(div)

            support_errors.append(float(row.get("support_position_error_m", 0)))
            height_errors.append(float(row.get("height_error", 0)))
            roll_vals.append(float(row.get("euler_roll_x", row.get("robot_roll_y", 0))))
            pitch_vals.append(float(row.get("euler_pitch_y", row.get("robot_pitch_x", 0))))
            wheel_vels.append(float(row.get("l_wheel_velocity", 0)))
            l_vels.append(float(row.get("l_hip_yaw_vel", 0)))
            r_vels.append(float(row.get("r_hip_yaw_vel", 0)))
        except:
            pass

    def pearson_corr(x, y):
        n = len(x)
        if n < 2:
            return 0.0
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den = (sum((xi - x_mean)**2 for xi in x) * sum((yi - y_mean)**2 for yi in y)) ** 0.5
        if den < 1e-10:
            return 0.0
        return num / den

    result = {}

    signals = {
        "support_error": support_errors,
        "height_error": height_errors,
        "roll": roll_vals,
        "pitch": pitch_vals,
        "wheel_vel": wheel_vels,
        "l_hip_yaw_vel": l_vels,
        "r_hip_yaw_vel": r_vels,
    }

    for name, vals in signals.items():
        if len(vals) == len(divergences) and len(divergences) > 10:
            corr = pearson_corr(divergences, vals)
            result[f"corr_div_{name}"] = round(corr, 4)

    return result


def analyze_reference_initialization(rows: list) -> dict:
    """Analyze reference and initialization."""
    result = {}

    if not rows:
        return result

    # First row (initial state)
    first = rows[0]
    last = rows[-1]

    try:
        result["l_ref_initial"] = float(first.get("l_hip_yaw_ref", 0))
        result["r_ref_initial"] = float(first.get("r_hip_yaw_ref", 0))
        result["l_pos_initial"] = float(first.get("l_hip_yaw_pos", 0))
        result["r_pos_initial"] = float(first.get("r_hip_yaw_pos", 0))
        result["l_error_initial"] = float(first.get("l_hip_yaw_error", 0))
        result["r_error_initial"] = float(first.get("r_hip_yaw_error", 0))

        result["l_ref_final"] = float(last.get("l_hip_yaw_ref", 0))
        result["r_ref_final"] = float(last.get("r_hip_yaw_ref", 0))
        result["l_pos_final"] = float(last.get("l_hip_yaw_pos", 0))
        result["r_pos_final"] = float(last.get("r_hip_yaw_pos", 0))
        result["l_error_final"] = float(last.get("l_hip_yaw_error", 0))
        result["r_error_final"] = float(last.get("r_hip_yaw_error", 0))

        # Check if refs are symmetric
        result["ref_symmetric"] = abs(result["l_ref_initial"] - result["r_ref_initial"]) < 1e-6

        # Check if initial divergence is present
        result["initial_divergence"] = result["l_error_initial"] - result["r_error_initial"]
        result["initial_divergence_abs"] = abs(result["initial_divergence"])

        # Direction of drift
        result["final_divergence"] = result["l_error_final"] - result["r_error_final"]
        result["final_divergence_abs"] = abs(result["final_divergence"])
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    output_dir = Path("outputs/hip_yaw_divergence_after_sign_fix_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Telemetry files (post-fix)
    post_fix_files = {
        "nominal": "outputs/hip_yaw_sign_convention_fix/step_e_5000/nominal_5000_telemetry.csv",
        "low_0p300": "outputs/hip_yaw_sign_convention_fix/step_e_5000/low_0p300_5000_telemetry.csv",
        "high_0p480": "outputs/hip_yaw_sign_convention_fix/step_e_5000/high_0p480_5000_telemetry.csv",
    }

    # Pre-fix telemetry
    pre_fix_files = {
        "nominal": "outputs/step_e_best_current_profile_5000_eval/nominal_5000_telemetry.csv",
        "low_0p300": "outputs/step_e_best_current_profile_5000_eval/low_0p300_5000_telemetry.csv",
        "high_0p480": "outputs/step_e_best_current_profile_5000_eval/high_0p480_5000_telemetry.csv",
    }

    results = {
        "post_fix": {},
        "pre_fix": {},
        "comparison": {},
    }

    all_events = {}
    all_mode_decomp = {}
    all_ref_init = {}
    all_correlations = {}
    all_torque_sign = {}

    # Analyze post-fix telemetry
    print("\n" + "="*60)
    print("ANALYZING POST-FIX TELEMETRY")
    print("="*60)

    for height, path in post_fix_files.items():
        print(f"\n{height.upper()}:")
        if not Path(path).exists():
            print(f"  WARNING: {path} not found")
            continue

        rows, columns = parse_telemetry(Path(path))
        print(f"  Steps: {len(rows)}")

        # Basic metrics
        metrics = analyze_telemetry(rows, height, columns)
        results["post_fix"][height] = metrics
        print(f"  Sign Correct L: {metrics.get('sign_correct_left_pct', 0):.1f}%")
        print(f"  Sign Correct R: {metrics.get('sign_correct_right_pct', 0):.1f}%")
        print(f"  Divergence RMS: {metrics.get('divergence_rms', 0):.4f} rad")
        print(f"  Divergence Max: {metrics.get('divergence_max', 0):.4f} rad")
        print(f"  Common Mode RMS: {metrics.get('common_mode_rms', 0):.4f} rad")

        # Event order
        events = compute_event_order(rows)
        all_events[height] = events
        print(f"  First divergence >0.07: step {events.get('divergence_gt_0.07', 'N/A')}")

        # Mode decomposition
        mode_decomp = compute_mode_decomposition(rows)
        all_mode_decomp[height] = mode_decomp
        print(f"  Div Torque RMS: {mode_decomp.get('div_torque_rms', 0):.4f} Nm")
        print(f"  Div Torque Max: {mode_decomp.get('div_torque_max', 0):.4f} Nm")

        # Reference initialization
        ref_init = analyze_reference_initialization(rows)
        all_ref_init[height] = ref_init
        print(f"  Initial divergence: {ref_init.get('initial_divergence', 0):.4f} rad")
        print(f"  Final divergence: {ref_init.get('final_divergence', 0):.4f} rad")

        # Correlations
        correlations = compute_correlations(rows)
        all_correlations[height] = correlations
        print(f"  Corr with support_error: {correlations.get('corr_div_support_error', 0):.3f}")
        print(f"  Corr with roll: {correlations.get('corr_div_roll', 0):.3f}")
        print(f"  Corr with l_hip_yaw_vel: {correlations.get('corr_div_l_hip_yaw_vel', 0):.3f}")

        # Torque sign analysis
        torque_sign = compute_torque_sign_analysis(rows)
        all_torque_sign[height] = torque_sign
        print(f"  L torque correct: {torque_sign.get('l_correct_pct', 0):.1f}%")
        print(f"  R torque correct: {torque_sign.get('r_correct_pct', 0):.1f}%")

        # Copy telemetry to output
        shutil.copy(path, output_dir / f"{height}_post_fix_telemetry.csv")

    # Analyze pre-fix telemetry
    print("\n" + "="*60)
    print("ANALYZING PRE-FIX TELEMETRY")
    print("="*60)

    for height, path in pre_fix_files.items():
        print(f"\n{height.upper()}:")
        if not Path(path).exists():
            print(f"  WARNING: {path} not found")
            continue

        rows, columns = parse_telemetry(Path(path))
        print(f"  Steps: {len(rows)}")

        metrics = analyze_telemetry(rows, height, columns)
        results["pre_fix"][height] = metrics
        print(f"  Sign Correct L: {metrics.get('sign_correct_left_pct', 0):.1f}%")
        print(f"  Sign Correct R: {metrics.get('sign_correct_right_pct', 0):.1f}%")
        print(f"  Divergence RMS: {metrics.get('divergence_rms', 0):.4f} rad")
        print(f"  Divergence Max: {metrics.get('divergence_max', 0):.4f} rad")

        shutil.copy(path, output_dir / f"{height}_pre_fix_telemetry.csv")

    # Compare pre vs post
    print("\n" + "="*60)
    print("PRE-FIX vs POST-FIX COMPARISON")
    print("="*60)

    for height in ["nominal", "low_0p300", "high_0p480"]:
        pre = results["pre_fix"].get(height, {})
        post = results["post_fix"].get(height, {})

        if not pre or not post:
            continue

        comparison = {
            "sign_correct_left": {
                "pre": pre.get("sign_correct_left_pct", 0),
                "post": post.get("sign_correct_left_pct", 0),
                "delta": post.get("sign_correct_left_pct", 0) - pre.get("sign_correct_left_pct", 0),
            },
            "sign_correct_right": {
                "pre": pre.get("sign_correct_right_pct", 0),
                "post": post.get("sign_correct_right_pct", 0),
                "delta": post.get("sign_correct_right_pct", 0) - pre.get("sign_correct_right_pct", 0),
            },
            "divergence_rms": {
                "pre": pre.get("divergence_rms", 0),
                "post": post.get("divergence_rms", 0),
                "delta": post.get("divergence_rms", 0) - pre.get("divergence_rms", 0),
            },
            "divergence_max": {
                "pre": pre.get("divergence_max", 0),
                "post": post.get("divergence_max", 0),
                "delta": post.get("divergence_max", 0) - pre.get("divergence_max", 0),
            },
            "common_mode_rms": {
                "pre": pre.get("common_mode_rms", 0),
                "post": post.get("common_mode_rms", 0),
                "delta": post.get("common_mode_rms", 0) - pre.get("common_mode_rms", 0),
            },
            "l_torque_rms": {
                "pre": pre.get("l_tau_raw_rms", 0),
                "post": post.get("l_tau_raw_rms", 0),
                "delta": post.get("l_tau_raw_rms", 0) - pre.get("l_tau_raw_rms", 0),
            },
            "r_torque_rms": {
                "pre": pre.get("r_tau_raw_rms", 0),
                "post": post.get("r_tau_raw_rms", 0),
                "delta": post.get("r_tau_raw_rms", 0) - pre.get("r_tau_raw_rms", 0),
            },
        }

        results["comparison"][height] = comparison

        print(f"\n{height.upper()}:")
        print(f"  Sign Correct L: {pre.get('sign_correct_left_pct', 0):.1f}% -> {post.get('sign_correct_left_pct', 0):.1f}% (delta: {comparison['sign_correct_left']['delta']:+.1f}%)")
        print(f"  Sign Correct R: {pre.get('sign_correct_right_pct', 0):.1f}% -> {post.get('sign_correct_right_pct', 0):.1f}% (delta: {comparison['sign_correct_right']['delta']:+.1f}%)")
        print(f"  Divergence RMS: {pre.get('divergence_rms', 0):.4f} -> {post.get('divergence_rms', 0):.4f} (delta: {comparison['divergence_rms']['delta']:+.4f})")
        print(f"  Common Mode RMS: {pre.get('common_mode_rms', 0):.4f} -> {post.get('common_mode_rms', 0):.4f} (delta: {comparison['common_mode_rms']['delta']:+.4f})")
        print(f"  L Torque RMS: {pre.get('l_tau_raw_rms', 0):.4f} -> {post.get('l_tau_raw_rms', 0):.4f} (delta: {comparison['l_torque_rms']['delta']:+.4f})")
        print(f"  R Torque RMS: {pre.get('r_tau_raw_rms', 0):.4f} -> {post.get('r_tau_raw_rms', 0):.4f} (delta: {comparison['r_torque_rms']['delta']:+.4f})")

    # Save all results
    results["events"] = all_events
    results["mode_decomposition"] = all_mode_decomp
    results["reference_initialization"] = all_ref_init
    results["correlations"] = all_correlations
    results["torque_sign"] = all_torque_sign

    with open(output_dir / "divergence_after_sign_fix_summary.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\n\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
