"""Deep mode decomposition and divergence driver analysis.

Analyzes whether divergence is driven by:
1. Systematic torque bias (wrong PD direction per joint)
2. Per-joint PD creates antisymmetric errors that grow
3. Coupling from sagittal/lateral/height controller
4. Insufficient authority at boundary heights
"""

import csv
import json
from pathlib import Path
from collections import defaultdict


def parse_telemetry(path: Path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def analyze_divergence_drivers(rows: list, height: str) -> dict:
    """Analyze what drives divergence growth."""
    result = {"height": height, "steps": len(rows)}

    # Collect time series
    l_errors, r_errors = [], []
    l_vels, r_vels = [], []
    l_torques, r_torques = [], []
    l_pos, r_pos = [], []
    l_ref, r_ref = [], []
    divergences = []
    common_modes = []
    l_torque_raw, r_torque_raw = [], []

    # Additional signals
    support_errors = []
    height_errors = []
    pitch_vals = []
    roll_vals = []
    wheel_vels = []

    for row in rows:
        try:
            l_e = float(row.get("l_hip_yaw_error", 0))
            r_e = float(row.get("r_hip_yaw_error", 0))
            l_v = float(row.get("l_hip_yaw_vel", 0))
            r_v = float(row.get("r_hip_yaw_vel", 0))
            l_t = float(row.get("l_hip_yaw_tau_shape_raw", 0))
            r_t = float(row.get("r_hip_yaw_tau_shape_raw", 0))
            l_p = float(row.get("l_hip_yaw_pos", 0))
            r_p = float(row.get("r_hip_yaw_pos", 0))
            l_r = float(row.get("l_hip_yaw_ref", 0))
            r_r = float(row.get("r_hip_yaw_ref", 0))

            l_errors.append(l_e)
            r_errors.append(r_e)
            l_vels.append(l_v)
            r_vels.append(r_v)
            l_torques.append(l_t)
            r_torques.append(r_t)
            l_pos.append(l_p)
            r_pos.append(r_p)
            l_ref.append(l_r)
            r_ref.append(r_r)
            divergences.append(l_e - r_e)
            common_modes.append((l_e + r_e) / 2)
            l_torque_raw.append(l_t)
            r_torque_raw.append(r_t)
        except:
            pass

        try:
            support_errors.append(float(row.get("support_position_error_m", 0)))
            height_errors.append(float(row.get("height_error", 0)))
            pitch_vals.append(float(row.get("euler_pitch_y", row.get("robot_pitch_x", 0))))
            roll_vals.append(float(row.get("euler_roll_x", row.get("robot_roll_y", 0))))
            wheel_vels.append(float(row.get("l_wheel_velocity", 0)))
        except:
            pass

    n = len(l_errors)
    if n == 0:
        return {"error": "No valid data"}

    # === Mode Decomposition ===
    # Common mode: (l + r) / 2
    # Divergence mode: (l - r) / 2
    common_errors = [(l + r) / 2 for l, r in zip(l_errors, r_errors)]
    div_errors = [(l - r) / 2 for l, r in zip(l_errors, r_errors)]
    common_torques = [(l + r) / 2 for l, r in zip(l_torque_raw, r_torque_raw)]
    div_torques = [(l - r) / 2 for l, r in zip(l_torque_raw, r_torque_raw)]

    result["common_error"] = {
        "max": max(abs(e) for e in common_errors),
        "rms": (sum(e**2 for e in common_errors) / n) ** 0.5,
        "initial": common_errors[0],
        "final": common_errors[-1],
    }
    result["div_error"] = {
        "max": max(abs(e) for e in div_errors),
        "rms": (sum(e**2 for e in div_errors) / n) ** 0.5,
        "initial": div_errors[0],
        "final": div_errors[-1],
    }
    result["common_torque"] = {
        "max": max(abs(t) for t in common_torques),
        "rms": (sum(t**2 for t in common_torques) / n) ** 0.5,
    }
    result["div_torque"] = {
        "max": max(abs(t) for t in div_torques),
        "rms": (sum(t**2 for t in div_torques) / n) ** 0.5,
    }

    # === Torque-Error Phase Analysis ===
    # Check if divergence torque opposes divergence error
    # If div_error > 0 and div_torque > 0 -> SAME sign = torque is ACCELERATING divergence
    # If div_error > 0 and div_torque < 0 -> OPPOSITE sign = torque is CORRECTING divergence
    n_correct = 0
    n_accelerating = 0
    for de, dt in zip(div_errors, div_torques):
        if abs(de) > 1e-6 and abs(dt) > 1e-6:
            if de * dt < 0:  # Opposite signs = correcting
                n_correct += 1
            else:  # Same signs = accelerating
                n_accelerating += 1

    total_nz = n_correct + n_accelerating
    if total_nz > 0:
        result["div_torque_behavior"] = {
            "correcting_pct": n_correct / total_nz * 100,
            "accelerating_pct": n_accelerating / total_nz * 100,
        }
    else:
        result["div_torque_behavior"] = {"correcting_pct": 0, "accelerating_pct": 0}

    # === Per-Joint Analysis ===
    # Check which joint drives divergence
    l_error_mag = max(abs(e) for e in l_errors) if l_errors else 0
    r_error_mag = max(abs(e) for e in r_errors) if r_errors else 0

    result["l_joint"] = {
        "error_max": l_error_mag,
        "error_final": l_errors[-1] if l_errors else 0,
        "vel_max": max(abs(v) for v in l_vels) if l_vels else 0,
        "torque_max": max(abs(t) for t in l_torque_raw) if l_torque_raw else 0,
        "pos_initial": l_pos[0] if l_pos else 0,
        "pos_final": l_pos[-1] if l_pos else 0,
        "ref_initial": l_ref[0] if l_ref else 0,
        "ref_final": l_ref[-1] if l_ref else 0,
    }
    result["r_joint"] = {
        "error_max": r_error_mag,
        "error_final": r_errors[-1] if r_errors else 0,
        "vel_max": max(abs(v) for v in r_vels) if r_vels else 0,
        "torque_max": max(abs(t) for t in r_torque_raw) if r_torque_raw else 0,
        "pos_initial": r_pos[0] if r_pos else 0,
        "pos_final": r_pos[-1] if r_pos else 0,
        "ref_initial": r_ref[0] if r_ref else 0,
        "ref_final": r_ref[-1] if r_ref else 0,
    }

    # === Divergence Growth Pattern ===
    # Check if divergence is monotonic or oscillatory
    div_deltas = []
    for i in range(1, len(div_errors)):
        div_deltas.append(div_errors[i] - div_errors[i-1])

    positive_deltas = sum(1 for d in div_deltas if d > 0)
    negative_deltas = sum(1 for d in div_deltas if d < 0)

    result["divergence_growth_pattern"] = {
        "total_deltas": len(div_deltas),
        "positive_deltas": positive_deltas,
        "negative_deltas": negative_deltas,
        "growth_ratio": positive_deltas / max(negative_deltas, 1),
        "pattern": "biased_growth" if positive_deltas > negative_deltas * 1.5 else
                   "biased_decline" if negative_deltas > positive_deltas * 1.5 else
                   "oscillatory",
    }

    # === Reference Drift Analysis ===
    # Check if left/right references are drifting apart
    ref_spread_initial = abs(l_ref[0] - r_ref[0]) if l_ref and r_ref else 0
    ref_spread_final = abs(l_ref[-1] - r_ref[-1]) if l_ref and r_ref else 0

    result["reference_drift"] = {
        "initial_spread": ref_spread_initial,
        "final_spread": ref_spread_final,
        "drifting": abs(ref_spread_final - ref_spread_initial) > 0.01,
    }

    # === Velocity Divergence Analysis ===
    div_vels = [l - r for l, r in zip(l_vels, r_vels)]
    result["div_velocity"] = {
        "max": max(abs(v) for v in div_vels) if div_vels else 0,
        "rms": (sum(v**2 for v in div_vels) / len(div_vels)) ** 0.5 if div_vels else 0,
        "initial": div_vels[0] if div_vels else 0,
        "final": div_vels[-1] if div_vels else 0,
    }

    # === Coupling Analysis ===
    # Check correlation between divergence and other signals
    if support_errors and divergences and len(support_errors) == len(divergences):
        corr = pearson_corr(divergences, support_errors)
        result["coupling_support"] = {"correlation": round(corr, 3)}
    else:
        result["coupling_support"] = {"correlation": None}

    if roll_vals and divergences and len(roll_vals) == len(divergences):
        corr = pearson_corr(divergences, roll_vals)
        result["coupling_roll"] = {"correlation": round(corr, 3)}
    else:
        result["coupling_roll"] = {"correlation": None}

    if pitch_vals and divergences and len(pitch_vals) == len(divergences):
        corr = pearson_corr(divergences, pitch_vals)
        result["coupling_pitch"] = {"correlation": round(corr, 3)}
    else:
        result["coupling_pitch"] = {"correlation": None}

    # === Error Integration ===
    # If velocity is nonzero and torque doesn't oppose it, error will grow
    result["velocity_divergence_analysis"] = {
        "div_vel_initial": div_vels[0] if div_vels else 0,
        "div_vel_final": div_vels[-1] if div_vels else 0,
        "div_error_initial": div_errors[0] if div_errors else 0,
        "div_error_final": div_errors[-1] if div_errors else 0,
    }

    return result


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


def compare_pre_post(pre: dict, post: dict, height: str) -> dict:
    """Compare pre-fix vs post-fix analysis."""
    comparison = {"height": height}

    # Key metrics
    keys = ["div_error_max", "div_error_rms", "div_torque_max", "div_torque_rms"]
    for key in keys:
        pre_val = pre.get(key, 0) if isinstance(pre.get(key), (int, float)) else pre.get(key, {}).get("max", 0) if isinstance(pre.get(key), dict) else 0
        post_val = post.get(key, 0) if isinstance(post.get(key), (int, float)) else post.get(key, {}).get("max", 0) if isinstance(post.get(key), dict) else 0

    # Torque behavior change
    pre_behavior = pre.get("div_torque_behavior", {})
    post_behavior = post.get("div_torque_behavior", {})

    comparison["torque_behavior"] = {
        "pre_correcting_pct": pre_behavior.get("correcting_pct", 0),
        "post_correcting_pct": post_behavior.get("correcting_pct", 0),
        "pre_accelerating_pct": pre_behavior.get("accelerating_pct", 0),
        "post_accelerating_pct": post_behavior.get("accelerating_pct", 0),
    }

    # Joint position drift
    l_drift_pre = pre.get("l_joint", {}).get("pos_final", 0) - pre.get("l_joint", {}).get("pos_initial", 0)
    r_drift_pre = pre.get("r_joint", {}).get("pos_final", 0) - pre.get("r_joint", {}).get("pos_initial", 0)
    l_drift_post = post.get("l_joint", {}).get("pos_final", 0) - post.get("l_joint", {}).get("pos_initial", 0)
    r_drift_post = post.get("r_joint", {}).get("pos_final", 0) - post.get("r_joint", {}).get("pos_initial", 0)

    comparison["position_drift"] = {
        "pre_l": round(l_drift_pre, 6),
        "pre_r": round(r_drift_pre, 6),
        "post_l": round(l_drift_post, 6),
        "post_r": round(r_drift_post, 6),
        "pre_div_drift": round(l_drift_pre - r_drift_pre, 6),
        "post_div_drift": round(l_drift_post - r_drift_post, 6),
    }

    # Growth pattern
    comparison["growth_pattern"] = {
        "pre": pre.get("divergence_growth_pattern", {}).get("pattern", "unknown"),
        "post": post.get("divergence_growth_pattern", {}).get("pattern", "unknown"),
        "pre_growth_ratio": pre.get("divergence_growth_pattern", {}).get("growth_ratio", 0),
        "post_growth_ratio": post.get("divergence_growth_pattern", {}).get("growth_ratio", 0),
    }

    return comparison


def main():
    output_dir = Path("outputs/hip_yaw_divergence_after_sign_fix_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    heights = ["nominal", "low_0p300", "high_0p480"]

    for height in heights:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {height.upper()}")
        print(f"{'='*60}")

        # Post-fix
        post_path = Path(f"outputs/hip_yaw_sign_convention_fix/step_e_5000/{height}_5000_telemetry.csv")
        pre_path = Path(f"outputs/step_e_best_current_profile_5000_eval/{height}_5000_telemetry.csv")

        post_rows = parse_telemetry(post_path)
        print(f"Post-fix steps: {len(post_rows)}")

        post_analysis = analyze_divergence_drivers(post_rows, height)
        print(f"  Div Error Max: {post_analysis.get('div_error', {}).get('max', 0):.4f} rad")
        print(f"  Div Error RMS: {post_analysis.get('div_error', {}).get('rms', 0):.4f} rad")
        print(f"  Div Torque Max: {post_analysis.get('div_torque', {}).get('max', 0):.4f} Nm")
        print(f"  Div Torque RMS: {post_analysis.get('div_torque', {}).get('rms', 0):.4f} Nm")
        print(f"  Torque behavior: {post_analysis.get('div_torque_behavior', {})}")
        print(f"  Growth pattern: {post_analysis.get('divergence_growth_pattern', {}).get('pattern', 'unknown')}")

        # Pre-fix
        pre_rows = parse_telemetry(pre_path)
        print(f"\nPre-fix steps: {len(pre_rows)}")

        pre_analysis = analyze_divergence_drivers(pre_rows, height)
        print(f"  Div Error Max: {pre_analysis.get('div_error', {}).get('max', 0):.4f} rad")
        print(f"  Div Error RMS: {pre_analysis.get('div_error', {}).get('rms', 0):.4f} rad")
        print(f"  Div Torque Max: {pre_analysis.get('div_torque', {}).get('max', 0):.4f} Nm")
        print(f"  Div Torque RMS: {pre_analysis.get('div_torque', {}).get('rms', 0):.4f} Nm")
        print(f"  Torque behavior: {pre_analysis.get('div_torque_behavior', {})}")
        print(f"  Growth pattern: {pre_analysis.get('divergence_growth_pattern', {}).get('pattern', 'unknown')}")

        # Compare
        comparison = compare_pre_post(pre_analysis, post_analysis, height)
        print(f"\nComparison:")
        print(f"  Position drift pre:  L={comparison['position_drift']['pre_l']:.4f}, R={comparison['position_drift']['pre_r']:.4f}, div={comparison['position_drift']['pre_div_drift']:.4f}")
        print(f"  Position drift post:  L={comparison['position_drift']['post_l']:.4f}, R={comparison['position_drift']['post_r']:.4f}, div={comparison['position_drift']['post_div_drift']:.4f}")
        print(f"  Growth pattern pre:   {comparison['growth_pattern']['pre']} (ratio={comparison['growth_pattern']['pre_growth_ratio']:.2f})")
        print(f"  Growth pattern post: {comparison['growth_pattern']['post']} (ratio={comparison['growth_pattern']['post_growth_ratio']:.2f})")

        results[height] = {
            "pre": pre_analysis,
            "post": post_analysis,
            "comparison": comparison,
        }

    # Save results
    with open(output_dir / "divergence_driver_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_dir / 'divergence_driver_analysis.json'}")

    return results


if __name__ == "__main__":
    main()
