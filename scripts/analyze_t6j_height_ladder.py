"""Phase 9D: T6J Height Ladder Analysis — compare with T6I baseline.

Reads T6J telemetry from t6j_height_ladder_2000_<label>/telemetry_2000.csv,
computes drift and stability metrics, and compares with T6I baseline data.
"""
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
T6J_MANIFEST = OUT_BASE / "t6j_height_ladder_setup_manifest.json"
T6I_SUMMARY = OUT_BASE / "t6i_height_ladder_2000_summary.json"
T6I_METRICS = OUT_BASE / "t6i_height_ladder_2000_metrics.csv"

# Physical drift column priority
DRIFT_COLS = [
    "active_pitch_crossing_signed_error_m",
    "sagittal_position_error_m",
    "support_position_error_m",
    "hip_yaw_comp_support_error_m",
]

# T6J-specific telemetry columns
T6J_COLS = [
    "t6j_bias_trim_active",
    "t6j_bias_safety_gate_pass",
    "t6j_bias_expected_direction_correct",
    "t6j_bias_trim_tau_nm",
    "t6j_bias_block_reason",
    "t6j_bias_mean_error_m",
]


def load_t6i_summary():
    with open(T6I_SUMMARY) as f:
        return {e["label"]: e for e in json.load(f)}


def find_drift_col(rows):
    for col in DRIFT_COLS:
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
        if vals and any(abs(v) > 1e-9 for v in vals):
            return col
    return DRIFT_COLS[0]


def compute_metrics(rows, label):
    drift_col = find_drift_col(rows)
    errors = [float(r[drift_col]) for r in rows]
    abs_errors = [abs(e) for e in errors]
    n = len(rows)

    # Basic drift metrics
    min_e = min(errors)
    max_e = max(errors)
    max_abs = max(abs_errors)
    final = errors[-1]
    mean = sum(errors) / n
    mean_abs = sum(abs_errors) / n
    p2p = max_e - min_e
    pos = sum(1 for e in errors if e > 0.001) / n * 100
    neg = sum(1 for e in errors if e < -0.001) / n * 100
    zero_x = sum(1 for i in range(1, n) if errors[i-1] * errors[i] < 0)
    out_008 = sum(1 for e in abs_errors if e > 0.08) / n * 100
    out_010 = sum(1 for e in abs_errors if e > 0.10) / n * 100
    out_015 = sum(1 for e in abs_errors if e > 0.15) / n * 100

    # Late-run (last 500)
    late = errors[-500:]
    late_mean = sum(late) / len(late)
    late_out_008 = sum(1 for e in late if abs(e) > 0.08) / len(late) * 100

    # Pitch/roll
    pitch = [float(r["pitch_x"]) for r in rows]
    roll = [float(r["roll_y"]) for r in rows]
    pitch_max = max(abs(v) for v in pitch)
    pitch_rms = math.sqrt(sum(v*v for v in pitch) / n)
    roll_max = max(abs(v) for v in roll)
    roll_rms = math.sqrt(sum(v*v for v in roll) / n)

    # Wheel velocity
    wv = [float(r["wheel_vel_mean_rad_s"]) for r in rows]
    wv_max = max(abs(v) for v in wv)
    wv_rms = math.sqrt(sum(v*v for v in wv) / n)
    wv_gt5 = sum(1 for v in wv if abs(v) > 5)
    wv_gt6 = sum(1 for v in wv if abs(v) > 6)
    wv_gt7 = sum(1 for v in wv if abs(v) > 7)

    # Contact / height
    contact_l = sum(1 for r in rows if float(r.get("contact_L", 1)) > 0.5) / n
    contact_r = sum(1 for r in rows if float(r.get("contact_R", 1)) > 0.5) / n
    com_z = [float(r["com_z"]) for r in rows if r.get("com_z") not in (None, "", "nan")]
    com_z_min = min(com_z) if com_z else 0
    com_z_mean = sum(com_z) / len(com_z) if com_z else 0
    com_z_max = max(com_z) if com_z else 0

    # Stability checks
    terminated = any(r.get("_terminated", "") == "True" for r in rows)
    wbc_flag = any(r.get("wbc_active", "") == "True" for r in rows)
    hidden_tau_vals = [abs(float(r.get("hidden_torque_max_nm", 0))) for r in rows]
    hidden_torque_max = max(hidden_tau_vals)
    own_viol_vals = [abs(float(r.get("ownership_violation_max", 0))) for r in rows]
    ownership_violation_max = max(own_viol_vals)

    # T6J-specific
    t6j_active = sum(1 for r in rows if r.get("t6j_bias_trim_active", "") == "True") / n * 100
    t6j_safety = sum(1 for r in rows if r.get("t6j_bias_safety_gate_pass", "") == "True") / n * 100
    t6j_dir = sum(1 for r in rows if r.get("t6j_bias_expected_direction_correct", "") == "True") / n * 100
    t6j_tau = [float(r.get("t6j_bias_trim_tau_nm", 0)) for r in rows]
    t6j_tau_min = min(t6j_tau)
    t6j_tau_max = max(t6j_tau)

    # T6J block reasons
    block_reasons = {}
    for r in rows:
        reason = r.get("t6j_bias_block_reason", "")
        if reason:
            block_reasons[reason] = block_reasons.get(reason, 0) + 1

    # Positive/negative duration
    pos_steps = sum(1 for e in errors if e > 0.001)
    neg_steps = sum(1 for e in errors if e < -0.001)

    # Survival
    survived = n
    if terminated:
        survived = n  # counts rows before termination

    # Classification
    classification = classify(
        survived=survived,
        max_abs=max_abs,
        final=final,
        out_008=out_008,
        out_010=out_010,
        out_015=out_015,
        wbc_flag=wbc_flag,
        hidden_torque_max=hidden_torque_max,
        ownership_violation_max=ownership_violation_max,
        terminated=terminated,
        label=label,
    )

    result = {
        "label": label,
        "drift_column": drift_col,
        "survived_steps": survived,
        "terminated": terminated,
        "min_error": min_e,
        "max_error": max_e,
        "max_abs_error": max_abs,
        "final_error": final,
        "mean_error": mean,
        "mean_abs_error": mean_abs,
        "p2p": p2p,
        "positive_pct": pos,
        "negative_pct": neg,
        "positive_steps": pos_steps,
        "negative_steps": neg_steps,
        "zero_crossings": zero_x,
        "outside_0p08_pct": out_008,
        "outside_0p10_pct": out_010,
        "outside_0p15_pct": out_015,
        "late_mean_error": late_mean,
        "late_outside_0p08_pct": late_out_008,
        "contact_l": contact_l,
        "contact_r": contact_r,
        "com_z_min": com_z_min,
        "com_z_mean": com_z_mean,
        "com_z_max": com_z_max,
        "pitch_max_deg": math.degrees(pitch_max),
        "pitch_rms_deg": math.degrees(pitch_rms),
        "roll_max_deg": math.degrees(roll_max),
        "roll_rms_deg": math.degrees(roll_rms),
        "wheel_vel_max": wv_max,
        "wheel_vel_rms": wv_rms,
        "wheel_vel_gt5": wv_gt5,
        "wheel_vel_gt6": wv_gt6,
        "wheel_vel_gt7": wv_gt7,
        "wbc_flag": wbc_flag,
        "hidden_torque_max": hidden_torque_max,
        "ownership_violation_max": ownership_violation_max,
        "t6j_active_pct": t6j_active,
        "t6j_safety_pct": t6j_safety,
        "t6j_direction_correct_pct": t6j_dir,
        "t6j_tau_min": t6j_tau_min,
        "t6j_tau_max": t6j_tau_max,
        "t6j_block_reasons": block_reasons,
        "classification": classification,
    }

    print(f"\n{'='*60}")
    print(f"ANALYSIS {label}")
    print(f"  Drift col: {drift_col}")
    print(f"  Survived: {survived}/{n}")
    print(f"  Max abs error: {max_abs:.4f}m")
    print(f"  Final error: {final:+.4f}m")
    print(f"  Mean error: {mean:+.4f}m")
    print(f"  Mean abs error: {mean_abs:.4f}m")
    print(f"  P2P: {p2p:.4f}m")
    print(f"  Outside ±0.08: {out_008:.1f}%")
    print(f"  Outside ±0.10: {out_010:.1f}%")
    print(f"  Outside ±0.15: {out_015:.1f}%")
    print(f"  Positive %: {pos:.1f}%")
    print(f"  Late mean: {late_mean:+.4f}m")
    print(f"  T6J active %: {t6j_active:.1f}%")
    print(f"  T6J safety %: {t6j_safety:.1f}%")
    print(f"  T6J direction correct %: {t6j_dir:.1f}%")
    print(f"  T6J tau range: [{t6j_tau_min:.4f}, {t6j_tau_max:.4f}] Nm")
    print(f"  Classification: {classification}")
    print(f"{'='*60}")

    return result


def classify(survived, max_abs, final, out_008, out_010, out_015,
             wbc_flag, hidden_torque_max, ownership_violation_max,
             terminated, label):
    """Classify a single height variant result."""
    # Hard failure: fall or WBC/hidden/ownership violation
    if terminated or wbc_flag or hidden_torque_max > 0.01 or ownership_violation_max > 0:
        return f"T6J_HEIGHT_{label.upper()}_2000_FAIL_STABILITY"

    # Hard failure: exceeded max_abs threshold by more than 0.002m
    if max_abs > 0.252:
        return f"T6J_HEIGHT_{label.upper()}_2000_FAIL_DRIFT"

    # Marginal: exceeded by <= 0.002m AND self-corrects
    if max_abs > 0.250:
        return f"T6J_HEIGHT_{label.upper()}_2000_PASS_WITH_MONITORING"

    # Normal pass
    return f"T6J_HEIGHT_{label.upper()}_2000_PASS"


def compare_with_t6i(t6j_result, t6i_data):
    """Compute delta vs T6I for shared metrics."""
    label = t6j_result["label"]
    if label not in t6i_data:
        return None

    t6i = t6i_data[label]
    # T6I summary uses mean_abs_error, not mean_error
    deltas = {
        "max_abs_error_delta": t6j_result["max_abs_error"] - t6i["max_abs_error"],
        "final_error_delta": t6j_result["final_error"] - t6i["final_error"],
        "mean_abs_error_delta": t6j_result["mean_abs_error"] - t6i["mean_abs_error"],
        "p2p_delta": t6j_result["p2p"] - t6i["p2p"],
        "outside_0p08_delta": t6j_result["outside_0p08_pct"] - t6i["outside_0p08_pct"],
        "outside_0p10_delta": t6j_result["outside_0p10_pct"] - t6i["outside_0p10_pct"],
        "outside_0p15_delta": t6j_result["outside_0p15_pct"] - t6i["outside_0p15_pct"],
        "t6i_classification": t6i["classification"],
    }
    return deltas


def write_metrics_csv(results, path):
    """Write per-setup metrics as CSV."""
    if not results:
        return
    fields = [
        "label", "survived_steps", "terminated", "drift_column",
        "max_abs_error", "final_error", "mean_error", "mean_abs_error",
        "p2p", "positive_pct", "negative_pct", "zero_crossings",
        "outside_0p08_pct", "outside_0p10_pct", "outside_0p15_pct",
        "late_mean_error", "late_outside_0p08_pct",
        "contact_l", "contact_r",
        "com_z_min", "com_z_mean", "com_z_max",
        "pitch_max_deg", "pitch_rms_deg", "roll_max_deg", "roll_rms_deg",
        "wheel_vel_max", "wheel_vel_rms", "wheel_vel_gt7",
        "wbc_flag", "hidden_torque_max", "ownership_violation_max",
        "t6j_active_pct", "t6j_safety_pct", "t6j_direction_correct_pct",
        "t6j_tau_min", "t6j_tau_max",
        "classification",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nMetrics CSV written: {path}")


def write_comparison_csv(results, deltas, t6i_data, path):
    """Write T6J vs T6I comparison as CSV."""
    if not deltas:
        return
    fields = [
        "label",
        "t6j_max_abs", "t6i_max_abs", "max_abs_delta",
        "t6j_final", "t6i_final", "final_delta",
        "t6j_mae", "t6i_mae", "mae_delta",
        "t6j_out_008", "t6i_out_008", "out_008_delta",
        "t6j_out_010", "t6i_out_010", "out_010_delta",
        "t6j_out_015", "t6i_out_015", "out_015_delta",
        "t6i_classification", "t6j_classification",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            label = r["label"]
            if label not in deltas or deltas[label] is None:
                continue
            d = deltas[label]
            t6i_entry = t6i_data.get(label, {})
            row = {
                "label": label,
                "t6j_max_abs": r["max_abs_error"],
                "t6i_max_abs": t6i_entry.get("max_abs_error"),
                "max_abs_delta": d["max_abs_error_delta"],
                "t6j_final": r["final_error"],
                "t6i_final": t6i_entry.get("final_error"),
                "final_delta": d["final_error_delta"],
                "t6j_mae": r["mean_abs_error"],
                "t6i_mae": t6i_entry.get("mean_abs_error"),
                "mae_delta": d["mean_abs_error_delta"],
                "t6j_out_008": r["outside_0p08_pct"],
                "t6i_out_008": t6i_entry.get("outside_0p08_pct"),
                "out_008_delta": d["outside_0p08_delta"],
                "t6j_out_010": r["outside_0p10_pct"],
                "t6i_out_010": t6i_entry.get("outside_0p10_pct"),
                "out_010_delta": d["outside_0p10_delta"],
                "t6j_out_015": r["outside_0p15_pct"],
                "t6i_out_015": t6i_entry.get("outside_0p15_pct"),
                "out_015_delta": d["outside_0p15_delta"],
                "t6i_classification": d["t6i_classification"],
                "t6j_classification": r["classification"],
            }
            writer.writerow(row)
    print(f"Comparison CSV written: {path}")


def main():
    # Load T6I baseline
    t6i_data = load_t6i_summary()
    print(f"Loaded T6I baseline for {len(t6i_data)} setups")

    # Load manifest
    with open(T6J_MANIFEST) as f:
        manifest = json.load(f)
    print(f"Manifest has {len(manifest)} entries")

    results = []
    deltas = {}
    completed = 0
    failed = 0
    skipped = 0

    for entry in manifest:
        label = entry["label"]
        telemetry_path = entry.get("telemetry_path", "")
        launch_status = entry.get("launch_status", "pending")

        if launch_status == "skipped_missing":
            skipped += 1
            print(f"\nSKIP {label}: setup missing")
            results.append({
                "label": label,
                "survived_steps": 0,
                "terminated": None,
                "max_abs_error": None,
                "final_error": None,
                "mean_error": None,
                "mean_abs_error": None,
                "p2p": None,
                "positive_pct": None,
                "negative_pct": None,
                "zero_crossings": None,
                "outside_0p08_pct": None,
                "outside_0p10_pct": None,
                "outside_0p15_pct": None,
                "late_mean_error": None,
                "late_outside_0p08_pct": None,
                "contact_l": None,
                "contact_r": None,
                "com_z_min": None,
                "com_z_mean": None,
                "com_z_max": None,
                "pitch_max_deg": None,
                "pitch_rms_deg": None,
                "roll_max_deg": None,
                "roll_rms_deg": None,
                "wheel_vel_max": None,
                "wheel_vel_rms": None,
                "wheel_vel_gt5": None,
                "wheel_vel_gt6": None,
                "wheel_vel_gt7": None,
                "wbc_flag": None,
                "hidden_torque_max": None,
                "ownership_violation_max": None,
                "t6j_active_pct": None,
                "t6j_safety_pct": None,
                "t6j_direction_correct_pct": None,
                "t6j_tau_min": None,
                "t6j_tau_max": None,
                "t6j_block_reasons": {},
                "classification": f"T6J_HEIGHT_{label.upper()}_2000_SETUP_MISSING",
            })
            continue

        if launch_status in ("failed", "error", "timeout"):
            failed += 1
            print(f"\nFAIL {label}: launch failed ({launch_status})")
            results.append({
                "label": label,
                "survived_steps": 0,
                "terminated": None,
                "classification": f"T6J_HEIGHT_{label.upper()}_2000_LAUNCH_FAILED",
            })
            deltas[label] = None
            continue

        if not telemetry_path or not Path(telemetry_path).exists():
            print(f"\nPENDING {label}: telemetry not yet available")
            results.append({
                "label": label,
                "survived_steps": 0,
                "terminated": None,
                "classification": f"T6J_HEIGHT_{label.upper()}_2000_RUNNING",
            })
            continue

        # Load telemetry
        with open(telemetry_path) as f:
            rows = list(csv.DictReader(f))

        n = len(rows)
        print(f"\n[{completed+failed+skipped+1}/{len(manifest)}] Analyzing {label}: {n} rows")

        result = compute_metrics(rows, label)
        results.append(result)
        completed += 1

        delta = compare_with_t6i(result, t6i_data)
        deltas[label] = delta

        if delta:
            print(f"  vs T6I:")
            print(f"    max_abs delta: {delta['max_abs_error_delta']:+.4f} m")
            print(f"    final delta: {delta['final_error_delta']:+.4f} m")
            print(f"    mean delta (MAE): {delta['mean_abs_error_delta']:+.4f} m")
            print(f"    outside ±0.08 delta: {delta['outside_0p08_delta']:+.1f} pp")
            print(f"    outside ±0.10 delta: {delta['outside_0p10_delta']:+.1f} pp")
            print(f"    outside ±0.15 delta: {delta['outside_0p15_delta']:+.1f} pp")

    # Write outputs
    metrics_path = OUT_BASE / "t6j_height_ladder_2000_metrics.csv"
    comparison_path = OUT_BASE / "t6j_height_ladder_2000_comparison.csv"
    summary_path = OUT_BASE / "t6j_height_ladder_2000_summary.json"

    write_metrics_csv(results, metrics_path)

    # Write comparison CSV (custom, simpler format)
    write_comparison_csv(results, deltas, t6i_data, comparison_path)

    # Write summary JSON
    summary_data = {
        "results": results,
        "deltas": deltas,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary JSON written: {summary_path}")

    # Print classification table
    print(f"\n{'='*60}")
    print("CLASSIFICATION TABLE")
    print(f"{'='*60}")
    print(f"{'Label':<15} {'Max Abs':>9} {'Final':>9} {'OOB±0.10':>9} {'T6J%':>7} {'Class'}")
    print(f"{'-'*60}")
    for r in results:
        max_a = f"{r['max_abs_error']:.4f}" if r['max_abs_error'] is not None else "N/A"
        fin = f"{r['final_error']:+.4f}" if r['final_error'] is not None else "N/A"
        oob = f"{r['outside_0p10_pct']:.1f}%" if r['outside_0p10_pct'] is not None else "N/A"
        t6j = f"{r['t6j_active_pct']:.0f}%" if r.get('t6j_active_pct') is not None else "N/A"
        print(f"{r['label']:<15} {max_a:>9} {fin:>9} {oob:>9} {t6j:>7} {r['classification']}")

    print(f"\nTotal: {completed} completed, {failed} failed, {skipped} skipped")

    return results, deltas


if __name__ == "__main__":
    main()
