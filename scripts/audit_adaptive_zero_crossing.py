"""
Phase 1: Audit whether current adaptive_support_centering_trim implements
hold-through-zero recentering behavior.

Uses high_0p480 5000-step telemetry.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

# Path to telemetry
TELEMETRY_PATH = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/adaptive_5000_high_0p480/telemetry_5000.csv")

def parse_csv(path):
    """Parse CSV and return headers and rows."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            rows.append(row)
    return headers, rows

def col_index(headers, name):
    """Find column index by name."""
    idx = headers.index(name)
    print(f"  {name} -> col {idx}")
    return idx

def main():
    print("=" * 80)
    print("PHASE 1: Re-audit current adaptive_support_centering_trim logic")
    print("=" * 80)
    print(f"\nLoading: {TELEMETRY_PATH}")

    headers, rows = parse_csv(TELEMETRY_PATH)
    print(f"Rows: {len(rows)}")

    # Find primary drift column (physical support drift priority)
    print("\n--- Column mapping ---")
    apc_err_idx = col_index(headers, "active_pitch_crossing_signed_error_m")
    sag_err_idx = col_index(headers, "sagittal_position_error_m")
    sup_err_idx = col_index(headers, "support_position_error_m")
    hip_yaw_idx = col_index(headers, "hip_yaw_comp_support_error_m")
    pitch_idx = col_index(headers, "euler_pitch_y")
    roll_idx = col_index(headers, "euler_roll_x")
    step_idx = col_index(headers, "step")
    adaptive_trim_idx = col_index(headers, "adaptive_bias_trim_active")
    adaptive_trim_tau_idx = col_index(headers, "adaptive_bias_tau_nm")
    adaptive_trim_target_idx = col_index(headers, "adaptive_bias_target_tau_nm")
    adaptive_trim_mean_err_idx = col_index(headers, "adaptive_bias_mean_error_m")
    apc_state_idx = col_index(headers, "active_pitch_crossing_state")
    apc_active_idx = col_index(headers, "active_pitch_crossing_active")
    apc_tau_idx = col_index(headers, "active_pitch_crossing_tau")

    # Extract primary drift signal (active_pitch_crossing_signed_error_m is physical)
    print("\n--- Extracting drift signal ---")
    drift_values = []
    sag_values = []
    sup_values = []
    hip_yaw_values = []
    pitch_values = []
    adaptive_trim_active_values = []
    adaptive_trim_tau_values = []
    adaptive_trim_target_values = []
    adaptive_trim_mean_err_values = []
    apc_state_values = []
    apc_active_values = []
    apc_tau_values = []

    for row in rows:
        try:
            drift = float(row[apc_err_idx])
        except (ValueError, IndexError):
            drift = 0.0
        try:
            sag = float(row[sag_err_idx])
        except (ValueError, IndexError):
            sag = 0.0
        try:
            sup = float(row[sup_err_idx])
        except (ValueError, IndexError):
            sup = 0.0
        try:
            hy = float(row[hip_yaw_idx])
        except (ValueError, IndexError):
            hy = 0.0
        try:
            pitch = float(row[pitch_idx])
        except (ValueError, IndexError):
            pitch = 0.0

        try:
            trim_active = row[adaptive_trim_idx].strip().lower() == "true"
        except (ValueError, IndexError):
            trim_active = False
        try:
            trim_tau = float(row[adaptive_trim_tau_idx])
        except (ValueError, IndexError):
            trim_tau = 0.0
        try:
            trim_target = float(row[adaptive_trim_target_idx])
        except (ValueError, IndexError):
            trim_target = 0.0
        try:
            trim_err = float(row[adaptive_trim_mean_err_idx])
        except (ValueError, IndexError):
            trim_err = 0.0

        try:
            apc_state = row[apc_state_idx].strip()
        except (ValueError, IndexError):
            apc_state = "N/A"
        try:
            apc_active = row[apc_active_idx].strip().lower() == "true"
        except (ValueError, IndexError):
            apc_active = False
        try:
            # apc_tau is a scalar float in this column
            apc_tau = float(row[apc_tau_idx])
        except (ValueError, IndexError):
            apc_tau = 0.0

        drift_values.append(drift)
        sag_values.append(sag)
        sup_values.append(sup)
        hip_yaw_values.append(hy)
        pitch_values.append(pitch)
        adaptive_trim_active_values.append(trim_active)
        adaptive_trim_tau_values.append(trim_tau)
        adaptive_trim_target_values.append(trim_target)
        adaptive_trim_mean_err_values.append(trim_err)
        apc_state_values.append(apc_state)
        apc_active_values.append(apc_active)
        apc_tau_values.append(apc_tau)

    n = len(drift_values)
    print(f"Extracted {n} samples")

    # Compute drift statistics
    print("\n" + "=" * 80)
    print("DRIFT STATISTICS (active_pitch_crossing_signed_error_m)")
    print("=" * 80)

    drift = drift_values
    min_d = min(drift)
    max_d = max(drift)
    p2p = max_d - min_d
    max_abs = max(abs(min_d), abs(max_d))
    mean_signed = sum(drift) / n
    median_signed = sorted(drift)[n // 2]
    pos_count = sum(1 for d in drift if d > 0)
    neg_count = sum(1 for d in drift if d < 0)
    zero_count = sum(1 for d in drift if abs(d) < 1e-9)
    pos_pct = pos_count / n * 100
    neg_pct = neg_count / n * 100

    # Zero crossings
    zero_crossings = 0
    for i in range(1, n):
        if (drift[i-1] >= 0 and drift[i] < 0) or (drift[i-1] < 0 and drift[i] >= 0):
            zero_crossings += 1

    # Positive and negative areas
    pos_area = sum(max(d, 0) for d in drift)
    neg_area = sum(abs(min(d, 0)) for d in drift)

    # Symmetry ratio
    sym_ratio = pos_area / neg_area if neg_area > 1e-9 else float('inf')

    # Time inside bands
    inside_003 = sum(1 for d in drift if abs(d) <= 0.03) / n * 100
    inside_005 = sum(1 for d in drift if abs(d) <= 0.05) / n * 100
    inside_008 = sum(1 for d in drift if abs(d) <= 0.08) / n * 100
    outside_008 = sum(1 for d in drift if abs(d) > 0.08) / n * 100
    outside_010 = sum(1 for d in drift if abs(d) > 0.10) / n * 100
    outside_015 = sum(1 for d in drift if abs(d) > 0.15) / n * 100

    print(f"min drift: {min_d:.6f} m")
    print(f"max drift: {max_d:.6f} m")
    print(f"P2P: {p2p:.6f} m")
    print(f"max abs drift: {max_abs:.6f} m")
    print(f"mean signed drift: {mean_signed:.6f} m")
    print(f"median signed drift: {median_signed:.6f} m")
    print(f"positive samples: {pos_count} ({pos_pct:.1f}%)")
    print(f"negative samples: {neg_count} ({neg_pct:.1f}%)")
    print(f"zero samples: {zero_count} ({zero_count/n*100:.1f}%)")
    print(f"zero crossings: {zero_crossings}")
    print(f"positive area: {pos_area:.6f}")
    print(f"negative area: {neg_area:.6f}")
    print(f"symmetry ratio (pos/neg area): {sym_ratio:.3f}")
    print(f"time inside ±0.03: {inside_003:.1f}%")
    print(f"time inside ±0.05: {inside_005:.1f}%")
    print(f"time inside ±0.08: {inside_008:.1f}%")
    print(f"time outside ±0.08: {outside_008:.1f}%")
    print(f"time outside ±0.10: {outside_010:.1f}%")
    print(f"time outside ±0.15: {outside_015:.1f}%")

    # Detect violation episodes
    print("\n" + "=" * 80)
    print("VIOLATION EPISODE DETECTION (|e| > 0.08)")
    print("=" * 80)

    VIOLATION_THRESHOLD = 0.08
    CROSS_TARGET = 0.02

    positive_violations = []  # e > +0.08
    negative_violations = []  # e < -0.08

    in_pos_violation = False
    in_neg_violation = False
    pos_viol_start = None
    neg_viol_start = None
    pos_viol_min_e = None
    neg_viol_max_e = None

    for i, e in enumerate(drift):
        # Positive violation
        if e > VIOLATION_THRESHOLD:
            if not in_pos_violation:
                in_pos_violation = True
                pos_viol_start = i
                pos_viol_min_e = e
            else:
                pos_viol_min_e = min(pos_viol_min_e, e)
        else:
            if in_pos_violation:
                positive_violations.append({
                    'start': pos_viol_start,
                    'end': i - 1,
                    'duration': i - pos_viol_start,
                    'min_error': pos_viol_min_e,
                })
                in_pos_violation = False

        # Negative violation
        if e < -VIOLATION_THRESHOLD:
            if not in_neg_violation:
                in_neg_violation = True
                neg_viol_start = i
                neg_viol_max_e = e
            else:
                neg_viol_max_e = max(neg_viol_max_e, e)  # most negative
        else:
            if in_neg_violation:
                negative_violations.append({
                    'start': neg_viol_start,
                    'end': i - 1,
                    'duration': i - neg_viol_start,
                    'max_error': neg_viol_max_e,
                })
                in_neg_violation = False

    # Flush
    if in_pos_violation:
        positive_violations.append({
            'start': pos_viol_start,
            'end': n - 1,
            'duration': n - pos_viol_start,
            'min_error': pos_viol_min_e,
        })
    if in_neg_violation:
        negative_violations.append({
            'start': neg_viol_start,
            'end': n - 1,
            'duration': n - neg_viol_start,
            'max_error': neg_viol_max_e,
        })

    print(f"Positive violation episodes (e > +0.08): {len(positive_violations)}")
    print(f"Negative violation episodes (e < -0.08): {len(negative_violations)}")

    # Analyze positive violation episodes: did controller command negative correction?
    print("\n--- Positive violation episode analysis ---")
    pos_episodes_with_neg_correction = 0
    pos_episodes_crossed_zero = 0
    pos_episodes_crossed_target = 0
    pos_episodes_released_early = 0

    for ep in positive_violations[:5]:  # Show first 5
        start = ep['start']
        end = ep['end']
        duration = ep['duration']

        # Check correction during this episode
        corrections = adaptive_trim_tau_values[start:min(end+50, n)]  # allow some margin
        has_neg_correction = any(c < -0.01 for c in corrections)
        has_neg_apc = any(c < -0.01 for c in apc_tau_values[start:min(end+50, n)])

        # Check if error ever went negative or crossed target
        episode_drift = drift[start:min(end+100, n)]
        crossed_zero = any(d < 0 for d in episode_drift)
        crossed_target = any(d < -CROSS_TARGET for d in episode_drift)

        if has_neg_correction or has_neg_apc:
            pos_episodes_with_neg_correction += 1

        if crossed_zero:
            pos_episodes_crossed_zero += 1
        if crossed_target:
            pos_episodes_crossed_target += 1

        # Released early = drift still positive after trim released
        if not has_neg_correction and not has_neg_apc:
            pos_episodes_released_early += 1

        print(f"  Episode {ep['start']}-{ep['end']} (dur={duration}): "
              f"min_e={ep['min_error']:.4f}, "
              f"neg_corr={has_neg_correction or has_neg_apc}, "
              f"crossed_0={crossed_zero}, "
              f"crossed_target={crossed_target}")

    # Analyze negative violation episodes: did controller command positive correction?
    print("\n--- Negative violation episode analysis ---")
    neg_episodes_with_pos_correction = 0
    neg_episodes_crossed_zero = 0
    neg_episodes_crossed_target = 0
    neg_episodes_released_early = 0

    for ep in negative_violations[:5]:  # Show first 5
        start = ep['start']
        end = ep['end']
        duration = ep['duration']

        corrections = adaptive_trim_tau_values[start:min(end+50, n)]
        has_pos_correction = any(c > 0.01 for c in corrections)
        has_pos_apc = any(c > 0.01 for c in apc_tau_values[start:min(end+50, n)])

        episode_drift = drift[start:min(end+100, n)]
        crossed_zero = any(d > 0 for d in episode_drift)
        crossed_target = any(d > CROSS_TARGET for d in episode_drift)

        if has_pos_correction or has_pos_apc:
            neg_episodes_with_pos_correction += 1

        if crossed_zero:
            neg_episodes_crossed_zero += 1
        if crossed_target:
            neg_episodes_crossed_target += 1

        if not has_pos_correction and not has_pos_apc:
            neg_episodes_released_early += 1

        print(f"  Episode {ep['start']}-{ep['end']} (dur={duration}): "
              f"max_e={ep['max_error']:.4f}, "
              f"pos_corr={has_pos_correction or has_pos_apc}, "
              f"crossed_0={crossed_zero}, "
              f"crossed_target={crossed_target}")

    # Classification
    print("\n" + "=" * 80)
    print("CLASSIFICATION")
    print("=" * 80)

    hold_through_zero_positive = pos_episodes_crossed_zero > 0 and pos_episodes_with_neg_correction > 0
    hold_through_zero_negative = neg_episodes_crossed_zero > 0 and neg_episodes_with_pos_correction > 0

    hold_until_cross_positive = pos_episodes_crossed_target > 0 and pos_episodes_with_neg_correction > 0
    hold_until_cross_negative = neg_episodes_crossed_target > 0 and neg_episodes_with_pos_correction > 0

    total_pos_eps = len(positive_violations)
    total_neg_eps = len(negative_violations)

    print(f"Positive violations: {total_pos_eps}")
    print(f"  - With negative correction commanded: {pos_episodes_with_neg_correction}")
    print(f"  - Crossed zero: {pos_episodes_crossed_zero}")
    print(f"  - Crossed target (-{CROSS_TARGET}): {pos_episodes_crossed_target}")
    print(f"  - Released early (no correction): {pos_episodes_released_early}")

    print(f"\nNegative violations: {total_neg_eps}")
    print(f"  - With positive correction commanded: {neg_episodes_with_pos_correction}")
    print(f"  - Crossed zero: {neg_episodes_crossed_zero}")
    print(f"  - Crossed target (+{CROSS_TARGET}): {neg_episodes_crossed_target}")
    print(f"  - Released early (no correction): {neg_episodes_released_early}")

    # Determine classification based on clear evidence
    if total_pos_eps == 0 and total_neg_eps == 0:
        classification = "CURRENT_LOGIC_ALREADY_CROSSES_ZERO"
        reason = "No violation episodes detected - drift stays within ±0.08"
    elif pos_pct > 85 and neg_pct < 15 and sym_ratio > 10:
        # Clear evidence: drift is overwhelmingly positive
        classification = "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO"
        reason = f"Drift is overwhelmingly positive ({pos_pct:.1f}%) with symmetry ratio {sym_ratio:.1f}. Controller does NOT force drift to cross to negative side."
    elif pos_episodes_released_early > total_pos_eps * 0.5 or neg_episodes_released_early > total_neg_eps * 0.5:
        classification = "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO"
        reason = "Majority of violation episodes released early without correction or crossing"
    elif pos_episodes_crossed_zero == 0 and neg_episodes_crossed_zero == 0:
        classification = "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO"
        reason = "No violation episodes crossed zero - drift does not oscillate around zero"
    elif pos_episodes_with_neg_correction == 0 and neg_episodes_with_pos_correction == 0:
        classification = "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO"
        reason = "No opposite-direction correction commanded during violations"
    elif pos_episodes_crossed_zero == 0 and total_pos_eps > 0:
        classification = "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO"
        reason = f"Positive violations ({total_pos_eps}) never crossed zero. Drift remains on one side."
    elif hold_through_zero_positive and hold_through_zero_negative:
        classification = "CURRENT_LOGIC_PARTIAL_HOLD_THROUGH_ZERO"
        reason = "Both directions show correction AND crossing - but check if correction held until crossing"
    else:
        classification = "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO"
        reason = "Drift does not exhibit symmetric crossing behavior. A zero-crossing recenter controller is needed."

    print(f"\n** Classification: {classification}")
    print(f"** Reason: {reason}")

    # Save audit results
    audit = {
        "classification": classification,
        "reason": reason,
        "n_samples": n,
        "drift_stats": {
            "min": min_d,
            "max": max_d,
            "p2p": p2p,
            "max_abs": max_abs,
            "mean_signed": mean_signed,
            "median_signed": median_signed,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "zero_crossings": zero_crossings,
            "positive_area": pos_area,
            "negative_area": neg_area,
            "symmetry_ratio": sym_ratio,
            "time_inside_003_pct": inside_003,
            "time_inside_005_pct": inside_005,
            "time_inside_008_pct": inside_008,
            "time_outside_008_pct": outside_008,
            "time_outside_010_pct": outside_010,
            "time_outside_015_pct": outside_015,
        },
        "positive_violations": {
            "count": total_pos_eps,
            "with_neg_correction": pos_episodes_with_neg_correction,
            "crossed_zero": pos_episodes_crossed_zero,
            "crossed_target": pos_episodes_crossed_target,
            "released_early": pos_episodes_released_early,
        },
        "negative_violations": {
            "count": total_neg_eps,
            "with_pos_correction": neg_episodes_with_pos_correction,
            "crossed_zero": neg_episodes_crossed_zero,
            "crossed_target": neg_episodes_crossed_target,
            "released_early": neg_episodes_released_early,
        },
    }

    output_path = Path("docs/validation/zero_crossing_recenter_logic_audit.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# zero_crossing_support_recenter — Logic Audit\n\n")
        f.write(f"**Classification:** `{classification}`\n\n")
        f.write(f"**Profile audited:** `adaptive_support_centering_trim`\n\n")
        f.write(f"**Telemetry:** `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/adaptive_5000_high_0p480/telemetry_5000.csv`\n\n")
        f.write(f"**Steps:** 5000 | **Height:** high_0p480\n\n")
        f.write(f"---\n\n")
        f.write(f"## Drift Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| min drift | {min_d:.6f} m |\n")
        f.write(f"| max drift | {max_d:.6f} m |\n")
        f.write(f"| P2P | {p2p:.6f} m |\n")
        f.write(f"| max abs | {max_abs:.6f} m |\n")
        f.write(f"| mean signed | {mean_signed:.6f} m |\n")
        f.write(f"| median signed | {median_signed:.6f} m |\n")
        f.write(f"| positive % | {pos_pct:.1f}% |\n")
        f.write(f"| negative % | {neg_pct:.1f}% |\n")
        f.write(f"| zero crossings | {zero_crossings} |\n")
        f.write(f"| positive area | {pos_area:.4f} |\n")
        f.write(f"| negative area | {neg_area:.4f} |\n")
        f.write(f"| symmetry ratio | {sym_ratio:.3f} |\n")
        f.write(f"| time inside ±0.03 | {inside_003:.1f}% |\n")
        f.write(f"| time inside ±0.05 | {inside_005:.1f}% |\n")
        f.write(f"| time inside ±0.08 | {inside_008:.1f}% |\n")
        f.write(f"| time outside ±0.08 | {outside_008:.1f}% |\n")
        f.write(f"| time outside ±0.10 | {outside_010:.1f}% |\n")
        f.write(f"| time outside ±0.15 | {outside_015:.1f}% |\n\n")
        f.write(f"---\n\n")
        f.write(f"## Violation Episode Analysis\n\n")
        f.write(f"Threshold: ±{VIOLATION_THRESHOLD} m\n\n")
        f.write(f"### Positive Violations (e > +0.08)\n\n")
        f.write(f"- Episodes: {total_pos_eps}\n")
        f.write(f"- With negative correction: {pos_episodes_with_neg_correction}\n")
        f.write(f"- Crossed zero: {pos_episodes_crossed_zero}\n")
        f.write(f"- Crossed target (-{CROSS_TARGET}): {pos_episodes_crossed_target}\n")
        f.write(f"- Released early: {pos_episodes_released_early}\n\n")
        f.write(f"### Negative Violations (e < -0.08)\n\n")
        f.write(f"- Episodes: {total_neg_eps}\n")
        f.write(f"- With positive correction: {neg_episodes_with_pos_correction}\n")
        f.write(f"- Crossed zero: {neg_episodes_crossed_zero}\n")
        f.write(f"- Crossed target (+{CROSS_TARGET}): {neg_episodes_crossed_target}\n")
        f.write(f"- Released early: {neg_episodes_released_early}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Classification\n\n")
        f.write(f"**{classification}**\n\n")
        f.write(f"{reason}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Conclusion\n\n")
        if classification == "CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO":
            f.write("The current `adaptive_support_centering_trim` does **NOT** implement hold-through-zero recentering.\n\n")
            f.write("**Evidence:**\n")
            if pos_episodes_released_early > 0:
                f.write(f"- {pos_episodes_released_early}/{total_pos_eps} positive violation episodes released without correction\n")
            if neg_episodes_released_early > 0:
                f.write(f"- {neg_episodes_released_early}/{total_neg_eps} negative violation episodes released without correction\n")
            if pos_episodes_crossed_zero == 0 and total_pos_eps > 0:
                f.write(f"- {total_pos_eps} positive violations never crossed zero\n")
            if neg_episodes_crossed_zero == 0 and total_neg_eps > 0:
                f.write(f"- {total_neg_eps} negative violations never crossed zero\n")
            f.write("\n**A new zero-crossing recenter controller is needed.**\n")
        elif classification == "CURRENT_LOGIC_ALREADY_CROSSES_ZERO":
            f.write("The current `adaptive_support_centering_trim` already keeps drift within ±0.08 band.\n")
            f.write("No new zero-crossing controller needed.\n")
        else:
            f.write("Inconclusive - more data needed.\n")

    print(f"\n** Audit report written to: {output_path}")

    return classification, audit

if __name__ == "__main__":
    classification, audit = main()
    print(f"\nFinal classification: {classification}")