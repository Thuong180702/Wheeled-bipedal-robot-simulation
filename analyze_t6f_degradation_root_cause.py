"""Root-cause analysis: Why T6F degrades drift despite transmitting torque.

Performs 6-phase systematic diagnosis:
1. Stepwise T5 vs T6F degradation audit
2. Torque direction and phase audit
3. Cap jump and rate-limit audit
4. Gain mismatch audit
5. Wheel saturation and velocity audit
6. Band logic and release audit
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_telemetry():
    """Load T5 and T6F telemetry."""
    # T5 reference (first 2000 steps)
    t5_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv")
    t5_df = pd.read_csv(t5_path).head(2001)

    # T6F screening
    t6f_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_screening/telemetry_t6f_high_0p480_2000.csv")
    t6f_df = pd.read_csv(t6f_path)

    print(f"Loaded T5: {len(t5_df)} rows")
    print(f"Loaded T6F: {len(t6f_df)} rows")

    return t5_df, t6f_df


def get_drift_column(df):
    """Get primary drift column."""
    priority = [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m"
    ]

    for col in priority:
        if col in df.columns:
            return col

    raise ValueError("No drift column found")


def phase1_stepwise_audit(t5_df, t6f_df, drift_col):
    """Phase 1: Stepwise degradation audit."""
    print("\n" + "="*80)
    print("PHASE 1: Stepwise T5 vs T6F Degradation Audit")
    print("="*80)

    t5_e = t5_df[drift_col].values
    t6f_e = t6f_df[drift_col].values

    # Compute derivatives
    t5_e_dot = np.gradient(t5_e)
    t6f_e_dot = np.gradient(t6f_e)

    # Convergence/divergence
    t5_moving_away = t5_e * t5_e_dot > 0
    t6f_moving_away = t6f_e * t6f_e_dot > 0

    # Event detection
    events = []

    # First crossing thresholds
    for thresh in [0.08, 0.10, 0.15]:
        t5_first = np.where(np.abs(t5_e) > thresh)[0]
        t6f_first = np.where(np.abs(t6f_e) > thresh)[0]

        if len(t5_first) > 0:
            events.append({
                "event": f"T5_first_cross_{thresh:.2f}",
                "step": int(t5_first[0]),
                "error": float(t5_e[t5_first[0]])
            })

        if len(t6f_first) > 0:
            events.append({
                "event": f"T6F_first_cross_{thresh:.2f}",
                "step": int(t6f_first[0]),
                "error": float(t6f_e[t6f_first[0]])
            })

    # Max error
    t5_max_idx = np.argmax(np.abs(t5_e))
    t6f_max_idx = np.argmax(np.abs(t6f_e))

    events.append({
        "event": "T5_max_abs_error",
        "step": int(t5_max_idx),
        "error": float(t5_e[t5_max_idx])
    })

    events.append({
        "event": "T6F_max_abs_error",
        "step": int(t6f_max_idx),
        "error": float(t6f_e[t6f_max_idx])
    })

    # Arch fix transitions (T6F only)
    if "arch_fix_active" in t6f_df.columns:
        arch_active = t6f_df["arch_fix_active"].values
        transitions = np.where(np.diff(arch_active.astype(int)) != 0)[0]

        for idx in transitions[:20]:  # Limit to first 20 transitions
            events.append({
                "event": "arch_fix_transition",
                "step": int(idx),
                "error": float(t6f_e[idx]),
                "active_after": bool(arch_active[idx + 1])
            })

    # e_dot sign changes
    t5_sign_changes = np.where(np.diff(np.sign(t5_e_dot)) != 0)[0]
    t6f_sign_changes = np.where(np.diff(np.sign(t6f_e_dot)) != 0)[0]

    for idx in t5_sign_changes[:10]:
        events.append({
            "event": "T5_e_dot_sign_change",
            "step": int(idx),
            "error": float(t5_e[idx])
        })

    for idx in t6f_sign_changes[:10]:
        events.append({
            "event": "T6F_e_dot_sign_change",
            "step": int(idx),
            "error": float(t6f_e[idx])
        })

    return {
        "events": events,
        "t5_moving_away_pct": float(100.0 * np.sum(t5_moving_away) / len(t5_e)),
        "t6f_moving_away_pct": float(100.0 * np.sum(t6f_moving_away) / len(t6f_e))
    }


def phase2_torque_phase_audit(t5_df, t6f_df, drift_col):
    """Phase 2: Torque direction and phase audit."""
    print("\n" + "="*80)
    print("PHASE 2: Torque Direction and Phase Audit")
    print("="*80)

    t5_e = t5_df[drift_col].values
    t6f_e = t6f_df[drift_col].values
    t5_e_dot = np.gradient(t5_e)
    t6f_e_dot = np.gradient(t6f_e)

    # Get torque columns
    if "tau_position" in t6f_df.columns:
        t6f_tau_pos = t6f_df["tau_position"].values
    elif "apcr1n_tau_position_after_cap" in t6f_df.columns:
        t6f_tau_pos = t6f_df["apcr1n_tau_position_after_cap"].values
    else:
        t6f_tau_pos = np.zeros(len(t6f_df))

    if "final_wheel_tau_with_apc" in t6f_df.columns:
        t6f_final_tau = t6f_df["final_wheel_tau_with_apc"].values
    else:
        t6f_final_tau = np.zeros(len(t6f_df))

    # Direction correctness
    opposes_drift = np.sign(t6f_final_tau) == -np.sign(t6f_e)
    opposes_e_dot = np.sign(t6f_final_tau) == -np.sign(t6f_e_dot)
    converging = t6f_e * t6f_e_dot < 0
    helps_convergence = opposes_drift & converging

    # Phase lag detection
    if "arch_fix_active" in t6f_df.columns:
        arch_active = t6f_df["arch_fix_active"].values.astype(bool)
    else:
        arch_active = np.zeros(len(t6f_df), dtype=bool)

    # Find arch_fix activations and e_dot reversals
    arch_starts = np.where(np.diff(arch_active.astype(int)) == 1)[0]
    e_dot_reversals = np.where(np.diff(np.sign(t6f_e_dot)) != 0)[0]

    # Measure delay from activation to reversal
    delays_activation_to_reversal = []
    for start_idx in arch_starts[:50]:  # Limit analysis
        future_reversals = e_dot_reversals[e_dot_reversals > start_idx]
        if len(future_reversals) > 0:
            delay = future_reversals[0] - start_idx
            delays_activation_to_reversal.append(delay)

    # Overshoot analysis
    overshoot_count = 0
    max_overshoot = 0.0

    for rev_idx in e_dot_reversals:
        if rev_idx < 10 or rev_idx >= len(t6f_e) - 10:
            continue

        # Check if error overshoots past zero after reversal
        e_before = t6f_e[rev_idx]
        e_after = t6f_e[min(rev_idx + 20, len(t6f_e) - 1)]

        if abs(e_after) > abs(e_before) and np.sign(e_before) != np.sign(e_after):
            overshoot = abs(e_after)
            max_overshoot = max(max_overshoot, overshoot)
            overshoot_count += 1

    return {
        "opposes_drift_pct": float(100.0 * np.sum(opposes_drift) / len(t6f_e)),
        "opposes_e_dot_pct": float(100.0 * np.sum(opposes_e_dot) / len(t6f_e)),
        "helps_convergence_pct": float(100.0 * np.sum(helps_convergence) / len(t6f_e)),
        "mean_delay_activation_to_reversal": float(np.mean(delays_activation_to_reversal)) if delays_activation_to_reversal else None,
        "overshoot_count": int(overshoot_count),
        "max_overshoot_m": float(max_overshoot)
    }


def phase3_cap_jump_audit(t5_df, t6f_df, drift_col):
    """Phase 3: Cap jump and rate-limit audit."""
    print("\n" + "="*80)
    print("PHASE 3: Cap Jump and Rate-Limit Audit")
    print("="*80)

    t6f_e = t6f_df[drift_col].values
    t6f_e_dot = np.gradient(t6f_e)

    # Get cap and torque columns
    if "effective_max_position_tau_after_arch_fix" in t6f_df.columns:
        effective_cap = t6f_df["effective_max_position_tau_after_arch_fix"].values
    else:
        effective_cap = 4.0 * np.ones(len(t6f_df))

    if "tau_position" in t6f_df.columns:
        tau_pos = t6f_df["tau_position"].values
    else:
        tau_pos = np.zeros(len(t6f_df))

    if "final_wheel_tau_with_apc" in t6f_df.columns:
        final_tau = t6f_df["final_wheel_tau_with_apc"].values
    else:
        final_tau = np.zeros(len(t6f_df))

    # Cap transitions
    cap_diff = np.diff(effective_cap)
    cap_4_to_65 = np.sum((effective_cap[:-1] == 4.0) & (effective_cap[1:] == 6.5))
    cap_65_to_7 = np.sum((effective_cap[:-1] == 6.5) & (effective_cap[1:] == 7.0))
    cap_7_to_4 = np.sum((effective_cap[:-1] == 7.0) & (effective_cap[1:] == 4.0))

    # Max deltas per step
    max_cap_delta = float(np.max(np.abs(cap_diff)))
    max_tau_pos_delta = float(np.max(np.abs(np.diff(tau_pos))))
    max_final_tau_delta = float(np.max(np.abs(np.diff(final_tau))))

    # Torque jerk (3rd derivative)
    tau_accel = np.diff(np.diff(final_tau))
    tau_jerk = np.diff(tau_accel)

    # Relationship between cap jumps and drift spikes
    large_cap_jumps = np.where(np.abs(cap_diff) > 1.0)[0]
    e_dot_spikes_after_jumps = []

    for jump_idx in large_cap_jumps:
        if jump_idx + 5 < len(t6f_e_dot):
            spike = np.max(np.abs(t6f_e_dot[jump_idx:jump_idx + 5]))
            e_dot_spikes_after_jumps.append(spike)

    return {
        "cap_4_to_65_count": int(cap_4_to_65),
        "cap_65_to_7_count": int(cap_65_to_7),
        "cap_7_to_4_count": int(cap_7_to_4),
        "max_cap_delta_per_step": max_cap_delta,
        "max_tau_pos_delta_per_step": max_tau_pos_delta,
        "max_final_tau_delta_per_step": max_final_tau_delta,
        "tau_jerk_rms": float(np.sqrt(np.mean(tau_jerk**2))) if len(tau_jerk) > 0 else 0.0,
        "mean_e_dot_spike_after_jump": float(np.mean(e_dot_spikes_after_jumps)) if e_dot_spikes_after_jumps else None
    }


def phase4_gain_mismatch_audit(t5_df, t6f_df, drift_col):
    """Phase 4: Gain mismatch audit."""
    print("\n" + "="*80)
    print("PHASE 4: Gain Mismatch Audit")
    print("="*80)

    t5_e = t5_df[drift_col].values
    t6f_e = t6f_df[drift_col].values

    # Get torque before clip
    if "tau_position_before_clip" in t5_df.columns:
        t5_tau_before = t5_df["tau_position_before_clip"].values
    else:
        t5_tau_before = np.zeros(len(t5_df))

    if "tau_position_before_clip" in t6f_df.columns:
        t6f_tau_before = t6f_df["tau_position_before_clip"].values
    else:
        t6f_tau_before = np.zeros(len(t6f_df))

    # Implied gain = tau_before / error (where error != 0)
    t5_nonzero = np.abs(t5_e) > 0.001
    t6f_nonzero = np.abs(t6f_e) > 0.001

    t5_implied_gain = np.abs(t5_tau_before[t5_nonzero]) / np.abs(t5_e[t5_nonzero])
    t6f_implied_gain = np.abs(t6f_tau_before[t6f_nonzero]) / np.abs(t6f_e[t6f_nonzero])

    # Compare by band
    if "arch_fix_active" in t6f_df.columns:
        arch_active = t6f_df["arch_fix_active"].values.astype(bool)
    else:
        arch_active = np.zeros(len(t6f_df), dtype=bool)

    t6f_normal = ~arch_active & t6f_nonzero
    t6f_raised = arch_active & t6f_nonzero

    gain_normal = np.abs(t6f_tau_before[t6f_normal]) / np.abs(t6f_e[t6f_normal]) if np.sum(t6f_normal) > 0 else np.array([])
    gain_raised = np.abs(t6f_tau_before[t6f_raised]) / np.abs(t6f_e[t6f_raised]) if np.sum(t6f_raised) > 0 else np.array([])

    # Check if raised cap multiplies same gain
    if "effective_max_position_tau_after_arch_fix" in t6f_df.columns:
        effective_cap = t6f_df["effective_max_position_tau_after_arch_fix"].values
    else:
        effective_cap = 4.0 * np.ones(len(t6f_df))

    if "tau_position" in t6f_df.columns:
        tau_after = t6f_df["tau_position"].values
    else:
        tau_after = np.zeros(len(t6f_df))

    # Response ratio: tau_after / error
    t6f_response_ratio = np.abs(tau_after[t6f_nonzero]) / np.abs(t6f_e[t6f_nonzero])
    response_normal = np.abs(tau_after[t6f_normal]) / np.abs(t6f_e[t6f_normal]) if np.sum(t6f_normal) > 0 else np.array([])
    response_raised = np.abs(tau_after[t6f_raised]) / np.abs(t6f_e[t6f_raised]) if np.sum(t6f_raised) > 0 else np.array([])

    return {
        "t5_mean_implied_gain": float(np.mean(t5_implied_gain)) if len(t5_implied_gain) > 0 else None,
        "t6f_mean_implied_gain": float(np.mean(t6f_implied_gain)) if len(t6f_implied_gain) > 0 else None,
        "t6f_gain_normal_band": float(np.mean(gain_normal)) if len(gain_normal) > 0 else None,
        "t6f_gain_raised_band": float(np.mean(gain_raised)) if len(gain_raised) > 0 else None,
        "t6f_response_normal_band": float(np.mean(response_normal)) if len(response_normal) > 0 else None,
        "t6f_response_raised_band": float(np.mean(response_raised)) if len(response_raised) > 0 else None,
        "gain_increases_with_cap": bool(len(gain_raised) > 0 and len(gain_normal) > 0 and np.mean(gain_raised) > np.mean(gain_normal))
    }


def phase5_wheel_audit(t5_df, t6f_df, drift_col):
    """Phase 5: Wheel saturation and velocity audit."""
    print("\n" + "="*80)
    print("PHASE 5: Wheel Saturation and Velocity Audit")
    print("="*80)

    t5_e = t5_df[drift_col].values
    t6f_e = t6f_df[drift_col].values
    t6f_e_dot = np.gradient(t6f_e)

    # Wheel velocity
    if "wheel_vel_mean_rad_s" in t5_df.columns:
        t5_wheel_vel = t5_df["wheel_vel_mean_rad_s"].values
    else:
        t5_wheel_vel = np.zeros(len(t5_df))

    if "wheel_vel_mean_rad_s" in t6f_df.columns:
        t6f_wheel_vel = t6f_df["wheel_vel_mean_rad_s"].values
    else:
        t6f_wheel_vel = np.zeros(len(t6f_df))

    t5_wheel_vel_abs = np.abs(t5_wheel_vel)
    t6f_wheel_vel_abs = np.abs(t6f_wheel_vel)

    # Wheel torque
    if "final_wheel_tau_with_apc" in t5_df.columns:
        t5_final_tau = t5_df["final_wheel_tau_with_apc"].values
    else:
        t5_final_tau = np.zeros(len(t5_df))

    if "final_wheel_tau_with_apc" in t6f_df.columns:
        t6f_final_tau = t6f_df["final_wheel_tau_with_apc"].values
    else:
        t6f_final_tau = np.zeros(len(t6f_df))

    # Velocity thresholds
    t5_above_5 = np.sum(t5_wheel_vel_abs > 5.0)
    t6f_above_5 = np.sum(t6f_wheel_vel_abs > 5.0)
    t5_above_6 = np.sum(t5_wheel_vel_abs > 6.0)
    t6f_above_6 = np.sum(t6f_wheel_vel_abs > 6.0)
    t5_above_7 = np.sum(t5_wheel_vel_abs > 7.0)
    t6f_above_7 = np.sum(t6f_wheel_vel_abs > 7.0)

    # Torque near motor cap (assume 7.5 Nm)
    t6f_tau_near_cap = np.sum(np.abs(t6f_final_tau) > 7.0)

    # Velocity sign vs drift sign
    t6f_vel_same_sign_as_e = np.sum(np.sign(t6f_wheel_vel) == np.sign(t6f_e))
    t6f_vel_same_sign_as_e_dot = np.sum(np.sign(t6f_wheel_vel) == np.sign(t6f_e_dot))

    # Wheel velocity continues after torque decreases
    tau_decreasing = np.diff(np.abs(t6f_final_tau)) < 0
    vel_increasing = np.diff(t6f_wheel_vel_abs) > 0
    vel_continues_after_tau_drop = np.sum(tau_decreasing & vel_increasing)

    return {
        "t5_wheel_vel_max": float(np.max(t5_wheel_vel_abs)),
        "t6f_wheel_vel_max": float(np.max(t6f_wheel_vel_abs)),
        "t5_wheel_vel_rms": float(np.sqrt(np.mean(t5_wheel_vel**2))),
        "t6f_wheel_vel_rms": float(np.sqrt(np.mean(t6f_wheel_vel**2))),
        "t5_above_5_count": int(t5_above_5),
        "t6f_above_5_count": int(t6f_above_5),
        "t5_above_6_count": int(t5_above_6),
        "t6f_above_6_count": int(t6f_above_6),
        "t5_above_7_count": int(t5_above_7),
        "t6f_above_7_count": int(t6f_above_7),
        "t6f_tau_near_cap_count": int(t6f_tau_near_cap),
        "t6f_vel_same_sign_as_drift_pct": float(100.0 * t6f_vel_same_sign_as_e / len(t6f_e)),
        "t6f_vel_continues_after_tau_drop_count": int(vel_continues_after_tau_drop)
    }


def phase6_band_logic_audit(t5_df, t6f_df, drift_col):
    """Phase 6: Band logic and release audit."""
    print("\n" + "="*80)
    print("PHASE 6: Band Logic and Release Audit")
    print("="*80)

    t6f_e = t6f_df[drift_col].values
    t6f_e_dot = np.gradient(t6f_e)

    # Band state
    if "apcr1n_tuned_band_state" in t6f_df.columns:
        band_state = t6f_df["apcr1n_tuned_band_state"].values
    else:
        band_state = np.zeros(len(t6f_df))

    if "arch_fix_active" in t6f_df.columns:
        arch_active = t6f_df["arch_fix_active"].values.astype(bool)
    else:
        arch_active = np.zeros(len(t6f_df), dtype=bool)

    # Count by band (assuming: 0=normal, 1=soft, 2=desired, 3=hard, 4=emergency)
    band_counts = {}
    for b in range(5):
        band_counts[f"band_{b}_count"] = int(np.sum(band_state == b))

    # Arch fix active by band
    arch_by_band = {}
    for b in [3, 4]:  # Hard and emergency
        mask = band_state == b
        if np.sum(mask) > 0:
            arch_by_band[f"arch_active_in_band_{b}_pct"] = float(100.0 * np.sum(arch_active & mask) / np.sum(mask))

    # Outside ±0.10 but not hard/emergency
    outside_010 = np.abs(t6f_e) > 0.10
    not_hard_emergency = (band_state < 3)
    outside_but_no_escalation = outside_010 & not_hard_emergency

    # Inside ±0.08 but still high torque
    inside_008 = np.abs(t6f_e) < 0.08
    if "effective_max_position_tau_after_arch_fix" in t6f_df.columns:
        effective_cap = t6f_df["effective_max_position_tau_after_arch_fix"].values
    else:
        effective_cap = 4.0 * np.ones(len(t6f_df))
    high_cap = effective_cap > 4.0
    inside_but_high_cap = inside_008 & high_cap

    # High torque while converging
    converging = t6f_e * t6f_e_dot < 0
    high_torque_while_converging = high_cap & converging

    return {
        **band_counts,
        **arch_by_band,
        "outside_010_but_no_escalation_count": int(np.sum(outside_but_no_escalation)),
        "inside_008_but_high_cap_count": int(np.sum(inside_but_high_cap)),
        "high_torque_while_converging_count": int(np.sum(high_torque_while_converging)),
        "high_torque_while_converging_pct": float(100.0 * np.sum(high_torque_while_converging) / len(t6f_e))
    }


def classify_root_cause(phase1, phase2, phase3, phase4, phase5, phase6):
    """Classify root cause based on all phases."""
    print("\n" + "="*80)
    print("ROOT CAUSE CLASSIFICATION")
    print("="*80)

    evidence = []

    # Phase 2: Torque direction
    if phase2["opposes_drift_pct"] < 70.0:
        evidence.append("WRONG_TORQUE_SIGN")
        print(f"[X] Torque opposes drift only {phase2['opposes_drift_pct']:.1f}% of time")
    else:
        print(f"[OK] Torque direction correct ({phase2['opposes_drift_pct']:.1f}%)")

    # Phase 2: Overshoot
    if phase2["overshoot_count"] > 10:
        evidence.append("PHASE_LAG_OVERSHOOT")
        print(f"[X] Overshoot detected: {phase2['overshoot_count']} events, max {phase2['max_overshoot_m']:.3f} m")
    else:
        print(f"[OK] Overshoot minimal ({phase2['overshoot_count']} events)")

    # Phase 3: Cap jumps
    if phase3["max_cap_delta_per_step"] > 2.0:
        evidence.append("ABRUPT_TORQUE_JUMPS")
        print(f"[X] Large cap jumps: {phase3['max_cap_delta_per_step']:.1f} Nm per step")
    else:
        print(f"[OK] Cap jumps moderate ({phase3['max_cap_delta_per_step']:.1f} Nm)")

    # Phase 4: Gain mismatch
    if phase4.get("t6f_response_raised_band") and phase4.get("t6f_response_normal_band"):
        ratio = phase4["t6f_response_raised_band"] / phase4["t6f_response_normal_band"]
        if ratio > 1.5:
            evidence.append("GAIN_MISMATCH")
            print(f"[X] Response ratio raised/normal: {ratio:.2f} (gain not reduced with raised cap)")
        else:
            print(f"[OK] Response ratio acceptable: {ratio:.2f}")

    # Phase 5: Wheel velocity
    if phase5["t6f_above_6_count"] > phase5["t5_above_6_count"] * 2:
        evidence.append("WHEEL_VELOCITY_OVERSHOOT")
        print(f"[X] T6F wheel velocity >6 rad/s: {phase5['t6f_above_6_count']} vs T5: {phase5['t5_above_6_count']}")
    else:
        print(f"[OK] Wheel velocity comparable to T5")

    # Phase 6: Band logic
    if phase6["high_torque_while_converging_pct"] > 20.0:
        evidence.append("HIGH_TORQUE_HELD_TOO_LONG")
        print(f"[X] High torque while converging: {phase6['high_torque_while_converging_pct']:.1f}%")
    else:
        print(f"[OK] Torque timing acceptable")

    # Classify
    if len(evidence) == 0:
        classification = "T6F_DEGRADATION_INCONCLUSIVE"
    elif len(evidence) == 1:
        classification = f"T6F_DEGRADATION_{evidence[0]}"
    else:
        classification = "T6F_DEGRADATION_MIXED_CAUSES"

    print(f"\n>> Classification: {classification}")
    print(f">> Evidence: {', '.join(evidence) if evidence else 'None identified'}")

    return classification, evidence


def recommend_next_candidate(classification, evidence, phase2, phase3, phase4, phase5, phase6):
    """Recommend next candidate based on root cause."""
    print("\n" + "="*80)
    print("NEXT CANDIDATE RECOMMENDATION")
    print("="*80)

    recommendations = []

    if "GAIN_MISMATCH" in evidence:
        recommendations.append("DESIGN_T6G_GAIN_SCHEDULED_ARCH_FIX")
        print("[+] T6G: Reduce position gain when cap > 4.0")

    if "ABRUPT_TORQUE_JUMPS" in evidence:
        recommendations.append("DESIGN_T6H_RATE_LIMITED_ARCH_FIX")
        print("[+] T6H: Ramp cap transitions, limit torque rate")

    if "HIGH_TORQUE_HELD_TOO_LONG" in evidence or "PHASE_LAG_OVERSHOOT" in evidence:
        recommendations.append("DESIGN_T6I_PHASE_AWARE_DECAY")
        print("[+] T6I: Decay authority after e_dot reversal")

    if "WHEEL_VELOCITY_OVERSHOOT" in evidence:
        recommendations.append("DESIGN_T6J_VELOCITY_BRAKE")
        print("[+] T6J: Phase-aware wheel braking")

    if len(recommendations) == 0:
        if classification == "T6F_DEGRADATION_INCONCLUSIVE":
            return "INCONCLUSIVE_NEED_MORE_TELEMETRY"
        else:
            return "REVERT_TO_T5_NO_NEXT_CANDIDATE"
    elif len(recommendations) == 1:
        return recommendations[0]
    else:
        return "DESIGN_TWO_CANDIDATES"


def main():
    """Main execution."""
    print("="*80)
    print("T6F DEGRADATION ROOT-CAUSE ANALYSIS")
    print("="*80)

    # Load telemetry
    t5_df, t6f_df = load_telemetry()
    drift_col = get_drift_column(t6f_df)
    print(f"Using drift column: {drift_col}")

    # Execute phases
    phase1 = phase1_stepwise_audit(t5_df, t6f_df, drift_col)
    phase2 = phase2_torque_phase_audit(t5_df, t6f_df, drift_col)
    phase3 = phase3_cap_jump_audit(t5_df, t6f_df, drift_col)
    phase4 = phase4_gain_mismatch_audit(t5_df, t6f_df, drift_col)
    phase5 = phase5_wheel_audit(t5_df, t6f_df, drift_col)
    phase6 = phase6_band_logic_audit(t5_df, t6f_df, drift_col)

    # Classify
    classification, evidence = classify_root_cause(phase1, phase2, phase3, phase4, phase5, phase6)
    recommendation = recommend_next_candidate(classification, evidence, phase2, phase3, phase4, phase5, phase6)

    # Save results
    output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "classification": classification,
        "evidence": evidence,
        "recommendation": recommendation,
        "phase1_stepwise": phase1,
        "phase2_torque_phase": phase2,
        "phase3_cap_jump": phase3,
        "phase4_gain_mismatch": phase4,
        "phase5_wheel": phase5,
        "phase6_band_logic": phase6
    }

    json_path = output_dir / "t6f_degradation_root_cause_summary.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVE] Results saved to: {json_path}")

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    print(f"Classification: {classification}")
    print(f"Recommendation: {recommendation}")
    print("="*80)

    return classification, recommendation


if __name__ == "__main__":
    main()

