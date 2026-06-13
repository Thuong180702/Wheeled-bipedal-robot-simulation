#!/usr/bin/env python3
"""
PHASE 1: APCR1g Early Transient Root-Cause Audit
=================================================
Analyze APCR1g vs APCR1f early transient to understand why APCR1g drifts worse
while having better pitch stability.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_telemetry():
    """Load APCR1f and APCR1g telemetry files."""
    apcr1f = pd.read_csv("outputs/hierarchical_controller_sim/telemetry_1781015926.csv")
    apcr1g = pd.read_csv("outputs/hierarchical_controller_sim/telemetry_1781015927.csv")
    return apcr1f, apcr1g


def compute_derivatives(df):
    """Compute e_dot from signed error."""
    df = df.copy()
    df["step"] = range(len(df))
    df["e_dot"] = df["active_pitch_crossing_signed_error_m"].diff() / 1.0
    df["e_dot"] = df["e_dot"].fillna(0)
    return df


def analyze_early_transient(df, profile_name, window_start=0, window_end=500):
    """Analyze early transient behavior in a given window."""
    window = df[(df["step"] >= window_start) & (df["step"] < window_end)].copy()

    e = window["active_pitch_crossing_signed_error_m"]
    e_dot = window["e_dot"]
    pitch = window["pitch_x"] * 180 / np.pi
    apc_active = window["active_pitch_crossing_active"]
    apc_tau = window["active_pitch_crossing_tau"]
    wheel_vel = window["wheel_vel_mean_rad_s"]

    results = {
        "profile": profile_name,
        "window": f"{window_start}-{window_end}",
        "n_steps": len(window),
        "min_e": float(e.min()),
        "max_e": float(e.max()),
        "mean_e": float(e.mean()),
        "final_e": float(e.iloc[-1]),
        "mean_e_dot": float(e_dot.mean()),
        "max_e_dot": float(e_dot.max()),
        "min_e_dot": float(e_dot.min()),
        "e_dot_positive_pct": float((e_dot > 0).mean() * 100),
        "e_dot_negative_pct": float((e_dot < 0).mean() * 100),
        "moving_away_pct": float(((e * e_dot) > 0).mean() * 100),
        "moving_toward_zero_pct": float(((e * e_dot) < 0).mean() * 100),
        "outside_pm0p08_pct": float((e.abs() > 0.08).mean() * 100),
        "outside_pm0p10_pct": float((e.abs() > 0.10).mean() * 100),
        "outside_pm0p12_pct": float((e.abs() > 0.12).mean() * 100),
        "outside_pm0p15_pct": float((e.abs() > 0.15).mean() * 100),
        "pitch_rms": float(np.sqrt((pitch**2).mean())),
        "pitch_max": float(pitch.max()),
        "pitch_min": float(pitch.min()),
        "pitch_mean": float(pitch.mean()),
        "apc_active_pct": float(apc_active.mean() * 100),
        "apc_tau_max": float(apc_tau.abs().max()),
        "apc_tau_mean": float(apc_tau.abs().mean()),
        "apc_tau_positive_pct": float((apc_tau > 0).mean() * 100),
        "apc_tau_negative_pct": float((apc_tau < 0).mean() * 100),
        "wheel_vel_max": float(wheel_vel.abs().max()),
        "wheel_vel_mean": float(wheel_vel.abs().mean()),
    }

    if len(apc_tau) > 1 and np.std(apc_tau.values) > 0 and np.std(e.values) > 0:
        results["tau_e_correlation"] = float(np.corrcoef(apc_tau.values, e.values)[0, 1])
        results["tau_e_dot_correlation"] = float(np.corrcoef(apc_tau.values, e_dot.values)[0, 1])
    else:
        results["tau_e_correlation"] = 0.0
        results["tau_e_dot_correlation"] = 0.0

    return results


def analyze_torque_response_when_moving_away(df, profile_name):
    """Analyze APCR torque response when drift is moving away from zero."""
    df = df.copy()

    # Filter to periods where drift is moving away
    moving_away = df[(df["active_pitch_crossing_signed_error_m"] * df["e_dot"]) > 0.01]

    if len(moving_away) < 10:
        return {
            "profile": profile_name,
            "n_moving_away_steps": len(moving_away),
            "insufficient_data": True,
        }

    # When e > 0 and e_dot > 0
    mask_pos = (moving_away["active_pitch_crossing_signed_error_m"] > 0.05) & (moving_away["e_dot"] > 0.001)
    e_pos_dot_pos = moving_away[mask_pos]

    # When e < 0 and e_dot < 0
    mask_neg = (moving_away["active_pitch_crossing_signed_error_m"] < -0.05) & (moving_away["e_dot"] < -0.001)
    e_neg_dot_neg = moving_away[mask_neg]

    results = {
        "profile": profile_name,
        "n_moving_away_steps": len(moving_away),
        "insufficient_data": False,
        "e_pos_dot_pos_count": len(e_pos_dot_pos),
        "e_pos_dot_pos_mean_tau": float(e_pos_dot_pos["active_pitch_crossing_tau"].mean()) if len(e_pos_dot_pos) > 0 else 0.0,
        "e_pos_dot_pos_tau_opposes_drift_pct": float((e_pos_dot_pos["active_pitch_crossing_tau"] < -0.1).mean() * 100) if len(e_pos_dot_pos) > 0 else 0.0,
        "e_neg_dot_neg_count": len(e_neg_dot_neg),
        "e_neg_dot_neg_mean_tau": float(e_neg_dot_neg["active_pitch_crossing_tau"].mean()) if len(e_neg_dot_neg) > 0 else 0.0,
        "e_neg_dot_neg_tau_opposes_drift_pct": float((e_neg_dot_neg["active_pitch_crossing_tau"] > 0.1).mean() * 100) if len(e_neg_dot_neg) > 0 else 0.0,
    }

    return results


def analyze_wheel_velocity_constraint(df, profile_name):
    """Analyze if wheel velocity is too restricted."""
    df = df.copy()

    early = df[df["step"] < 500]
    late = df[(df["step"] >= 1500) & (df["step"] < 2000)]

    results = {
        "profile": profile_name,
        "wheel_vel_max": float(df["wheel_vel_mean_rad_s"].abs().max()),
        "wheel_vel_mean": float(df["wheel_vel_mean_rad_s"].abs().mean()),
        "wheel_vel_std": float(df["wheel_vel_mean_rad_s"].abs().std()),
        "wheel_vel_max_0_500": float(early["wheel_vel_mean_rad_s"].abs().max()),
        "wheel_vel_mean_0_500": float(early["wheel_vel_mean_rad_s"].abs().mean()),
        "wheel_vel_max_1500_2000": float(late["wheel_vel_mean_rad_s"].abs().max()) if len(late) > 0 else 0.0,
        "wheel_vel_mean_1500_2000": float(late["wheel_vel_mean_rad_s"].abs().mean()) if len(late) > 0 else 0.0,
    }

    if len(df) > 1 and np.std(df["wheel_vel_mean_rad_s"].values) > 0 and np.std(df["active_pitch_crossing_signed_error_m"].values) > 0:
        results["wheel_vel_e_correlation"] = float(np.corrcoef(df["wheel_vel_mean_rad_s"].values, df["active_pitch_crossing_signed_error_m"].values)[0, 1])
    else:
        results["wheel_vel_e_correlation"] = 0.0

    return results


def analyze_phase_brake_behavior(df, profile_name):
    """Analyze phase brake behavior."""
    df = df.copy()

    results = {"profile": profile_name}

    if "phase_recenter_active" in df.columns:
        phase_active = df["phase_recenter_active"]
        results["phase_recenter_active_pct"] = float(phase_active.mean() * 100)
        active_moving_away = df[(phase_active == 1) & ((df["active_pitch_crossing_signed_error_m"] * df["e_dot"]) > 0)]
        results["phase_active_while_moving_away_count"] = len(active_moving_away)
        if len(df[df["phase_recenter_active"] == 1]) > 0:
            results["phase_active_while_moving_away_pct"] = float(len(active_moving_away) / len(df[df["phase_recenter_active"] == 1]) * 100)
        else:
            results["phase_active_while_moving_away_pct"] = 0.0

    if "hysteresis_recenter_active" in df.columns:
        hyst_active = df["hysteresis_recenter_active"]
        results["hysteresis_active_pct"] = float(hyst_active.mean() * 100)
        active_moving_away = df[(hyst_active == 1) & ((df["active_pitch_crossing_signed_error_m"] * df["e_dot"]) > 0)]
        results["hysteresis_active_while_moving_away_count"] = len(active_moving_away)

    return results


def identify_critical_events(df, profile_name):
    """Identify critical events in early transient."""
    df = df.copy()

    events = []

    for threshold in [0.08, 0.10, 0.12, 0.15]:
        exceed = df[df["active_pitch_crossing_signed_error_m"] > threshold]
        if len(exceed) > 0:
            first_exceed = exceed.iloc[0]
            events.append({
                "type": f"first_exceed_{threshold}",
                "step": int(first_exceed["step"]),
                "e": float(first_exceed["active_pitch_crossing_signed_error_m"]),
                "e_dot": float(first_exceed["e_dot"]),
                "apc_tau": float(first_exceed["active_pitch_crossing_tau"]),
            })
        else:
            events.append({"type": f"first_exceed_{threshold}", "step": -1, "e": 0.0, "e_dot": 0.0, "apc_tau": 0.0})

    e_dot_sign = (df["e_dot"] > 0).astype(int)
    e_dot_changes = e_dot_sign.diff().fillna(0)
    reversals = df[e_dot_changes != 0]

    reversal_events = []
    for _, row in reversals.head(20).iterrows():
        step = int(row["step"])
        reversal_events.append({
            "step": step,
            "e": float(row["active_pitch_crossing_signed_error_m"]),
            "e_dot_before": float(df.iloc[step - 1]["e_dot"]) if step > 0 else 0.0,
            "e_dot_after": float(row["e_dot"]),
        })

    return {
        "profile": profile_name,
        "threshold_exceed_events": events,
        "e_dot_reversals": reversal_events,
        "n_e_dot_reversals": len(reversals),
    }


def compare_profiles(apcr1f, apcr1g):
    """Compare APCR1f vs APCR1g early transient behavior."""
    comparison = {}

    for window in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]:
        w_name = f"{window[0]}-{window[1]}"
        f1_results = analyze_early_transient(apcr1f, "APCR1f", window[0], window[1])
        g1_results = analyze_early_transient(apcr1g, "APCR1g", window[0], window[1])

        comparison[f"window_{w_name}"] = {
            "APCR1f": f1_results,
            "APCR1g": g1_results,
            "delta": {
                "max_e_diff": g1_results["max_e"] - f1_results["max_e"],
                "mean_e_diff": g1_results["mean_e"] - f1_results["mean_e"],
                "outside_pm0p15_diff": g1_results["outside_pm0p15_pct"] - f1_results["outside_pm0p15_pct"],
                "pitch_rms_diff": g1_results["pitch_rms"] - f1_results["pitch_rms"],
                "apc_active_diff": g1_results["apc_active_pct"] - f1_results["apc_active_pct"],
                "apc_tau_max_diff": g1_results["apc_tau_max"] - f1_results["apc_tau_max"],
                "wheel_vel_max_diff": g1_results["wheel_vel_max"] - f1_results["wheel_vel_max"],
                "moving_away_diff": g1_results["moving_away_pct"] - f1_results["moving_away_pct"],
            }
        }

    return comparison


def determine_root_cause(apcr1f, apcr1g):
    """Determine the root cause of APCR1g drift regression."""
    f1_0_500 = analyze_early_transient(apcr1f, "APCR1f", 0, 500)
    g1_0_500 = analyze_early_transient(apcr1g, "APCR1g", 0, 500)
    f1_torque = analyze_torque_response_when_moving_away(apcr1f, "APCR1f")
    g1_torque = analyze_torque_response_when_moving_away(apcr1g, "APCR1g")
    f1_wheel = analyze_wheel_velocity_constraint(apcr1f, "APCR1f")
    g1_wheel = analyze_wheel_velocity_constraint(apcr1g, "APCR1g")
    f1_brake = analyze_phase_brake_behavior(apcr1f, "APCR1f")
    g1_brake = analyze_phase_brake_behavior(apcr1g, "APCR1g")

    causes = {
        "WRONG_TORQUE_PHASE": 0,
        "PHASE_BRAKE_TOO_EARLY": 0,
        "PITCH_PRIORITY_OVER_SUPPORT": 0,
        "WHEEL_VELOCITY_TOO_RESTRICTED": 0,
        "PREDICTIVE_OVERSHOOT": 0,
    }

    # Evidence 1: Torque sign analysis
    if g1_torque.get("e_pos_dot_pos_count", 0) > 0:
        tau_opposes_pct = g1_torque.get("e_pos_dot_pos_tau_opposes_drift_pct", 0)
        if tau_opposes_pct < 50:
            causes["WRONG_TORQUE_PHASE"] += 2
        elif tau_opposes_pct < 80:
            causes["WRONG_TORQUE_PHASE"] += 1

    # Evidence 2: Phase brake reducing correction
    if g1_brake.get("phase_active_while_moving_away_count", 0) > 50:
        causes["PHASE_BRAKE_TOO_EARLY"] += 2

    # Evidence 3: Pitch priority over support
    pitch_diff = g1_0_500["pitch_rms"] - f1_0_500["pitch_rms"]
    drift_diff = g1_0_500["max_e"] - f1_0_500["max_e"]
    if pitch_diff < 0 and drift_diff > 0:
        causes["PITCH_PRIORITY_OVER_SUPPORT"] += 2

    # Evidence 4: Wheel velocity too restricted
    wheel_diff = g1_0_500["wheel_vel_max"] - f1_0_500["wheel_vel_max"]
    if wheel_diff < -0.5:
        causes["WHEEL_VELOCITY_TOO_RESTRICTED"] += 2
    elif wheel_diff < 0:
        causes["WHEEL_VELOCITY_TOO_RESTRICTED"] += 1

    # Evidence 5: APCR active more but drift worse
    apc_diff = g1_0_500["apc_active_pct"] - f1_0_500["apc_active_pct"]
    if apc_diff > 20 and drift_diff > 0.1:
        causes["PREDICTIVE_OVERSHOOT"] += 1

    dominant_cause = max(causes, key=causes.get)
    max_score = causes[dominant_cause]

    if max_score == 0:
        classification = "APCR1G_DRIFT_CAUSE_INCONCLUSIVE"
    else:
        classification_map = {
            "WRONG_TORQUE_PHASE": "APCR1G_DRIFT_WORSE_FROM_WRONG_TORQUE_PHASE",
            "PHASE_BRAKE_TOO_EARLY": "APCR1G_DRIFT_WORSE_FROM_PHASE_BRAKE_TOO_EARLY",
            "PITCH_PRIORITY_OVER_SUPPORT": "APCR1G_DRIFT_WORSE_FROM_PITCH_PRIORITY_OVER_SUPPORT",
            "WHEEL_VELOCITY_TOO_RESTRICTED": "APCR1G_DRIFT_WORSE_FROM_WHEEL_VELOCITY_TOO_RESTRICTED",
            "PREDICTIVE_OVERSHOOT": "APCR1G_DRIFT_WORSE_FROM_PREDICTIVE_OVERSHOOT",
        }
        classification = classification_map[dominant_cause]

    return {
        "classification": classification,
        "cause_scores": causes,
        "evidence": {
            "torque_response": {"APCR1f": f1_torque, "APCR1g": g1_torque},
            "wheel_velocity": {"APCR1f": f1_wheel, "APCR1g": g1_wheel},
            "phase_brake": {"APCR1f": f1_brake, "APCR1g": g1_brake},
            "early_transient_0_500": {"APCR1f": f1_0_500, "APCR1g": g1_0_500},
        }
    }


def create_event_csv(apcr1f, apcr1g):
    """Create CSV of critical events for both profiles."""
    events_f1 = identify_critical_events(apcr1f, "APCR1f")
    events_g1 = identify_critical_events(apcr1g, "APCR1g")

    rows = []
    for event in events_f1["threshold_exceed_events"]:
        rows.append({
            "profile": "APCR1f",
            "event_type": event["type"],
            "step": event["step"],
            "e": event["e"],
            "e_dot": event["e_dot"],
            "apc_tau": event["apc_tau"],
        })
    for event in events_g1["threshold_exceed_events"]:
        rows.append({
            "profile": "APCR1g",
            "event_type": event["type"],
            "step": event["step"],
            "e": event["e"],
            "e_dot": event["e_dot"],
            "apc_tau": event["apc_tau"],
        })

    return pd.DataFrame(rows)


def main():
    print("PHASE 1: APCR1g Early Transient Root-Cause Audit")
    print("=" * 60)

    print("\nLoading telemetry...")
    apcr1f, apcr1g = load_telemetry()

    print("Computing derivatives...")
    apcr1f = compute_derivatives(apcr1f)
    apcr1g = compute_derivatives(apcr1g)

    print("Comparing profiles...")
    comparison = compare_profiles(apcr1f, apcr1g)

    print("Determining root cause...")
    root_cause = determine_root_cause(apcr1f, apcr1g)

    print("Creating event CSV...")
    events_df = create_event_csv(apcr1f, apcr1g)

    print("\nSaving outputs...")
    audit = {
        "classification": root_cause["classification"],
        "cause_scores": root_cause["cause_scores"],
        "comparison": comparison,
        "root_cause_analysis": root_cause["evidence"],
    }

    with open(OUTPUT_DIR / "apcr1g_early_transient_root_cause_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    events_df.to_csv(OUTPUT_DIR / "apcr1g_early_transient_events.csv", index=False)

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"\nClassification: {root_cause['classification']}")
    print(f"\nCause Scores:")
    for cause, score in root_cause["cause_scores"].items():
        print(f"  {cause}: {score}")

    print("\nEarly Transient Comparison (0-500 steps):")
    f1 = root_cause["evidence"]["early_transient_0_500"]["APCR1f"]
    g1 = root_cause["evidence"]["early_transient_0_500"]["APCR1g"]

    print(f"\n  {'Metric':<30} {'APCR1f':>12} {'APCR1g':>12} {'Delta':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'max_e (m)':<30} {f1['max_e']:>12.4f} {g1['max_e']:>12.4f} {g1['max_e']-f1['max_e']:>+12.4f}")
    print(f"  {'outside_pm0.15 (%)':<30} {f1['outside_pm0p15_pct']:>12.1f} {g1['outside_pm0p15_pct']:>12.1f} {g1['outside_pm0p15_pct']-f1['outside_pm0p15_pct']:>+12.1f}")
    print(f"  {'pitch_rms (deg)':<30} {f1['pitch_rms']:>12.2f} {g1['pitch_rms']:>12.2f} {g1['pitch_rms']-f1['pitch_rms']:>+12.2f}")
    print(f"  {'apc_active (%)':<30} {f1['apc_active_pct']:>12.1f} {g1['apc_active_pct']:>12.1f} {g1['apc_active_pct']-f1['apc_active_pct']:>+12.1f}")
    print(f"  {'apc_tau_max (Nm)':<30} {f1['apc_tau_max']:>12.3f} {g1['apc_tau_max']:>12.3f} {g1['apc_tau_max']-f1['apc_tau_max']:>+12.3f}")
    print(f"  {'wheel_vel_max (rad/s)':<30} {f1['wheel_vel_max']:>12.2f} {g1['wheel_vel_max']:>12.2f} {g1['wheel_vel_max']-f1['wheel_vel_max']:>+12.2f}")
    print(f"  {'moving_away (%)':<30} {f1['moving_away_pct']:>12.1f} {g1['moving_away_pct']:>12.1f} {g1['moving_away_pct']-f1['moving_away_pct']:>+12.1f}")

    print("\nWheel Velocity Analysis:")
    f1_wheel = root_cause["evidence"]["wheel_velocity"]["APCR1f"]
    g1_wheel = root_cause["evidence"]["wheel_velocity"]["APCR1g"]
    print(f"  {'wheel_vel_max':<25} {f1_wheel['wheel_vel_max']:>10.2f} {g1_wheel['wheel_vel_max']:>10.2f}")
    print(f"  {'wheel_vel_max_0_500':<25} {f1_wheel['wheel_vel_max_0_500']:>10.2f} {g1_wheel['wheel_vel_max_0_500']:>10.2f}")
    print(f"  {'wheel_vel_max_1500_2000':<25} {f1_wheel['wheel_vel_max_1500_2000']:>10.2f} {g1_wheel['wheel_vel_max_1500_2000']:>10.2f}")

    print("\nTorque Response When Moving Away:")
    f1_torque = root_cause["evidence"]["torque_response"]["APCR1f"]
    g1_torque = root_cause["evidence"]["torque_response"]["APCR1g"]
    print(f"  {'n_moving_away':<25} {f1_torque.get('n_moving_away_steps', 0):>10} {g1_torque.get('n_moving_away_steps', 0):>10}")
    print(f"  {'insufficient_data':<25} {f1_torque.get('insufficient_data', False):>10} {g1_torque.get('insufficient_data', False):>10}")
    if not f1_torque.get('insufficient_data', False):
        print(f"  {'e_pos_dot_pos_count':<25} {f1_torque.get('e_pos_dot_pos_count', 0):>10} {g1_torque.get('e_pos_dot_pos_count', 0):>10}")
        print(f"  {'tau_opposes_drift_pct':<25} {f1_torque.get('e_pos_dot_pos_tau_opposes_drift_pct', 0):>10.1f} {g1_torque.get('e_pos_dot_pos_tau_opposes_drift_pct', 0):>10.1f}")

    print("\n" + "=" * 60)
    print("Files saved:")
    print(f"  - {OUTPUT_DIR / 'apcr1g_early_transient_root_cause_audit.json'}")
    print(f"  - {OUTPUT_DIR / 'apcr1g_early_transient_events.csv'}")

    return root_cause["classification"]


if __name__ == "__main__":
    classification = main()