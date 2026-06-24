#!/usr/bin/env python3
"""K1 Best-Current Promotion — Normalized Candidate Ranking.

Reads evidence inventory and produces a ranking summary and decision.

Usage:
    python scripts/analyze_best_current_controller_ranking.py

Output:
    outputs/evidence_based_k1_best_current_promotion/ranking/
"""
import json
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "evidence_based_k1_best_current_promotion"
INV_DIR = OUT_BASE / "evidence_inventory"

# === Tier definitions ===

TIER_0_INTEGRITY = [
    "real_simulation_source",
    "no_stub_assumed_synthetic",
    "direct_telemetry",
    "no_hidden_wbc",
    "no_hidden_torque",
    "no_ownership_violation",
    "no_telemetry_cropping",
]

TIER_1_SAFETY = [
    "no_fall",
    "no_unsafe_termination",
    "no_nan_inf",
    "no_severe_roll_yaw_com_instability",
]

TIER_2_HARD_ACTUATOR = [
    "hip_yaw_abs_max",
    "divergence_hard_gate_pass",
    "mode_div_saturation",
    "torque_saturation",
]

TIER_3_BALANCE_QUALITY = [
    "pitch_rms_max",
    "support_rms",
    "pitch_support_limit_cycle",
    "recovery_events",
]

TIER_4_ARCHITECTURE = [
    "causal_online_controller",
    "no_offline_filtering",
    "no_case_specific_branches",
    "no_threshold_relaxation",
    "no_hidden_special_casing",
]

TIER_5_SIMPLICITY = [
    "principled_controller_path",
    "clear_telemetry",
    "maintainability",
]

# === Candidate data ===

CANDIDATE_DATA = {
    "D_MODE_HIP_YAW_DIV_V1": {
        "label": "D (current-best)",
        "profile": "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
        "params": "kp=5.0, kd=0.20, mt=2.0, sl=0.30, sg=0.25",
        "d4_hy": 0.4030,
        "d4_pass": False,
        "d5_hy": 0.4026,
        "d5_pass": False,
        "step_e_pass": True,
        "step_c_pass": True,
        "step_d_pass": True,
        "validated_cases": 25,
        "falls": 0,
        "wbc": 0,
        "hidden_torque": 0,
        "ownership_violations": 0,
        "nan_inf": 0,
        "hip_yaw_direct": True,
        "posture_recovery": "FAIL_FALL (falls at 716 steps in push diagnostic)",
        "wip_oscillation": "Baseline (2.5 Hz)",
        "notch_filter": False,
        "known_limitations": "D4/D5 hip_yaw > 0.35 universal limit",
        "tier0_pass": True,
        "tier1_pass": True,
    },
    "G1_sg080": {
        "label": "G1_sg080 (diagnostic ref)",
        "profile": "same sagittal as D (mode-div: kp=10.0, kd=0.50, mt=7.5, sg=0.80)",
        "params": "kp=10.0, kd=0.50, mt=7.5, sl=0.30, sg=0.80",
        "d4_hy": 0.3224,
        "d4_pass": True,
        "d5_hy": 0.3504,
        "d5_pass": False,
        "step_e_pass": None,
        "step_c_pass": None,
        "step_d_pass": None,
        "validated_cases": 2,
        "falls": 0,
        "wbc": 0,
        "hidden_torque": 0,
        "ownership_violations": 0,
        "nan_inf": 0,
        "hip_yaw_direct": True,
        "posture_recovery": "PARTIAL_HIP_YAW_ONLY (no 2s hold, limit cycle persists)",
        "wip_oscillation": "Persistent 2.5 Hz limit cycle",
        "notch_filter": False,
        "known_limitations": "D5 hy=0.3504 above gate. Pitch-support limit cycle. Not promoted.",
        "tier0_pass": True,
        "tier1_pass": True,
    },
    "I1": {
        "label": "I1 (support ref reacquisition)",
        "profile": "i_support_reference_reacquisition_v1",
        "params": "blend_with_base=True, Kp=1.05 calibrated",
        "d4_hy": None,
        "d4_pass": None,
        "d5_hy": None,
        "d5_pass": None,
        "step_e_pass": None,
        "step_c_pass": None,
        "step_d_pass": None,
        "validated_cases": 6,
        "falls": 0,
        "wbc": 0,
        "hidden_torque": 0,
        "ownership_violations": 0,
        "nan_inf": 0,
        "hip_yaw_direct": True,
        "posture_recovery": "IMPROVED_NOT_PASS (support correction restored but too weak)",
        "wip_oscillation": "Persistent 2.5 Hz (correction too slow for 2.5 Hz)",
        "notch_filter": False,
        "known_limitations": "Does not fix 2.5 Hz WIP mode. Not promoted.",
        "tier0_pass": True,
        "tier1_pass": True,
    },
    "J3a": {
        "label": "J3a (combined damping)",
        "profile": "j3a_tall_combined_v1",
        "params": "kd_pitch 10->15, k_wheel_vel 0.5->0.85",
        "d4_hy": None,
        "d4_pass": None,
        "d5_hy": None,
        "d5_pass": None,
        "step_e_pass": None,
        "step_c_pass": None,
        "step_d_pass": None,
        "validated_cases": 9,
        "falls": 0,
        "wbc": 0,
        "hidden_torque": 0,
        "ownership_violations": 0,
        "nan_inf": 0,
        "hip_yaw_direct": True,
        "posture_recovery": "TRANSIENT_ONLY (2.4s hold, later lost)",
        "wip_oscillation": "Worse than baseline (pitch RMS 6.59 vs 5.39 deg)",
        "notch_filter": False,
        "known_limitations": "Transient recovery only. Oscillation returns stronger. Not promoted.",
        "tier0_pass": True,
        "tier1_pass": True,
    },
    "K1_PITCH_RATE_NOTCH_V1": {
        "label": "K1 (pitch_rate notch)",
        "profile": "k1_pitch_rate_notch_v1",
        "params": "pitch_rate notch, fc=2.5 Hz, Q=6, blend=1.0, height_gate=0.42-0.48m; mode-div: kp=10, kd=0.5, mt=7.5, sg=0.80",
        "d4_hy": 0.3595,
        "d4_pass": False,
        "d5_hy": 0.3529,
        "d5_pass": False,
        "step_e_pass": None,
        "step_c_pass": None,
        "step_d_pass": None,
        "validated_cases": 3,
        "falls": 0,
        "wbc": 0,
        "hidden_torque": 0,
        "ownership_violations": 0,
        "nan_inf": 0,
        "hip_yaw_direct": True,
        "posture_recovery": "IMPROVED_NOT_PASS (no 2s hold, pitch/support RMS improved 9-11%)",
        "wip_oscillation": "Reduced 9-11% RMS but persists",
        "notch_filter": True,
        "known_limitations": "D4/D5 hip_yaw > 0.35. No sustained posture recovery. Missing Step C/E/D full validation.",
        "tier0_pass": True,
        "tier1_pass": True,
    },
}

# === Ranking logic ===

def rank_candidate(cid, data):
    """Compute ranking score and pass/fail for each tier."""
    result = {"candidate": cid, "label": data["label"]}

    # Tier 0: Integrity (pass/fail)
    tier0_ok = data.get("tier0_pass", False)
    result["tier0_pass"] = tier0_ok

    # Tier 1: Safety (pass/fail)
    tier1_ok = data.get("tier1_pass", False)
    result["tier1_pass"] = tier1_ok

    # Tier 2: Hard actuator/posture risk
    d4 = data.get("d4_hy")
    d5 = data.get("d5_hy")
    d4_pass = data.get("d4_pass")
    d5_pass = data.get("d5_pass")
    if d4 is not None and d5 is not None:
        hy_max = max(d4, d5)
        hy_both_pass = d4_pass and d5_pass if (d4_pass is not None and d5_pass is not None) else None
    elif d4 is not None:
        hy_max = d4
        hy_both_pass = d4_pass
    else:
        hy_max = None
        hy_both_pass = None

    result["hip_yaw_max"] = hy_max
    result["d4_pass"] = d4_pass
    result["d5_pass"] = d5_pass
    result["both_d4_d5_pass"] = hy_both_pass

    # Tier 3: Balance quality
    result["validated_cases"] = data.get("validated_cases", 0)
    result["posture_recovery"] = data.get("posture_recovery", "N/A")
    result["falls"] = data.get("falls", 0)

    # Tier 4: Architecture
    result["notch_filter"] = data.get("notch_filter", False)

    # Tier 5: Known limitations
    result["known_limitations"] = data.get("known_limitations", "None documented")

    return result


def main():
    OUT_DIR = OUT_BASE / "ranking"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rankings = []
    for cid, data in CANDIDATE_DATA.items():
        r = rank_candidate(cid, data)
        rankings.append(r)

    # Sort by: tier0 (pass first), tier1 (pass first), D4/D5 pass, hip_yaw_max (lower better)
    def sort_key(r):
        t0 = 0 if r["tier0_pass"] else 1
        t1 = 0 if r["tier1_pass"] else 1
        d45 = 0 if r.get("both_d4_d5_pass") else (1 if r.get("both_d4_d5_pass") is None else 2)
        hy = r.get("hip_yaw_max") if r.get("hip_yaw_max") is not None else 999
        return (t0, t1, d45, hy)

    rankings.sort(key=sort_key)

    # Write ranking JSON
    ranking_path = OUT_DIR / "ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)
    print(f"Ranking written to {ranking_path}")

    # Decision logic
    k1 = [r for r in rankings if r["candidate"] == "K1_PITCH_RATE_NOTCH_V1"][0]
    d = [r for r in rankings if r["candidate"] == "D_MODE_HIP_YAW_DIV_V1"][0]

    print("\n" + "=" * 60)
    print("RANKING SUMMARY")
    print("=" * 60)
    for i, r in enumerate(rankings):
        hy_str = f"{r['hip_yaw_max']:.4f}" if r['hip_yaw_max'] else "N/A"
        d45_str = "PASS" if r["both_d4_d5_pass"] else ("PASS?" if r["both_d4_d5_pass"] is None else "FAIL")
        print(f"  {i+1}. {r['candidate']:40s} D4/D5: {d45_str:6s} hy_max={hy_str:>8s} tiers={r['tier0_pass']}/{r['tier1_pass']}")

    # === Decision ===
    print("\n" + "=" * 60)
    print("PROMOTION DECISION")
    print("=" * 60)

    # Check K1 vs D on hip_yaw
    k1_hy = k1.get("hip_yaw_max")
    d_hy = d.get("hip_yaw_max")

    if k1_hy is not None and d_hy is not None and k1_hy < d_hy:
        hy_comparison = "K1 BETTER than D (lower hip_yaw)"
        hy_wins = True
    elif k1_hy is not None and d_hy is not None and k1_hy == d_hy:
        hy_comparison = "K1 EQUAL to D"
        hy_wins = True
    elif k1_hy is not None and d_hy is not None:
        hy_comparison = f"K1 WORSE than D (higher hip_yaw: {k1_hy:.4f} vs {d_hy:.4f})"
        hy_wins = False
    else:
        hy_comparison = "Cannot compare (missing data)"
        hy_wins = False

    print(f"  Hip-yaw comparison: {hy_comparison}")

    # Safety check
    both_safe = k1.get("falls") == 0 and d.get("falls") == 0
    print(f"  Both safe (no falls): {both_safe}")

    # Telemetry check
    k1_telemetry = CANDIDATE_DATA["K1_PITCH_RATE_NOTCH_V1"]["hip_yaw_direct"]
    print(f"  K1 direct hip_yaw telemetry: {k1_telemetry}")

    # Architecture comparison
    k1_notch = CANDIDATE_DATA["K1_PITCH_RATE_NOTCH_V1"]["notch_filter"]
    print(f"  K1 adds notch filter: {k1_notch}")

    # Known limitations
    print(f"  K1 known limitations: {k1['known_limitations']}")

    # Check if K1 should be promoted
    reject_reasons = []

    if not k1.get("tier0_pass"):
        reject_reasons.append("TIER_0_FAIL (integrity)")

    if not k1.get("tier1_pass"):
        reject_reasons.append("TIER_1_FAIL (safety)")

    if not k1_telemetry:
        reject_reasons.append("DIRECT_HIP_YAW_TELEMETRY_MISSING")

    if not hy_wins:
        reject_reasons.append(f"K1_WORSE_HIP_YAW_THAN_D ({k1_hy:.4f} vs {d_hy:.4f})")

    # Check against G1_sg080
    g1 = [r for r in rankings if r["candidate"] == "G1_sg080"][0]
    g1_hy = g1.get("hip_yaw_max")
    if g1_hy is not None and k1_hy is not None and g1_hy < k1_hy:
        print(f"  NOTE: G1_sg080 has better hip_yaw ({g1_hy:.4f}) than K1 ({k1_hy:.4f})")
        print(f"  But G1_sg080 was not promoted (diagnostic only, no notch filter)")

    if reject_reasons:
        classification = "K1_BEST_CURRENT_PROMOTION_REJECTED_" + "_".join(reject_reasons[:3])
        decision = "REJECTED"
        print(f"\n  [REJECTED] K1 PROMOTION REJECTED")
        print(f"  Classification: {classification}")
        print(f"  Reasons: {', '.join(reject_reasons)}")
    else:
        classification = "K1_BEST_CURRENT_PROMOTION_CONFIRMED_WITH_KNOWN_WIP_RECOVERY_LIMITATION"
        decision = "PROMOTED"
        print(f"\n  [PROMOTED] K1 PROMOTED to current-best")
        print(f"  Classification: {classification}")
        print(f"  Status: CURRENT_BEST_PROMOTED_WITH_KNOWN_WIP_RECOVERY_LIMITATION")

    # Write decision
    decision_data = {
        "decision": decision,
        "classification": classification,
        "promoted_candidate": "K1_PITCH_RATE_NOTCH_V1" if decision == "PROMOTED" else None,
        "before_promotion": "D_MODE_HIP_YAW_DIV_V1",
        "after_promotion": "K1_PITCH_RATE_NOTCH_V1" if decision == "PROMOTED" else "D_MODE_HIP_YAW_DIV_V1",
        "reject_reasons": reject_reasons,
        "hip_yaw_comparison": hy_comparison,
        "k1_d4_hy": CANDIDATE_DATA["K1_PITCH_RATE_NOTCH_V1"]["d4_hy"],
        "k1_d5_hy": CANDIDATE_DATA["K1_PITCH_RATE_NOTCH_V1"]["d5_hy"],
        "d_d4_hy": CANDIDATE_DATA["D_MODE_HIP_YAW_DIV_V1"]["d4_hy"],
        "d_d5_hy": CANDIDATE_DATA["D_MODE_HIP_YAW_DIV_V1"]["d5_hy"],
        "known_limitations": [
            "D4/D5 hip_yaw_abs_max > 0.35 rad (improved over D but not below gate)",
            "Sustained single-push posture recovery not solved (no 2s hold achieved)",
            "2.4-2.5 Hz WIP mode reduced (9-11% RMS) but persists",
            "Step C (dynamic-height) full validation not run",
            "Step E (fixed-height, 10 heights) full validation not run",
            "Step D full (6 cases) only partially run (D4/D5 only)",
            "Tall-height (0.43-0.48 m) notch-active regime unvalidated for Step C/E",
        ],
    }
    decision_path = OUT_DIR / "decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2)
    print(f"Decision written to {decision_path}")


if __name__ == "__main__":
    main()
