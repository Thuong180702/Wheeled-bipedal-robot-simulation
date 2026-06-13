"""APCR1m blend behavior audit - Phase 5.

Analyzes why APCR1m's blend behavior doesn't prevent drift despite being active.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load APCR1m telemetry
BASE_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
APCR1M_CSV = BASE_DIR / "apcr1m_low_0p300_1000_full_telemetry" / "telemetry.csv"


def main():
    print("=" * 80)
    print("APCR1m BLEND BEHAVIOR AUDIT")
    print("=" * 80)

    df = pd.read_csv(APCR1M_CSV)
    print(f"Loaded {len(df)} rows")

    results = {}

    # 1. Startup Guard Analysis
    print("\n--- STARTUP GUARD ANALYSIS ---")
    startup_guard = df["apcr1m_startup_guard_active"]
    print(f"Startup guard active: {startup_guard.sum()} steps ({startup_guard.mean()*100:.1f}%)")

    # Check pitch blend during startup
    during_startup = df[startup_guard]
    blend_during_startup = during_startup["apcr1m_pitch_blend_active"]
    print(f"Blend active during startup guard: {blend_during_startup.sum()} steps")

    # Check tau_pitch difference during startup
    tau_before_startup = during_startup["apcr1m_tau_pitch_before_blend"]
    tau_after_startup = during_startup["apcr1m_tau_pitch_after_blend"]
    print(f"tau_pitch before (startup): mean={tau_before_startup.mean():.4f}, abs_mean={tau_before_startup.abs().mean():.4f}")
    print(f"tau_pitch after (startup): mean={tau_after_startup.mean():.4f}, abs_mean={tau_after_startup.abs().mean():.4f}")
    print(f"tau_pitch reduction during startup: {(tau_before_startup.abs().mean() - tau_after_startup.abs().mean()):.4f} Nm")

    results["startup_guard"] = {
        "active_steps": int(startup_guard.sum()),
        "active_pct": float(startup_guard.mean() * 100),
        "blend_active_during_guard": int(blend_during_startup.sum()),
        "tau_before_abs_mean_during_guard": float(tau_before_startup.abs().mean()),
        "tau_after_abs_mean_during_guard": float(tau_after_startup.abs().mean()),
    }

    # 2. Blend Activation Analysis
    print("\n--- BLEND ACTIVATION ANALYSIS ---")
    blend_active = df["apcr1m_pitch_blend_active"]
    print(f"Blend active: {blend_active.sum()} steps ({blend_active.mean()*100:.1f}%)")

    # Scale distribution
    scale = df["apcr1m_pitch_blend_scale"]
    print("\nBlend scale distribution:")
    for val in [0.0, 0.25, 0.5, 1.0]:
        count = (scale == val).sum()
        print(f"  scale={val:.2f}: {count} steps ({count/len(df)*100:.1f}%)")

    results["blend_activation"] = {
        "active_steps": int(blend_active.sum()),
        "active_pct": float(blend_active.mean() * 100),
        "scale_0.0_count": int((scale == 0.0).sum()),
        "scale_0.0_pct": float((scale == 0.0).mean() * 100),
        "scale_0.25_count": int((scale == 0.25).sum()),
        "scale_0.5_count": int((scale == 0.5).sum()),
        "scale_1.0_count": int((scale == 1.0).sum()),
    }

    # 3. Block Reason Analysis
    print("\n--- BLEND BLOCK REASON ANALYSIS ---")
    block_reason = df["apcr1m_pitch_blend_block_reason"]
    print("Block reason distribution:")
    for reason, count in block_reason.value_counts().items():
        print(f"  {reason}: {count} steps ({count/len(df)*100:.1f}%)")

    results["block_reason"] = block_reason.value_counts().to_dict()

    # 4. RECENTER State Analysis
    print("\n--- RECENTER STATE ANALYSIS ---")
    recenter_active = df["apcr1m_recenter_active"]
    print(f"RECENTER active: {recenter_active.sum()} steps ({recenter_active.mean()*100:.1f}%)")

    # Get signed error
    signed_error = df["active_pitch_crossing_signed_error_m"]

    # RECENTER from positive vs negative
    recenter_positive = (recenter_active) & (signed_error > 0)
    recenter_negative = (recenter_active) & (signed_error < 0)
    recenter_neutral = recenter_active & (signed_error == 0)

    print(f"RECENTER from positive: {recenter_positive.sum()} steps")
    print(f"RECENTER from negative: {recenter_negative.sum()} steps")
    print(f"RECENTER at neutral: {recenter_neutral.sum()} steps")

    results["recenter_states"] = {
        "total_active_steps": int(recenter_active.sum()),
        "active_pct": float(recenter_active.mean() * 100),
        "from_positive_steps": int(recenter_positive.sum()),
        "from_negative_steps": int(recenter_negative.sum()),
        "at_neutral_steps": int(recenter_neutral.sum()),
    }

    # 5. Safety Gates Analysis
    print("\n--- SAFETY GATES ANALYSIS ---")
    pitch_safe = df["apcr1m_pitch_safe"]
    height_safe = df["apcr1m_height_safe"]
    contact_safe = df["apcr1m_contact_safe"]
    roll_safe = df["apcr1m_roll_safe"]
    pitch_rate_safe = df["apcr1m_pitch_rate_safe"]

    print(f"pitch_safe: {pitch_safe.sum()} steps ({pitch_safe.mean()*100:.1f}%)")
    print(f"height_safe: {height_safe.sum()} steps ({height_safe.mean()*100:.1f}%)")
    print(f"contact_safe: {contact_safe.sum()} steps ({contact_safe.mean()*100:.1f}%)")
    print(f"roll_safe: {roll_safe.sum()} steps ({roll_safe.mean()*100:.1f}%)")
    print(f"pitch_rate_safe: {pitch_rate_safe.sum()} steps ({pitch_rate_safe.mean()*100:.1f}%)")

    # All safe
    all_safe = pitch_safe & height_safe & contact_safe & roll_safe & pitch_rate_safe
    print(f"\nALL safe: {all_safe.sum()} steps ({all_safe.mean()*100:.1f}%)")

    # Breakdown by reason
    unsafe_startup = (startup_guard) & (~all_safe)
    unsafe_not_startup = (~startup_guard) & (~all_safe)

    print(f"\nUnsafe during startup: {unsafe_startup.sum()} steps")
    print(f"Unsafe NOT during startup: {unsafe_not_startup.sum()} steps")

    results["safety_gates"] = {
        "pitch_safe_pct": float(pitch_safe.mean() * 100),
        "height_safe_pct": float(height_safe.mean() * 100),
        "contact_safe_pct": float(contact_safe.mean() * 100),
        "roll_safe_pct": float(roll_safe.mean() * 100),
        "pitch_rate_safe_pct": float(pitch_rate_safe.mean() * 100),
        "all_safe_pct": float(all_safe.mean() * 100),
        "unsafe_during_startup": int(unsafe_startup.sum()),
        "unsafe_not_during_startup": int(unsafe_not_startup.sum()),
    }

    # 6. Tau Pitch Analysis During RECENTER
    print("\n--- TAU PITCH ANALYSIS DURING RECENTER ---")

    recenter_df = df[recenter_active]

    tau_before_recenter = recenter_df["apcr1m_tau_pitch_before_blend"]
    tau_after_recenter = recenter_df["apcr1m_tau_pitch_after_blend"]

    print(f"tau_pitch before blend during RECENTER:")
    print(f"  mean: {tau_before_recenter.mean():.4f} Nm")
    print(f"  abs_mean: {tau_before_recenter.abs().mean():.4f} Nm")
    print(f"  max: {tau_before_recenter.max():.4f} Nm")
    print(f"  min: {tau_before_recenter.min():.4f} Nm")

    print(f"\ntau_pitch after blend during RECENTER:")
    print(f"  mean: {tau_after_recenter.mean():.4f} Nm")
    print(f"  abs_mean: {tau_after_recenter.abs().mean():.4f} Nm")
    print(f"  max: {tau_after_recenter.max():.4f} Nm")
    print(f"  min: {tau_after_recenter.min():.4f} Nm")

    # Reduction
    before_abs = tau_before_recenter.abs().mean()
    after_abs = tau_after_recenter.abs().mean()
    reduction = before_abs - after_abs
    reduction_pct = (reduction / before_abs * 100) if before_abs != 0 else 0

    print(f"\nTau pitch reduction during RECENTER:")
    print(f"  reduction: {reduction:.4f} Nm")
    print(f"  reduction %: {reduction_pct:.1f}%")

    # Steps where tau_pitch still fights APCR
    # If RECENTER is active and tau_pitch has same sign as error, it fights recovery
    tau_fights_apcr = (
        (recenter_active) &
        (
            ((signed_error > 0) & (tau_after_recenter > 0)) |
            ((signed_error < 0) & (tau_after_recenter < 0))
        )
    )
    print(f"\nSteps where tau_pitch fights APCR after blend: {tau_fights_apcr.sum()} ({tau_fights_apcr.mean()*100:.1f}%)")

    results["tau_pitch_during_recenter"] = {
        "before_abs_mean": float(before_abs),
        "after_abs_mean": float(after_abs),
        "reduction_nm": float(reduction),
        "reduction_pct": float(reduction_pct),
        "tau_fights_apcr_steps": int(tau_fights_apcr.sum()),
        "tau_fights_apcr_pct": float(tau_fights_apcr.mean() * 100),
    }

    # 7. Blend effectiveness by error band
    print("\n--- BLEND EFFECTIVENESS BY ERROR BAND ---")

    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15]:
        large_error = abs(signed_error) > threshold
        if large_error.sum() > 0:
            large_df = df[large_error]
            scale_large = large_df["apcr1m_pitch_blend_scale"]
            blend_active_large = large_df["apcr1m_pitch_blend_active"]
            tau_before_large = large_df["apcr1m_tau_pitch_before_blend"]
            tau_after_large = large_df["apcr1m_tau_pitch_after_blend"]

            print(f"\n|e| > {threshold:.2f}m ({large_error.sum()} steps):")
            print(f"  blend active: {blend_active_large.sum()} ({blend_active_large.mean()*100:.1f}%)")
            print(f"  avg scale: {scale_large.mean():.3f}")
            print(f"  tau_before abs_mean: {tau_before_large.abs().mean():.4f} Nm")
            print(f"  tau_after abs_mean: {tau_after_large.abs().mean():.4f} Nm")

    # 8. Drift correlation with blend
    print("\n--- DRIFT CORRELATION WITH BLEND ---")

    # When blend is active, is drift smaller or larger?
    blend_on = df[blend_active]
    blend_off = df[~blend_active]

    error_on = abs(blend_on["active_pitch_crossing_signed_error_m"])
    error_off = abs(blend_off["active_pitch_crossing_signed_error_m"])

    print(f"When blend ON: |e| mean = {error_on.mean():.4f}m, max = {error_on.max():.4f}m")
    print(f"When blend OFF: |e| mean = {error_off.mean():.4f}m, max = {error_off.max():.4f}m")

    results["drift_correlation"] = {
        "blend_on_mean_abs_error": float(error_on.mean()),
        "blend_on_max_abs_error": float(error_on.max()),
        "blend_off_mean_abs_error": float(error_off.mean()),
        "blend_off_max_abs_error": float(error_off.max()),
    }

    # 9. Final classification
    print("\n" + "=" * 80)
    print("BLEND BEHAVIOR CLASSIFICATION")
    print("=" * 80)

    # Check if blend is working
    if results["tau_pitch_during_recenter"]["reduction_pct"] < 10:
        classification = "APCR1M_BLEND_BLOCKED_TOO_OFTEN"
        reason = f"tau_pitch reduction is only {results['tau_pitch_during_recenter']['reduction_pct']:.1f}%"
    elif results["safety_gates"]["all_safe_pct"] < 50:
        classification = "APCR1M_BLEND_SAFETY_GATES_BLOCKING"
        reason = f"Only {results['safety_gates']['all_safe_pct']:.1f}% of steps have all safety gates passing"
    elif results["tau_pitch_during_recenter"]["tau_fights_apcr_pct"] > 20:
        classification = "APCR1M_BLEND_TAU_PITCH_STILL_FIGHTING"
        reason = f"tau_pitch still fights APCR in {results['tau_pitch_during_recenter']['tau_fights_apcr_pct']:.1f}% of RECENTER steps"
    elif blend_active.mean() < 0.3:
        classification = "APCR1M_BLEND_TOO_WEAK"
        reason = f"Blend is only active {blend_active.mean()*100:.1f}% of the time"
    else:
        classification = "APCR1M_BLEND_WORKING_AS_DESIGNED"
        reason = "Blend is functioning but drift is caused by other factors"

    print(f"\nClassification: {classification}")
    print(f"Reason: {reason}")
    print(f"\nKey findings:")
    print(f"  - Startup guard active: {results['startup_guard']['active_pct']:.1f}%")
    print(f"  - Blend active: {results['blend_activation']['active_pct']:.1f}%")
    print(f"  - RECENTER active: {results['recenter_states']['active_pct']:.1f}%")
    print(f"  - All safety gates: {results['safety_gates']['all_safe_pct']:.1f}%")
    print(f"  - tau_pitch reduction: {results['tau_pitch_during_recenter']['reduction_pct']:.1f}%")
    print(f"  - tau_pitch fights APCR: {results['tau_pitch_during_recenter']['tau_fights_apcr_pct']:.1f}%")

    results["classification"] = classification
    results["classification_reason"] = reason

    # Save results
    output_path = BASE_DIR / "apcr1m_pitch_blend_behavior_audit.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
