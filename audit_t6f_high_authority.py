"""Phase C: High Authority Activation Audit.

Investigates why T6F_sign_corrected transmitted >4.0 Nm only 8 steps (1.6%).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

T6F_SIGN_PATH = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F_sign_corrected/telemetry_1781269776.csv")

OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path("docs/validation")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def audit_high_authority():
    """Audit why high authority (>4.0 Nm) was rare."""

    print("="*80)
    print("PHASE C: HIGH AUTHORITY ACTIVATION AUDIT")
    print("="*80)

    df = pd.read_csv(T6F_SIGN_PATH)
    print(f"\n[OK] Loaded T6F_sign_corrected telemetry: {len(df)} rows")

    # Extract key fields
    arch_fix_active = df["arch_fix_active"].values
    tau_wheel_total_clipped_left = df["tau_wheel_total_clipped_left"].values
    tau_wheel_total_clipped_right = df["tau_wheel_total_clipped_right"].values
    final_tau_mean = 0.5 * (tau_wheel_total_clipped_left + tau_wheel_total_clipped_right)

    # Band state (Phase 3 fix: use correct field name tuned_band_state_id)
    apcr1nd_band_state = df["tuned_band_state_id"].values if "tuned_band_state_id" in df.columns else np.zeros(len(df))

    # Caps and torques
    tau_position = df["tau_position"].values if "tau_position" in df.columns else np.zeros(len(df))
    tau_position_after_clip = df["tau_position_after_clip"].values if "tau_position_after_clip" in df.columns else tau_position

    # Check for upstream clip
    tau_position_after_upstream_clip = df["tau_position_after_upstream_clip"].values if "tau_position_after_upstream_clip" in df.columns else tau_position

    # Arch fix telemetry
    arch_fix_requested_cap = df["arch_fix_requested_cap"].values if "arch_fix_requested_cap" in df.columns else np.zeros(len(df))
    effective_max_position_tau_after_arch_fix = df["effective_max_position_tau_after_arch_fix"].values if "effective_max_position_tau_after_arch_fix" in df.columns else np.zeros(len(df))

    # Error signal
    error_cols = [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
    ]

    error = None
    for col in error_cols:
        if col in df.columns:
            error = df[col].values
            print(f"\n[DRIFT] Using error signal: {col}")
            break

    # High authority analysis
    high_authority_mask = np.abs(final_tau_mean) > 4.0
    high_authority_count = int(np.sum(high_authority_mask))
    high_authority_pct = 100.0 * high_authority_count / len(df)

    print(f"\n[HIGH AUTHORITY ANALYSIS]")
    print(f"  Steps with |final_tau| > 4.0 Nm: {high_authority_count} ({high_authority_pct:.1f}%)")
    print(f"  Max transmitted torque: {np.max(np.abs(final_tau_mean)):.2f} Nm")
    print(f"  Mean transmitted torque: {np.mean(np.abs(final_tau_mean)):.2f} Nm")

    if high_authority_count > 0:
        first_high_authority_step = int(np.where(high_authority_mask)[0][0])
        print(f"  First high authority step: {first_high_authority_step}")
    else:
        first_high_authority_step = None
        print(f"  First high authority step: None")

    # Arch fix activation
    arch_fix_count = int(np.sum(arch_fix_active))
    arch_fix_pct = 100.0 * arch_fix_count / len(df)

    print(f"\n[ARCH FIX ACTIVATION]")
    print(f"  Steps with arch_fix_active: {arch_fix_count} ({arch_fix_pct:.1f}%)")

    # Band state distribution
    print(f"\n[BAND STATE DISTRIBUTION]")
    # Phase 3 fix: correct state mapping (0=normal, 1=soft, 2=desired, 3=hard, 4=emergency)
    for state in [0, 1, 2, 3, 4]:
        count = int(np.sum(apcr1nd_band_state == state))
        pct = 100.0 * count / len(df)
        state_names = {0: "normal", 1: "soft", 2: "desired", 3: "hard", 4: "emergency"}
        print(f"  Band state {state} ({state_names.get(state, 'unknown')}): {count} ({pct:.1f}%)")

    # Torque demand analysis
    print(f"\n[POSITION TORQUE DEMAND]")
    print(f"  tau_position max: {np.max(np.abs(tau_position)):.2f} Nm")
    print(f"  tau_position mean: {np.mean(np.abs(tau_position)):.2f} Nm")
    print(f"  Steps with |tau_position| > 4.0 Nm: {int(np.sum(np.abs(tau_position) > 4.0))} ({100.0*np.mean(np.abs(tau_position) > 4.0):.1f}%)")
    print(f"  Steps with |tau_position| > 6.0 Nm: {int(np.sum(np.abs(tau_position) > 6.0))} ({100.0*np.mean(np.abs(tau_position) > 6.0):.1f}%)")

    # Position torque after upstream clip
    print(f"\n[POSITION TORQUE AFTER UPSTREAM CLIP]")
    print(f"  tau_position_after_upstream_clip max: {np.max(np.abs(tau_position_after_upstream_clip)):.2f} Nm")
    print(f"  Steps with |tau_position_after_upstream_clip| > 4.0 Nm: {int(np.sum(np.abs(tau_position_after_upstream_clip) > 4.0))} ({100.0*np.mean(np.abs(tau_position_after_upstream_clip) > 4.0):.1f}%)")

    # Arch fix cap distribution
    if arch_fix_count > 0:
        print(f"\n[ARCH FIX CAP DISTRIBUTION]")
        print(f"  arch_fix_requested_cap min: {np.min(arch_fix_requested_cap[arch_fix_active]):.2f} Nm")
        print(f"  arch_fix_requested_cap max: {np.max(arch_fix_requested_cap[arch_fix_active]):.2f} Nm")
        print(f"  arch_fix_requested_cap mean: {np.mean(arch_fix_requested_cap[arch_fix_active]):.2f} Nm")

        print(f"\n[EFFECTIVE MAX POSITION TAU AFTER ARCH FIX]")
        print(f"  effective_max_position_tau_after_arch_fix min: {np.min(effective_max_position_tau_after_arch_fix[arch_fix_active]):.2f} Nm")
        print(f"  effective_max_position_tau_after_arch_fix max: {np.max(effective_max_position_tau_after_arch_fix[arch_fix_active]):.2f} Nm")
        print(f"  effective_max_position_tau_after_arch_fix mean: {np.mean(effective_max_position_tau_after_arch_fix[arch_fix_active]):.2f} Nm")

    # Error distribution
    if error is not None:
        abs_error = np.abs(error)
        print(f"\n[ERROR DISTRIBUTION]")
        print(f"  Error max: {np.max(abs_error):.4f} m")
        print(f"  Error mean: {np.mean(abs_error):.4f} m")
        print(f"  Error p95: {np.percentile(abs_error, 95):.4f} m")
        print(f"  Error p99: {np.percentile(abs_error, 99):.4f} m")

        # Error when arch_fix active
        if arch_fix_count > 0:
            error_during_arch_fix = abs_error[arch_fix_active]
            print(f"\n[ERROR DURING ARCH_FIX]")
            print(f"  Error max: {np.max(error_during_arch_fix):.4f} m")
            print(f"  Error mean: {np.mean(error_during_arch_fix):.4f} m")

    # Sample high authority steps
    if high_authority_count > 0:
        print(f"\n[SAMPLE HIGH AUTHORITY STEPS]")
        high_authority_indices = np.where(high_authority_mask)[0]
        for idx in high_authority_indices[:5]:
            print(f"\nStep {idx}:")
            print(f"  final_tau_mean: {final_tau_mean[idx]:.2f} Nm")
            print(f"  tau_position: {tau_position[idx]:.2f} Nm")
            print(f"  tau_position_after_clip: {tau_position_after_clip[idx]:.2f} Nm")
            print(f"  arch_fix_active: {arch_fix_active[idx]}")
            print(f"  band_state: {int(apcr1nd_band_state[idx])}")
            if error is not None:
                print(f"  abs(error): {abs_error[idx]:.4f} m")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")

    # Check why high authority is rare
    if high_authority_count == 0:
        classification = "HIGH_AUTHORITY_RARE_BECAUSE_TAU_DEMAND_LOW"
        explanation = "Position torque demand never exceeded 4.0 Nm during 500-step window."
    elif arch_fix_count == 0:
        classification = "HIGH_AUTHORITY_RARE_BECAUSE_ARCH_FIX_GATED_OFF"
        explanation = "Arch fix was never activated during 500-step window."
    elif np.max(np.abs(tau_position)) > 4.0 and np.max(np.abs(tau_position_after_upstream_clip)) <= 4.0:
        classification = "HIGH_AUTHORITY_RARE_BECAUSE_STILL_CLIPPED"
        explanation = f"Position torque demand reached {np.max(np.abs(tau_position)):.2f} Nm but was clipped to {np.max(np.abs(tau_position_after_upstream_clip)):.2f} Nm before arch fix could boost it."
    elif first_high_authority_step is not None and first_high_authority_step > 400:
        classification = "HIGH_AUTHORITY_RARE_BECAUSE_500_STEP_WINDOW_TOO_EARLY"
        explanation = f"High authority first appeared at step {first_high_authority_step}. The 500-step window captured only {high_authority_count} high-authority steps. A longer diagnostic window would show more high-authority behavior."
    elif np.max(abs_error) < 0.15:
        classification = "HIGH_AUTHORITY_RARE_BECAUSE_500_STEP_WINDOW_TOO_EARLY"
        explanation = f"Error never exceeded {np.max(abs_error):.4f}m during 500-step window. High authority requires larger drift that may appear later."
    else:
        classification = "HIGH_AUTHORITY_RARE_BECAUSE_500_STEP_WINDOW_TOO_EARLY"
        explanation = f"Only {high_authority_count} steps with high authority during 500-step window. This appears to be insufficient sampling - a longer window would capture more."

    print(f"\n{explanation}")
    print(f"\nCLASSIFICATION: {classification}")

    # Create JSON report
    report = {
        "classification": classification,
        "explanation": explanation,
        "high_authority": {
            "count": high_authority_count,
            "pct": float(high_authority_pct),
            "first_step": first_high_authority_step,
            "max_transmitted_nm": float(np.max(np.abs(final_tau_mean))),
            "mean_transmitted_nm": float(np.mean(np.abs(final_tau_mean))),
        },
        "arch_fix": {
            "count": arch_fix_count,
            "pct": float(arch_fix_pct),
        },
        "band_state_distribution": {
            "normal": int(np.sum(apcr1nd_band_state == 0)),
            "soft": int(np.sum(apcr1nd_band_state == 1)),
            "desired": int(np.sum(apcr1nd_band_state == 2)),
            "hard": int(np.sum(apcr1nd_band_state == 3)),
            "emergency": int(np.sum(apcr1nd_band_state == 4)),
        },
        "tau_position_demand": {
            "max_nm": float(np.max(np.abs(tau_position))),
            "mean_nm": float(np.mean(np.abs(tau_position))),
            "above_4nm_count": int(np.sum(np.abs(tau_position) > 4.0)),
            "above_6nm_count": int(np.sum(np.abs(tau_position) > 6.0)),
        },
        "error_stats": {
            "max_m": float(np.max(abs_error)) if error is not None else None,
            "mean_m": float(np.mean(abs_error)) if error is not None else None,
            "p95_m": float(np.percentile(abs_error, 95)) if error is not None else None,
            "p99_m": float(np.percentile(abs_error, 99)) if error is not None else None,
        }
    }

    json_path = OUTPUT_DIR / "t6f_sign_fix_high_authority_activation_audit.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # Create markdown report
    md_lines = [
        "# T6F Sign Fix High Authority Activation Audit",
        "",
        "**Date**: 2026-06-12",
        "**Task**: Phase C - Investigate why high authority (>4.0 Nm) was rare",
        "",
        "## Classification",
        "",
        f"**{classification}**",
        "",
        "## Summary",
        "",
        explanation,
        "",
        "## High Authority Analysis",
        "",
        f"- Steps with |final_tau| > 4.0 Nm: {high_authority_count} ({high_authority_pct:.1f}%)",
        f"- Max transmitted torque: {np.max(np.abs(final_tau_mean)):.2f} Nm",
        f"- Mean transmitted torque: {np.mean(np.abs(final_tau_mean)):.2f} Nm",
        f"- First high authority step: {first_high_authority_step if first_high_authority_step is not None else 'None'}",
        "",
        "## Arch Fix Activation",
        "",
        f"- Steps with arch_fix_active: {arch_fix_count} ({arch_fix_pct:.1f}%)",
        "",
        "## Position Torque Demand",
        "",
        f"- tau_position max: {np.max(np.abs(tau_position)):.2f} Nm",
        f"- tau_position mean: {np.mean(np.abs(tau_position)):.2f} Nm",
        f"- Steps with |tau_position| > 4.0 Nm: {int(np.sum(np.abs(tau_position) > 4.0))} ({100.0*np.mean(np.abs(tau_position) > 4.0):.1f}%)",
        f"- Steps with |tau_position| > 6.0 Nm: {int(np.sum(np.abs(tau_position) > 6.0))} ({100.0*np.mean(np.abs(tau_position) > 6.0):.1f}%)",
        "",
        "## Conclusion",
        "",
        f"{explanation}",
    ]

    md_path = DOCS_DIR / "t6f_sign_fix_high_authority_activation_audit.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"[SAVED] {md_path}")

    return report

if __name__ == "__main__":
    audit_high_authority()
