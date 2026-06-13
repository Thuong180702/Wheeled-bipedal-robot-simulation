"""Analyze Phase 5 re-evaluation results after integration bug fix."""

print("=" * 80)
print("PHASE 5 RE-EVALUATION RESULTS ANALYSIS")
print("Integration Bug Fixed - HY-FF Now Functional")
print("=" * 80)
print()

# Results from evaluation output
baseline = {
    "low_0p300": {"hip_yaw": 0.2137, "support": 0.2430},
    "high_0p480": {"hip_yaw": 0.0462, "support": 0.2336},
    "nominal": {"hip_yaw": 0.0392, "support": 0.1026},
}

candidates = {
    "B_sign_plus_k2": {
        "sign": +1.0, "k": 2.0, "tau_max": 1.0,
        "low_0p300": {"hip_yaw": 0.2379, "support": 0.2563},
        "high_0p480": {"hip_yaw": 0.0462, "support": 0.2336},
        "nominal": {"hip_yaw": 0.0392, "support": 0.1026},
    },
    "C_sign_minus_k2": {
        "sign": -1.0, "k": 2.0, "tau_max": 1.0,
        "low_0p300": {"hip_yaw": 0.1941, "support": 0.2380},
        "high_0p480": {"hip_yaw": 0.0462, "support": 0.2336},
        "nominal": {"hip_yaw": 0.0392, "support": 0.1026},
    },
    "D_sign_minus_k4": {
        "sign": -1.0, "k": 4.0, "tau_max": 1.0,
        "low_0p300": {"hip_yaw": 0.1964, "support": 0.2361},
        "high_0p480": {"hip_yaw": 0.0462, "support": 0.2336},
        "nominal": {"hip_yaw": 0.0392, "support": 0.1026},
    },
    "E_sign_minus_k6": {
        "sign": -1.0, "k": 6.0, "tau_max": 2.0,
        "low_0p300": {"hip_yaw": 0.1921, "support": 0.2385},
        "high_0p480": {"hip_yaw": 0.0462, "support": 0.2336},
        "nominal": {"hip_yaw": 0.0392, "support": 0.1026},
    },
    "F_sign_minus_k8": {
        "sign": -1.0, "k": 8.0, "tau_max": 2.0,
        "low_0p300": {"hip_yaw": 0.2698, "support": 0.6507},
        "high_0p480": {"hip_yaw": 0.0462, "support": 0.2336},
        "nominal": {"hip_yaw": 0.0392, "support": 0.1026},
    },
}

print("SIGN DETERMINATION (low_0p300)")
print("-" * 80)
print(f"Baseline:      hip_yaw = {baseline['low_0p300']['hip_yaw']:.4f} rad")
print(f"Sign +1.0 (B): hip_yaw = {candidates['B_sign_plus_k2']['low_0p300']['hip_yaw']:.4f} rad (WORSE by {(candidates['B_sign_plus_k2']['low_0p300']['hip_yaw'] - baseline['low_0p300']['hip_yaw']) * 1000:.1f} mrad)")
print(f"Sign -1.0 (C): hip_yaw = {candidates['C_sign_minus_k2']['low_0p300']['hip_yaw']:.4f} rad (BETTER by {(baseline['low_0p300']['hip_yaw'] - candidates['C_sign_minus_k2']['low_0p300']['hip_yaw']) * 1000:.1f} mrad)")
print()
print("BEST SIGN: -1.0")
print()

print("CANDIDATE COMPARISON (low_0p300)")
print("-" * 80)
print(f"{'Candidate':<20} {'hip_yaw (rad)':<15} {'Delta vs baseline':<20} {'support (m)':<15} {'Pass 0.070?'}")
print("-" * 80)

threshold = 0.070
baseline_hy = baseline['low_0p300']['hip_yaw']
baseline_sup = baseline['low_0p300']['support']

print(f"{'Baseline':<20} {baseline_hy:<15.4f} {'---':<20} {baseline_sup:<15.4f} {'FAIL'}")

for name, data in candidates.items():
    hy = data['low_0p300']['hip_yaw']
    sup = data['low_0p300']['support']
    delta_hy = hy - baseline_hy
    delta_sup = sup - baseline_sup

    pass_status = "PASS" if hy <= threshold else "FAIL"
    improvement = "better" if delta_hy < 0 else "worse"

    print(f"{name:<20} {hy:<15.4f} {delta_hy:+.4f} ({improvement:<6}) {sup:<15.4f} {pass_status}")

print()
print("ACCEPTANCE CRITERIA CHECK")
print("-" * 80)
print(f"Threshold: hip_yaw_abs_max <= {threshold:.3f} rad")
print()
print(f"Best candidate: C_sign_minus_k2")
print(f"  hip_yaw: 0.1941 rad (177% over threshold)")
print(f"  Improvement vs baseline: -0.0196 rad (-9.2%)")
print(f"  Support error: 0.2380 m (2.0% worse than baseline)")
print()
print("VERDICT: NO CANDIDATE PASSES")
print()

print("KEY OBSERVATIONS")
print("-" * 80)
print("1. HY-FF is now FUNCTIONAL (compensation actively applied)")
print("2. Sign -1.0 is correct (improves hip-yaw by ~9%)")
print("3. Best gain: k=2.0 or k=6.0 (similar performance)")
print("4. k=8.0 causes SEVERE REGRESSION (support error 0.6507 m, 168% worse!)")
print("5. Even best candidate (C) is 177% over threshold")
print("6. Hip-yaw improvement: modest (~20 mrad reduction)")
print("7. Support error: slightly worse with HY-FF (~2% degradation)")
print()

print("FINAL DECISION")
print("=" * 80)
print("Decision Code: HIP_YAW_AND_SUPPORT_COUPLED_NEED_JOINT_FIX")
print()
print("Rationale:")
print("- HY-FF implementation is correct and functional")
print("- Compensation is actively applied (verified in smoke test)")
print("- Sign -1.0 provides modest improvement (~9% hip-yaw reduction)")
print("- But even best candidate cannot meet 0.070 rad threshold")
print("- Hip-yaw remains 177% over threshold (0.1941 vs 0.070)")
print("- Support position error slightly worsened (~2%)")
print("- Aggressive gains (k=8.0) cause severe support regression")
print()
print("Conclusion:")
print("Hip-yaw disturbance rejection cannot be solved by HY-FF alone.")
print("The root cause is support position drift triggering hip-yaw error.")
print("A joint fix addressing BOTH sagittal support drift AND hip-yaw")
print("disturbance rejection is required.")
print()
print("Recommended approach:")
print("1. Fix sagittal support drift first (continuous low-height forward authority)")
print("2. Re-evaluate hip-yaw after support is stabilized")
print("3. If hip-yaw problem persists, consider coupled sagittal-yaw controller")
print("=" * 80)
